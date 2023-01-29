from ray.rllib.algorithms.ddpg.ddpg_tf_policy import *
from deer.agents.xadqn.xadqn_torch_policy import xa_postprocess_nstep_and_prio

def loss(self, model, dist_class, train_batch):
	twin_q = self.config["twin_q"]
	gamma = self.config["gamma"]
	n_step = self.config["n_step"]
	use_huber = self.config["use_huber"]
	huber_threshold = self.config["huber_threshold"]
	l2_reg = self.config["l2_reg"]

	input_dict = SampleBatch(
		obs=train_batch[SampleBatch.CUR_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=True
	)
	input_dict_next = SampleBatch(
		obs=train_batch[SampleBatch.NEXT_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=True
	)

	model_out_t, _ = model(input_dict, [], None)
	model_out_tp1, _ = model(input_dict_next, [], None)
	target_model_out_tp1, _ = self.target_model(input_dict_next, [], None)

	self._target_q_func_vars = self.target_model.variables()

	# Policy network evaluation.
	policy_t = model.get_policy_output(model_out_t)
	policy_tp1 = self.target_model.get_policy_output(target_model_out_tp1)

	# Action outputs.
	if self.config["smooth_target_policy"]:
		target_noise_clip = self.config["target_noise_clip"]
		clipped_normal_sample = tf.clip_by_value(
			tf.random.normal(
				tf.shape(policy_tp1), stddev=self.config["target_noise"]
			),
			-target_noise_clip,
			target_noise_clip,
		)
		policy_tp1_smoothed = tf.clip_by_value(
			policy_tp1 + clipped_normal_sample,
			self.action_space.low * tf.ones_like(policy_tp1),
			self.action_space.high * tf.ones_like(policy_tp1),
		)
	else:
		# No smoothing, just use deterministic actions.
		policy_tp1_smoothed = policy_tp1

	# Q-net(s) evaluation.
	# prev_update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
	# Q-values for given actions & observations in given current
	q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

	# Q-values for current policy (no noise) in given current state
	q_t_det_policy = model.get_q_values(model_out_t, policy_t)

	if twin_q:
		twin_q_t = model.get_twin_q_values(
			model_out_t, train_batch[SampleBatch.ACTIONS]
		)

	# Target q-net(s) evaluation.
	q_tp1 = self.target_model.get_q_values(
		target_model_out_tp1, policy_tp1_smoothed
	)

	if twin_q:
		twin_q_tp1 = self.target_model.get_twin_q_values(
			target_model_out_tp1, policy_tp1_smoothed
		)

	q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
	if twin_q:
		twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
		q_tp1 = tf.minimum(q_tp1, twin_q_tp1)

	q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
	q_tp1_best_masked = (
		1.0 - tf.cast(train_batch[SampleBatch.DONES], tf.float32)
	) * q_tp1_best

	# Compute RHS of bellman equation.
	q_t_selected_target = tf.stop_gradient(
		tf.cast(train_batch[SampleBatch.REWARDS], tf.float32)
		+ gamma**n_step * q_tp1_best_masked
	)

	# Compute the error (potentially clipped).
	if twin_q:
		td_error = q_t_selected - q_t_selected_target
		twin_td_error = twin_q_t_selected - q_t_selected_target
		if use_huber:
			errors = huber_loss(td_error, huber_threshold) + huber_loss(
				twin_td_error, huber_threshold
			)
		else:
			errors = 0.5 * tf.math.square(td_error) + 0.5 * tf.math.square(
				twin_td_error
			)
	else:
		td_error = q_t_selected - q_t_selected_target
		if use_huber:
			errors = huber_loss(td_error, huber_threshold)
		else:
			errors = 0.5 * tf.math.square(td_error)

	prio_weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
	critic_loss = tf.reduce_mean(prio_weights * errors)
	actor_loss = -tf.reduce_mean(prio_weights * q_t_det_policy)

	# Add l2-regularization if required.
	if l2_reg is not None:
		for var in self.model.policy_variables():
			if "bias" not in var.name:
				actor_loss += l2_reg * tf.nn.l2_loss(var)
		for var in self.model.q_variables():
			if "bias" not in var.name:
				critic_loss += l2_reg * tf.nn.l2_loss(var)

	# Model self-supervised losses.
	if self.config["use_state_preprocessor"]:
		# Expand input_dict in case custom_loss' need them.
		input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
		input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
		input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
		input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
		input_dict['policy_signature'] = train_batch.get('policy_signature',None)
		if log_once("ddpg_custom_loss"):
			logger.warning(
				"You are using a state-preprocessor with DDPG and "
				"therefore, `custom_loss` will be called on your Model! "
				"Please be aware that DDPG now uses the ModelV2 API, which "
				"merges all previously separate sub-models (policy_model, "
				"q_model, and twin_q_model) into one ModelV2, on which "
				"`custom_loss` is called, passing it "
				"[actor_loss, critic_loss] as 1st argument. "
				"You may have to change your custom loss function to handle "
				"this."
			)
		[actor_loss, critic_loss] = model.custom_loss(
			[actor_loss, critic_loss], input_dict
		)

	# Store values for stats function.
	self.actor_loss = actor_loss
	self.critic_loss = critic_loss
	self.td_error = td_error
	self.q_t = q_t

	# Return one loss value (even though we treat them separately in our
	# 2 optimizers: actor and critic).
	return self.critic_loss + self.actor_loss

class NewComputeTDErrorMixin:
	def __init__(self):
		@make_tf_callable(self.get_session(), dynamic_shape=True)
		def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights, policy_signature=None):
			d = {
				SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t),
				SampleBatch.ACTIONS: tf.convert_to_tensor(act_t),
				SampleBatch.REWARDS: tf.convert_to_tensor(rew_t),
				SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1),
				SampleBatch.DONES: tf.convert_to_tensor(done_mask),
				PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights),
			}
			if policy_signature is not None:
				d["policy_signature"] = policy_signature
			input_dict = SampleBatch(d)
			# Do forward pass on loss to update td errors attribute
			# (one TD-error value per item in batch to update PR weights).
			self.loss(self.model, None, input_dict)
			# `self.td_error` is set in loss_fn.
			return self.td_error

		self.compute_td_error = compute_td_error

class XADDPGTF1Policy(DDPGTF1Policy):
	def __init__(self, observation_space, action_space, config, *, existing_inputs = None, existing_model = None):
		super().__init__(observation_space, action_space, config, existing_inputs = existing_inputs, existing_model = existing_model)
		NewComputeTDErrorMixin.__init__(self)
		self.maybe_initialize_optimizer_and_loss()
		TargetNetworkMixin.__init__(self)

	def postprocess_trajectory(self, sample_batch, other_agent_batches = None, episode = None):
		return xa_postprocess_nstep_and_prio(self, sample_batch, other_agent_batches, episode)

	def loss(self, model, dist_class, train_batch):
		return loss(self, model, dist_class, train_batch)

class XADDPGTF2Policy(DDPGTF2Policy):
	def __init__(self, observation_space, action_space, config, *, existing_inputs = None, existing_model = None):
		super().__init__(observation_space, action_space, config, existing_inputs = existing_inputs, existing_model = existing_model)
		NewComputeTDErrorMixin.__init__(self)
		self.maybe_initialize_optimizer_and_loss()
		TargetNetworkMixin.__init__(self)

	def postprocess_trajectory(self, sample_batch, other_agent_batches = None, episode = None):
		return xa_postprocess_nstep_and_prio(self, sample_batch, other_agent_batches, episode)

	def loss(self, model, dist_class, train_batch):
		return loss(self, model, dist_class, train_batch)
