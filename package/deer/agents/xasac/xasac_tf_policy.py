"""
TensorFlow policy class used for SAC.
"""
from ray.rllib.algorithms.sac.sac_tf_policy import *
from ray.rllib.utils.tf_utils import make_tf_callable
from ray.rllib.algorithms.sac.sac_tf_policy import _get_dist_class
from deer.agents.xadqn.xadqn_torch_policy import xa_postprocess_nstep_and_prio

def xasac_actor_critic_loss(policy, model, dist_class, train_batch):
	"""Constructs the loss for the Soft Actor Critic.

	Args:
		policy: The Policy to calculate the loss for.
		model (ModelV2): The Model to calculate the loss for.
		dist_class (Type[ActionDistribution]: The action distr. class.
		train_batch: The training data.

	Returns:
		Union[TensorType, List[TensorType]]: A single loss tensor or a list
			of loss tensors.
	"""
	# Should be True only for debugging purposes (e.g. test cases)!
	deterministic = policy.config["_deterministic_loss"]

	_is_training = policy._get_is_training_placeholder()
	# Get the base model output from the train batch.
	model_out_t, _ = model(
		SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=_is_training),
		[],
		None,
	)

	# Get the base model output from the next observations in the train batch.
	model_out_tp1, _ = model(
		SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=_is_training),
		[],
		None,
	)

	# Get the target model's base outputs from the next observations in the
	# train batch.
	target_model_out_tp1, _ = policy.target_model(
		SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=_is_training),
		[],
		None,
	)

	# Discrete actions case.
	if model.discrete:
		# Get all action probs directly from pi and form their logp.
		action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
		log_pis_t = tf.nn.log_softmax(action_dist_inputs_t, -1)
		policy_t = tf.math.exp(log_pis_t)

		action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
		log_pis_tp1 = tf.nn.log_softmax(action_dist_inputs_tp1, -1)
		policy_tp1 = tf.math.exp(log_pis_tp1)

		# Q-values.
		q_t, _ = model.get_q_values(model_out_t)
		# Target Q-values.
		q_tp1, _ = policy.target_model.get_q_values(target_model_out_tp1)
		if policy.config["twin_q"]:
			twin_q_t, _ = model.get_twin_q_values(model_out_t)
			twin_q_tp1, _ = policy.target_model.get_twin_q_values(target_model_out_tp1)
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)
		q_tp1 -= model.alpha * log_pis_tp1

		# Actually selected Q-values (from the actions batch).
		one_hot = tf.one_hot(
			train_batch[SampleBatch.ACTIONS], depth=q_t.shape.as_list()[-1]
		)
		q_t_selected = tf.reduce_sum(q_t * one_hot, axis=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.reduce_sum(twin_q_t * one_hot, axis=-1)
		# Discrete case: "Best" means weighted by the policy (prob) outputs.
		q_tp1_best = tf.reduce_sum(tf.multiply(policy_tp1, q_tp1), axis=-1)
		q_tp1_best_masked = (
			1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)
		) * q_tp1_best
	# Continuous actions case.
	else:
		# Sample simgle actions from distribution.
		action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
		action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
		action_dist_t = action_dist_class(action_dist_inputs_t, policy.model)
		policy_t = (
			action_dist_t.sample()
			if not deterministic
			else action_dist_t.deterministic_sample()
		)
		log_pis_t = tf.expand_dims(action_dist_t.logp(policy_t), -1)

		action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
		action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, policy.model)
		policy_tp1 = (
			action_dist_tp1.sample()
			if not deterministic
			else action_dist_tp1.deterministic_sample()
		)
		log_pis_tp1 = tf.expand_dims(action_dist_tp1.logp(policy_tp1), -1)

		# Q-values for the actually selected actions.
		q_t, _ = model.get_q_values(
			model_out_t, tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32)
		)
		if policy.config["twin_q"]:
			twin_q_t, _ = model.get_twin_q_values(
				model_out_t, tf.cast(train_batch[SampleBatch.ACTIONS], tf.float32)
			)

		# Q-values for current policy in given current state.
		q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
		if policy.config["twin_q"]:
			twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
			q_t_det_policy = tf.reduce_min(
				(q_t_det_policy, twin_q_t_det_policy), axis=0
			)

		# target q network evaluation
		q_tp1, _ = policy.target_model.get_q_values(target_model_out_tp1, policy_tp1)
		if policy.config["twin_q"]:
			twin_q_tp1, _ = policy.target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1
			)
			# Take min over both twin-NNs.
			q_tp1 = tf.reduce_min((q_tp1, twin_q_tp1), axis=0)

		q_t_selected = tf.squeeze(q_t, axis=len(q_t.shape) - 1)
		if policy.config["twin_q"]:
			twin_q_t_selected = tf.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
		q_tp1 -= model.alpha * log_pis_tp1

		q_tp1_best = tf.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
		q_tp1_best_masked = (
			1.0 - tf.cast(train_batch[SampleBatch.TERMINATEDS], tf.float32)
		) * q_tp1_best

	# Compute RHS of bellman equation for the Q-loss (critic(s)).
	q_t_selected_target = tf.stop_gradient(
		tf.cast(train_batch[SampleBatch.REWARDS], tf.float32)
		+ policy.config["gamma"] ** policy.config["n_step"] * q_tp1_best_masked
	)

	# Compute the TD-error (potentially clipped).
	base_td_error = tf.math.abs(q_t_selected - q_t_selected_target)
	if policy.config["twin_q"]:
		twin_td_error = tf.math.abs(twin_q_t_selected - q_t_selected_target)
		td_error = 0.5 * (base_td_error + twin_td_error)
	else:
		td_error = base_td_error

	# Calculate one or two critic losses (2 in the twin_q case).
	prio_weights = tf.cast(train_batch[PRIO_WEIGHTS], tf.float32)
	critic_loss = [tf.reduce_mean(prio_weights * huber_loss(base_td_error))]
	if policy.config["twin_q"]:
		critic_loss.append(tf.reduce_mean(prio_weights * huber_loss(twin_td_error)))

	# Alpha- and actor losses.
	# Note: In the papers, alpha is used directly, here we take the log.
	# Discrete case: Multiply the action probs as weights with the original
	# loss terms (no expectations needed).
	if model.discrete:
		alpha_loss = tf.reduce_mean(
			prio_weights * tf.reduce_sum(
				tf.multiply(
					tf.stop_gradient(policy_t),
					-model.log_alpha
					* tf.stop_gradient(log_pis_t + model.target_entropy),
				),
				axis=-1,
			)
		)
		actor_loss = tf.reduce_mean(
			prio_weights * tf.reduce_sum(
				tf.multiply(
					# NOTE: No stop_grad around policy output here
					# (compare with q_t_det_policy for continuous case).
					policy_t,
					tf.stop_gradient(model.alpha) * log_pis_t - tf.stop_gradient(q_t),
				),
				axis=-1,
			)
		)
	else:
		alpha_loss = -tf.reduce_mean(
			prio_weights * model.log_alpha * tf.stop_gradient(log_pis_t + model.target_entropy)
		)
		actor_loss = tf.reduce_mean(prio_weights * (tf.stop_gradient(model.alpha) * log_pis_t - q_t_det_policy))

	# Save for stats function.
	policy.policy_t = policy_t
	policy.q_t = q_t
	policy.td_error = td_error
	policy.actor_loss = actor_loss
	policy.critic_loss = critic_loss
	policy.alpha_loss = alpha_loss
	policy.alpha_value = model.alpha
	policy.target_entropy = model.target_entropy

	# In a custom apply op we handle the losses separately, but return them
	# combined in one loss here.
	return actor_loss + tf.math.add_n(critic_loss) + alpha_loss

class TFComputeTDErrorMixin:
	def __init__(self):
		@make_tf_callable(self.get_session(), dynamic_shape=True)
		def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights, policy_signature=None):
			# Do forward pass on loss to update td errors attribute
			# (one TD-error value per item in batch to update PR weights).
			input_dict = {
				SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t),
				SampleBatch.ACTIONS: tf.convert_to_tensor(act_t),
				SampleBatch.REWARDS: tf.convert_to_tensor(rew_t),
				SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1),
				SampleBatch.TERMINATEDS: tf.convert_to_tensor(done_mask),
				PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights),
			}
			if policy_signature is not None:
				input_dict["policy_signature"] = policy_signature
			xasac_actor_critic_loss(
				self,
				self.model,
				None,
				input_dict,
			)
			# `self.td_error` is set in loss_fn.
			return self.td_error

		self.compute_td_error = compute_td_error

def tf_setup_mid_mixins(policy, obs_space, action_space, config):
	TFComputeTDErrorMixin.__init__(policy)

XASACTFPolicy = SACTFPolicy.with_updates(
	name="XASACTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=xasac_actor_critic_loss,
	before_loss_init=tf_setup_mid_mixins,
	mixins=[TargetNetworkMixin, ActorCriticOptimizerMixin, TFComputeTDErrorMixin],
)