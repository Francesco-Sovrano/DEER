"""
PyTorch policy class used for TD3 and DDPG.
"""
from ray.rllib.algorithms.ddpg.ddpg_torch_policy import *
from deer.agents.xadqn.xadqn_torch_policy import xa_postprocess_nstep_and_prio
from deer.agents.xaddpg.xaddpg import XADDPGConfig

class NewComputeTDErrorMixin:
	def __init__(self):
		def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights, policy_signature=None):
			d = {
				SampleBatch.CUR_OBS: obs_t,
				SampleBatch.ACTIONS: act_t,
				SampleBatch.REWARDS: rew_t,
				SampleBatch.NEXT_OBS: obs_tp1,
				SampleBatch.DONES: done_mask,
				PRIO_WEIGHTS: importance_weights,
			}
			if policy_signature is not None:
				d["policy_signature"] = policy_signature
			input_dict = self._lazy_tensor_dict(SampleBatch(d))
			# Do forward pass on loss to update td errors attribute
			# (one TD-error value per item in batch to update PR weights).
			self.loss(self.model, None, input_dict)

			# `self.model.td_error` is set within actor_critic_loss call.
			return self.model.tower_stats["td_error"].detach().numpy()

		self.compute_td_error = compute_td_error

class XADDPGTorchPolicy(DDPGTorchPolicy):
	def __init__(self, observation_space, action_space, config):
		config = dict(XADDPGConfig().to_dict(), **config)

		# Create global step for counting the number of update operations.
		self.global_step = 0

		# Validate action space for DDPG
		validate_spaces(self, observation_space, action_space)

		TorchPolicyV2.__init__(
			self,
			observation_space,
			action_space,
			config,
			max_seq_len=config["model"]["max_seq_len"],
		)

		NewComputeTDErrorMixin.__init__(self)

		# TODO: Don't require users to call this manually.
		self._initialize_loss_from_dummy_batch()

		TargetNetworkMixin.__init__(self)

	def postprocess_trajectory(self, sample_batch, other_agent_batches = None, episode = None):
		return xa_postprocess_nstep_and_prio(self, sample_batch, other_agent_batches, episode)

	def loss(self, model, dist_class, train_batch):
		target_model = self.target_models[model]

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
		target_model_out_tp1, _ = target_model(input_dict_next, [], None)

		# Policy network evaluation.
		# prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
		policy_t = model.get_policy_output(model_out_t)
		# policy_batchnorm_update_ops = list(
		#	set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

		policy_tp1 = target_model.get_policy_output(target_model_out_tp1)

		# Action outputs.
		if self.config["smooth_target_policy"]:
			target_noise_clip = self.config["target_noise_clip"]
			clipped_normal_sample = torch.clamp(
				torch.normal(
					mean=torch.zeros(policy_tp1.size()), std=self.config["target_noise"]
				).to(policy_tp1.device),
				-target_noise_clip,
				target_noise_clip,
			)

			policy_tp1_smoothed = torch.min(
				torch.max(
					policy_tp1 + clipped_normal_sample,
					torch.tensor(
						self.action_space.low,
						dtype=torch.float32,
						device=policy_tp1.device,
					),
				),
				torch.tensor(
					self.action_space.high,
					dtype=torch.float32,
					device=policy_tp1.device,
				),
			)
		else:
			# No smoothing, just use deterministic actions.
			policy_tp1_smoothed = policy_tp1

		# Q-net(s) evaluation.
		# prev_update_ops = set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS))
		# Q-values for given actions & observations in given current
		q_t = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])

		# Q-values for current policy (no noise) in given current state
		q_t_det_policy = model.get_q_values(model_out_t, policy_t)

		if twin_q:
			twin_q_t = model.get_twin_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS]
			)
		# q_batchnorm_update_ops = list(
		#	 set(tf1.get_collection(tf.GraphKeys.UPDATE_OPS)) - prev_update_ops)

		# Target q-net(s) evaluation.
		q_tp1 = target_model.get_q_values(target_model_out_tp1, policy_tp1_smoothed)

		if twin_q:
			twin_q_tp1 = target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1_smoothed
			)

		q_t_selected = torch.squeeze(q_t, axis=len(q_t.shape) - 1)
		if twin_q:
			twin_q_t_selected = torch.squeeze(twin_q_t, axis=len(q_t.shape) - 1)
			q_tp1 = torch.min(q_tp1, twin_q_tp1)

		q_tp1_best = torch.squeeze(input=q_tp1, axis=len(q_tp1.shape) - 1)
		q_tp1_best_masked = (1.0 - train_batch[SampleBatch.DONES].float()) * q_tp1_best

		# Compute RHS of bellman equation.
		q_t_selected_target = (
			train_batch[SampleBatch.REWARDS] + gamma**n_step * q_tp1_best_masked
		).detach()

		# Compute the error (potentially clipped).
		if twin_q:
			td_error = q_t_selected - q_t_selected_target
			twin_td_error = twin_q_t_selected - q_t_selected_target
			if use_huber:
				errors = huber_loss(td_error, huber_threshold) + huber_loss(
					twin_td_error, huber_threshold
				)
			else:
				errors = 0.5 * (
					torch.pow(td_error, 2.0) + torch.pow(twin_td_error, 2.0)
				)
		else:
			td_error = q_t_selected - q_t_selected_target
			if use_huber:
				errors = huber_loss(td_error, huber_threshold)
			else:
				errors = 0.5 * torch.pow(td_error, 2.0)

		critic_loss = torch.mean(train_batch[PRIO_WEIGHTS] * errors)
		actor_loss = -torch.mean(train_batch[PRIO_WEIGHTS] * q_t_det_policy)

		# Add l2-regularization if required.
		if l2_reg is not None:
			for name, var in model.policy_variables(as_dict=True).items():
				if "bias" not in name:
					actor_loss += l2_reg * l2_loss(var)
			for name, var in model.q_variables(as_dict=True).items():
				if "bias" not in name:
					critic_loss += l2_reg * l2_loss(var)

		# Model self-supervised losses.
		if self.config["use_state_preprocessor"]:
			# Expand input_dict in case custom_loss' need them.
			input_dict[SampleBatch.ACTIONS] = train_batch[SampleBatch.ACTIONS]
			input_dict[SampleBatch.REWARDS] = train_batch[SampleBatch.REWARDS]
			input_dict[SampleBatch.DONES] = train_batch[SampleBatch.DONES]
			input_dict[SampleBatch.NEXT_OBS] = train_batch[SampleBatch.NEXT_OBS]
			input_dict['policy_signature'] = train_batch.get('policy_signature',None)
			[actor_loss, critic_loss] = model.custom_loss(
				[actor_loss, critic_loss], input_dict
			)

		# Store values for stats function in model (tower), such that for
		# multi-GPU, we do not override them during the parallel loss phase.
		model.tower_stats["q_t"] = q_t
		model.tower_stats["actor_loss"] = actor_loss
		model.tower_stats["critic_loss"] = critic_loss
		# TD-error tensor in final stats
		# will be concatenated and retrieved for each individual batch item.
		model.tower_stats["td_error"] = td_error

		# Return two loss terms (corresponding to the two optimizers, we create).
		return [actor_loss, critic_loss]
