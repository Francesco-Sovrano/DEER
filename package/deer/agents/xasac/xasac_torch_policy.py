"""
PyTorch policy class used for SAC.
"""
from ray.rllib.algorithms.sac.sac_torch_policy import *
from ray.rllib.algorithms.sac.sac_torch_policy import _get_dist_class
from deer.agents.xadqn.xadqn_torch_policy import xa_postprocess_nstep_and_prio
from ray.rllib.utils.torch_utils import (
	concat_multi_gpu_td_errors,
	huber_loss,
)
import numpy as np

def xasac_actor_critic_loss(policy, model, dist_class, train_batch):
	"""Constructs the loss for the Soft Actor Critic.

	Args:
		policy: The Policy to calculate the loss for.
		model (ModelV2): The Model to calculate the loss for.
		dist_class (Type[TorchDistributionWrapper]: The action distr. class.
		train_batch: The training data.

	Returns:
		Union[TensorType, List[TensorType]]: A single loss tensor or a list
			of loss tensors.
	"""
	# Look up the target model (tower) using the model tower.
	target_model = policy.target_models[model]

	# Should be True only for debugging purposes (e.g. test cases)!
	deterministic = policy.config["_deterministic_loss"]

	model_out_t, _ = model(
		SampleBatch(obs=train_batch[SampleBatch.CUR_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=True), [], None
	)

	model_out_tp1, _ = model(
		SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=True), [], None
	)

	target_model_out_tp1, _ = target_model(
		SampleBatch(obs=train_batch[SampleBatch.NEXT_OBS], policy_signature=train_batch.get('policy_signature',None), _is_training=True), [], None
	)

	alpha = torch.exp(model.log_alpha)

	# Discrete case.
	if model.discrete:
		# Get all action probs directly from pi and form their logp.
		action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
		log_pis_t = F.log_softmax(action_dist_inputs_t, dim=-1)
		policy_t = torch.exp(log_pis_t)
		action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
		log_pis_tp1 = F.log_softmax(action_dist_inputs_tp1, -1)
		policy_tp1 = torch.exp(log_pis_tp1)
		# Q-values.
		q_t, _ = model.get_q_values(model_out_t)
		# Target Q-values.
		q_tp1, _ = target_model.get_q_values(target_model_out_tp1)
		if policy.config["twin_q"]:
			twin_q_t, _ = model.get_twin_q_values(model_out_t)
			twin_q_tp1, _ = target_model.get_twin_q_values(target_model_out_tp1)
			q_tp1 = torch.min(q_tp1, twin_q_tp1)
		q_tp1 -= alpha * log_pis_tp1

		# Actually selected Q-values (from the actions batch).
		one_hot = F.one_hot(
			train_batch[SampleBatch.ACTIONS].long(), num_classes=q_t.size()[-1]
		)
		q_t_selected = torch.sum(q_t * one_hot, dim=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = torch.sum(twin_q_t * one_hot, dim=-1)
		# Discrete case: "Best" means weighted by the policy (prob) outputs.
		q_tp1_best = torch.sum(torch.mul(policy_tp1, q_tp1), dim=-1)
		q_tp1_best_masked = (1.0 - train_batch[SampleBatch.TERMINATEDS].float()) * q_tp1_best
	# Continuous actions case.
	else:
		# Sample single actions from distribution.
		action_dist_class = _get_dist_class(policy, policy.config, policy.action_space)
		action_dist_inputs_t, _ = model.get_action_model_outputs(model_out_t)
		action_dist_t = action_dist_class(action_dist_inputs_t, model)
		policy_t = (
			action_dist_t.sample()
			if not deterministic
			else action_dist_t.deterministic_sample()
		)
		log_pis_t = torch.unsqueeze(action_dist_t.logp(policy_t), -1)
		action_dist_inputs_tp1, _ = model.get_action_model_outputs(model_out_tp1)
		action_dist_tp1 = action_dist_class(action_dist_inputs_tp1, model)
		policy_tp1 = (
			action_dist_tp1.sample()
			if not deterministic
			else action_dist_tp1.deterministic_sample()
		)
		log_pis_tp1 = torch.unsqueeze(action_dist_tp1.logp(policy_tp1), -1)

		# Q-values for the actually selected actions.
		q_t, _ = model.get_q_values(model_out_t, train_batch[SampleBatch.ACTIONS])
		if policy.config["twin_q"]:
			twin_q_t, _ = model.get_twin_q_values(
				model_out_t, train_batch[SampleBatch.ACTIONS]
			)

		# Q-values for current policy in given current state.
		q_t_det_policy, _ = model.get_q_values(model_out_t, policy_t)
		if policy.config["twin_q"]:
			twin_q_t_det_policy, _ = model.get_twin_q_values(model_out_t, policy_t)
			q_t_det_policy = torch.min(q_t_det_policy, twin_q_t_det_policy)

		# Target q network evaluation.
		q_tp1, _ = target_model.get_q_values(target_model_out_tp1, policy_tp1)
		if policy.config["twin_q"]:
			twin_q_tp1, _ = target_model.get_twin_q_values(
				target_model_out_tp1, policy_tp1
			)
			# Take min over both twin-NNs.
			q_tp1 = torch.min(q_tp1, twin_q_tp1)

		q_t_selected = torch.squeeze(q_t, dim=-1)
		if policy.config["twin_q"]:
			twin_q_t_selected = torch.squeeze(twin_q_t, dim=-1)
		q_tp1 -= alpha * log_pis_tp1

		q_tp1_best = torch.squeeze(input=q_tp1, dim=-1)
		q_tp1_best_masked = (1.0 - train_batch[SampleBatch.TERMINATEDS].float()) * q_tp1_best

	# compute RHS of bellman equation
	q_t_selected_target = (
		train_batch[SampleBatch.REWARDS]
		+ (policy.config["gamma"] ** policy.config["n_step"]) * q_tp1_best_masked
	).detach()

	# Compute the TD-error (potentially clipped).
	base_td_error = torch.abs(q_t_selected - q_t_selected_target)
	if policy.config["twin_q"]:
		twin_td_error = torch.abs(twin_q_t_selected - q_t_selected_target)
		td_error = 0.5 * (base_td_error + twin_td_error)
	else:
		td_error = base_td_error

	critic_loss = [torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(base_td_error))]
	if policy.config["twin_q"]:
		critic_loss.append(
			torch.mean(train_batch[PRIO_WEIGHTS] * huber_loss(twin_td_error))
		)

	# Alpha- and actor losses.
	# Note: In the papers, alpha is used directly, here we take the log.
	# Discrete case: Multiply the action probs as weights with the original
	# loss terms (no expectations needed).
	if model.discrete:
		weighted_log_alpha_loss = policy_t.detach() * (
			-model.log_alpha * (log_pis_t + model.target_entropy).detach()
		)
		# Sum up weighted terms and mean over all batch items.
		alpha_loss = torch.mean(train_batch[PRIO_WEIGHTS] * torch.sum(weighted_log_alpha_loss, dim=-1))
		# Actor loss.
		actor_loss = torch.mean(
			train_batch[PRIO_WEIGHTS] * torch.sum(
				torch.mul(
					# NOTE: No stop_grad around policy output here
					# (compare with q_t_det_policy for continuous case).
					policy_t,
					alpha.detach() * log_pis_t - q_t.detach(),
				),
				dim=-1,
			)
		)
	else:
		alpha_loss = -torch.mean(
			train_batch[PRIO_WEIGHTS] * model.log_alpha * (log_pis_t + model.target_entropy).detach()
		)
		# Note: Do not detach q_t_det_policy here b/c is depends partly
		# on the policy vars (policy sample pushed through Q-net).
		# However, we must make sure `actor_loss` is not used to update
		# the Q-net(s)' variables.
		actor_loss = torch.mean(train_batch[PRIO_WEIGHTS] * (alpha.detach() * log_pis_t - q_t_det_policy))

	# Store values for stats function in model (tower), such that for
	# multi-GPU, we do not override them during the parallel loss phase.
	model.tower_stats["q_t"] = q_t
	model.tower_stats["policy_t"] = policy_t
	model.tower_stats["log_pis_t"] = log_pis_t
	model.tower_stats["actor_loss"] = actor_loss
	model.tower_stats["critic_loss"] = critic_loss
	model.tower_stats["alpha_loss"] = alpha_loss

	# TD-error tensor in final stats
	# will be concatenated and retrieved for each individual batch item.
	model.tower_stats["td_error"] = td_error

	# Return all loss terms corresponding to our optimizers.
	return tuple([actor_loss] + critic_loss + [alpha_loss])

class TorchComputeTDErrorMixin:
	"""Mixin class calculating TD-error (part of critic loss) per batch item.

	- Adds `policy.compute_td_error()` method for TD-error calculation from a
	  batch of observations/actions/rewards/etc..
	"""

	def __init__(self):
		def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights, policy_signature=None):
			d = {
				SampleBatch.CUR_OBS: obs_t,
				SampleBatch.ACTIONS: act_t,
				SampleBatch.REWARDS: rew_t,
				SampleBatch.NEXT_OBS: obs_tp1,
				SampleBatch.TERMINATEDS: done_mask,
				PRIO_WEIGHTS: importance_weights,
			}
			if policy_signature is not None:
				d["policy_signature"] = policy_signature
			input_dict = self._lazy_tensor_dict(d)
			# Do forward pass on loss to update td errors attribute
			# (one TD-error value per item in batch to update PR weights).
			xasac_actor_critic_loss(self, self.model, None, input_dict)

			# `self.model.td_error` is set within actor_critic_loss call.
			# Return its updated value here.
			return self.model.tower_stats["td_error"]

		# Assign the method to policy (self) for later usage.
		self.compute_td_error = compute_td_error

def torch_setup_late_mixins(policy, obs_space, action_space, config):
	TorchComputeTDErrorMixin.__init__(policy)
	TargetNetworkMixin.__init__(policy)

XASACTorchPolicy = SACTorchPolicy.with_updates(
	name="XASACTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=xasac_actor_critic_loss,
	before_loss_init=torch_setup_late_mixins,
	mixins=[TargetNetworkMixin, TorchComputeTDErrorMixin],
)
