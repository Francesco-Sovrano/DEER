"""
PyTorch policy class used for SAC.
"""
from ray.rllib.algorithms.dqn.dqn_torch_policy import *
import numpy as np

from deer.experience_buffers.replay_ops import add_policy_signature

def xa_postprocess_nstep_and_prio(policy, batch, other_agent=None, episode=None):
	# N-step Q adjustments.
	if policy.config["n_step"] > 1:
		adjust_nstep(policy.config["n_step"], policy.config["gamma"], batch)
	if PRIO_WEIGHTS not in batch:
		batch[PRIO_WEIGHTS] = np.ones_like(batch[SampleBatch.REWARDS])
	if policy.config["buffer_options"]["priority_id"] == "td_errors":
		if policy.config["model"]["custom_model_config"].get("add_nonstationarity_correction", False):
			# print('a', batch.count)
			batch = add_policy_signature(batch, policy)
			batch["td_errors"] = policy.compute_td_error(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS], batch[SampleBatch.DONES], batch[PRIO_WEIGHTS], batch["policy_signature"])
		else:
			batch["td_errors"] = policy.compute_td_error(batch[SampleBatch.CUR_OBS], batch[SampleBatch.ACTIONS], batch[SampleBatch.REWARDS], batch[SampleBatch.NEXT_OBS], batch[SampleBatch.DONES], batch[PRIO_WEIGHTS])
	return batch

def xadqn_q_losses(policy, model, _, train_batch):
	"""Constructs the loss for DQNTorchPolicy.

	Args:
		policy (Policy): The Policy to calculate the loss for.
		model (ModelV2): The Model to calculate the loss for.
		train_batch (SampleBatch): The training data.

	Returns:
		TensorType: A single loss tensor.
	"""

	config = policy.config
	# Q-network evaluation.
	q_t, q_logits_t, q_probs_t, _ = compute_q_values(
		policy,
		model,
		{"obs": train_batch[SampleBatch.CUR_OBS], 'policy_signature': train_batch.get('policy_signature',None)},
		explore=False,
		is_training=True,
	)

	# Target Q-network evaluation.
	q_tp1, q_logits_tp1, q_probs_tp1, _ = compute_q_values(
		policy,
		policy.target_models[model],
		{"obs": train_batch[SampleBatch.NEXT_OBS], 'policy_signature': train_batch.get('policy_signature',None)},
		explore=False,
		is_training=True,
	)

	# Q scores for actions which we know were selected in the given state.
	one_hot_selection = F.one_hot(
		train_batch[SampleBatch.ACTIONS].long(), policy.action_space.n
	)
	q_t_selected = torch.sum(
		torch.where(q_t > FLOAT_MIN, q_t, torch.tensor(0.0, device=q_t.device))
		* one_hot_selection,
		1,
	)
	q_logits_t_selected = torch.sum(
		q_logits_t * torch.unsqueeze(one_hot_selection, -1), 1
	)

	# compute estimate of best possible value starting from state at t + 1
	if config["double_q"]:
		(
			q_tp1_using_online_net,
			q_logits_tp1_using_online_net,
			q_dist_tp1_using_online_net,
			_,
		) = compute_q_values(
			policy,
			model,
			{"obs": train_batch[SampleBatch.NEXT_OBS], 'policy_signature': train_batch.get('policy_signature',None)},
			explore=False,
			is_training=True,
		)
		q_tp1_best_using_online_net = torch.argmax(q_tp1_using_online_net, 1)
		q_tp1_best_one_hot_selection = F.one_hot(
			q_tp1_best_using_online_net, policy.action_space.n
		)
		q_tp1_best = torch.sum(
			torch.where(
				q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
			)
			* q_tp1_best_one_hot_selection,
			1,
		)
		q_probs_tp1_best = torch.sum(
			q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
		)
	else:
		q_tp1_best_one_hot_selection = F.one_hot(
			torch.argmax(q_tp1, 1), policy.action_space.n
		)
		q_tp1_best = torch.sum(
			torch.where(
				q_tp1 > FLOAT_MIN, q_tp1, torch.tensor(0.0, device=q_tp1.device)
			)
			* q_tp1_best_one_hot_selection,
			1,
		)
		q_probs_tp1_best = torch.sum(
			q_probs_tp1 * torch.unsqueeze(q_tp1_best_one_hot_selection, -1), 1
		)

	q_loss = QLoss(
		q_t_selected,
		q_logits_t_selected,
		q_tp1_best,
		q_probs_tp1_best,
		train_batch[PRIO_WEIGHTS],
		train_batch[SampleBatch.REWARDS],
		train_batch[SampleBatch.DONES].float(),
		config["gamma"],
		config["n_step"],
		config["num_atoms"],
		config["v_min"],
		config["v_max"],
	)

	# Store values for stats function in model (tower), such that for
	# multi-GPU, we do not override them during the parallel loss phase.
	model.tower_stats["td_error"] = q_loss.td_error
	# TD-error tensor in final stats
	# will be concatenated and retrieved for each individual batch item.
	model.tower_stats["q_loss"] = q_loss

	return q_loss.loss

class TorchComputeTDErrorMixin:
	def __init__(self):
		def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights, policy_signature=None):
			input_dict = {
				SampleBatch.CUR_OBS: obs_t,
				SampleBatch.ACTIONS: act_t,
				SampleBatch.REWARDS: rew_t,
				SampleBatch.NEXT_OBS: obs_tp1,
				SampleBatch.DONES: done_mask,
				PRIO_WEIGHTS: importance_weights,
			}
			if policy_signature is not None:
				input_dict["policy_signature"] = policy_signature
			input_dict = self._lazy_tensor_dict(input_dict)
			# Do forward pass on loss to update td error attribute
			xadqn_q_losses(self, self.model, None, input_dict)
			return self.model.tower_stats["q_loss"].td_error
		self.compute_td_error = compute_td_error

def torch_before_loss_init(policy, obs_space, action_space, config):
	TorchComputeTDErrorMixin.__init__(policy)
	TargetNetworkMixin.__init__(policy)

XADQNTorchPolicy = DQNTorchPolicy.with_updates(
	name="XADQNTorchPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=xadqn_q_losses,
	before_loss_init=torch_before_loss_init,
	mixins=[
		TargetNetworkMixin,
		TorchComputeTDErrorMixin,
		LearningRateSchedule,
	],
)