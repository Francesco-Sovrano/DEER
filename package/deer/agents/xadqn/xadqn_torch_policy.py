"""
PyTorch policy class used for SAC.
"""
from ray.rllib.algorithms.dqn.dqn_torch_policy import *
from ray.rllib.evaluation.postprocessing import adjust_nstep
import numpy as np
import random

def add_policy_signature(batch, policy): # Experience replay in MARL may suffer from non-stationarity. To avoid this issue a solution is to condition each agent’s value function on a fingerprint that disambiguates the age of the data sampled from the replay memory. To stabilise experience replay, it should be sufficient if each agent’s observations disambiguate where along this trajectory the current training sample originated from. # cit. [2017]Stabilising Experience Replay for Deep Multi-Agent Reinforcement Learning
	# train_step = np.array((policy.num_grad_updates,), dtype=np.float32)
	# if train_step > 0: print(train_step)
	policy_exploration_state = policy.get_exploration_state()
	if len(policy_exploration_state) > 1:
		policy_exploration_state_items = policy_exploration_state.items()
		policy_exploration_state_items = filter(lambda x: x[0].startswith('cur'), policy_exploration_state_items)
		# policy_exploration_state_items=list(policy_exploration_state_items)
		# print(policy_exploration_state_items)
		policy_entropy_var = next(map(lambda x: x[-1], policy_exploration_state_items), None)
	else:
		policy_entropy_var = 0
	
	# policy_entropy_var = np.array((policy_entropy_var,), dtype=np.float32)
	model_entropy_var = policy.model.get_entropy_var()
	if model_entropy_var is None:
		model_entropy_var = 0 #np.array((0,), dtype=np.float32)
	# print("policy_signature:", model_entropy_var,policy_entropy_var)
	# assert train_step != 0 or policy_entropy_var != 0 or model_entropy_var != 0, "Invalid policy signature!"
	# batch["policy_signature"] = np.array((policy.num_grad_updates/1000,policy_entropy_var,model_entropy_var), dtype=np.float32)
	batch["policy_signature"] = np.array((0 if policy_entropy_var!=0 or model_entropy_var!=0 else policy.num_grad_updates/100,policy_entropy_var,model_entropy_var), dtype=np.float32)
	batch["policy_signature"] = np.tile(batch["policy_signature"],(batch.count,1))
	return batch

def sample_from_beta_with_mean(target_mean, alpha=1):
	beta = alpha / target_mean - alpha
	return np.random.beta(alpha, beta)

def special_adjust_nstep(n_step, gamma, batch):
	assert not any(
		batch[SampleBatch.DONES][:-1]
	), "Unexpected done in middle of trajectory!"

	len_ = len(batch)

	# Shift NEXT_OBS and DONES.
	batch[SampleBatch.NEXT_OBS] = np.concatenate(
		[
			batch[SampleBatch.OBS][n_step:],
			np.stack([batch[SampleBatch.NEXT_OBS][-1]] * min(n_step, len_)),
		],
		axis=0,
	)
	batch[SampleBatch.DONES] = np.concatenate(
		[
			batch[SampleBatch.DONES][n_step - 1 :],
			np.tile(batch[SampleBatch.DONES][-1], min(n_step - 1, len_)),
		],
		axis=0,
	)

	# Change rewards in place.
	for i in range(len_):
		if i+n_step >= len_:
			break
		idx_of_highest_rewards = batch[SampleBatch.REWARDS][i+1:i+n_step].argmax(axis=-1)
		batch[SampleBatch.REWARDS][i] += (
			gamma**(idx_of_highest_rewards-i) * batch[SampleBatch.REWARDS][idx_of_highest_rewards]
		)
		# batch[SampleBatch.REWARDS][i] += (
		# 	gamma**(n_step-1) * batch[SampleBatch.REWARDS][n_step-1]
		# )

def xa_postprocess_nstep_and_prio(policy, batch, other_agent=None, episode=None):
	n_step = random.uniform(0,1)*policy.config['n_step'] if policy.config['n_step_random_sampling'] else policy.config['n_step'] # double-check that the random seed initialization done within the algorithm constructor is reaching this point
	n_step = int(np.ceil(n_step))
	# N-step Q adjustments.
	if n_step > 1:
		if policy.config['n_step_annealing_scheduler']['fn']:
			special_adjust_nstep(n_step, policy.config["gamma"], batch)
		else:
			adjust_nstep(n_step, policy.config["gamma"], batch)
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