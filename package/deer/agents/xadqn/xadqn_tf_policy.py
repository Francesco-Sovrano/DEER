"""
PyTorch policy class used for SAC.
"""
from ray.rllib.algorithms.dqn.dqn_tf_policy import *
from ray.rllib.utils.tf_utils import make_tf_callable
from deer.agents.xadqn.xadqn_torch_policy import xa_postprocess_nstep_and_prio

def xadqn_q_losses(policy, model, _, train_batch):
	"""Constructs the loss for DQNTFPolicy.

	Args:
		policy (Policy): The Policy to calculate the loss for.
		model (ModelV2): The Model to calculate the loss for.
		train_batch (SampleBatch): The training data.

	Returns:
		TensorType: A single loss tensor.
	"""
	config = policy.config
	# q network evaluation
	q_t, q_logits_t, q_dist_t, _ = compute_q_values(
		policy,
		model,
		SampleBatch({"obs": train_batch[SampleBatch.CUR_OBS], 'policy_signature': train_batch.get('policy_signature',None)}),
		state_batches=None,
		explore=False,
	)

	# target q network evalution
	q_tp1, q_logits_tp1, q_dist_tp1, _ = compute_q_values(
		policy,
		policy.target_model,
		SampleBatch({"obs": train_batch[SampleBatch.NEXT_OBS], 'policy_signature': train_batch.get('policy_signature',None)}),
		state_batches=None,
		explore=False,
	)
	if not hasattr(policy, "target_q_func_vars"):
		policy.target_q_func_vars = policy.target_model.variables()

	# q scores for actions which we know were selected in the given state.
	one_hot_selection = tf.one_hot(
		tf.cast(train_batch[SampleBatch.ACTIONS], tf.int32), policy.action_space.n
	)
	q_t_selected = tf.reduce_sum(q_t * one_hot_selection, 1)
	q_logits_t_selected = tf.reduce_sum(
		q_logits_t * tf.expand_dims(one_hot_selection, -1), 1
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
			SampleBatch({"obs": train_batch[SampleBatch.NEXT_OBS], 'policy_signature': train_batch.get('policy_signature',None)}),
			state_batches=None,
			explore=False,
		)
		q_tp1_best_using_online_net = tf.argmax(q_tp1_using_online_net, 1)
		q_tp1_best_one_hot_selection = tf.one_hot(
			q_tp1_best_using_online_net, policy.action_space.n
		)
		q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
		q_dist_tp1_best = tf.reduce_sum(
			q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1
		)
	else:
		q_tp1_best_one_hot_selection = tf.one_hot(
			tf.argmax(q_tp1, 1), policy.action_space.n
		)
		q_tp1_best = tf.reduce_sum(q_tp1 * q_tp1_best_one_hot_selection, 1)
		q_dist_tp1_best = tf.reduce_sum(
			q_dist_tp1 * tf.expand_dims(q_tp1_best_one_hot_selection, -1), 1
		)

	policy.q_loss = QLoss(
		q_t_selected,
		q_logits_t_selected,
		q_tp1_best,
		q_dist_tp1_best,
		train_batch[PRIO_WEIGHTS],
		train_batch[SampleBatch.REWARDS],
		tf.cast(train_batch[SampleBatch.DONES], tf.float32),
		config["gamma"],
		config["n_step"],
		config["num_atoms"],
		config["v_min"],
		config["v_max"],
	)

	return policy.q_loss.loss

class TFComputeTDErrorMixin:
	"""Assign the `compute_td_error` method to the DQNTFPolicy

	This allows us to prioritize on the worker side.
	"""

	def __init__(self):
		@make_tf_callable(self.get_session(), dynamic_shape=True)
		def compute_td_error(obs_t, act_t, rew_t, obs_tp1, done_mask, importance_weights, policy_signature=None):
			# Do forward pass on loss to update td error attribute
			input_dict = {
				SampleBatch.CUR_OBS: tf.convert_to_tensor(obs_t),
				SampleBatch.ACTIONS: tf.convert_to_tensor(act_t),
				SampleBatch.REWARDS: tf.convert_to_tensor(rew_t),
				SampleBatch.NEXT_OBS: tf.convert_to_tensor(obs_tp1),
				SampleBatch.DONES: tf.convert_to_tensor(done_mask),
				PRIO_WEIGHTS: tf.convert_to_tensor(importance_weights),
			}
			if policy_signature is not None:
				input_dict["policy_signature"] = policy_signature
			xadqn_q_losses(
				self,
				self.model,
				None,
				input_dict,
			)

			return self.q_loss.td_error

		self.compute_td_error = compute_td_error

def tf_before_loss_init(policy, obs_space, action_space, config):
	LearningRateSchedule.__init__(policy, config["lr"], config["lr_schedule"])
	TFComputeTDErrorMixin.__init__(policy)

XADQNTFPolicy = DQNTFPolicy.with_updates(
	name="XADQNTFPolicy",
	postprocess_fn=xa_postprocess_nstep_and_prio,
	loss_fn=xadqn_q_losses,
	before_loss_init=tf_before_loss_init,
	mixins=[
		TargetNetworkMixin,
		TFComputeTDErrorMixin,
		LearningRateSchedule,
	],
)