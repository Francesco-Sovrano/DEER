from ray.rllib.policy.sample_batch import SampleBatch, MultiAgentBatch, DEFAULT_POLICY_ID
from deer.experience_buffers.replay_buffer import SimpleReplayBuffer, LocalReplayBuffer, get_batch_infos
from deer.experience_buffers.clustering_scheme import *

import numpy as np

def get_clustered_replay_buffer(config):
	assert config.batch_mode == "complete_episodes" or not config.clustering_options["cluster_with_episode_type"], f"This algorithm requires 'complete_episodes' as batch_mode when 'cluster_with_episode_type' is True"
	clustering_scheme_type = config.clustering_options.get("clustering_scheme", None)
	# no need for unclustered_buffer if clustering_scheme_type is none
	ratio_of_samples_from_unclustered_buffer = config.clustering_options["ratio_of_samples_from_unclustered_buffer"] if clustering_scheme_type else 0
	local_replay_buffer = LocalReplayBuffer(
		config.buffer_options, 
		learning_starts=config.num_steps_sampled_before_learning_starts,
		seed=config.seed,
		cluster_selection_policy=config.clustering_options["cluster_selection_policy"],
		ratio_of_samples_from_unclustered_buffer=ratio_of_samples_from_unclustered_buffer,
	)
	clustering_scheme = ClusterManager(clustering_scheme_type, config.clustering_options["clustering_scheme_options"])
	return local_replay_buffer, clustering_scheme

def assign_types(multi_batch, clustering_scheme, batch_fragment_length, with_episode_type=True, training_step=None):
	if not isinstance(multi_batch, MultiAgentBatch):
		multi_batch = MultiAgentBatch({DEFAULT_POLICY_ID: multi_batch}, multi_batch.count)
	
	# if not with_episode_type:
	# 	batch_list = multi_batch.timeslices(batch_fragment_length) if multi_batch.count > batch_fragment_length else [multi_batch]
	# 	for i,batch in enumerate(batch_list):
	# 		for pid,sub_batch in batch.policy_batches.items():
	# 			get_batch_infos(sub_batch)['batch_type'] = clustering_scheme.get_batch_type(sub_batch, training_step=training_step, episode_step=i, agent_id=pid)		
	# 	return batch_list

	batch_dict = {}
	for pid,meta_batch in multi_batch.policy_batches.items():
		batch_dict[pid] = []
		batch_list = meta_batch.split_by_episode() if with_episode_type else [meta_batch]
		for batch in batch_list:
			sub_batch_count = int(np.ceil(len(batch)/batch_fragment_length))
			sub_batch_list = [
				batch[i*batch_fragment_length : (i+1)*batch_fragment_length]
				for i in range(sub_batch_count)
			] if len(batch) > batch_fragment_length else [batch]
			episode_type = clustering_scheme.get_episode_type(sub_batch_list) if with_episode_type else None
			for i,sub_batch in enumerate(sub_batch_list):
				batch_type = clustering_scheme.get_batch_type(
					sub_batch, 
					episode_type=episode_type, 
					training_step=training_step, 
					episode_step=i, 
					agent_id=pid
				)
				sub_batch[SampleBatch.INFOS] = [{'batch_type': batch_type,'training_step': training_step}] # remove unnecessary infos to save some memory
			batch_dict[pid] += sub_batch_list
	return [
		MultiAgentBatch(
			{
				pid: b
				for pid,b in zip(batch_dict.keys(),b_list)
			},
			b_list[0].count
		)
		for b_list in zip(*batch_dict.values())
	]

def add_policy_signature(batch, policy):
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

def add_buffer_metrics(results, buffer):
	results['buffer']=buffer.stats()
	return results
