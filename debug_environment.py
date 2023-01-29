import gym
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import deer.utils.plot_lib as plt
import sys
import os
from environments import *

HORIZON = 2**8
VISIBILITY_RADIUS = 16
N_AGENTS = 9

PLOT_EPISODE = False
if PLOT_EPISODE:
	OUTPUT_DIR = './demo_episode'
	os.makedirs(OUTPUT_DIR, exist_ok=True)

def get_default_environment_MAGraphDelivery_options(num_agents, reward_fn, fairness_type_fn, fairness_reward_fn, discrete_actions=None, spawn_on_sources_only=True):
	target_junctions_number = num_agents//3
	max_deliveries_per_target = 2
	source_junctions_number = 2
	assert max_deliveries_per_target
	assert target_junctions_number
	return {
		'num_agents': num_agents,
		'n_discrete_actions': discrete_actions,
		'reward_fn': reward_fn, # one of the following: 'frequent', 'more_frequent', 'sparse', 'unitary_frequent', 'unitary_more_frequent', 'unitary_sparse'
		'fairness_type_fn': fairness_type_fn, # one of the following: None, 'simple', 'engineered'
		'fairness_reward_fn': fairness_reward_fn, # one of the following: None, 'simple', 'engineered', 'unitary_engineered'
		'visibility_radius': VISIBILITY_RADIUS,
		'spawn_on_sources_only': spawn_on_sources_only,
		'max_refills_per_source': float('inf'),
		'max_deliveries_per_target': max_deliveries_per_target,#(num_agents//target_junctions_number)+2,
		'target_junctions_number': target_junctions_number,
		'source_junctions_number': source_junctions_number,
		################################
		'max_dimension': 64,
		'junctions_number': 64,
		'max_roads_per_junction': 4,
		'junction_radius': 1,
		'max_distance_to_path': .5, # meters
		################################
		'random_seconds_per_step': False, # whether to sample seconds_per_step from an exponential distribution
		'mean_seconds_per_step': 1, # in average, a step every n seconds
		################################
		# information about speed parameters: http://www.ijtte.com/uploads/2012-10-01/5ebd8343-9b9c-b1d4IJTTE%20vol2%20no3%20%287%29.pdf
		'min_speed': 0.5, # m/s
		'max_speed': 1.5, # m/s
		'print_debug_info': True,
	}

env_config = get_default_environment_MAGraphDelivery_options(
	num_agents=N_AGENTS, 
	reward_fn='unitary_frequent', 
	fairness_type_fn=None, 
	fairness_reward_fn=None, 
	discrete_actions=False, 
	spawn_on_sources_only=True
)

env = PartWorldSomeAgents_GraphDelivery({"culture": 'ComplexHeterogeneity', **env_config})
env.seed(38)
# env = CescoDriveV0()
multiagent = isinstance(env, MultiAgentEnv)
render_modes = env.metadata['render.modes']

def print_screen(screens_directory, step):
	filename = os.path.join(screens_directory, f'frame{step}.jpg')
	if 'rgb_array' in render_modes:
		plt.rgb_array_image(
			env.render(mode='rgb_array'), 
			filename
		)
	elif 'ansi' in render_modes:
		plt.ascii_image(
			env.render(mode='ansi'), 
			filename
		)
	elif 'ascii' in render_modes:
		plt.ascii_image(
			env.render(mode='ascii'), 
			filename
		)
	elif 'human' in render_modes:
		old_stdout = sys.stdout
		sys.stdout = StringIO()
		env.render(mode='human')
		with closing(sys.stdout):
			plt.ascii_image(
				sys.stdout.getvalue(), 
				filename
			)
		sys.stdout = old_stdout
	else:
		raise Exception(f"No compatible render mode (rgb_array,ansi,ascii,human) in {render_modes}.")
	return filename

def run_one_episode(env, name):
	if PLOT_EPISODE:
		episode_dir = os.path.join(OUTPUT_DIR, name)
		os.makedirs(episode_dir, exist_ok=True)
	state = env.reset()
	step = 0
	sum_reward = 0
	if PLOT_EPISODE:
		file_list = [print_screen(episode_dir, step)]
	if multiagent:
		done_dict = {i: False for i in state.keys()}
		done_dict['__all__'] = False
		while not done_dict['__all__'] and step <= HORIZON:
			t = time.time()
			step += 1
			action_dict = {
				i: env.action_space.sample()
				for i in state.keys()
				if not done_dict.get(i,True)
			}
			state_dict, reward_dict, done_dict, info_dict = env.step(action_dict)
			state = state_dict
			sum_reward += sum(reward_dict.values())
			print(f'step {step} took {time.time()-t:.3f} seconds')
			if PLOT_EPISODE:
				file_list.append(print_screen(episode_dir, step))
			env.render()
			# time.sleep(0.25)
	else:
		done = False
		while not done and step <= HORIZON:
			t = time.time()
			step += 1
			action = env.action_space.sample()
			state, reward, done, info = env.step(action)
			sum_reward += reward
			print(f'step {step} took {time.time()-t:.3f} seconds')
			if PLOT_EPISODE:
				file_list.append(print_screen(OUTPUT_DIR, step))
			env.render()
			# time.sleep(0.25)
	if PLOT_EPISODE:
		gif_filename = os.path.join(episode_dir, 'episode.gif')
		plt.make_gif(file_list=file_list, gif_path=gif_filename)
	return sum_reward

sum_reward = run_one_episode(env, 'episode_1')
sum_reward = run_one_episode(env, 'episode_2')
