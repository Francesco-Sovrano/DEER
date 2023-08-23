import gym
import time
from environment import *

env = GraphDrive({"reward_fn": 'frequent_reward_explanation_engineering_v1', "culture_level": "Hard"})
# env = CescoDriveV0()

def run_one_episode (env):
	env.seed(38)
	env.reset()
	sum_reward = 0
	done = False
	while not done:
		action = env.action_space.sample()
		state, reward, done, info = env.step(action)
		sum_reward += reward
		env.render()
		time.sleep(0.25)
	return sum_reward

sum_reward = run_one_episode(env)