from ray.tune.registry import register_env
######### Add new environment below #########

# import gym
# def build_env_with_agent_groups(env_class, config):
# 	env = env_class(config)
# 	grouping = {"group_1": list(range(config.get('num_agents',1)))}
# 	obs_space = gym.spaces.Tuple([env.observation_space]*config.get('num_agents',1))
# 	act_space = gym.spaces.Tuple([env.action_space]*config.get('num_agents',1))
# 	return env.with_agent_groups(grouping, obs_space=obs_space, act_space=act_space)

### GridPlanning
from .grid_planning import GridPlanning
register_env("GridPlanning", lambda config: GridPlanning(config))
# register_env("GridPlanning-Group", lambda config: build_env_with_agent_groups(GridPlanning,config))
