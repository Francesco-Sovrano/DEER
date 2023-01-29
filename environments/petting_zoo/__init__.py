from ray.tune.registry import register_env
######### Add new environment below #########

### PettingZoo
from ray.rllib.env.wrappers.pettingzoo_env import PettingZooEnv, ParallelPettingZooEnv
import logging
logger = logging.getLogger('pettingzoo.utils.env_logger')
logger.setLevel(logging.ERROR)

from pettingzoo.butterfly import pistonball_v6
register_env("pistonball_v6", lambda config: ParallelPettingZooEnv(pistonball_v6.parallel_env(**config)))

from pettingzoo.classic import hanabi_v4
register_env("hanabi_v4", lambda config: PettingZooEnv(hanabi_v4.env(**config)))
