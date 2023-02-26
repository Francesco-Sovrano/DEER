from ray.tune.registry import register_env
######### Add new environment below #########

###########################################################################
### Multi-Agent Graph Delivery
from .full_world_all_agents_env import FullWorldAllAgents_GraphDelivery
from .part_world_some_agents_env import PartWorldSomeAgents_GraphDelivery
from .full_world_some_agents_env import FullWorldSomeAgents_GraphDelivery
culture_list = ["Heterogeneity","ComplexHeterogeneity"]

for culture in culture_list:
	register_env(f"GraphDelivery-SeeAllAgents-{culture}", lambda config: FullWorldAllAgents_GraphDelivery({"culture": culture, **config}))
register_env("GraphDelivery-SeeAllAgents", lambda config: FullWorldAllAgents_GraphDelivery({"culture": None, **config}))

for culture in culture_list:
	register_env(f"GraphDelivery-SeePartWorld-{culture}", lambda config: PartWorldSomeAgents_GraphDelivery({"culture": culture, **config}))
register_env("GraphDelivery-SeePartWorld", lambda config: PartWorldSomeAgents_GraphDelivery({"culture": None, **config}))

for culture in culture_list:
	register_env(f"GraphDelivery-{culture}", lambda config: FullWorldSomeAgents_GraphDelivery({"culture": culture, **config}))
register_env("GraphDelivery", lambda config: FullWorldSomeAgents_GraphDelivery({"culture": None, **config}))
