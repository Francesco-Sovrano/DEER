from ray.tune.registry import register_env
######### Add new environment below #########

###########################################################################
### Multi-Agent Graph Delivery
from .env.multi_agent_graph_delivery.full_world_all_agents_env import FullWorldAllAgents_GraphDelivery
from .env.multi_agent_graph_delivery.part_world_some_agents_env import PartWorldSomeAgents_GraphDelivery
from .env.multi_agent_graph_delivery.full_world_some_agents_env import FullWorldSomeAgents_GraphDelivery
culture_list = ["Heterogeneity","ComplexHeterogeneity"]

for culture in culture_list:
	register_env(f"MAGraphDelivery-FullWorldAllAgents-{culture}", lambda config: FullWorldAllAgents_GraphDelivery({"culture": culture, **config}))
register_env("MAGraphDelivery-FullWorldAllAgents", lambda config: FullWorldAllAgents_GraphDelivery({"culture": None, **config}))

for culture in culture_list:
	register_env(f"MAGraphDelivery-PartWorldSomeAgents-{culture}", lambda config: PartWorldSomeAgents_GraphDelivery({"culture": culture, **config}))
register_env("MAGraphDelivery-PartWorldSomeAgents", lambda config: PartWorldSomeAgents_GraphDelivery({"culture": None, **config}))

for culture in culture_list:
	register_env(f"MAGraphDelivery-FullWorldSomeAgents-{culture}", lambda config: FullWorldSomeAgents_GraphDelivery({"culture": culture, **config}))
register_env("MAGraphDelivery-FullWorldSomeAgents", lambda config: FullWorldSomeAgents_GraphDelivery({"culture": None, **config}))
