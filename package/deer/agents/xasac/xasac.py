"""
XASAC - eXplanation-Aware Soft Actor-Critic
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from deer.agents.xadqn import init_xa_config, PolicySignatureListCollector, XADQN
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.sac.sac import SACConfig
from deer.agents.xasac.xasac_tf_policy import XASACTFPolicy
from deer.agents.xasac.xasac_torch_policy import XASACTorchPolicy

class XASACConfig(SACConfig):

	def __init__(self, algo_class=None):
		"""Initializes a DQNConfig instance."""
		super().__init__(algo_class=algo_class or XASAC)

		init_xa_config(self)
		self.n_step = 1
		self.buffer_options['clustering_xi'] = 4

	@override(SACConfig)
	def validate(self):
		# Call super's validation method.
		super().validate()

		if self.model["custom_model_config"].get("add_nonstationarity_correction", False):
			self.sample_collector = PolicySignatureListCollector

########################
# XASAC Policy
########################

class XASAC(XADQN):
	def __init__(self, *args, **kwargs):
		self._allow_unknown_subkeys += ["policy_model_config", "q_model_config"]
		super().__init__(*args, **kwargs)

	@classmethod
	@override(XADQN)
	def get_default_config(cls):
		return XASACConfig()

	@classmethod
	@override(XADQN)
	def get_default_policy_class(self, config):
		return XASACTorchPolicy if config['framework'] == "torch" else XASACTFPolicy
