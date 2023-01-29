"""
XADDPG - eXplanation-Aware Deep Deterministic Policy Gradient
==============================================

Detailed documentation:
https://docs.ray.io/en/master/rllib-algorithms.html#deep-deterministic-policy-gradients-ddpg-td3
"""  # noqa: E501

from deer.agents.xadqn import init_xa_config, PolicySignatureListCollector, XADQN
from ray.rllib.utils.annotations import override
from ray.rllib.algorithms.ddpg.ddpg import DDPGConfig
from ray.rllib.algorithms.td3.td3 import TD3Config

class XADDPGConfig(DDPGConfig):

	def __init__(self, algo_class=None):
		"""Initializes a DDPGConfig instance."""
		super().__init__(algo_class=algo_class or XADDPG)

		init_xa_config(self)
		self.n_step = 1
		self.buffer_options['clustering_xi'] = 4

	@override(DDPGConfig)
	def validate(self):
		# Call super's validation method.
		super().validate()

		if self.model["custom_model_config"].get("add_nonstationarity_correction", False):
			self.sample_collector = PolicySignatureListCollector

class XATD3Config(TD3Config):

	def __init__(self, algo_class=None):
		"""Initializes a TD3Config instance."""
		super().__init__(algo_class=algo_class or XATD3)

		init_xa_config(self)
		self.n_step = 1
		self.buffer_options['clustering_xi'] = 4

	@override(TD3Config)
	def validate(self):
		# Call super's validation method.
		super().validate()

		if self.model["custom_model_config"].get("add_nonstationarity_correction", False):
			self.sample_collector = PolicySignatureListCollector

########################
# XADDPG Execution Plan
########################

class XADDPG(XADQN):
	@classmethod
	@override(XADQN)
	def get_default_config(cls):
		return XADDPGConfig()

	@classmethod
	@override(XADQN)
	def get_default_policy_class(self, config):
		if config['framework'] == "torch":
			from deer.agents.xaddpg.xaddpg_torch_policy import XADDPGTorchPolicy
			return XADDPGTorchPolicy
		elif config["framework"] == "tf":
			from deer.agents.xaddpg.xaddpg_tf_policy import XADDPGTF1Policy
			return XADDPGTF1Policy
		else:
			from deer.agents.xaddpg.xaddpg_tf_policy import XADDPGTF2Policy
			return XADDPGTF2Policy

class XATD3(XADQN):
	@classmethod
	@override(XADQN)
	def get_default_config(cls):
		return XATD3Config()

	@classmethod
	@override(XADQN)
	def get_default_policy_class(self, config):
		if config['framework'] == "torch":
			from deer.agents.xaddpg.xaddpg_torch_policy import XADDPGTorchPolicy
			return XADDPGTorchPolicy
		elif config["framework"] == "tf":
			from deer.agents.xaddpg.xaddpg_tf_policy import XADDPGTF1Policy
			return XADDPGTF1Policy
		else:
			from deer.agents.xaddpg.xaddpg_tf_policy import XADDPGTF2Policy
			return XADDPGTF2Policy
