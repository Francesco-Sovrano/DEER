import gym
import os
import numpy as np

from ray.rllib.algorithms.ddpg.ddpg_torch_model import DDPGTorchModel
from ray.rllib.models.torch.misc import SlimFC
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.utils import get_activation_fn
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.utils.typing import ModelConfigDict, TensorType
from ray.rllib.models.modelv2 import restore_original_dimensions

torch, nn = try_import_torch()
# torch.set_num_threads(os.cpu_count())
# torch.set_num_interop_threads(os.cpu_count())

class TorchAdaptiveMultiHeadDDPG:

	@staticmethod
	def init(preprocessing_model):
		class TorchAdaptiveMultiHeadDDPGInner(DDPGTorchModel):

			policy_signature_size = 3

			def __init__(self, obs_space, action_space, num_outputs, model_config, name, actor_hiddens=None, actor_hidden_activation="relu", critic_hiddens=None, critic_hidden_activation="relu", twin_q=False, add_layer_norm=False):
				num_outputs = preprocessing_model(obs_space, model_config['custom_model_config']).get_num_outputs()
				self.add_nonstationarity_correction = model_config['custom_model_config'].get("add_nonstationarity_correction", False)
				if self.add_nonstationarity_correction:
					print("Adding nonstationarity corrections")
					num_outputs += self.policy_signature_size
				super().__init__(obs_space, action_space, num_outputs, model_config, name, actor_hiddens, actor_hidden_activation, critic_hiddens, critic_hidden_activation, twin_q, add_layer_norm)
				self.preprocessing_model_policy = preprocessing_model(obs_space, model_config['custom_model_config'])
				self.preprocessing_model_q = preprocessing_model(obs_space, model_config['custom_model_config'])
				self.preprocessing_model_twin_q = preprocessing_model(obs_space, model_config['custom_model_config'])

			# def __call__(self, input_dict, state = None, seq_lens = None):
			# 	print('u',input_dict)
			# 	return super().__call__(input_dict, state, seq_lens)

			def forward(self, input_dict, state, seq_lens):
				if self.add_nonstationarity_correction:
					if "policy_signature" not in input_dict:
						print("Adding dummy policy_signature")
						input_dict["policy_signature"] = torch.from_numpy(np.zeros((input_dict.count,self.policy_signature_size), dtype=np.float32))
					return {"obs":input_dict["obs"],"policy_signature":input_dict["policy_signature"]}, state
				return input_dict["obs"], state

			def get_policy_output(self, model_out):
				if self.add_nonstationarity_correction:
					model_out = torch.concat((
						self.preprocessing_model_policy(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_policy(model_out)
				return self.policy_model(model_out)

			def get_q_values(self, model_out, actions = None):
				if self.add_nonstationarity_correction:
					model_out = torch.concat((
						self.preprocessing_model_q(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_q(model_out)
				return self.q_model(torch.cat([model_out, actions], -1))

			def get_twin_q_values(self, model_out, actions = None):
				if self.add_nonstationarity_correction:
					model_out = torch.concat((
						self.preprocessing_model_twin_q(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_twin_q(model_out)
				return self.twin_q_model(torch.cat([model_out, actions], -1))

			def policy_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_policy.variables(as_dict) + super().policy_variables(as_dict)
				p_dict = super().policy_variables(as_dict)
				p_dict.update(self.preprocessing_model_policy.variables(as_dict))
				return p_dict

			def q_variables(self, as_dict=False):
				if as_dict:
					return {
						**self.preprocessing_model_q.state_dict(),
						**self.q_model.state_dict(),
						**(self.preprocessing_model_twin_q.state_dict() if self.twin_q_model else {}),
						**(self.twin_q_model.state_dict() if self.twin_q_model else {}),
					}
				return list(self.preprocessing_model_q.parameters()) + list(self.q_model.parameters()) + (
					list(self.preprocessing_model_twin_q.parameters()) if self.twin_q_model else []
				) + (
					list(self.twin_q_model.parameters()) if self.twin_q_model else []
				)

			def get_entropy_var(self):
				return None

		return TorchAdaptiveMultiHeadDDPGInner
