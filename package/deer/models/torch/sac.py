import os
import gym
import numpy as np

from ray.rllib.algorithms.sac.sac_torch_model import SACTorchModel
from ray.rllib.utils.framework import try_import_torch
from ray.rllib.policy.view_requirement import ViewRequirement

torch, nn = try_import_torch()
# torch.set_num_threads(os.cpu_count())
# torch.set_num_interop_threads(os.cpu_count())

class TorchAdaptiveMultiHeadNet:

	@staticmethod
	def init(policy_preprocessing_model,value_preprocessing_model):
		class TorchAdaptiveMultiHeadNetInner(SACTorchModel):
			"""
			Data flow:
			`obs` -> forward() (should stay a noop method!) -> `model_out`
			`model_out` -> get_policy_output() -> pi(actions|obs)
			`model_out`, `actions` -> get_q_values() -> Q(s, a)
			`model_out`, `actions` -> get_twin_q_values() -> Q_twin(s, a)
			"""
			policy_signature_size = 3

			def __init__(self,obs_space,action_space,num_outputs,model_config,name,policy_model_config = None,q_model_config = None,twin_q = False,initial_alpha = 1.0,target_entropy = None):
				self.add_nonstationarity_correction = model_config['custom_model_config'].get("add_nonstationarity_correction", False)
				if self.add_nonstationarity_correction:
					print("Adding nonstationarity corrections")
				super().__init__(
					obs_space,
					action_space,
					num_outputs,
					model_config,
					name,
					policy_model_config = policy_model_config,
					q_model_config = q_model_config,
					twin_q = twin_q,
					initial_alpha = initial_alpha,
					target_entropy = target_entropy
				)

			def forward(self, input_dict, state, seq_lens):
				if self.add_nonstationarity_correction:
					# print(input_dict)
					if "policy_signature" not in input_dict:
						print("Adding dummy policy_signature")
						input_dict["policy_signature"] = torch.from_numpy(np.zeros((input_dict.count,self.policy_signature_size), dtype=np.float32))
					return {"obs":input_dict["obs"],"policy_signature":input_dict["policy_signature"]}, state
				return super().forward(input_dict, state, seq_lens)

			def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
				self.preprocessing_model_policy = policy_preprocessing_model(obs_space, self.model_config['custom_model_config'])
				preprocessed_input_size = self.preprocessing_model_policy.get_num_outputs()
				if self.add_nonstationarity_correction:
					preprocessed_input_size += self.policy_signature_size
				preprocessed_obs_space_policy = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(preprocessed_input_size,), dtype=np.float32)
				model = super().build_policy_model(preprocessed_obs_space_policy, num_outputs, policy_model_config, name)
				return model

			def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
				if name == "twin_q":
					self.preprocessing_model_twin_q = value_preprocessing_model(obs_space, self.model_config['custom_model_config'])
					preprocessed_input_size = self.preprocessing_model_twin_q.get_num_outputs()
				elif name == "q":
					self.preprocessing_model_q = value_preprocessing_model(obs_space, self.model_config['custom_model_config'])
					preprocessed_input_size = self.preprocessing_model_q.get_num_outputs()
				if self.add_nonstationarity_correction:
					preprocessed_input_size += self.policy_signature_size
				preprocessed_obs_space_q = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=(preprocessed_input_size,), dtype=np.float32)
				model = super().build_q_model(preprocessed_obs_space_q, action_space, num_outputs, q_model_config, name)
				return model

			def get_policy_output(self, model_out):
				if self.add_nonstationarity_correction:
					# print(model_out["policy_signature"].shape, self.preprocessing_model_policy(model_out["obs"]).shape)
					model_out = torch.concat((
						self.preprocessing_model_policy(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_policy(model_out)
				return super().get_policy_output(model_out)

			def get_action_model_outputs(self, model_out, state_in=None, seq_lens=None):
				if self.add_nonstationarity_correction:
					# print(model_out["policy_signature"].shape, self.preprocessing_model_policy(model_out["obs"]).shape)
					model_out = torch.concat((
						self.preprocessing_model_policy(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_policy(model_out)
				return super().get_action_model_outputs(model_out, state_in=state_in, seq_lens=seq_lens)

			def get_q_values(self, model_out, actions = None):
				if self.add_nonstationarity_correction:
					model_out = torch.concat((
						self.preprocessing_model_q(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_q(model_out)
				return self._get_q_value(model_out, actions, self.q_net)

			def get_twin_q_values(self, model_out, actions = None):
				if self.add_nonstationarity_correction:
					model_out = torch.concat((
						self.preprocessing_model_twin_q(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.preprocessing_model_twin_q(model_out)
				return self._get_q_value(model_out, actions, self.twin_q_net)

			def policy_variables(self):
				return self.preprocessing_model_policy.variables() + super().policy_variables()

			def q_variables(self):
				q_vars = self.preprocessing_model_q.variables() + self.q_net.variables()
				if not self.twin_q_net:
					return q_vars
				# sac_torch_policy::optimizer_fn uses index to separate q variables from twin_q variables. So, here, variables should be returned accordingly
				twin_q_vars = self.preprocessing_model_twin_q.variables() + self.twin_q_net.variables()
				return q_vars + twin_q_vars

			def get_entropy_var(self):
				alpha = np.exp(self.log_alpha.detach().numpy())
				return alpha

		return TorchAdaptiveMultiHeadNetInner
