from ray.rllib.algorithms.dqn.dqn_torch_model import DQNTorchModel
from ray.rllib.utils.framework import try_import_torch
import os
import numpy as np

torch, nn = try_import_torch()
# torch.set_num_threads(os.cpu_count())
# torch.set_num_interop_threads(os.cpu_count())

class TorchAdaptiveMultiHeadDQN:

	@staticmethod
	def init(preprocessing_model):
		class TorchAdaptiveMultiHeadDQNInner(DQNTorchModel):
			
			policy_signature_size = 3

			def __init__(self, obs_space, action_space, num_outputs, model_config, name, *, q_hiddens = (256,), dueling = False, dueling_activation = "relu", num_atoms = 1, use_noisy = False, v_min = -10.0, v_max = 10.0, sigma0 = 0.5, add_layer_norm = False):
				preprocessed_input_size = preprocessing_model(obs_space, model_config['custom_model_config']).get_num_outputs()
				self.add_nonstationarity_correction = model_config['custom_model_config'].get("add_nonstationarity_correction", False)
				if self.add_nonstationarity_correction:
					print("Adding nonstationarity corrections")
					preprocessed_input_size += self.policy_signature_size
				super().__init__(
					obs_space=obs_space, 
					action_space=action_space, 
					num_outputs=preprocessed_input_size, 
					model_config=model_config, 
					name=name, 
					q_hiddens=q_hiddens, 
					dueling=dueling, 
					dueling_activation=dueling_activation,
					num_atoms=num_atoms, 
					use_noisy=use_noisy, 
					v_min=v_min, 
					v_max=v_max, 
					sigma0=sigma0, 
					add_layer_norm=add_layer_norm
				)
				self.preprocessing_model = preprocessing_model(obs_space, model_config['custom_model_config'])

			def forward(self, input_dict, state, seq_lens):
				model_out = self.preprocessing_model(input_dict['obs'])
				if self.add_nonstationarity_correction:
					if "policy_signature" not in input_dict:
						print("Adding dummy policy_signature")
						input_dict["policy_signature"] = torch.from_numpy(np.zeros((input_dict.count,self.policy_signature_size), dtype=np.float32))
					model_out = torch.concat((model_out,input_dict['policy_signature']), dim=-1)
				return model_out, state

			def variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model.variables(as_dict) + super().variables(as_dict)
				v = self.preprocessing_model.variables(as_dict)
				v.update(super().variables(as_dict))
				return v

			def get_entropy_var(self):
				return None

		return TorchAdaptiveMultiHeadDQNInner
