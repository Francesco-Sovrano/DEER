from ray.rllib.algorithms.sac.sac_tf_model import SACTFModel

import numpy as np
import gym
from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()


class TFAdaptiveMultiHeadNet:

	@staticmethod
	def init(get_input_layers_and_keras_layers, get_input_list_from_input_dict):
		class TFAdaptiveMultiHeadNetInner(SACTFModel):
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

			def build_policy_model(self, obs_space, num_outputs, policy_model_config, name):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.policy_preprocessing_model = tf.keras.Model(inputs, last_layer)
				self.policy_preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=last_layer.shape[1:], dtype=np.float32)
				return super().build_policy_model(self.policy_preprocessed_obs_space, num_outputs, policy_model_config, name)

			def build_q_model(self, obs_space, action_space, num_outputs, q_model_config, name):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.value_preprocessing_model = tf.keras.Model(inputs, last_layer)
				self.value_preprocessed_obs_space = gym.spaces.Box(low=float('-inf'), high=float('inf'), shape=last_layer.shape[1:], dtype=np.float32)
				return super().build_q_model(self.value_preprocessed_obs_space, action_space, num_outputs, q_model_config, name)

			def get_policy_output(self, model_out):
				model_out = self.policy_preprocessing_model(get_input_list_from_input_dict({"obs": model_out}))
				return super().get_policy_output(model_out)

			def get_action_model_outputs(self, model_out, state_in=None, seq_lens=None):
				if self.add_nonstationarity_correction:
					# print(model_out["policy_signature"].shape, self.policy_preprocessing_model(model_out["obs"]).shape)
					model_out = torch.concat((
						self.policy_preprocessing_model(model_out["obs"]),
						model_out["policy_signature"]
					), dim=-1)
				else:
					model_out = self.policy_preprocessing_model(model_out)
				return super().get_action_model_outputs(model_out, state_in=state_in, seq_lens=seq_lens)

			def get_q_values(self, model_out, actions = None):
				model_out = self.value_preprocessing_model(model_out)
				return self._get_q_value(model_out, actions, self.q_net)

			def get_twin_q_values(self, model_out, actions = None):
				model_out = self.value_preprocessing_model(model_out)
				return self._get_q_value(model_out, actions, self.twin_q_net)

			def policy_variables(self):
				return self.policy_preprocessing_model.variables() + super().policy_variables()

			def q_variables(self):
				return self.value_preprocessing_model.variables() + super().q_variables()

		return TFAdaptiveMultiHeadNetInner
