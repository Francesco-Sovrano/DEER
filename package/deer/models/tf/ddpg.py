from ray.rllib.algorithms.ddpg.ddpg_tf_model import DDPGTFModel

from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

class TFAdaptiveMultiHeadDDPG:

	@staticmethod
	def init(get_input_layers_and_keras_layers, get_input_list_from_input_dict):
		class TFAdaptiveMultiHeadDDPGInner(DDPGTFModel):
			def __init__(self, obs_space, action_space, num_outputs, model_config, name, actor_hiddens=(256, 256), actor_hidden_activation="relu", critic_hiddens=(256, 256), critic_hidden_activation="relu", twin_q=False, add_layer_norm=False):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.preprocessing_model = tf.keras.Model(inputs, last_layer)
				# self.register_variables(self.preprocessing_model.variables)
				super().__init__(
					obs_space=obs_space, 
					action_space=action_space, 
					num_outputs=last_layer.shape[1], 
					model_config=model_config, 
					name=name, 
					actor_hiddens=actor_hiddens, 
					actor_hidden_activation=actor_hidden_activation, 
					critic_hiddens=critic_hiddens, 
					critic_hidden_activation=critic_hidden_activation, 
					twin_q=twin_q, 
					add_layer_norm=add_layer_norm
				)

			def forward(self, input_dict, state, seq_lens):
				return input_dict["obs"], state

			def get_policy_output(self, model_out):
				model_out = self.preprocessing_model_policy(model_out)
				return self.policy_model(model_out)

			def get_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self.q_model(torch.cat([model_out, actions], -1))

			def get_twin_q_values(self, model_out, actions = None):
				model_out = self.preprocessing_model_q(model_out)
				return self.twin_q_model(torch.cat([model_out, actions], -1))

			def policy_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_policy.variables(as_dict) + super().policy_variables(as_dict)
				p_dict = super().policy_variables(as_dict)
				p_dict.update(self.preprocessing_model_policy.variables(as_dict))
				return p_dict

			def q_variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model_q.variables(as_dict) + super().q_variables(as_dict)
				q_dict = super().q_variables(as_dict)
				q_dict.update(self.preprocessing_model_q.variables(as_dict))
				return q_dict

			def get_entropy_var(self):
				return None

		return TFAdaptiveMultiHeadDDPGInner
