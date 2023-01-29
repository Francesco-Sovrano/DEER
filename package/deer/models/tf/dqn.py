from ray.rllib.algorithms.dqn.distributional_q_tf_model import DistributionalQTFModel

from ray.rllib.utils.framework import try_import_tf
tf1, tf, tfv = try_import_tf()

class TFAdaptiveMultiHeadDQN:

	@staticmethod
	def init(get_input_layers_and_keras_layers, get_input_list_from_input_dict):
		class TFAdaptiveMultiHeadDQNInner(DistributionalQTFModel):
			def __init__(self, obs_space, action_space, num_outputs, model_config, name, q_hiddens = (256, ), dueling = False, num_atoms = 1, use_noisy = False, v_min = -10.0, v_max = 10.0, sigma0 = 0.5, add_layer_norm = False):
				inputs, last_layer = get_input_layers_and_keras_layers(obs_space)
				self.preprocessing_model = tf.keras.Model(inputs, last_layer)
				# self.register_variables(self.preprocessing_model.variables)
				super().__init__(
					obs_space=obs_space, 
					action_space=action_space, 
					num_outputs=last_layer.shape[1], 
					model_config=model_config, 
					name=name, 
					q_hiddens=q_hiddens, 
					dueling=dueling, 
					num_atoms=num_atoms, 
					use_noisy=use_noisy, 
					v_min=v_min, 
					v_max=v_max, 
					sigma0=sigma0, 
					add_layer_norm=add_layer_norm
				)

			def forward(self, input_dict, state, seq_lens):
				model_out = self.preprocessing_model(get_input_list_from_input_dict(input_dict))
				return model_out, state

			def variables(self, as_dict=False):
				if not as_dict:
					return self.preprocessing_model.variables(as_dict) + super().variables(as_dict)
				v = self.preprocessing_model.variables(as_dict)
				v.update(super().variables(as_dict))
				return v

			def get_entropy_var(self):
				return None

		return TFAdaptiveMultiHeadDQNInner
