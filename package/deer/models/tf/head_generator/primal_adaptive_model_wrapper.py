from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.models.tf.misc import normc_initializer as tf_normc_initializer
# from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
import gym
import numpy as np

tf1, tf, tfv = try_import_tf()
# torch, nn = try_import_torch()

RNN_SIZE = 512
DIAG_MVMT = False  # Diagonal movements allowed?
A_SIZE = 5 + int(DIAG_MVMT) * 4
KEEP_PROB1 = 1  # was 0.5
KEEP_PROB2 = 1  # was 0.7
GOAL_REPR_SIZE = 12

def get_input_layers_and_keras_layers(obs_space, **args):
	def _build_net(inputs, goal_pos, rnn_size, a_size):
		def conv_mlp(kernel_size, output_size):
			return tf.keras.Sequential(layers=[
				tf.keras.layers.Reshape([-1, 1, kernel_size, 1]),
				tf.keras.layers.Conv2D(
					filters=output_size, 
					kernel_size=kernel_size, 
					strides=1, 
					padding='VALID', 
					activation=tf.nn.relu, 
					data_format="channels_last",
					kernel_initializer=tf.keras.initializers.VarianceScaling(),
				)
			])

		def VGG_Block():
			def conv_2d(kernel_size, output_size):
				return tf.keras.layers.Conv2D(
					filters=output_size, 
					kernel_size=[kernel_size[0], kernel_size[1]], 
					strides=1, 
					padding='SAME', 
					activation=tf.nn.relu, 
					data_format="channels_last",
					kernel_initializer=tf.keras.initializers.VarianceScaling(),
				)
				
			return tf.keras.Sequential(layers=[
				conv_2d([3, 3], rnn_size // 4),
				conv_2d([3, 3], rnn_size // 4),
				conv_2d([3, 3], rnn_size // 4),
				tf.keras.layers.MaxPool2D(pool_size=[2, 2]),
			])

		inputs = tf.transpose(inputs, perm=[0, 2, 3, 1])
		vgg1 = VGG_Block()(inputs)
		vgg2 = VGG_Block()(vgg1)

		flat = tf.keras.Sequential(layers=[
			tf.keras.layers.Conv2D(
				filters=rnn_size - GOAL_REPR_SIZE, 
				kernel_size=[2, 2], 
				strides=1, 
				padding='VALID', 
				activation=tf.nn.relu, 
				data_format="channels_last",
				kernel_initializer=tf.keras.initializers.VarianceScaling(),
			),
			tf.keras.layers.Flatten(),
		])(vgg2)

		goal_layer = tf.keras.layers.Dense(GOAL_REPR_SIZE, activation=tf.nn.relu)(goal_pos)

		hidden_input = tf.keras.layers.Concatenate(axis=1)([flat, goal_layer])

		d2 = tf.keras.Sequential(layers=[
			tf.keras.layers.Dense(rnn_size, activation=tf.nn.relu),
			tf.keras.layers.Dropout(1-KEEP_PROB1),
			tf.keras.layers.Dense(rnn_size, activation=None),
			tf.keras.layers.Dropout(1-KEEP_PROB2),
		])(hidden_input)

		h3 = tf.nn.relu(d2 + hidden_input)
		# Recurrent network for temporal dependencies
		# return tf.keras.Sequential(layers=[
		# 	tf.keras.layers.RNN(
		# 		tf.nn.rnn_cell.BasicLSTMCell(rnn_size, state_is_tuple=True), 
		# 		return_sequences=False, return_state=False, go_backwards=False,
		# 		stateful=False, unroll=False, time_major=False
		# 	),
		# 	tf.keras.layers.Flatten(),
		# ])(tf.expand_dims(self.h3, [0]))
		return tf.keras.layers.Flatten()(h3)

	fc_head = obs_space.original_space['goal']
	goal_pos = tf.keras.layers.Input(shape=fc_head.shape)

	cnn_head = obs_space.original_space['map']
	map_layer = tf.keras.layers.Input(shape=cnn_head.shape)

	inputs = [map_layer,goal_pos]

	last_layer = _build_net(map_layer, goal_pos, RNN_SIZE, A_SIZE)
	return inputs, last_layer

def get_input_list_from_input_dict(input_dict, **args):
	obs = input_dict['obs']
	goal_pos = obs["goal"]
	map_layer = obs["map"]
	this_input = [map_layer,goal_pos]
	return this_input
