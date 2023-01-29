from ray.rllib.utils.framework import try_import_tf
# from ray.rllib.utils.framework import get_activation_fn, try_import_torch
from ray.rllib.models.tf.misc import normc_initializer as tf_normc_initializer
# from ray.rllib.models.torch.misc import normc_initializer as torch_normc_initializer
import gym
import numpy as np
import logging
import itertools
logger = logging.getLogger(__name__)

tf1, tf, tfv = try_import_tf()

KERAS_LAYER_DICT = {}
def get_from_keras_layers_dict(key, build_fn):
	if key not in KERAS_LAYER_DICT:
		logger.warning(f'Building layer: {key}')
		KERAS_LAYER_DICT[key] = build_fn(key)
	return KERAS_LAYER_DICT[key]

def get_input_layers_and_keras_layers(obs_space, **args):
	cnn_inputs = []
	cnn_heads = []
	fc_inputs = []
	fc_heads = []
	if hasattr(obs_space, 'original_space'):
		obs_space = obs_space.original_space

	def build_inputs(_obs_space, _type):
		_heads = []
		_inputs_dict = {}
		for _key in filter(lambda x: x.startswith(_type), sorted(_obs_space.spaces.keys())):
			obs_original_space = _obs_space[_key]
			if isinstance(obs_original_space, gym.spaces.Dict):
				space_iter = obs_original_space.spaces.items()
				_permutation_invariant = False
			else:
				if not isinstance(obs_original_space, gym.spaces.Tuple):
					obs_original_space = [obs_original_space]
				space_iter = enumerate(obs_original_space)
				_permutation_invariant = True
			_inputs = [
				tf.keras.layers.Input(shape=_head.shape)
				for _name,_head in space_iter
			]
			_inputs_dict[(_key,_permutation_invariant)] = _inputs
		return _inputs_dict

	def build_heads(_inputs_dict, _layers_build_fn, _layers_aggregator_fn):
		_heads = []
		for (_key,_permutation_invariant),_inputs in _inputs_dict.items():
			_layers = _layers_build_fn(_key,_inputs)
			_heads.append(_layers_aggregator_fn(_key,_layers,_permutation_invariant))
		return _heads

	def fc_layers_aggregator_fn(_key,_layers,_permutation_invariant): # Permutation invariant aggregator
		assert _layers
		assert _key
		_splitted_units = _key.split('-')
		_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
		if not _units:
			# logger.warning('No units specified: concatenating inputs')
			return tf.keras.layers.Concatenate(axis=-1)(_layers)

		#### FC net
		if len(_layers) <= 1:
			# logger.warning(f'Building dense layer with {_units} units on 1 layer')
			return get_from_keras_layers_dict(
				f'fc_layers_aggregator_fn_dense_1_{_key}', 
				lambda n: tf.keras.layers.Dense(_units, activation='relu', name=n)
			)(_layers[0])

		#### Concat net
		if not _permutation_invariant:
			# logger.warning(f'Building concat layer with {_units} units on {len(_layers)} layers')
			return get_from_keras_layers_dict(
				f'fc_layers_aggregator_fn_shared_hypernet_layer_{_key}', 
				lambda n: tf.keras.Sequential(name=n, layers=[
					tf.keras.layers.Concatenate(axis=-1),
					tf.keras.layers.Dense(_units, activation='relu')
				])
			)(_layers)

		#### Permutation Invariant net
		# logger.warning(f'Building permutation invariant layer with {_units} units on {len(_layers)} layers')
		k = _layers[0].shape[-1]
		_shared_hypernet_layer = get_from_keras_layers_dict(
			f'fc_layers_aggregator_fn_shared_hypernet_layer_{_key}_{k}', 
			lambda n: tf.keras.Sequential(name=n, layers=[
				tf.keras.layers.Dense(k*_units, activation='relu'),
				# tf.keras.layers.Dense(k*_units, activation='sigmoid'),
				tf.keras.layers.Reshape((k,_units)),
			])
		)
		
		_weights = list(map(_shared_hypernet_layer,_layers))
		
		_layers = list(map(tf.keras.layers.Reshape((1, k)), _layers))
		_layers = [
			tf.linalg.matmul(l,w)
			for l,w in zip(_layers,_weights)
		]
		_layers = list(map(tf.keras.layers.Flatten(), _layers))
		return tf.keras.layers.Add()(_layers)
		# _shared_layer = tf.keras.layers.Dense(_units, activation='relu')
		# _layers = list(map(_shared_layer, _layers))
		# return tf.keras.layers.Maximum()(_layers)

	def cnn_layers_build_fn(_key,_inputs):
		return [
			get_from_keras_layers_dict(
				f"cnn_layers_build_fn_{_key}_layer{i}", 
				lambda n: tf.keras.Sequential(name=n, layers=[
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv1',  filters=32, kernel_size=8, strides=4, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv2',  filters=64, kernel_size=4, strides=2, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Conv2D(name=f'CNN{i}_Conv3',  filters=64, kernel_size=4, strides=1, padding='SAME', activation=tf.nn.relu, kernel_initializer=tf_normc_initializer(1.0)),
					tf.keras.layers.Flatten(),
				])
			)(layer)
			for i,layer in enumerate(_inputs)
		]

	def fc_layers_build_fn(_key,_inputs):
		return [
			tf.keras.layers.Flatten()(layer)
			for i,layer in enumerate(_inputs)
		]

	def apply_obs_to_main_model(obs):
		logger.warning('Applying obs to main model..')
		cnn_inputs_dict = build_inputs(obs, 'cnn')
		fc_inputs_dict = build_inputs(obs, 'fc')
		inputs = [i for i_list in cnn_inputs_dict.values() for i in i_list]+[i for i_list in fc_inputs_dict.values() for i in i_list]
		if inputs:
			cnn_heads = build_heads(cnn_inputs_dict, cnn_layers_build_fn, fc_layers_aggregator_fn)
			if len(cnn_heads) > 1: 
				cnn_heads = [tf.keras.layers.Concatenate(axis=-1)(cnn_heads)]
			fc_heads = build_heads(fc_inputs_dict, fc_layers_build_fn, fc_layers_aggregator_fn)
			if len(fc_heads) > 1: 
				fc_heads = [tf.keras.layers.Concatenate(axis=-1)(fc_heads)]

			last_layer = fc_heads + cnn_heads
		
			if len(last_layer) > 1: last_layer = tf.keras.layers.Concatenate()(last_layer)
			else: last_layer = last_layer[0]
			last_layer = tf.keras.layers.Flatten()(last_layer)
		else:
			logger.warning('N.B.: Flattening all observations!')
			def get_input_recursively(_obs_space):
				if isinstance(_obs_space, gym.spaces.Dict):
					return list(itertools.chain.from_iterable(map(get_input_recursively,_obs_space.spaces.values())))
				elif isinstance(_obs_space, gym.spaces.Tuple):
					return list(itertools.chain.from_iterable(map(get_input_recursively,_obs_space.spaces)))
				return [tf.keras.layers.Input(shape=_obs_space.shape)]
				
			inputs = get_input_recursively(obs)
			last_layer = list(map(tf.keras.layers.Flatten(), inputs))
			last_layer = tf.keras.layers.Concatenate(axis=-1)(last_layer)
		return inputs, last_layer
	
	return apply_obs_to_main_model(obs_space)

def get_input_list_from_input_dict(input_dict, **args):
	obs = input_dict['obs']
	assert isinstance(obs, dict)
	cnn_inputs = []
	fc_inputs = []
	other_inputs = []
	for k,v in sorted(obs.items(), key=lambda x:x[0]):
		if k.startswith("cnn"):
			cnn_inputs.append(v)
		elif k.startswith("fc"):
			fc_inputs.append(v)
		else:
			other_inputs.append(v)
	input_list = cnn_inputs + fc_inputs + other_inputs
	flattened_input_list = []
	for i in input_list:
		if isinstance(i,dict):
			flattened_input_list += i.values()
		elif isinstance(i,list):
			flattened_input_list += i
		else:
			flattened_input_list.append(i)
	return flattened_input_list
