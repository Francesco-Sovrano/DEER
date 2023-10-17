from ray.rllib.utils.framework import try_import_torch
import gym
import numpy as np
import logging
import itertools

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()


def get_input_recursively(_obs_space, valid_key_fn=lambda x: True):
	if isinstance(_obs_space, gym.spaces.Dict):
		space_iter = (v for k,v in _obs_space.spaces.items() if valid_key_fn(k))
		return list(itertools.chain.from_iterable(map(get_input_recursively,space_iter)))
	elif isinstance(_obs_space, gym.spaces.Tuple):
		return list(itertools.chain.from_iterable(map(get_input_recursively,_obs_space.spaces)))
	elif isinstance(_obs_space, dict):
		space_iter = (v for k,v in _obs_space.items() if valid_key_fn(k))
		return list(itertools.chain.from_iterable(map(get_input_recursively,space_iter)))
	elif isinstance(_obs_space, (list,tuple)):
		return list(itertools.chain.from_iterable(map(get_input_recursively,_obs_space)))
	return [_obs_space]


class Permute(nn.Module):

	def __init__(self, dims):
		super(Permute, self).__init__()
		self.dims = dims

	def forward(self, x):
		return x.permute(self.dims)


class AdaptiveModel(nn.Module):
	def __init__(self, obs_space, config):
		super().__init__()
		if hasattr(obs_space, 'original_space'):
			obs_space = obs_space.original_space

		self.obs_space = obs_space
		self._num_outputs = None
		self.sub_model_dict = {}

		###### FC
		fc_inputs_shape_dict = self.get_inputs_shape_dict(obs_space, 'fc')
		# print(1, fc_inputs_shape_dict)
		if fc_inputs_shape_dict:
			self.sub_model_dict['fc'] = [
				self.fc_head_build(_key,_input_list)
				for (_key,_),_input_list in fc_inputs_shape_dict.items()
			]

		###### CNN
		cnn_inputs_shape_dict = self.get_inputs_shape_dict(obs_space, 'cnn')
		if cnn_inputs_shape_dict:
			self.sub_model_dict['cnn'] = [
				self.cnn_head_build(_key, _input_list)
				for (_key, _), _input_list in cnn_inputs_shape_dict.items()
			]

		###### Others
		other_inputs_list = get_input_recursively(obs_space, lambda k: not k.startswith('fc') and not k.startswith('cnn'))
		if other_inputs_list:
			self.sub_model_dict[''] = [[nn.Flatten()] for l in other_inputs_list]
		if not self.sub_model_dict.get('cnn',None) and not self.sub_model_dict.get('fc',None):
			assert other_inputs_list
			logger.warning('N.B.: Flattening all observations!')

	def variables(self, as_dict = False):
		if as_dict:
			return self.state_dict()
		return list(self.parameters())

	def forward(self, x):
		output_list = []
		inputs_dict = self.get_inputs_dict(x)
		for _key,_input_list in inputs_dict.items():
			sub_output_list = []
			for _sub_input_list,_model_list in zip(_input_list, self.sub_model_dict[_key]):
				key_output_list = [
					_model(_input)
					for _input,_model in zip(_sub_input_list, _model_list)
				]
				key_output = torch.cat(key_output_list, -1) if len(key_output_list)>1 else key_output_list[0]
				key_output = torch.flatten(key_output, start_dim=1)
				sub_output_list.append(key_output)
			output_list.append(torch.cat(sub_output_list, -1) if len(sub_output_list)>1 else sub_output_list[0])
		output = torch.cat(output_list, -1) if len(output_list)>1 else output_list[0]
		output = torch.flatten(output, start_dim=1)
		return output

	def get_num_outputs(self):
		if self._num_outputs is None:
			def get_random_input_recursively(_obs_space):
				if isinstance(_obs_space, gym.spaces.Dict):
					return {
						k: get_random_input_recursively(v)
						for k,v in _obs_space.spaces.items()
					}
				elif isinstance(_obs_space, gym.spaces.Tuple):
					return list(map(get_random_input_recursively, _obs_space.spaces))
				return torch.rand(1,*_obs_space.shape)
			random_obs = get_random_input_recursively(self.obs_space)
			self._num_outputs = self.forward(random_obs).shape[-1]
		return self._num_outputs

	@staticmethod
	def get_inputs_shape_dict(_obs_space, _type):
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
				_head.shape
				for _name,_head in space_iter
			]
			_inputs_dict[(_key,_permutation_invariant)] = _inputs
		return _inputs_dict

	@staticmethod
	def get_inputs_dict(_obs):
		assert isinstance(_obs, dict)
		
		inputs_dict = {}
		for k,v in sorted(_obs.items(), key=lambda x:x[0]):
			if k.startswith('cnn'):
				if 'cnn' not in inputs_dict:
					inputs_dict['cnn'] = []
				inputs_dict['cnn'].append(v)
			elif k.startswith('fc'):
				if 'fc' not in inputs_dict:
					inputs_dict['fc'] = []
				inputs_dict['fc'].append(v)
			else:
				if '' not in inputs_dict:
					inputs_dict[''] = []
				inputs_dict[''] += get_input_recursively(v, lambda k: not k.startswith('fc') and not k.startswith('cnn'))
			
		for _type,_input_list in inputs_dict.items():
			inputs_dict[_type] = [
				list(i.values())
				if isinstance(i,dict) else
				(
					i
					if isinstance(i,list) else
					[i]
				)
				for i in _input_list
			]
		return inputs_dict

	@staticmethod
	def cnn_head_build(_key,_input_list):
		_splitted_units = _key.split('-')
		_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
		return [
			nn.Sequential(
				Permute((0, 3, 1, 2)),
				nn.Conv2d(in_channels=input_shape[-1] , out_channels=_units//2+1, kernel_size=9, stride=4, padding=4),
				nn.ReLU(),
				nn.Conv2d(in_channels=_units//2+1 , out_channels=_units, kernel_size=5, stride=2, padding=2),
				nn.ReLU(),
				nn.Conv2d(in_channels=_units , out_channels=_units, kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Flatten(),
			)
			for i,input_shape in enumerate(_input_list)
		]

	@staticmethod
	def fc_head_build(_key,_input_list):
		_splitted_units = _key.split('-')
		# print(_splitted_units, _input_list)
		_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
		return [
			nn.Sequential(
				nn.Flatten(),
				nn.Linear(in_features=np.prod(input_shape, dtype='int'), out_features=_units),
				nn.ReLU(),
			)
			for i,input_shape in enumerate(_input_list)
		]


class SiameseAdaptiveModel(nn.Module):
	def __init__(self, obs_space, config):
		super().__init__()
		if hasattr(obs_space, 'original_space'):
			obs_space = obs_space.original_space

		self.obs_space = obs_space
		self._num_outputs = None
		self.sub_model_dict = {}

		assert isinstance(obs_space, gym.spaces.Dict), 'SiameseAdaptiveModel only works with Dict observation spaces.'

		super_dict = {k: {} for k in obs_space.spaces.keys()}
		for k, v in obs_space.spaces.items():
			if isinstance(v, gym.spaces.Dict):
				super_dict[k]['fc_inputs_shape_dict'] = self.get_inputs_shape_dict(v, 'fc')
				super_dict[k]['cnn_inputs_shape_dict'] = self.get_inputs_shape_dict(v, 'cnn')
			super_dict[k]['other_inputs_list'] = get_input_recursively(v, lambda k: not k.startswith('fc') and not k.startswith('cnn'))

		for k, v in super_dict.items():
			self.sub_model_dict[k] = {}
			###### FC
			fc_inputs_shape_dict = v.get('fc_inputs_shape_dict', None)
			if fc_inputs_shape_dict:
				self.sub_model_dict[k]['fc'] = [
					self.fc_head_build(_key, _input_list)
					for (_key, _), _input_list in fc_inputs_shape_dict.items()
				]

			###### CNN
			cnn_inputs_shape_dict = v.get('cnn_inputs_shape_dict', None)
			if cnn_inputs_shape_dict:
				self.sub_model_dict[k]['cnn'] = [
					self.cnn_head_build(_key, _input_list)
					for (_key, _), _input_list in cnn_inputs_shape_dict.items()
				]

			###### Others
			other_inputs_list = v.get('other_inputs_list', None)
			if other_inputs_list:
				self.sub_model_dict[k][''] = [[nn.Flatten()] for l in other_inputs_list]

		print(self.sub_model_dict)

	def variables(self, as_dict=False):
		if as_dict:
			return self.state_dict()
		return list(self.parameters())

	def forward(self, x):
		output_list = []
		inputs_dict = self.get_inputs_dict(x)
		for _key, _input_list in inputs_dict.items():
			sub_output_list = []
			for _sub_input_list, _model_list in zip(
					_input_list, self.sub_model_dict[_key]):
				key_output_list = [
					_model(_input)
					for _input, _model in zip(_sub_input_list, _model_list)
				]
				key_output = torch.cat(key_output_list, -1) if len(
					key_output_list) > 1 else key_output_list[0]
				key_output = torch.flatten(key_output, start_dim=1)
				sub_output_list.append(key_output)
			output_list.append(
				torch.cat(sub_output_list, -1) if len(sub_output_list) > 1 else
				sub_output_list[0])
		output = torch.cat(output_list, -1) if len(output_list) > 1 else \
		output_list[0]
		output = torch.flatten(output, start_dim=1)
		return output

	def get_num_outputs(self):
		if self._num_outputs is None:
			def get_random_input_recursively(_obs_space):
				if isinstance(_obs_space, gym.spaces.Dict):
					return {
						k: get_random_input_recursively(v)
						for k, v in _obs_space.spaces.items()
					}
				elif isinstance(_obs_space, gym.spaces.Tuple):
					return list(
						map(get_random_input_recursively, _obs_space.spaces))
				return torch.rand(1, *_obs_space.shape)

			random_obs = get_random_input_recursively(self.obs_space)
			self._num_outputs = self.forward(random_obs).shape[-1]
		return self._num_outputs

	@staticmethod
	def get_inputs_shape_dict(_obs_space, _type):
		_heads = []
		_inputs_dict = {}
		for _key in filter(lambda x: x.startswith(_type),
						   sorted(_obs_space.spaces.keys())):
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
				_head.shape
				for _name, _head in space_iter
			]
			_inputs_dict[(_key, _permutation_invariant)] = _inputs
		return _inputs_dict

	@staticmethod
	def get_inputs_dict(_obs):
		assert isinstance(_obs, dict)

		inputs_dict = {}
		for k, v in sorted(_obs.items(), key=lambda x: x[0]):
			if k.startswith('cnn'):
				if 'cnn' not in inputs_dict:
					inputs_dict['cnn'] = []
				inputs_dict['cnn'].append(v)
			elif k.startswith('fc'):
				if 'fc' not in inputs_dict:
					inputs_dict['fc'] = []
				inputs_dict['fc'].append(v)
			else:
				if '' not in inputs_dict:
					inputs_dict[''] = []
				inputs_dict[''] += get_input_recursively(v, lambda
					k: not k.startswith('fc') and not k.startswith('cnn'))

		for _type, _input_list in inputs_dict.items():
			inputs_dict[_type] = [
				list(i.values())
				if isinstance(i, dict) else
				(
					i
					if isinstance(i, list) else
					[i]
				)
				for i in _input_list
			]
		return inputs_dict

	@staticmethod
	def cnn_head_build(_key, _input_list):
		_splitted_units = _key.split('-')
		_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
		return [
			nn.Sequential(
				Permute((0, 3, 1, 2)),
				nn.Conv2d(in_channels=input_shape[-1],
						  out_channels=_units // 2 + 1, kernel_size=9,
						  stride=4, padding=4),
				nn.ReLU(),
				nn.Conv2d(in_channels=_units // 2 + 1, out_channels=_units,
						  kernel_size=5, stride=2, padding=2),
				nn.ReLU(),
				nn.Conv2d(in_channels=_units, out_channels=_units,
						  kernel_size=3, stride=1, padding=1),
				nn.ReLU(),
				nn.Flatten(),
			)
			for i, input_shape in enumerate(_input_list)
		]

	@staticmethod
	def fc_head_build(_key, _input_list):
		_splitted_units = _key.split('-')
		# print(_splitted_units, _input_list)
		_units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
		return [
			nn.Sequential(
				nn.Flatten(),
				nn.Linear(in_features=np.prod(input_shape, dtype='int'),
						  out_features=_units),
				nn.ReLU(),
			)
			for i, input_shape in enumerate(_input_list)
		]
