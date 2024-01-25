from deer.models.torch.head_generator.adaptive_model_wrapper import AdaptiveModel
from ray.rllib.utils.framework import try_import_torch
import logging
import numpy as np
import gymnasium as gym
import torch_geometric

logger = logging.getLogger(__name__)
torch, nn = try_import_torch()

two_pi = 2*np.pi
pi = np.pi


def rotate(x, y, theta=0):
	sin_theta = np.sin(theta)
	cos_theta = np.cos(theta)
	return x*cos_theta-y*sin_theta, x*sin_theta+y*cos_theta


def shift_and_rotate(xv, yv, dx, dy, theta=0):
	return rotate(xv+dx, yv+dy, theta)


class RelativePosition:
	def __init__(self):
		pass

	def __call__(self, data):
		(row, col), pos, deg, pseudo = data.edge_index, data.pos, data.deg, data.edge_attr

		xy = pos[row] - pos[col]
		sin_theta = torch.sin(-deg[col])
		cos_theta = torch.cos(-deg[col])
		x = xy[:, 0][:, None]
		y = xy[:, 1][:, None]
		relative_xy = torch.cat(
			[
				x*cos_theta-y*sin_theta,
				x*sin_theta+y*cos_theta
			], 
			dim=-1
		)
		
		relative_xy = relative_xy.view(-1, 1) if relative_xy.dim() == 1 else relative_xy
		if pseudo is not None:
			pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
			data.edge_attr = torch.cat([pseudo, relative_xy.type_as(pseudo)], dim=-1)
		else:
			data.edge_attr = relative_xy

		return data

	def __repr__(self) -> str:
		return self.__class__.__name__


class RelativeOrientation:
	def __init__(self, norm=False):
		self.norm = norm

	def __call__(self, data):
		(row, col), orientation, pseudo = data.edge_index, data.deg, data.edge_attr

		relative_orientation = torch.remainder(orientation[row] - orientation[col], 2*np.pi)
		if self.norm:
			relative_orientation /= 2*np.pi
		relative_orientation = relative_orientation.view(-1, 1) if relative_orientation.dim() == 1 else relative_orientation

		if pseudo is not None:
			pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
			data.edge_attr = torch.cat(
				[pseudo, relative_orientation.type_as(pseudo)], dim=-1)
		else:
			data.edge_attr = relative_orientation

		return data

	def __repr__(self) -> str:
		return self.__class__.__name__


class CommAdaptiveModel(AdaptiveModel):
	def __init__(self, obs_space, config):
		# print('CommAdaptiveModel', config)
		if hasattr(obs_space, 'original_space'):
			obs_space = obs_space.original_space
		super().__init__(obs_space['all_agents_relative_features_list'][0], config)
		self.obs_space = obs_space

		# GNN
		agent_features_size = self.get_agent_features_size()
		logger.warning(f"Agent features size: {agent_features_size}")
		self.n_agents = obs_space['all_agents_absolute_position_vector'].shape[0]
		self.n_leaders = obs_space['all_leaders_absolute_position_vector'].shape[0] if 'all_leaders_absolute_position_vector' in obs_space else 0
		self.n_agents_and_leaders = self.n_agents + self.n_leaders
		self.max_num_neighbors = config.get('max_num_neighbors', self.n_agents_and_leaders-1)
		self.message_size = config.get('message_size', agent_features_size)
		self.comm_range = torch.Tensor([config.get('comm_range', 10.)])
		self.gnn = torch_geometric.nn.GATv2Conv( # https://pytorch-geometric.readthedocs.io/en/latest/modules/nn.html?highlight=GATv2Conv#torch_geometric.nn.conv.GENConv
			in_channels=agent_features_size,
			out_channels=self.message_size,
			edge_dim=2+1, # position + orientation
			# add_self_loops=True,
			# fill_value=0, # The way to generate edge features of self-loops (in case :obj:`edge_dim != None`).
			# heads=1,
		)
		# self.post_proc = torch.nn.LayerNorm(agent_features_size + self.message_size)
		logger.warning(f"Building keras layers for Comm model with {self.n_agents} agents, {self.n_leaders} leaders and communication range {self.comm_range[0]} for maximum {self.max_num_neighbors} neighbours")
		# self.use_beta = True

	def get_agent_features_size(self):
		def get_random_input_recursively(_obs_space):
			if isinstance(_obs_space, gym.spaces.Dict):
				return {
					k: get_random_input_recursively(v)
					for k,v in _obs_space.spaces.items()
				}
			elif isinstance(_obs_space, gym.spaces.Tuple):
				return list(map(get_random_input_recursively, _obs_space.spaces))
			return torch.rand(1,*_obs_space.shape)
		random_obs = get_random_input_recursively(self.obs_space['all_agents_relative_features_list'][0])
		return super().forward(random_obs).data.shape[-1]

	def forward(self, x):
		super_forward = super().forward
		
		# this_agent_id_mask = x['this_agent_id_mask'][:,:,None] # add extra dimension
		this_agent_id = x['this_agent_id'][:,:,None].to(torch.long) # add extra dimension
		all_agents_features = torch.stack(list(map(super_forward, x['all_agents_relative_features_list'])), dim=1)
		# main_output = torch.sum(all_agents_features*this_agent_id_mask, dim=1)
		main_output = torch.squeeze(torch.take_along_dim(all_agents_features,this_agent_id,dim=1),dim=1)
		
		all_agents_positions = x['all_agents_absolute_position_vector']
		if self.n_leaders:
			all_agents_positions = torch.cat(
				[
					x['all_leaders_absolute_position_vector'], 
					all_agents_positions
				], 
				dim=1
			)
		all_agents_orientations = x['all_agents_absolute_orientation_vector']

		device = all_agents_positions.device
		batch_size = all_agents_positions.shape[0]

		# build graphs
		graphs = torch_geometric.data.Batch()
		graphs.batch = torch.repeat_interleave(
			torch.arange(batch_size), 
			self.n_agents_and_leaders, 
			dim=0
		).to(device)
		graphs.pos = all_agents_positions.reshape(-1, all_agents_positions.shape[-1])
		graphs.deg = all_agents_orientations.reshape(-1, all_agents_orientations.shape[-1])
		graphs.x = all_agents_features.reshape(-1, all_agents_features.shape[-1])#.detach() # do not propagate gradient to senders
		if self.n_leaders:
			all_agents_types = torch.zeros(batch_size, self.n_agents_and_leaders, 1, device=device)
			all_agents_types[:, :self.n_leaders] = 1.0
			all_agents_types = all_agents_types.reshape(-1, all_agents_types.shape[-1])
			graphs.x = torch.cat([graphs.x, all_agents_types], dim=1)
		graphs = torch_geometric.transforms.RadiusGraph(r=self.comm_range, loop=False, max_num_neighbors=self.max_num_neighbors)(graphs) # Creates edges based on node positions pos to all points within a given distance (functional name: radius_graph).
		graphs = RelativePosition()(graphs) # Saves the relative positions of linked nodes in its edge attributes
		graphs = RelativeOrientation(norm=True)(graphs) # Saves the relative orientations in its edge attributes

		# process graphs
		gnn_output = self.gnn(graphs.x, graphs.edge_index, edge_attr=graphs.edge_attr)
		# assert not gnn_output.isnan().any()
		gnn_output = gnn_output.view(-1, self.n_agents_and_leaders, self.message_size) # reshape GNN outputs
		if self.n_leaders:
			gnn_output = gnn_output[:, self.n_leaders:]
		# message_from_others = torch.sum(gnn_output*this_agent_id_mask, dim=1)
		message_from_others = torch.squeeze(torch.take_along_dim(gnn_output,this_agent_id,dim=1),dim=1)

		# build output
		# output = message_from_others
		output = torch.cat([main_output, message_from_others], dim=1)
		# output = self.post_proc(output)
		return output
