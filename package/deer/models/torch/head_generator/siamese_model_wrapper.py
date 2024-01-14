from deer.models.torch.head_generator.adaptive_model_wrapper import *


class SiameseAdaptiveModel(nn.ModuleDict):
    def __init__(self, obs_space, embedding_size=64, env="GridDrive-Hard"):
        super().__init__()
        if hasattr(obs_space, 'original_space'):
            obs_space = obs_space.original_space

        self.obs_space = obs_space
        self.embedding_size = embedding_size
        self._num_outputs = None
        self.sub_model_dict = {}

        assert isinstance(obs_space,
                          gym.spaces.Dict), 'SiameseAdaptiveModel only works with Dict observation spaces.'

        # This is to keep compatible with the original AdaptiveModel
        super_dict = {k: {} for k in obs_space.spaces.keys()}
        for k, v in obs_space.spaces.items():
            if isinstance(v, gym.spaces.Dict):
                super_dict[k][
                    'fc_inputs_shape_dict'] = self.get_inputs_shape_dict(v,
                                                                         'fc')
                super_dict[k][
                    'cnn_inputs_shape_dict'] = self.get_inputs_shape_dict(v,
                                                                          'cnn')
            super_dict[k]['other_inputs_list'] = get_input_recursively(v,
                                                                       lambda
                                                                           k: not k.startswith(
                                                                           'fc') and not k.startswith(
                                                                           'cnn'))

        for k, v in super_dict.items():
            # self.sub_model_dict[k] = {}
            # FC
            fc_inputs_shape_dict = v.get('fc_inputs_shape_dict', None)
            fc_head = nn.ModuleList([nn.Identity()])
            if fc_inputs_shape_dict:
                fc_head = nn.ModuleList([
                    self.fc_head_build(_key, _input_list)
                    for (_key, _), _input_list in fc_inputs_shape_dict.items()
                ])

            # CNN
            cnn_inputs_shape_dict = v.get('cnn_inputs_shape_dict', None)
            cnn_head = nn.ModuleList([nn.Identity()])
            if cnn_inputs_shape_dict:
                cnn_head = nn.ModuleList([
                    self.cnn_head_build(_key, _input_list)
                    for (_key, _), _input_list in cnn_inputs_shape_dict.items()
                ])

            self[k] = nn.ModuleDict({
                'fc': fc_head,
                'cnn': cnn_head,
            })
            # Others
            other_inputs_list = v.get('other_inputs_list', None)
            if other_inputs_list:
                self[k] = nn.ModuleList([nn.Flatten()
                                         for _ in other_inputs_list])

        # TODO: back to a hacky solution, lazy is not supported on cluster
        if env == "GridDrive-Hard":
            in_dim = 586
        elif env == "GridDrive-Medium":
            in_dim = 554
        elif env == "GridDrive-Easy":
            in_dim = 514
        else:
            raise NotImplementedError(f"env {env} not supported for "
                                      f"SiameseAdaptiveModel")

        self.last_fc = nn.Sequential(
            nn.Linear(in_features=in_dim, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=embedding_size),
            nn.ReLU(),
        )

    def variables(self, as_dict=False):
        if as_dict:
            return self.state_dict()
        return list(self.parameters())

    def forward(self, x):
        assert x, 'forwarding an empty input'
        output_list = []
        if isinstance(x, SampleBatch):
            inputs_dicts = [self.get_inputs_dict(x)]
        elif isinstance(x, Iterable):
            inputs_dicts = [self.get_inputs_dict(x_i) for x_i in x]
        else:
            inputs_dicts = [self.get_inputs_dict(x)]
        for inputs_dict in inputs_dicts:
            out_list = []
            for _key, _input_list in inputs_dict.items():
                if _key not in self:
                    continue
                sub_output_list = []
                if isinstance(self[_key], nn.ModuleDict):
                    # for _sub_input_list, _model_list in zip(
                    #         _input_list, self[_key]['cnn']):
                    key_output_list = self[_key]['cnn'][0][0](
                        torch.Tensor(_input_list[0][0]))
                    key_output = torch.cat(key_output_list, -1) if len(
                        key_output_list) > 1 else key_output_list[0]
                    # key_output = torch.flatten(key_output, start_dim=1)
                    sub_output_list.append(key_output)

                    # for _sub_input_list, _model_list in zip(
                    #         _input_list, self[_key]['fc']):
                    key_output_list = self[_key]['fc'][0][0](
                        torch.Tensor(_input_list[1][0]))
                    key_output = torch.cat(key_output_list, -1) if len(
                        key_output_list) > 1 else key_output_list[0]
                    # key_output = torch.flatten(key_output, start_dim=1)
                    sub_output_list.append(key_output)
                else:
                    for _sub_input_list, _model_list in zip(
                            _input_list, self[_key]):
                        key_output_list = _model_list(
                            torch.Tensor(np.array(_sub_input_list)))
                        key_output = torch.cat(key_output_list, -1) if len(
                            key_output_list) > 1 else key_output_list[0]
                        # key_output = torch.flatten(key_output, start_dim=1)
                        sub_output_list.append(key_output)
                out_list.append(
                    torch.cat(sub_output_list, -1) if len(sub_output_list) > 1 else
                    sub_output_list[0])
            output = torch.cat(out_list, -1) if len(out_list) > 1 else out_list[0]
            output_list.append(output)
        # output = torch.flatten(output, start_dim=1)
        stacked_output = torch.stack(output_list)
        return self.last_fc(stacked_output)

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
        assert isinstance(_obs, dict), (f'SiameseAdaptiveModel only works '
                                        f'with Dict observation spaces. '
                                        f'{type(_obs)} is not supported.')

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
                if k not in inputs_dict:
                    inputs_dict[k] = []
                inputs_dict[k] += get_input_recursively(v)

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
        return nn.ModuleList([
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
        ] if _units else [
            nn.Sequential(
                Permute((0, 3, 1, 2)),
                nn.Conv2d(in_channels=input_shape[-1], out_channels=32,
                          kernel_size=8, stride=4, padding=4),
                nn.ReLU(),
                nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4,
                          stride=2, padding=2),
                nn.ReLU(),
                nn.Conv2d(in_channels=64, out_channels=64, kernel_size=4,
                          stride=1, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            for i, input_shape in enumerate(_input_list)
        ])

    @staticmethod
    def fc_head_build(_key, _input_list):
        _splitted_units = _key.split('-')
        # print(_splitted_units, _input_list)
        _units = int(_splitted_units[-1]) if len(_splitted_units) > 1 else 0
        return nn.ModuleList([
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features=np.prod(input_shape, dtype='int'),
                          out_features=_units),
                nn.ReLU(),
            )
            for i, input_shape in enumerate(_input_list)
        ] if _units else [
            nn.Flatten()
            for i, input_shape in enumerate(_input_list)
        ])
