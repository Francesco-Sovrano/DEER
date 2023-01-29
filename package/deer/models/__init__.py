model_catalog_dict = {}

### Tensorflow
from deer.models.tf.head_generator.primal_adaptive_model_wrapper import get_input_layers_and_keras_layers as get_input_layers_and_keras_layers_primal, get_input_list_from_input_dict as get_input_list_from_input_dict_primal
from deer.models.tf.head_generator.adaptive_model_wrapper import get_input_layers_and_keras_layers, get_input_list_from_input_dict
from deer.models.tf.head_generator.comm_adaptive_model_wrapper import get_input_layers_and_keras_layers as get_input_layers_and_keras_layers_comm, get_input_list_from_input_dict as get_input_list_from_input_dict_comm
from deer.models.tf.dqn import TFAdaptiveMultiHeadDQN
from deer.models.tf.ddpg import TFAdaptiveMultiHeadDDPG
from deer.models.tf.sac import TFAdaptiveMultiHeadNet as TFAdaptiveMultiHeadNetSAC
# from deer.models.tf.appo import TFAdaptiveMultiHeadNet as TFAdaptiveMultiHeadNetAPPO
model_catalog_dict['tf'] = {
    'dqn': {
        "adaptive_multihead_network": TFAdaptiveMultiHeadDQN.init(get_input_layers_and_keras_layers, get_input_list_from_input_dict),
        "primal_adaptive_multihead_network": TFAdaptiveMultiHeadDQN.init(get_input_layers_and_keras_layers_primal, get_input_list_from_input_dict_primal),
        "comm_adaptive_multihead_network": TFAdaptiveMultiHeadDQN.init(get_input_layers_and_keras_layers_comm, get_input_list_from_input_dict_comm),
    },
    'ddpg': {
        "adaptive_multihead_network": TFAdaptiveMultiHeadDDPG.init(get_input_layers_and_keras_layers, get_input_list_from_input_dict),
        "primal_adaptive_multihead_network": TFAdaptiveMultiHeadDDPG.init(get_input_layers_and_keras_layers_primal, get_input_list_from_input_dict_primal),
        "comm_adaptive_multihead_network": TFAdaptiveMultiHeadDDPG.init(get_input_layers_and_keras_layers_comm, get_input_list_from_input_dict_comm),
    },
    'sac': {
        "adaptive_multihead_network": TFAdaptiveMultiHeadNetSAC.init(get_input_layers_and_keras_layers, get_input_list_from_input_dict),
        "primal_adaptive_multihead_network": TFAdaptiveMultiHeadNetSAC.init(get_input_layers_and_keras_layers_primal, get_input_list_from_input_dict_primal),
        "comm_adaptive_multihead_network": TFAdaptiveMultiHeadNetSAC.init(get_input_layers_and_keras_layers_comm, get_input_list_from_input_dict_comm),
    },
    # 'ppo': {
    #     "adaptive_multihead_network": TFAdaptiveMultiHeadNetAPPO.init(get_input_layers_and_keras_layers, get_input_list_from_input_dict),
    #     "primal_adaptive_multihead_network": TFAdaptiveMultiHeadNetAPPO.init(get_input_layers_and_keras_layers_primal, get_input_list_from_input_dict_primal),
    #     "comm_adaptive_multihead_network": TFAdaptiveMultiHeadNetAPPO.init(get_input_layers_and_keras_layers_comm, get_input_list_from_input_dict_comm),
    # },
}
model_catalog_dict['tf']['td3'] = model_catalog_dict['tf']['ddpg']

### PyTorch      
from deer.models.torch.head_generator.adaptive_model_wrapper import AdaptiveModel
from deer.models.torch.head_generator.comm_adaptive_model_wrapper import CommAdaptiveModel
from deer.models.torch.dqn import TorchAdaptiveMultiHeadDQN
from deer.models.torch.ddpg import TorchAdaptiveMultiHeadDDPG
from deer.models.torch.sac import TorchAdaptiveMultiHeadNet as TorchAdaptiveMultiHeadSAC
# from deer.models.torch.appo import TorchAdaptiveMultiHeadNet as TorchAdaptiveMultiHeadAPPO

model_catalog_dict['torch'] = {
    'dqn': {
        "adaptive_multihead_network": TorchAdaptiveMultiHeadDQN.init(AdaptiveModel),
        "comm_adaptive_multihead_network": TorchAdaptiveMultiHeadDQN.init(CommAdaptiveModel),
    },
    'ddpg': {
        "adaptive_multihead_network": TorchAdaptiveMultiHeadDDPG.init(AdaptiveModel),
        "comm_adaptive_multihead_network": TorchAdaptiveMultiHeadDDPG.init(CommAdaptiveModel),
    },
    'sac': {
        "adaptive_multihead_network": TorchAdaptiveMultiHeadSAC.init(AdaptiveModel,AdaptiveModel),
        "comm_adaptive_multihead_network": TorchAdaptiveMultiHeadSAC.init(CommAdaptiveModel,CommAdaptiveModel),
    },
    # 'ppo': {
    #     "adaptive_multihead_network": TorchAdaptiveMultiHeadSAC.init(AdaptiveModel),
    #     "comm_adaptive_multihead_network": TorchAdaptiveMultiHeadSAC.init(CommAdaptiveModel),
    # },
}
model_catalog_dict['torch']['td3'] = model_catalog_dict['torch']['ddpg']

def get_algorithm_label_from_name(alg_name):
    for l in ['dqn','ddpg','td3','sac','ppo']:
        if alg_name.endswith(l):
            return l
    return None

def get_framework_label_from_name(framework):
    return 'torch' if framework.startswith('torch') else 'tf'

def get_model_catalog_dict(alg_name, framework):
    return model_catalog_dict[get_framework_label_from_name(framework)][get_algorithm_label_from_name(alg_name)]
