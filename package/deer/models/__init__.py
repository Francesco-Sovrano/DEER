model_catalog_dict = {}

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
