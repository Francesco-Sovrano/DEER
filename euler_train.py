from pathlib import Path
import os
import copy
import pprint
import argparse
import argunparse
import subprocess
import random
from datetime import datetime

import numpy as np

from deer.utils.workflow import train
from deer.models import get_model_catalog_dict

from deer.agents.xadqn import XADQN, XADQNConfig
from environment import CustomEnvironmentCallbacks

EXPERIENCE_BUFFER_SIZE = 2 ** 14
CENTRALISED_TRAINING = True


def copy_dict_and_update(d, u):
    new_dict = copy.deepcopy(d)
    new_dict.update(u)
    return new_dict


def copy_dict_and_update_with_key(d, k, u):
    new_dict = copy.deepcopy(d)
    if k not in new_dict:
        new_dict[k] = {}
    new_dict[k].update(u)
    return new_dict


default_options = {
    "framework": "torch",
    "model": {
        "custom_model": "adaptive_multihead_network",
    },
    "no_done_at_end": False,
    "grad_clip": None,
     "gamma": 0.999, "seed": 42,
    "train_batch_size": 2 ** 8,
    "min_train_timesteps_per_iteration": 1,
}


algorithm_options = {
    "grad_clip": None,  # no need of gradient clipping with huber loss
    "dueling": True,
    "double_q": True,
    "num_atoms": 21,
    "v_max": 2 ** 5,
    "v_min": -1,
}


xa_default_options = {
    "buffer_options": {
        "prioritized_replay": True,
        "centralised_buffer": True,  # for MARL
        'global_size': EXPERIENCE_BUFFER_SIZE,
        'priority_id': 'td_errors',
        'priority_lower_limit': 0,
        'priority_aggregation_fn': 'np.mean',
        'prioritization_alpha': 0.6,
        'prioritization_importance_beta': 0.4,
        'prioritization_importance_eta': 1e-2,
        'prioritization_epsilon': 1e-6,
        'cluster_size': None,
        'cluster_prioritisation_strategy': 'sum',
        'cluster_level_weighting': True,
        'clustering_xi': 2,
        'prioritized_drop_probability': 1,
        'global_distribution_matching': False,
        'stationarity_window_size': None,
        'stationarity_smoothing_factor': 1,
        'max_age_window': None,
    },
    "clustering_options": {
        'clustering_scheme': [
            'Why',
            # 'Who',
            'How_Well',
            # 'How_Fair',
            # 'Where',
            # 'What',
            # 'How_Many'
            # 'UWho',
            # 'UWhich_CoopStrategy',
        ],
        "clustering_scheme_options": {
            "n_clusters": {
                "who": 4,
                # "why": 8,
                # "what": 8,
            },
            "default_n_clusters": 8,
            "frequency_independent_clustering": False,
            # Setting this to True can be memory expensive, especially for WHO explanations
            "agent_action_sliding_window": 2 ** 3,
            "episode_window_size": 2 ** 6,
            "batch_window_size": 2 ** 8,
            "training_step_window_size": 2 ** 2,
        },
        "cluster_selection_policy": "min",
        "cluster_with_episode_type": False,
        "cluster_overview_size": 1,
        "collect_cluster_metrics": True,
        "ratio_of_samples_from_unclustered_buffer": 0,
    },
}


def run_training(args):
    os.environ["TUNE_RESULT_DIR"] = str(args.results_dir)
    import ray
    from ray.tune.registry import _global_registry, ENV_CREATOR
    from ray.rllib.policy.sample_batch import DEFAULT_POLICY_ID
    from ray.rllib.models import ModelCatalog

    SELECTED_ENV = "GridDrive-Easy"
    CONFIG = default_options
    CONFIG = copy_dict_and_update(CONFIG, xa_default_options)
    CONFIG = copy_dict_and_update(CONFIG, algorithm_options)
    CONFIG["callbacks"] = CustomEnvironmentCallbacks

    # Setup MARL training strategy: centralised or decentralised
    env = _global_registry.get(ENV_CREATOR, SELECTED_ENV)(
        CONFIG.get("env_config", {}))
    obs_space = env.observation_space
    act_space = env.action_space

    # policy_graphs = {DEFAULT_POLICY_ID: (None, obs_space, act_space, CONFIG)}
    policy_graphs = {DEFAULT_POLICY_ID}
    policy_mapping_fn = lambda agent_id: DEFAULT_POLICY_ID

    CONFIG["multiagent"] = {
        "policies": policy_graphs,
        "policy_mapping_fn": policy_mapping_fn,
        "policies_to_train": None,
        "observation_fn": None,
        "replay_mode": "independent",
    }
    print('Config:', CONFIG)

    # Register models
    for k, v in get_model_catalog_dict(
            'dqn', CONFIG.get("framework", 'tf')).items():
        ModelCatalog.register_custom_model(k, v)

    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=args.cpus,
             include_dashboard=False)
    train(XADQN, XADQNConfig, CONFIG, SELECTED_ENV,
          test_every_n_step=args.eval_freq,
          stop_training_after_n_step=args.total_n_steps)


def submit_jobs(args):

    # If the same seed should be used for all experiments
    random.seed(43)
    if isinstance(args.seed, list) and len(args.seed) == 1 \
            and int(args.seed[0]) == -1:
        seeds = random.sample(range(1, 1000), args.repetitions)
    elif isinstance(args.seed, list) and len(args.seed) == 1:
        seeds = np.repeat(args.seed[0], args.repetitions)
    elif isinstance(args.seed, list) and len(args.seed) > 1:
        assert len(args.seed) == args.repetitions
        seeds = args.seed
    elif args.seed is None:
        seeds = random.sample(range(1, 1000), args.repetitions)
    else:
        raise NotImplemented

    lsf_out_dir = args.results_dir / 'outs'
    lsf_out_dir.mkdir(parents=True, exist_ok=True, mode=0o755)

    if args.no_gpu:
        args.any_gpu = True

    args_dict = vars(args).copy()
    for i in range(args.repetitions):
        seed = seeds[i]

        time = datetime.now().strftime('%H%M%S%f')
        run_id = f"deer_{args.run_id}_{i}_seed_{seed}_{time}"
        print(f"Run {run_id} with seed: {seed}")
        euler_slurm = (f"sbatch --mem-per-cpu={args.memory} "
                       f"-n {args.cpus} "
                       f"{'--gpus=1 ' if not args.no_gpu else ''}"
                       f"{'--gpus=rtx_3090:1 ' if not args.any_gpu else ''}"
                       f"-J {run_id} "
                       f"-o {lsf_out_dir / run_id} "
                       f"--time={args.time}:00:00 --wrap ")
        euler_lsf = (f"bsub -R rusage[mem={args.memory}"
                     f"{',ngpus_excl_p=1' if not args.any_gpu else ''}] "
                     f"{'-R select[gpu_model0==NVIDIAGeForceGTX1080Ti] ' if not args.any_gpu else ''}"
                     f"-o {lsf_out_dir / run_id} "
                     f"-W {args.time}:00 ")
        args_dict['seed'] = seed
        args_dict['run_id'] = run_id
        args_dict['repetitions'] = 1

        rks = []
        list_str = ""
        for k, v in args_dict.items():
            if isinstance(v, list):
                list_str += f" --{k} {' '.join([str(x) for x in v])}"
                rks.append(k)

        new_args_dict = args_dict.copy()
        for k in rks:
            del new_args_dict[k]
        unparser = argunparse.ArgumentUnparser()
        newargs = unparser.unparse(**new_args_dict)
        newargs = newargs + list_str + f" --no_submit"

        command = ''
        if args.batch_system == 'slurm':
            command = f"python euler_train.py {newargs}"
            subprocess.check_output(args=euler_slurm.split() + [command])
        elif args.batch_system == 'lsf':
            command = euler_lsf + f"python euler_train.py {newargs}"
            subprocess.check_output(args=command.split())
        else:
            NotImplemented(f"batch system {args.batch_system} is not "
                           f"supported on the cluster.")
        print(f"Command: {command}")


def run_experiments(args):
    pass


def main():
    euler_res = "/cluster/project/jbuhmann/workspace/deer/results/"
    parser = argparse.ArgumentParser(description="Deer experiments")
    parser.add_argument("--results_dir", type=Path, default=Path(euler_res),
                        help="Path in which results of training are/will be "
                             "located")
    parser.add_argument("-n", "--run_id", type=Path,
                        help="directory name for results to be saved")
    parser.add_argument("--ml_config_path",
                        type=Path,
                        default=Path("configs/config_sb3.yaml"),
                        help="Path to the ml-agents or sb3 config. "
                             "Ex: 'configs/fast_ppo_config_linear_lr.yaml'")

    parser.add_argument("--seed",
                        nargs="+",
                        type=int,
                        default=None,
                        help="Random seed to use. If None, randomly generated "
                             "seeds for each experiment will be used. If a "
                             "single value, the same seed will be used for "
                             "all the experiments. If the a list of seeds, "
                             "each seed will be used for each of the "
                             "experiments. In this case, the length of seed "
                             "must be the same as number of repetitions (-r)."
                             "if it's set to -1 the same set of  seeds will "
                             "be used for the sake of  experiments' "
                             "consistency.")
    parser.add_argument("--resume",
                        action='store_true',
                        help="Resume training or inference")
    parser.add_argument("--experiments", type=str, nargs='+')
    parser.add_argument("--inference",
                        action='store_true',
                        help="Run inference")

    parser.add_argument("--render_mode",
                        type=str, default='rgb_array',
                        help="rendering the environment")
    parser.add_argument("--eval_freq", type=int, default=1000000)
    parser.add_argument("--anim_freq", type=int, default=None)
    parser.add_argument("--episode_max_length", type=int, default=15000)
    parser.add_argument("--total_n_steps", type=int, default=40000000)
    parser.add_argument("--env", type=str, default='GridDrive-Easy')

    parser.add_argument("--algo", type=str, default='dqnar',
                        help='sqn, ppo, dqnar, ppoar, qlearning or sarsa')

    # Cluster arguments
    parser.add_argument('--no_submit', default=False, required=False,
                        action='store_true',
                        help='Do not submit to Euler. Default will submit.')
    parser.add_argument('--any_gpu', default=False, required=False,
                        action='store_true',
                        help='Do not select a specific GPU in oder to wait '
                             'less in the queue. Selecting a specific GPU is '
                             'important for reproducibility in order to run '
                             'deterministic experiments, but this could be '
                             'ignored sometimes.')
    parser.add_argument('--no_gpu', default=False, required=False,
                        action='store_true',
                        help='Do not request any gpus.')
    parser.add_argument('--cpus', default=1, required=False, type=int)
    parser.add_argument("--batch_system",
                        type=str,
                        default='slurm',
                        choices=["slurm", "lsf"],
                        help="Which batch system to use for submission. "
                             "lsf is becoming  obsolete on Euler very soon.")
    parser.add_argument("-s", "--scratch", action='store_true',
                        help="Whether to first copy the dataset to the "
                             "scratch storage of Leonhard. Do not use on "
                             "other systems than Leonhard.")
    parser.add_argument("-m", "--memory",
                        type=int,
                        default=20000,
                        help="Memory allocated for each leonhard job. This "
                             "will be ignored of Leonhard is not selected.")
    parser.add_argument("-t", "--time",
                        type=int,
                        default=4,
                        help="Number of hours requested for the job on "
                             "Leonhard. For virtual models usually it "
                             "requires more time than this default value.")
    parser.add_argument("-r", "--repetitions",
                        type=int,
                        default=1,
                        help="Number of repetitions to run_mlagents the same "
                             "experiment")

    args = parser.parse_args()
    print("   Experiment parameters: ")
    print("-" * 100)
    pprint.pprint(vars(args), indent=5)
    print("-" * 100)

    if args.no_submit:
        run_training(args)
    elif args.experiments is not None:
        run_experiments(args)
    else:
        submit_jobs(args)


if __name__ == '__main__':
    main()
