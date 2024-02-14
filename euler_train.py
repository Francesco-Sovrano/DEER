from pathlib import Path
import os
import yaml
import pprint
import argparse
import argunparse
import subprocess
import random
import numpy as np
from datetime import datetime

import ray
from ray.rllib.models import ModelCatalog
from ray import tune
from ray.air.config import RunConfig
from ray.rllib.algorithms.sac.sac import SAC, SACConfig
from deer.models import get_model_catalog_dict
from deer.agents.xadqn import XADQN, XADQNConfig
from deer.agents.xasac import XASAC, XASACConfig
from environment import CustomEnvironmentCallbacks


def run_mujoco_siamese_experiments(args):
    # Run Siamese experiments
    envs = ['HalfCheetah-v4', 'Hopper-v4', 'Walker2d-v4', 'Ant-v4',
            'Humanoid-v4']
    methods = ["siamese"]
    er_buffer_size = [1024, 2048, 4096, 8192]
    siamese_embedding_size = [512, 2048]
    siamese_update_frequency = [1000, 30000]
    siamese_loss_margin = [2**0, 2**20]
    update_steps = 5

    new_args_dict = vars(args).copy()
    res_dir = args.results_dir / 'xasac_siamese'
    new_args_dict['repetitions'] = 1
    new_args_dict['time'] = 100
    new_args_dict['memory'] = 20000
    new_args_dict['total_n_steps'] = 3000000
    new_args_dict['no_gpu'] = True
    new_args_dict['algo'] = 'xasac'
    new_args_dict['ml_config_path'] = Path("configs/xasac_siamese_config.yaml")

    for env in envs:
        for method in methods:
            for b_size in er_buffer_size:
                for embedding_size in siamese_embedding_size:
                    for update_freq in siamese_update_frequency:
                        for loss_margin in siamese_loss_margin:
                            print(f"Running experiment with env {env}, "
                                  f"method {method}, "
                                  f"buffer_size={b_size}, "
                                  f"embedding_size={embedding_size}, "
                                  f"update_frequency={update_freq}, "
                                  f"loss_margin={loss_margin}")

                            run_res_dir = res_dir / env / f"method_{method}" \
                                                          f"_buffer_{b_size}" \
                                                          f"_embedding_" \
                                                          f"{embedding_size}" \
                                                          f"_update_" \
                                                          f"{update_freq}" \
                                                          f"_margin_" \
                                                          f"{loss_margin}"
                            run_res_dir.mkdir(parents=True, exist_ok=True)
                            new_args_dict['results_dir'] = run_res_dir
                            new_args_dict['run_id'] = f"{args.algo}_env_{env}"
                            new_args_dict['env'] = env

                            with open(args.ml_config_path) as file:
                                configs = yaml.load(file, Loader=yaml.FullLoader)
                            time = datetime.now().strftime('%H%M%S%f')
                            name = f"{args.algo}_{env}_{time}"
                            new_config_path = run_res_dir / (
                                    name + "_" + Path(args.ml_config_path).name)

                            if method == "siamese":
                                configs[env]['siamese'] = {
                                    'use_siamese': True,
                                    'buffer_size': 512,
                                    'embedding_size': embedding_size,
                                    'update_frequency': update_freq,
                                    'update_steps': update_steps,
                                    'loss_margin': loss_margin
                                }
                                configs[env]['buffer_options']['global_size'] = b_size
                            else:
                                raise ValueError("This should not happen")

                            with open(new_config_path, 'w') as file:
                                yaml.dump(configs, file)

                            new_args_dict['ml_config_path'] = new_config_path
                            exp_args = argparse.Namespace(**new_args_dict)
                            submit_jobs(exp_args)

    # Run baseline experiments
    methods = ["gt_clustering", "no_clustering"]
    er_buffer_size = [4096, 8192, 100000, 1000000]
    new_args_dict = vars(args).copy()
    res_dir = args.results_dir / 'xasac_siamese'
    new_args_dict['repetitions'] = 1
    new_args_dict['time'] = 100
    new_args_dict['memory'] = 20000
    new_args_dict['total_n_steps'] = 3000000
    new_args_dict['no_gpu'] = True
    new_args_dict['algo'] = 'xasac'
    new_args_dict['ml_config_path'] = Path("configs/xasac_siamese_config.yaml")
    for env in envs:
        for method in methods:
            for b_size in er_buffer_size:
                print(f"Running experiment with env {env}, method {method}")

                run_res_dir = res_dir / env / f"method_{method}_buffer_{b_size}"
                run_res_dir.mkdir(parents=True, exist_ok=True)
                new_args_dict['results_dir'] = run_res_dir
                new_args_dict['run_id'] = f"{args.algo}_env_{env}"
                new_args_dict['env'] = env

                with open(args.ml_config_path) as file:
                    configs = yaml.load(file, Loader=yaml.FullLoader)
                time = datetime.now().strftime('%H%M%S%f')
                name = f"{args.algo}_{env}_{time}"
                new_config_path = run_res_dir / (
                        name + "_" + Path(args.ml_config_path).name)

                if method == "siamese":
                    raise ValueError("This should not happen")
                else:
                    configs[env]['siamese']['use_siamese'] = False
                    if method == "no_clustering":
                        configs[env]['clustering_options'][
                            'clustering_scheme'] = None
                    configs[env]['buffer_options']['global_size'] = b_size

                with open(new_config_path, 'w') as file:
                    yaml.dump(configs, file)

                new_args_dict['ml_config_path'] = new_config_path
                exp_args = argparse.Namespace(**new_args_dict)
                submit_jobs(exp_args)


def run_griddrive_siamese_experiments(args):

    # Run Siamese experiments
    envs = ['GridDrive-Easy', 'GridDrive-Medium', 'GridDrive-Hard']
    methods = ["siamese"]
    siamese_buffer_size = [100, 500]
    siamese_embedding_size = [512, 2048]
    siamese_update_frequency = [1000, 30000]
    siamese_loss_margin = [2**0, 2**20]

    new_args_dict = vars(args).copy()
    res_dir = args.results_dir / 'xadqn_siamese'
    new_args_dict['repetitions'] = 1
    new_args_dict['time'] = 100
    new_args_dict['memory'] = 20000
    new_args_dict['total_n_steps'] = 40000000
    new_args_dict['no_gpu'] = False
    new_args_dict['algo'] = 'xadqn'
    new_args_dict['ml_config_path'] = Path("configs/xadqn_siamese_config.yaml")

    for env in envs:
        for method in methods:
            for size in siamese_buffer_size:
                for embedding_size in siamese_embedding_size:
                    for update_freq in siamese_update_frequency:
                        for loss_margin in siamese_loss_margin:
                            print(f"Running experiment with env {env}, "
                                  f"method {method}, "
                                  f"buffer_size={size}, "
                                  f"embedding_size={embedding_size}, "
                                  f"update_frequency={update_freq}, "
                                  f"loss_margin={loss_margin}")

                            run_res_dir = res_dir / env / f"method_{method}" \
                                                          f"_buffer_{size}" \
                                                          f"_embedding_" \
                                                          f"{embedding_size}" \
                                                          f"_update_" \
                                                          f"{update_freq}" \
                                                          f"_margin_" \
                                                          f"{loss_margin}"
                            run_res_dir.mkdir(parents=True, exist_ok=True)
                            new_args_dict['results_dir'] = run_res_dir
                            new_args_dict['run_id'] = f"{args.algo}_env_{env}"
                            new_args_dict['env'] = env

                            with open(args.ml_config_path) as file:
                                configs = yaml.load(file, Loader=yaml.FullLoader)
                            time = datetime.now().strftime('%H%M%S%f')
                            name = f"{args.algo}_{env}_{time}"
                            new_config_path = run_res_dir / (
                                    name + "_" + Path(args.ml_config_path).name)

                            if method == "siamese":
                                configs[env]['siamese'] = {
                                    'use_siamese': True,
                                    'buffer_size': size,
                                    'embedding_size': embedding_size,
                                    'update_frequency': update_freq,
                                    'loss_margin': loss_margin
                                }
                            else:
                                raise ValueError("This should not happen")

                            with open(new_config_path, 'w') as file:
                                yaml.dump(configs, file)

                            new_args_dict['ml_config_path'] = new_config_path
                            exp_args = argparse.Namespace(**new_args_dict)
                            submit_jobs(exp_args)

    # Run baseline experiments
    methods = ["clustering", "no_clustering"]
    new_args_dict = vars(args).copy()
    res_dir = args.results_dir / 'xadqn_siamese'
    new_args_dict['repetitions'] = 1
    new_args_dict['time'] = 100
    new_args_dict['memory'] = 20000
    new_args_dict['total_n_steps'] = 40000000
    new_args_dict['no_gpu'] = True
    new_args_dict['algo'] = 'xadqn'
    new_args_dict['ml_config_path'] = Path("configs/xadqn_siamese_config.yaml")
    for env in envs:
        for method in methods:
            print(f"Running experiment with env {env}, method {method}")

            run_res_dir = res_dir / env / f"method_{method}"
            run_res_dir.mkdir(parents=True, exist_ok=True)
            new_args_dict['results_dir'] = run_res_dir
            new_args_dict['run_id'] = f"{args.algo}_env_{env}"
            new_args_dict['env'] = env

            with open(args.ml_config_path) as file:
                configs = yaml.load(file, Loader=yaml.FullLoader)
            time = datetime.now().strftime('%H%M%S%f')
            name = f"{args.algo}_{env}_{time}"
            new_config_path = run_res_dir / (
                    name + "_" + Path(args.ml_config_path).name)

            if method == "siamese":
                raise ValueError("This should not happen")
            else:
                configs[env]['siamese']['use_siamese'] = False
                if method == "no_clustering":
                    configs[env]['clustering_options'][
                        'clustering_scheme'] = None

            with open(new_config_path, 'w') as file:
                yaml.dump(configs, file)

            new_args_dict['ml_config_path'] = new_config_path
            exp_args = argparse.Namespace(**new_args_dict)
            submit_jobs(exp_args)


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
                       f"{f'--gpus={args.gpus} ' if not args.no_gpu else ''}"
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

        env_var = os.environ.copy()
        env_var["TUNE_RESULT_DIR"] = str(args.results_dir)
        command = ''
        if args.batch_system == 'slurm':
            command = f"python euler_train.py {newargs}"
            subprocess.check_output(args=euler_slurm.split() + [command],
                                    env=env_var)
        elif args.batch_system == 'lsf':
            command = euler_lsf + f"python euler_train.py {newargs}"
            subprocess.check_output(args=command.split(),
                                    env=env_var)
        else:
            NotImplemented(f"batch system {args.batch_system} is not "
                           f"supported on the cluster.")
        print(f"Command: {command}")


def run_training(args):
    os.environ["TUNE_RESULT_DIR"] = str(args.results_dir.absolute())

    num_gpus = args.gpus
    if args.no_gpu:
        num_gpus = 0

    if args.algo == 'xadqn':
        algo_class = XADQN
        algo_config = XADQNConfig()
        alg_name = 'dqn'
    elif args.algo == 'xasac':
        algo_class = XASAC
        algo_config = XASACConfig()
        alg_name = 'sac'
    elif args.algo == 'sac':
        algo_class = SAC
        algo_config = SACConfig()
        alg_name = 'sac'
    else:
        raise ValueError(f"Unknown algorithm {args.algo}")

    framework = 'torch'
    if args.ml_config_path is not None:
        with open(args.ml_config_path) as file:
            configs = yaml.load(file, Loader=yaml.FullLoader)
            configs = configs.get(args.env, None)
            framework = configs.get("framework", 'torch')

            configs["callbacks"] = CustomEnvironmentCallbacks
            algo_config = algo_config.from_dict(configs)
    algo_config = algo_config.environment(args.env)
    print('Config:', algo_config.to_dict())

    # Register models
    for k, v in get_model_catalog_dict(alg_name, framework).items():
        ModelCatalog.register_custom_model(k, v)

    run_config = RunConfig(name=args.run_id)
    local_mode = False  # Set to True to debug locally
    ray.shutdown()
    ray.init(ignore_reinit_error=True, num_cpus=args.cpus, num_gpus=num_gpus,
             include_dashboard=False, local_mode=local_mode)
    tuner = tune.Tuner(
        algo_class,
        param_space=algo_config.to_dict(),
        run_config=run_config,
    )
    tuner.fit()


def run_experiments(args):
    for exp in args.experiments:
        if exp == 'all' or exp == "xadqn_siamese":
            run_griddrive_siamese_experiments(args)
        elif exp == 'all' or exp == "xasac_siamese":
            run_mujoco_siamese_experiments(args)
        else:
            raise ValueError(f'Unknown experiment {exp}')


def main():
    euler_res = "/cluster/project/jbuhmann/workspace/deer/results/tests"
    parser = argparse.ArgumentParser(description="Deer experiments")
    parser.add_argument("--results_dir", type=Path, default=Path(euler_res),
                        help="Path in which results of training are/will be "
                             "located")
    parser.add_argument("-n", "--run_id", type=Path,
                        help="directory name for results to be saved")
    parser.add_argument("--ml_config_path",
                        type=Path, default=None,
                        help="Path to the ml-agents or sb3 config. "
                             "Ex: 'configs/xadqn_siamese_config.yaml'")

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
    parser.add_argument("--env", type=str, default='GridDrive-Hard')

    parser.add_argument("--algo", type=str, default='xadqn',
                        help='xadqn for now')

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
    parser.add_argument('--gpus', default=1, required=False, type=int)
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
