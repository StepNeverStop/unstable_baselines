from ntpath import join
import os
import sys
# sys.path.append(os.path.join(os.getcwd(), './'))
# sys.path.append(os.path.join(os.getcwd(), '../../'))
import gym
import click
from unstable_baselines.common.logger import Logger
from unstable_baselines.baselines.sac.trainer import SACTrainer
from unstable_baselines.baselines.sac.agent import SACAgent
from unstable_baselines.common.agents import RandomAgent
from unstable_baselines.common.util import set_device_and_logger, load_config, set_global_seed
from unstable_baselines.common.buffer import ReplayBuffer
from unstable_baselines.common.env_wrapper import get_env, ScaleRewardWrapper

@click.command(context_settings=dict(
    ignore_unknown_options=True,
    allow_extra_args=True,
))
@click.argument("config-path",type=str)
@click.option("--log-dir", default=os.path.join("logs", "sac"))
@click.option("--gpu", type=int, default=-1)
@click.option("--print-log", type=bool, default=True)
@click.option("--seed", type=int, default=35)
@click.option("--info", type=str, default="")
@click.argument('args', nargs=-1)
def main(config_path, log_dir, gpu, print_log, seed, info, args):
    print(args)
    #todo: add load and update parameters function
    args = load_config(config_path, args)

    #set global seed
    set_global_seed(seed)

    #initialize logger
    env_name = args['env_name']
    logger = Logger(log_dir, env_name, prefix = info, print_to_terminal=print_log)

    #set device and logger
    set_device_and_logger(gpu, logger)

    #save args
    logger.log_str_object("parameters", log_dict = args)

    #initialize environment
    logger.log_str("Initializing Environment")
    train_env = get_env(env_name)
    eval_env = get_env(env_name)
    joint_action_space = train_env.joint_action_space
    joint_observation_space = train_env.joint_observation_space
    observation_space = joint_observation_space[0]
    action_space = joint_action_space[0]
    print(action_space)
    print(joint_action_space)

    #initialize buffer
    logger.log_str("Initializing Buffer")
    buffer = ReplayBuffer(observation_space, action_space, **args['buffer'])

    #initialize agent
    logger.log_str("Initializing Agent")
    agent = SACAgent(observation_space, action_space, **args['agent'])
    #initialize opponent agent
    opponent_agent_type = args['common']['opponent_agent']['type']
    if opponent_agent_type == "random":
        opponent_agent = RandomAgent(observation_space, action_space, **args['opponent_agent'])
    elif opponent_agent_type == "delayed":
        raise NotImplementedError
    elif opponent_agent_type == "self_play":
        raise NotImplementedError
    #initialize trainer
    logger.log_str("Initializing Trainer")
    trainer  = SACTrainer(
        agent,
        opponent_agent,
        train_env,
        eval_env,
        buffer,
        **args['trainer']
    )

    
    logger.log_str("Started training")
    trainer.train()


if __name__ == "__main__":
    main()