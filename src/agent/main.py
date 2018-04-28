import tensorflow as tf
import random
import sys
import argparse

import bot


hyper = {
    'gamma': 0.9,
    'epsilon': 0.6,
    'target_epsilon': 0.01,
    'epsilon_strength': 50,
    'memory_size': 10000,
    'batch_size': 24,
    'episodes': 400,
    'ep_max_steps': 30,
    'hidden_nodes': 24,
    'bias': 0.01,
    'tau': 0.1,
    'trace_size': 12,
    'dropout': 0.2
}


parser = argparse.ArgumentParser(
    description=
        """
        LSTM-DDDQN bot:
        Declare environment with gym, coin or coinlist.
        Example usage: --gym CartPole-v0
        """
)
parser.add_argument(
    '--gym', 
    help='name of openAI gym env'
)
parser.add_argument(
    '--coin', 
    help='name of coin'
)
parser.add_argument(
    '--coinlist', 
    help='filename location w/list of coins, delimeter=newline'
)
parser.add_argument(
    '--render', 
    action='store_true',
    help='render openAI gym environment'
)
parser.add_argument(
    '--shuffle', 
    action='store_true',
    help='random coin selection on each episode'
)
parser.add_argument(
    '--verbose',
    action='store_true',
    help='print reward of each episode'
)
parser.add_argument(
    '--lstm',
    type=int,
    default=1,
    help='1 = lstm network, 0 = feed forward only'
)
parser.add_argument(
    '--seed',
    type=int,
    default=0,
    help='use seed to control randomness'
)
parser.add_argument(
    '--gamma',
    default=hyper['gamma'],
    type=float,
    help='discount of future reward'
)
parser.add_argument(
    '--epsilon',
    default=hyper['epsilon'],
    type=float,
    help='exploration start rate'
)
parser.add_argument(
    '--target_epsilon',
    default=hyper['target_epsilon'],
    type=float,
    help='exploration convergence goal'
)
parser.add_argument(
    '--epsilon_strength',
    default=hyper['epsilon_strength'],
    type=int,
    help='higher = more exploration time'
)
parser.add_argument(
    '--memory_size',
    default=hyper['memory_size'],
    type=int,
    help='size of experience replay memory'
)
parser.add_argument(
    '--batch_size',
    default=hyper['batch_size'],
    type=int,
    help='batch size of NN'
)
parser.add_argument(
    '--episodes',
    default=hyper['episodes'],
    type=int,
    help='max number of episodes'
)
parser.add_argument(
    '--ep_max_steps',
    default=hyper['ep_max_steps'],
    type=int,
    help='number of env steps per episode'
)
parser.add_argument(
    '--hidden_nodes',
    default=hyper['hidden_nodes'],
    type=int,
    help='number of hidden nodes for each layer in NN'
)
parser.add_argument(
    '--bias',
    default=hyper['bias'],
    type=float,
    help='bias for each layer in network'
)
parser.add_argument(
    '--tau',
    default=hyper['tau'],
    type=float,
    help='convergence rate from online to target network'
)
parser.add_argument(
    '--trace_size',
    default=hyper['trace_size'],
    type=int,
    help='length of each sample of episode for lstm time sequence'
)
parser.add_argument(
    '--dropout',
    default=hyper['dropout'],
    type=float,
    help='dropout rate of network'
)


def env_parser(args):
    try:
        gym, env_name = get_env(args)
        if not env_name:
            raise AssertionError 
        return gym, env_name
    except(AssertionError):
        print('env must be declared, see --help')
        sys.exit(1)
    except(OSError):
        print('coinlist file not found')
        sys.exit(1)


def get_env(args):
    if (args.gym):
        import gym
        env_name = args.gym
    else:
        sys.path.append("..")
        from gym_crypto.env import Env
        gym = Env()
        env_name = get_coin(args)
    return gym, env_name


def get_coin(args):
    if args.coin:
        return [args.coin]
    elif args.coinlist:
        return open(args.coinlist).read().splitlines()
    else:
        return None


def set_seed(seed):
    tf.set_random_seed(seed)
    random.seed(seed)


def get_bot(args):
    if args.lstm:
        return bot.BotLSTM(gym, env_name, args)
    else:
        return bot.BotFF(gym, env_name, args)

if __name__ == '__main__':
    tf.reset_default_graph()
    args = parser.parse_args()
    if args.seed:
        set_seed(args.seed)
    gym, env_name = env_parser(args)
    bot = get_bot(args)
    rewards = bot.q_train()


