import os
import sys

import numpy as np
import torch
import argparse
import airsim

from torch.utils.tensorboard import SummaryWriter
from model.ou_noise import OrnsteinUhlenbeckActionNoise as OUNoise
from env.multirotor import Multirotor
from utils import save_results, make_dir, plot_rewards, save_args

curr_path = os.path.dirname(os.path.abspath(__file__))  # current path
parent_path = os.path.dirname(curr_path)  # parent path
sys.path.append(parent_path)  # add to system path

def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def train():

    set_seed(args.seed)

    print('Start training!')
    print(f'Env:{args.env_name}, Algorithm:{args.algo_name}, Device:{args.device}')

    # sensor = 0 means UAV only use distance sensors, sensor = 1
    # means UAV use distance sensors and depth camera
    human = 0
    if args.algorithm == 0:             # DDPG
        file_name = "DDPG"
        if args.sensors == 0:
            sensor = 0
        else:
            sensor = 1
    elif args.algorithm == 1:           # DDPG-H
        file_name = "DDPG-H"
        sensor = 1
        human = 1
    elif args.algorithm == 2:           # TD3
        file_name = "TD3"
        sensor = 1
    else:                               # TD3-H
        file_name = "TD3-H"
        sensor = 1
        human = 1

    # the dimension of UAV state and action
    # only use distance sensors
    if sensor == 0:
        s_dim = 20
    # distance sensors and depth camera
    else:
        s_dim = 49
    # human-in-the-loop need to add a dimension
    # to characterize the avoidance state
    if human == 1:
       s_dim += 1
    # action space: ax, ay, az
    a_dim = 3

    start_epoch = 0

    if args.algorithm == 0 or args.algorithm == 1:
        from model.DDPG import DRL
        DRL = DRL(a_dim, s_dim, sensor)

        ou_noise = OUNoise(mu=np.zeros(a_dim), decay_period=args.maximum_train_episode * 0.5)  # noise of action
        rewards = []
        ma_rewards = []
        writer = SummaryWriter('./train_image')
        for i_ep in range(args.maximum_train_episode):
            env = Multirotor(sensor, human)
            state = env.get_state()  # get UAV state
            ou_noise.reset()  # reset the noise
            ep_reward = 0
            finish_step = 0
            final_distance = state[3] * env.init_distance
            for i_step in range(args.maximum_step):
                finish_step = finish_step + 1

                '''
                Section DRL's actting
                '''
                action = DRL.choose_action(state)
                action = action + ou_noise(i_step)
                action = np.clip(action, -1, 1)

                '''
                Section environment update
                '''
                next_state, reward, done = env.step(action)
                ep_reward += reward

                '''
                Section DRL store
                '''
                DRL.memory.push(state, action, reward, next_state, done)

                replay_len = len(DRL.memory)
                k = 1 + replay_len / args.memory_capacity
                update_times = int(k * args.update_times)
                for _ in range(update_times):
                    DRL.update()

                state = next_state
                print(
                    '\rEpisode: {}\tStep: {}\tReward: {:.2f}\tDistance: {:.2f}'.format(i_ep + 1, i_step + 1, ep_reward,
                                                                                       state[3] * env.init_distance),
                    end="")
                final_distance = state[3] * env.init_distance
                if done:
                    break
            print('\rEpisode: {}\tFinish step: {}\tReward: {:.2f}\tFinal distance: {:.2f}'.format(i_ep + 1, finish_step,
                                                                                                  ep_reward,
                                                                                                  final_distance))
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            writer.add_scalars(main_tag='train',
                               tag_scalar_dict={
                                   'reward': ep_reward,
                                   'ma_reward': ma_rewards[-1]
                               },
                               global_step=i_ep)
            if (i_ep + 1) % 10 == 0:
                DRL.save(path=args.model_path)
        writer.close()
        print('Finish training!')
        return rewards, ma_rewards

    else:
        from model.TD3 import DRL
        DRL = DRL(a_dim, s_dim, sensor)

        # set TD3's exploration rate
        exploration_rate = args.initial_exploration_rate

        total_step = 0
        a_loss, c_loss = 0, 0
        rewards = []
        ma_rewards = []
        writer = SummaryWriter('./train_image')

        for i_ep in range(start_epoch, args.maximum_train_episode):
            env = Multirotor(sensor, human)  # connect to Airsim
            state = env.get_state()  # get UAV state
            ep_reward = 0
            step = 0
            finish_step = 0
            final_distance = state[3] * env.init_distance
            for i_step in range(args.maximum_step):
                finish_step = finish_step + 1

                '''
                Section DRL's actting
                '''
                action = DRL.choose_action(state)
                # add a Gaussian noise to the DRL action
                action = np.clip(np.random.normal(action, exploration_rate), -1, 1)

                '''
                Section environment update
                '''
                state_, reward, done = env.step(action)

                '''
                Section DRL store
                '''
                DRL.memory.push(state, action, reward, state_, done)

                '''
                Section DRL update
                '''
                learn_threshold = 256
                if total_step > learn_threshold:
                    c_loss, a_loss = DRL.learn()
                    loss_critic.append(np.average(c_loss))
                    loss_actor.append(np.average(a_loss))
                    # Decrease the exploration rate
                    exploration_rate = exploration_rate * args.exploration_decay_rate if exploration_rate > args.cutoff_exploration_rate else 0.05

                ep_reward += reward
                state = state_
                total_step += 1
                step += 1
                print('\rEpisode: {}\tStep: {}\tReward: {:.2f}\tDistance: {:.2f}'.format(i_ep + 1, i_step + 1, ep_reward, state[3] * env.init_distance), end="")
                final_distance = state[3] * env.init_distance
                if done:
                    break
            print('\rEpisode: {}\tFinish step: {}\tReward: {:.2f}\tFinal distance: {:.2f}'.format(i_ep + 1, finish_step, ep_reward, final_distance))rewards.append(ep_reward)
            rewards.append(ep_reward)
            if ma_rewards:
                ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
            else:
                ma_rewards.append(ep_reward)
            writer.add_scalars(main_tag='train',
                               tag_scalar_dict={
                                   'reward': ep_reward,
                                   'ma_reward': ma_rewards[-1]
                               },
                               global_step=i_ep)
            if (i_ep + 1) % 10 == 0:
                DRL.save(path=args.model_path)
        writer.close()
        print('Finish training!')
        return rewards, ma_rewards


if __name__ == '__main__':
    # Arguments
    parser = argparse.ArgumentParser(description="Training")
    parser.add_argument('--algorithm', type=int, help="RL algorithm(0 for DDPG, 1 for DDPG-H, 2 for TD3, 3 for TD3-H)")
    parser.add_argument('--env_name', type=str, help="name of environment", default='UE4 and Airsim')
    parser.add_argument('--seed', type=int, help="random seed", default=1)
    parser.add_argument('--num', type=int, help='the number of UAVs', default=1)
    parser.add_argument('--update_times', type=int, help="update times", default=1)
    parser.add_argument('--sensors', type=int, help="the sensors of UAV(0 for distance sensors only, 1 for distance sensor and depth camera)", default=0)
    parser.add_argument('--maximum_train_episode', type=int, help='maximum training episode number (default:1500)', default=1500)
    parser.add_argument('--maximum_step', type=int, help='maximum training step number of one episode (default:500)', default=500)
    parser.add_argument('--maximum_test_episode', type=int, help='maximum training episode number (default:100)', default=100)
    parser.add_argument('--memory_capacity', type=int, help="memory capacity", default=2 ** 17)
    parser.add_argument("--initial_exploration_rate", type=float, help="initial explore policy variance (default: 0.5)", default=0.5)
    parser.add_argument("--cutoff_exploration_rate", type=float, help="minimum explore policy variance (default: 0.05)", default=0.05)
    parser.add_argument("--exploration_decay_rate", type=float, help="decay factor of explore policy variance (default: 0.99988)", default=0.99988)
    parser.add_argument('--result_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + file_name + '/results/')
    parser.add_argument('--model_path', default=curr_path + "/outputs/" + parser.parse_args().env_name + '/' + file_name + '/models/')
    parser.add_argument('--save_fig', type=bool, help="if save figure or not", default=True)
    parser.add_argument('--device', type=str, help='run on which device (default: cuda)', default='cuda')
    args = parser.parse_args()

    # Run
    train()

    # Save
    make_dir(args.result_path, args.model_path)
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()

    rewards, ma_rewards = train()
    save_args(args)
    save_results(rewards, ma_rewards, tag='train', path=args.result_path)
    # plot
    plot_rewards(rewards, ma_rewards, args, tag="train")
