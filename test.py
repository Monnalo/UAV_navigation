import airsim
import numpy as np

from torch.utils.tensorboard import SummaryWriter
from env.multirotor import Multirotor
from utils import save_results, plot_rewards


def set_seed(seed):

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def test():

    set_seed(args.seed)

    print('Start testing')
    print(f'Env:{args.env_name}, Algorithm:{args.algo_name}, Device:{args.device}')

    human = 0
    if args.algorithm == 0:    # DDPG
        if args.sensors == 0:
            sensor = 0
        else:
            sensor = 1
    elif args.algorithm == 1:  # DDPG-H
        sensor = 1
        human = 1
    elif args.algorithm == 2:  # TD3
        sensor = 1
    else:                      # TD3-H
        sensor = 1
        human = 1

    if sensor == 0:
        s_dim = 20
    else:
        s_dim = 49

    if human == 1:
       s_dim += 1

    a_dim = 3

    start_epoch = 0

    if args.algorithm == 0 or args.algorithm == 1:
        from model.DDPG import DRL
        DRL = DRL(a_dim, s_dim, sensor)
    else:
        from model.TD3 import DRL
        DRL = DRL(a_dim, s_dim, sensor)

    rewards = []
    ma_rewards = []
    writer = SummaryWriter('./test_image')
    success = 0
    for i_ep in range(args.test_eps):
        env = Multirotor(sensor, human)
        state = env.get_state()
        ep_reward = 0
        finish_step = 0
        final_distance = state[3] * env.init_distance
        for i_step in range(args.max_step):
            finish_step = finish_step + 1
            action = DRL.choose_action(state)
            state_, reward, done = env.step(action)
            ep_reward += reward
            state = state_
            print('\rEpisode: {}\tStep: {}\tReward: {:.2f}\tDistance: {}'.format(i_ep + 1, i_step + 1, ep_reward,
                                                                                 state[3] * env.init_distance), end="")
            final_distance = state[3] * env.init_distance
            if done:
                break
        print('\rEpisode: {}\tFinish step: {}\tReward: {:.2f}\tFinal distance: {}'.format(i_ep + 1, finish_step,
                                                                                          ep_reward, final_distance))
        if final_distance <= 30.0:
            success += 1
        rewards.append(ep_reward)
        if ma_rewards:
            ma_rewards.append(0.9 * ma_rewards[-1] + 0.1 * ep_reward)
        else:
            ma_rewards.append(ep_reward)
        # writer.add_scalars(main_tag='test',
        #                    tag_scalar_dict={
        #                        'reward': ep_reward,
        #                        'ma_reward': ma_rewards[-1]
        #                    },
        #                    global_step=i_ep)
    print('Finish testing!')
    print('Average Reward: {}\tSuccess Rate: {}'.format(np.mean(rewards), success / args.test_eps))
    writer.close()

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
    test()

    set_seed(cfg.seed)
    # connect to the AirSim simulator
    client = airsim.MultirotorClient()

    DRL.load(path=args.model_path)
    rewards, ma_rewards = test()
    # save_results(rewards, ma_rewards, tag='test', path=args.result_path)
    # plot_rewards(rewards, ma_rewards, args, tag="test")
