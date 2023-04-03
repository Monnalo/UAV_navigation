import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from model.replay_buffer import ReplayBuffer
from model.network_DDPG import Actor, Critic
from model.util import hard_update, soft_update

BATCH_SIZE = 256
GAMMA = 0.98
LR_C = 0.001
LR_A = 0.0001
TAU = 0.01


class DRL:
    def __init__(self, action_dim, state_dim, LR_C = LR_C, LR_A = LR_A):

        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU

        self.memory = ReplayBuffer(args.memory_capacity)

        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.target_actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), LR_A)

        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.target_critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), LR_C)

        # copy the parma from the online network to the target network
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(param.data)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(param.data)

    def choose_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)

        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()

        return action

    def update(self):
        if len(self.memory) < self.batch_size:
            return
        # batched state, batched action, batched reward, batched next state
        bs, ba, br, bs_, done = self.memory.sample(self.batch_size)
        bs = torch.tensor(bs, dtype=torch.float).to(self.device)
        bs_ = torch.tensor(bs_, dtype=torch.float).to(self.device)
        ba = torch.tensor(ba, dtype=torch.float).to(self.device)
        br = torch.tensor(br, dtype=torch.float).unsqueeze(1).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)

        # calculate the policy loss
        policy_loss = self.critic(bs, self.actor(bs))
        policy_loss = -policy_loss.mean()

        # calculate the expected_value by target network
        ba_ = self.target_actor(bs_)
        target_value = self.target_critic(bs_, ba_ .detach())
        expected_value = reward + (1.0 - done) * self.gamma * target_value
        expected_value = torch.clamp(expected_value, -np.inf, np.inf)

        value = self.critic(bs, ba)
        mse_loss = nn.MSELoss()
        value_loss = mse_loss(value, expected_value.detach())

        # online network update
        self.actor_optimizer.zero_grad()
        policy_loss.backward()
        self.actor_optimizer.step()
        self.critic_optimizer.zero_grad()
        value_loss.backward()
        self.critic_optimizer.step()

        # target network update, soft update
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.soft_tau) + param.data * self.soft_tau)

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))


if __name__ == '__main__':
    print("")
