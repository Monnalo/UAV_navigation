import pickle
import numpy as np

import torch
import torch.optim as optim
import torch.nn as nn

from model.replay_buffer import ReplayBuffer
from model.network_TD3 import Actor, Critic
from model.util import hard_update, soft_update

BATCH_SIZE = 256
GAMMA = 0.98
LR_C = 0.001
LR_A = 0.0001
LR_I = 0.01
TAU = 0.01
POLICY_NOISE = 0.2
POLICY_FREQ = 1
NOISE_CLIP = 0.5


class DRL:
        
    def __init__(self, action_dim, state_dim, LR_C = LR_C, LR_A = LR_A):
        
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.tau = TAU
        self.policy_noise = POLICY_NOISE
        self.noise_clip = NOISE_CLIP
        self.policy_freq = POLICY_FREQ
        self.itera = 0

        self.memory = Memory(MEMORY_CAPACITY)
        
        self.actor = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_target = Actor(self.state_dim, self.action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), LR_A)
        
        self.critic = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_target = Critic(self.state_dim, self.action_dim).to(self.device)
        self.critic_optimizers = torch.optim.Adam(self.critic.parameters(), LR_C)
        
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

    def choose_action(self, state):

        state = torch.tensor(state, dtype=torch.float).to(self.device)
        state = state.unsqueeze(0)

        action = self.actor.forward(state).detach()
        action = action.squeeze(0).cpu().numpy()
        action = np.clip(action, -1, 1)

        return action
            
    def learn(self):
        if len(self.memory) < self.batch_size:
            return
        # batched state, batched action, batched reward, batched next state
        bs, ba, br, bs_, done = self.memory.sample(self.batch_size)
        bs = torch.tensor(bs, dtype=torch.float).to(self.device)
        bs_ = torch.tensor(bs_, dtype=torch.float).to(self.device)
        ba = torch.tensor(ba, dtype=torch.float).to(self.device)
        br = torch.tensor(br, dtype=torch.float).unsqueeze(1).to(self.device)
        done = torch.tensor(done, dtype=torch.float32).unsqueeze(1).to(self.device)
        
        # initialize the loss variables
        loss_c, loss_a = 0, 0

        # calculate the predicted values of the critic
        with torch.no_grad():
            noise = (torch.randn_like(ba) * self.policy_noise).clamp(0, 1)
            a_ = (self.actor_target(bs_).detach() + noise).clamp(0, 1)
            target_q1, target_q2 = self.critic_target([bs_, a_])
            target_q1 = target_q1.detach()
            target_q2 = target_q2.detach()
            target_q = torch.min(target_q1, target_q2)
            y_expected = br + self.gamma * target_q   
        y_predicted1, y_predicted2 = self.critic.forward([bs,ba])    
        errors = y_expected - y_predicted1
        
        # update the critic
        critic_loss = nn.MSELoss()
        loss_critic = critic_loss(y_predicted1,y_expected)+critic_loss(y_predicted2,y_expected)
        self.critic_optimizers.zero_grad()
        loss_critic.backward()
        self.critic_optimizers.step()
        
        # update the actor
        if self.itera % self.policy_freq == 0:

            pred_a = self.actor.forward(bs)
            loss_actor = (-self.critic.forward([bs,pred_a])[0])

            self.actor_optimizer.zero_grad()
            loss_actor.mean().backward()
            self.actor_optimizer.step()

            soft_update(self.actor_target, self.actor, self.tau)
            soft_update(self.critic_target, self.critic, self.tau)

            loss_a = loss_actor.mean().item()

        loss_c = loss_critic.mean().item()

        self.itera += 1

        self.memory.batch_update(tree_idx, abs(errors.detach().cpu().numpy()) )

        return loss_c, loss_a

    def save(self, path):
        torch.save(self.actor.state_dict(), path + 'checkpoint.pt')

    def load(self, path):
        self.actor.load_state_dict(torch.load(path + 'checkpoint.pt'))


if __name__ == '__main__':
    print("")

        
        
        
        
        
        
        
