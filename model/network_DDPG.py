import torch
import torch.nn as nn
import torch.nn.functional as F


class Actor(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=256, init_w=3e-3):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(nb_states, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.fc5 = nn.Linear(hidden, nb_actions)

        nn.init.uniform_(self.linear5.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.linear5.bias.detach(), a=-init_w, b=init_w)

        self.relu = nn.ReLU()
        self.sig = nn.Tanh()

    def forward(self, state):
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.relu(self.fc4(x))
        action = self.sig(self.fc5(x))
        return out * 0.7


class Critic(nn.Module):
    def __init__(self, nb_states, nb_actions, hidden=256, init_w=3e-3):
        super(Critic,self).__init__()

        self.fc1 = nn.Linear(nb_states + nb_actions, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, 128)
        self.fc5 = nn.Linear(128, 1)
        
        self.relu = nn.ReLU()

        nn.init.uniform_(self.fc5.weight.detach(), a=-init_w, b=init_w)
        nn.init.uniform_(self.fc5.bias.detach(), a=-init_w, b=init_w)

    def forward(self, inp):
        x, a = inp
        x = torch.cat([x, a], 1)

        q = self.relu(self.fc1(x))
        q = self.relu(self.fc2(q))
        q = self.relu(self.fc3(q))
        q = self.relu(self.fc4(q))
        q = self.fc5(q)
        
        return q


    
    
    
    
