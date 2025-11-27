import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, epsilon, lr=1e-3, lr_critic=1e-3,
                 gamma=0.95, tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau

        # actor networks (homogeneous)
        self.actor = Actor(state_dim, max_action)
        self.actor_target = Actor(state_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # critic networks (centralized)
        total_state_dim = state_dim
        total_action_dim = action_dim
        self.critic = Critic(total_state_dim, total_action_dim)
        self.critic_target = Critic(total_state_dim, total_action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        exploring = False
        if random.random() < self.epsilon:
            exploring = True
            action = np.random.normal(0, self.max_action, size=self.action_dim)
        return [exploring, action]

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class Actor(nn.Module):
    def __init__(self, state_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.dir = nn.Linear(128, 2)
        self.distance = nn.Linear(128, 1)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))

        direction = F.normalize(self.dir(x), dim=-1)
        actual_distance = torch.sigmoid(self.distance(x)) * self.max_action

        delta = direction * actual_distance
        return delta

class Critic(nn.Module):
    def __init__(self, total_state_dim, total_action_dim):
        super().__init__()
        self.fc1 = nn.Linear(total_state_dim + total_action_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, 1)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)
