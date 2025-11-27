import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np

class MADDPGAgent:
    def __init__(self, personal_state_dim, observed_object_state_dim, observable_enemies, observable_friendlies, action_dim, max_action, epsilon, lr=1e-3, lr_critic=1e-3, gamma=0.95, tau=0.01):
        self.personal_state_dim = personal_state_dim
        self.observed_object_state_dim = observed_object_state_dim
        self.observable_enemies = observable_enemies
        self.observable_friendlies = observable_friendlies
        self.total_state_dim = personal_state_dim + observed_object_state_dim * (observable_enemies + observable_friendlies)
        self.action_dim = action_dim
        self.max_action = max_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau

        # actor networks (homogeneous)
        self.actor = Actor(self.total_state_dim, max_action)
        self.actor_target = Actor(self.total_state_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # critic networks (centralized)
        total_action_dim = action_dim
        self.critic = Critic(self.total_state_dim, total_action_dim)
        self.critic_target = Critic(self.total_state_dim, total_action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        exploring = False
        if random.random() < self.epsilon:
            exploring = True
            noise_std = float(self.max_action)
            action = action + np.random.normal(0, noise_std, size=self.action_dim)
        return (exploring, action)

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

        direction = F.normalize(self.dir(x) + 1e-8, dim=-1)
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
