import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, epsilon, lr=1e-3, lr_critic=1e-3,
                 gamma=0.95, tau=0.01):
        self.state_dim = state_dim
        self.embed_dim = 32
        self.action_dim = action_dim
        self.max_action = max_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau

        # actor networks (homogeneous)
        self.actor = Actor(state_dim, self.embed_dim, max_action)
        self.actor_target = Actor(state_dim, self.embed_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # critic networks (centralized)
        total_state_dim = state_dim
        total_action_dim = action_dim
        self.critic = Critic(total_state_dim, total_action_dim)
        self.critic_target = Critic(total_state_dim, total_action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, enemy_feats, state):
        enemy_feats = torch.tensor(enemy_feats, dtype=torch.float32)   # (N, 4)
        state = torch.tensor(state, dtype=torch.float32)

        action = self.actor(enemy_feats, state).detach().cpu().numpy()
        if random.random() < self.epsilon:
            action = np.asarray(np.random.normal(0, self.max_action, size=self.action_dim))
        return action

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class Actor(nn.Module):
    def __init__(self, state_dim, embed_dim, max_action):
        super().__init__()
        self.attn_pool = AttentionPool(input_dim=4, embed_dim=32)

        self.fc1 = nn.Linear(state_dim + embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dir = nn.Linear(128, 2)
        self.distance = nn.Linear(128, 1)

        self.max_action = max_action

    def forward(self, enemy_feats, state):
        # enemy_feats: (N,4)
        # state: either (state_dim,) or (1,state_dim)

        enemy_embed = self.attn_pool(enemy_feats)  # (embed_dim,)

        # final concat
        x = torch.cat([enemy_embed, state], dim=-1)  # (embed_dim + state_dim,)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        direction = F.normalize(self.dir(x), dim=-1)
        actual_distance = torch.sigmoid(self.distance(x)) * self.max_action
        result = direction * actual_distance

        return result

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

class AttentionPool(nn.Module):
    def __init__(self, input_dim=4, embed_dim=32):
        super().__init__()
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)

        # learned query vector
        self.query = nn.Parameter(torch.randn(embed_dim))

    def forward(self, enemy_feats):
        """
        enemy_feats: (N, 4) for N enemies
        returns: (embed_dim,)
        """
        # print("enemy_feats.shape entering pool:", enemy_feats.shape)
        # print(enemy_feats)
        K = self.key(enemy_feats)  # (N, D)
        V = self.value(enemy_feats)  # (N, D)

        # q @ k_i  → attention logits
        attn_logits = (K @ self.query)  # (N,)
        attn_weights = F.softmax(attn_logits, dim=0)

        # weighted sum of values
        pooled = (attn_weights.unsqueeze(-1) * V).sum(dim=-2)
        return pooled
