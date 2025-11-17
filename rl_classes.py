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
        # enemy_feats: (N,4) or list of (N_i, 4) tensors for batching
        # state: (state_dim,) or (B, state_dim)

        # Handle both single and batched inputs
        if isinstance(enemy_feats, list):
            # Batched case
            enemy_embed, _ = self.attn_pool.forward_batched(enemy_feats)  # (B, embed_dim)
            x = torch.cat([enemy_embed, state], dim=-1)  # (B, embed_dim + state_dim)
        else:
            # Single case
            enemy_embed = self.attn_pool(enemy_feats)  # (embed_dim,)
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
        K = self.key(enemy_feats)  # (N, D)
        V = self.value(enemy_feats)  # (N, D)

        # q @ k_i  → attention logits
        attn_logits = (K @ self.query)  # (N,)
        attn_weights = F.softmax(attn_logits, dim=0)

        # weighted sum of values
        pooled = (attn_weights.unsqueeze(-1) * V).sum(dim=-2)
        return pooled

    def forward_batched(self, enemy_feats_list, max_enemies=None):
        """
        Batched attention processing for multiple agents.

        Args:
            enemy_feats_list: List of (N_i, 4) tensors for batch_size agents
            max_enemies: Maximum number of enemies to pad to (auto-compute if None)

        Returns:
            pooled_embeddings: (batch_size, embed_dim)
            attention_masks: (batch_size, max_enemies) bool tensor
        """
        if max_enemies is None:
            max_enemies = max(ef.shape[0] for ef in enemy_feats_list)

        batch_size = len(enemy_feats_list)
        device = next(self.parameters()).device

        # Pad all enemy feature sequences to max_enemies
        padded_feats = torch.zeros(batch_size, max_enemies, 4, device=device)
        attention_masks = torch.zeros(batch_size, max_enemies, dtype=torch.bool, device=device)

        for i, ef in enumerate(enemy_feats_list):
            n_enemies = ef.shape[0]
            padded_feats[i, :n_enemies] = ef.to(device)
            attention_masks[i, :n_enemies] = True

        # Compute keys and values for all agents at once
        K = self.key(padded_feats)  # (batch_size, max_enemies, embed_dim)
        V = self.value(padded_feats)  # (batch_size, max_enemies, embed_dim)

        # Compute attention logits
        attn_logits = (K @ self.query)  # (batch_size, max_enemies)

        # Mask padding positions before softmax
        attn_logits = attn_logits.masked_fill(~attention_masks, float('-inf'))
        attn_weights = F.softmax(attn_logits, dim=1)  # (batch_size, max_enemies)
        attn_weights = attn_weights.masked_fill(~attention_masks, 0.0)

        # Weighted sum of values
        pooled = (attn_weights.unsqueeze(-1) * V).sum(dim=1)  # (batch_size, embed_dim)

        return pooled, attention_masks
