import numpy as np
import math
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Toggle debugging prints for attention
DEBUG_ATTENTION = False
_ATT_DEBUG_COUNT = 0


class MADDPGAgent:
    def __init__(self, state_dim, feats_dim_1d, action_dim, max_action, epsilon, lr=1e-3,
                 gamma=0.95, tau=0.01):
        self.state_dim = state_dim
        self.feats_dim_1d = feats_dim_1d
        self.embed_dim = 16
        self.action_dim = action_dim
        self.max_action = max_action
        self.epsilon = epsilon
        self.gamma = gamma
        self.tau = tau

        # actor networks (homogeneous)
        self.actor = Actor(state_dim, self.feats_dim_1d, self.embed_dim, max_action)
        self.actor_target = Actor(state_dim, self.feats_dim_1d, self.embed_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        # critic networks (centralized)
        total_state_dim = state_dim
        total_action_dim = action_dim
        self.critic = Critic(total_state_dim, total_action_dim)
        self.critic_target = Critic(total_state_dim, total_action_dim)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def select_action(self, enemy_feats, state):
        """
        enemy_feats: (N, feats_dim_1d) numpy
        state:       (state_dim,)     numpy
        """
        enemy_feats = torch.tensor(enemy_feats, dtype=torch.float32)   # (N, feats_dim_1d)
        state = torch.tensor(state, dtype=torch.float32)               # (state_dim,)

        action = self.actor(enemy_feats, state).detach().cpu().numpy()

        # exploration
        exploring = False
        if random.random() < self.epsilon:
            exploring = True
            noise_std = self.max_action
            action = np.asarray(np.random.normal(0, noise_std, size=self.action_dim))

        return action, exploring

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)


class Actor(nn.Module):
    def __init__(self, state_dim, feats_dim_1d, embed_dim, max_action):
        super().__init__()
        self.state_dim = state_dim
        self.feat_dim = feats_dim_1d
        self.embed_dim = embed_dim

        # Attention over enemy features
        self.attn_pool = AttentionPool(feats_dim_1d, embed_dim)

        # Query network: **state → query** (no enemy mean in here anymore)
        self.query_net = nn.Linear(state_dim, embed_dim)

        self.fc1 = nn.Linear(state_dim + embed_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.dir = nn.Linear(128, 2)
        self.distance = nn.Linear(128, 1)

        # weight initialization
        for m in [self.fc1, self.fc2, self.dir, self.distance, self.query_net]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        self.max_action = max_action

    def forward(self, enemy_feats, state):
        """
        enemy_feats: (N, feat_dim)  OR list of length B with tensors (N_i, feat_dim)
        state:       (state_dim,)   OR (B, state_dim)
        """
        # -------- Batched case: list of tensors --------
        if isinstance(enemy_feats, list):
            # state: (B, state_dim)
            # Normalize shapes for each sample
            enemy_feats_norm = [normalize_feats(f) for f in enemy_feats]

            # Pad to (B, maxN, feat_dim) and mask
            enemy_feats_pad, enemy_mask = pad_and_stack(enemy_feats_norm)

            # Query from state only
            # (B, embed_dim)
            query = torch.tanh(self.query_net(state))

            # Attention over padded enemies with mask + distance bias
            enemy_embed = self.attn_pool.forward_batched(
                enemy_feats_pad, query_batch=query, mask=enemy_mask
            )  # (B, embed_dim)

            x = torch.cat([enemy_embed, state], dim=-1)  # (B, embed_dim + state_dim)

        # -------- Single-agent case: tensor --------
        else:
            # enemy_feats: (N, feat_dim)
            # state: (state_dim,)
            device = state.device
            f = normalize_feats(enemy_feats).to(device)   # (N, feat_dim) or (0, feat_dim)
            N = f.shape[0]

            if N == 0:
                # no visible enemies → zero embedding
                enemy_embed = torch.zeros(self.embed_dim, device=device)
            else:
                # query from state only
                query = torch.tanh(self.query_net(state))  # (embed_dim,)
                enemy_embed = self.attn_pool(f, query)     # (embed_dim,)

            x = torch.cat([enemy_embed, state], dim=-1)    # (embed_dim + state_dim,)

        # common MLP head
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
        # init
        for m in [self.fc1, self.fc2, self.out]:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, states, actions):
        x = torch.cat([states, actions], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class AttentionPool(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.key = nn.Linear(input_dim, embed_dim)
        self.value = nn.Linear(input_dim, embed_dim)
        self.embed_dim = embed_dim
        self.input_dim = input_dim

        nn.init.xavier_uniform_(self.key.weight)
        nn.init.zeros_(self.key.bias)
        nn.init.xavier_uniform_(self.value.weight)
        nn.init.zeros_(self.value.bias)

    # ---------------------------------------------------------
    # Single-agent mode (enemy_feats: (N, input_dim))
    # ---------------------------------------------------------
    def forward(self, enemy_feats, query, mask=None):
        """
        enemy_feats: (N, input_dim)
        query:      (embed_dim,)
        mask:       (N,) boolean or None
        """
        device = next(self.parameters()).device
        enemy_feats = enemy_feats.to(device)
        query = query.to(device)

        N = enemy_feats.shape[0]

        if N == 0:
            # No enemies → return zero embedding
            return torch.zeros(self.embed_dim, device=device)

        K = self.key(enemy_feats)   # (N, D)
        V = self.value(enemy_feats) # (N, D)

        # Distance-based bias (closer enemies → higher logit)
        dist = torch.norm(enemy_feats[:, :2], dim=1)        # (N,)
        distance_bias = torch.exp(-0.01 * dist)             # (N,)

        attn_logits = (K @ query) + distance_bias           # (N,)

        # If no mask provided, make an "all valid" mask
        if mask is None:
            mask = torch.ones(N, dtype=torch.bool, device=device)

        # Mask invalid positions (if any)
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        # Softmax over valid enemies
        attn_weights = F.softmax(attn_logits, dim=0)
        attn_weights = attn_weights.masked_fill(~mask, 0.0)

        # Weighted sum
        pooled = (attn_weights.unsqueeze(-1) * V).sum(dim=0)   # (D,)
        return pooled

    # ---------------------------------------------------------
    # Batched attention (enemy_feats already padded by caller)
    # enemy_feats: (B, maxN, input_dim)
    # query_batch: (B, embed_dim)
    # mask:        (B, maxN)
    # ---------------------------------------------------------
    def forward_batched(self, enemy_feats, query_batch, mask):
        """
        enemy_feats: (B, maxN, input_dim)
        query_batch: (B, embed_dim)
        mask:        (B, maxN) bool
        """
        device = next(self.parameters()).device
        enemy_feats = enemy_feats.to(device)
        query_batch = query_batch.to(device)
        mask = mask.to(device)

        B, maxN, _ = enemy_feats.shape

        # Compute keys & values
        K = self.key(enemy_feats)   # (B, maxN, D)
        V = self.value(enemy_feats) # (B, maxN, D)

        # Distance-based bias (per enemy)
        dist = torch.norm(enemy_feats[..., :2], dim=-1)       # (B, maxN)
        distance_bias = torch.exp(-0.01 * dist)               # (B, maxN)

        # Attention logits: (B, maxN)
        attn_logits = torch.einsum("bnd,bd->bn", K, query_batch) + distance_bias

        # Mask invalid entries
        attn_logits = attn_logits.masked_fill(~mask, float('-inf'))

        # Softmax over enemies
        attn_weights = F.softmax(attn_logits, dim=1)
        attn_weights = attn_weights.masked_fill(~mask, 0.0)

        # Weighted pooling
        pooled = (attn_weights.unsqueeze(-1) * V).sum(dim=1)  # (B, D)
        return pooled


def pad_and_stack(feats_list):
    """
    feats_list: list of (Ni, feat_dim) tensors, Ni may be 0
    Returns:
        padded: (B, maxN, feat_dim)
        mask:   (B, maxN) bool
    """
    # Pick device from first tensor
    device = feats_list[0].device if isinstance(feats_list[0], torch.Tensor) else torch.device("cpu")

    # Determine feat_dim robustly
    if feats_list[0].numel() > 0 and feats_list[0].dim() == 2:
        feat_dim = feats_list[0].shape[1]
    else:
        feat_dim = 4  # fallback; if you change feature dim, it's OK because non-empty cases dominate

    B = len(feats_list)
    maxN = max(f.shape[0] for f in feats_list)

    padded = torch.zeros(B, maxN, feat_dim, device=device)
    mask   = torch.zeros(B, maxN, dtype=torch.bool, device=device)

    for i, f in enumerate(feats_list):
        f = f.to(device)
        n = f.shape[0]
        if n > 0:
            padded[i, :n] = f
            mask[i, :n] = 1

    return padded, mask


def normalize_feats(f):
    """
    Normalize enemy feature tensor shape.
    Input:
        f: (N, feat_dim) or (feat_dim,) or empty
    Output:
        (N, feat_dim) or (0, feat_dim)
    """
    f = torch.as_tensor(f, dtype=torch.float32)

    if f.dim() == 1:
        return f.unsqueeze(0)  # (1, feat_dim)

    if f.dim() == 2:
        return f

    # Empty or weird → (0, 4) as a safe default
    return torch.zeros(0, 4, dtype=torch.float32, device=f.device)
