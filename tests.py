import torch
import numpy as np
import math
import pytest

from rl_classes import (
    normalize_feats,
    pad_and_stack,
    AttentionPool,
    Actor,
    Critic,
    MADDPGAgent,
)


def test_normalize_feats_variants():
    # 1D input -> becomes (1, feat_dim)
    v = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    out = normalize_feats(v)
    assert isinstance(out, torch.Tensor)
    assert out.shape == (1, 4)

    # 2D input -> unchanged
    v2 = np.array([[1.0, 2.0, 3.0, 4.0], [0, 0, 0, 0]], dtype=np.float32)
    out2 = normalize_feats(v2)
    assert out2.shape == (2, 4)

    # Weird/empty input -> returns (0, 4)
    out3 = normalize_feats(np.array([], dtype=np.float32))
    assert out3.shape[0] == 0 and out3.shape[1] == 4


def test_pad_and_stack_various_lengths():
    a = torch.zeros(2, 4)
    b = torch.zeros(0, 4)
    c = torch.ones(3, 4)

    padded, mask = pad_and_stack([a, b, c])
    assert padded.shape == (3, 3, 4)
    assert mask.shape == (3, 3)
    # first row has 2 valid
    assert mask[0].sum().item() == 2
    # second row had zero
    assert mask[1].sum().item() == 0
    # third row has 3 valid
    assert mask[2].sum().item() == 3


def test_pad_and_stack_all_empty():
    a = torch.zeros(0, 4)
    b = torch.zeros(0, 4)
    padded, mask = pad_and_stack([a, b])
    # when all empty, padded created with maxN=1
    assert padded.shape[0] == 2
    assert padded.shape[1] == 1
    assert mask.sum().item() == 0


def test_attention_forward_batched_handles_all_masked_row():
    torch.manual_seed(0)
    pool = AttentionPool(input_dim=4, embed_dim=8)

    # Batch size 2: first has 2 enemies, second has none (masked)
    enemy_feats = torch.zeros(2, 2, 4)
    enemy_feats[0, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
    enemy_feats[0, 1] = torch.tensor([0.0, 1.0, 0.0, 0.0])

    query = torch.randn(2, 8)
    mask = torch.tensor([[1, 1], [0, 0]], dtype=torch.bool)

    pooled = pool.forward_batched(enemy_feats, query, mask)
    assert pooled.shape == (2, 8)
    # second row was fully masked -> should be zeros
    assert torch.allclose(pooled[1], torch.zeros_like(pooled[1]))


def test_attention_forward_single_empty_returns_zero():
    pool = AttentionPool(input_dim=4, embed_dim=6)
    empty = torch.zeros(0, 4)
    q = torch.zeros(6)
    out = pool.forward(empty, q)
    assert out.shape == (6,)
    assert torch.allclose(out, torch.zeros_like(out))


def test_actor_single_and_batched():
    torch.manual_seed(1)
    state_dim = 6
    feats_dim = 4
    embed_dim = 8
    max_action = 7

    actor = Actor(state_dim, feats_dim, embed_dim, max_action)

    # single-agent, no enemies
    enemy_feats = torch.zeros(0, feats_dim)
    state = torch.zeros(state_dim)
    out = actor(enemy_feats, state)
    assert out.shape == (2,)
    assert out.abs().max().item() <= float(max_action) + 1e-6

    # batched case with list of enemy tensors
    enemy_list = [torch.zeros(0, feats_dim), torch.randn(2, feats_dim)]
    states_b = torch.zeros(2, state_dim)
    out_b = actor(enemy_list, states_b)
    assert out_b.shape == (2, 2)
    assert (out_b.abs() <= float(max_action) + 1e-6).all()


def test_critic_forward():
    critic = Critic(total_state_dim=6, total_action_dim=2)
    states = torch.zeros(3, 6)
    actions = torch.zeros(3, 2)
    out = critic(states, actions)
    assert out.shape == (3, 1)


def test_maddpgagent_select_action_and_soft_update():
    # small agent
    agent = MADDPGAgent(state_dim=6, feats_dim_1d=4, action_dim=2, max_action=5, epsilon=1.0, lr=1e-3, gamma=0.9, tau=0.5)

    # with epsilon=1.0, should always explore (random draw < 1.0)
    enemy_feats = np.zeros((0, 4), dtype=np.float32)
    state = np.zeros(6, dtype=np.float32)
    action, exploring = agent.select_action(enemy_feats, state)
    assert isinstance(action, np.ndarray)
    assert action.shape == (2,)
    assert exploring is True
    assert np.all(action <= 5 + 1e-6) and np.all(action >= -5 - 1e-6)

    # Test soft update: set source params to ones and target to zeros; after soft_update target should be tau*1 + (1-tau)*0
    for p in agent.actor.parameters():
        p.data.fill_(1.0)
    for tp in agent.actor_target.parameters():
        tp.data.zero_()

    agent.tau = 0.3
    agent.soft_update(agent.actor_target, agent.actor)

    # Check at least one parameter updated to ~0.3
    found = False
    for tp in agent.actor_target.parameters():
        if torch.allclose(tp.data, torch.full_like(tp.data, 0.3)):
            found = True
            break
    # It's sufficient that some parameter tensors changed toward 0.3
    assert found or any((tp.data - 0.3).abs().max().item() < 1e-4 for tp in agent.actor_target.parameters())


def test_attention_prefers_closer_enemy_single_and_batched():
    """Conceptual test: when keys do not depend on query, distance bias should make attention prefer closer enemies."""
    torch.manual_seed(0)
    # make embed_dim == input_dim so we can set value to identity
    pool = AttentionPool(input_dim=4, embed_dim=4)

    # Zero the key transform so logits come purely from distance_bias
    pool.key.weight.data.zero_()
    pool.key.bias.data.zero_()

    # Make the value transform an identity mapping (so pooled vector is weighted avg of raw features)
    pool.value.weight.data.copy_(torch.eye(4))
    pool.value.bias.data.zero_()

    # two enemies: close and far
    enemies = torch.tensor([[1.0, 0.0, 0.0, 0.0], [10.0, 0.0, 0.0, 0.0]])
    q = torch.zeros(4)

    pooled = pool.forward(enemies, q)

    dist_to_close = torch.norm(pooled - enemies[0])
    dist_to_far = torch.norm(pooled - enemies[1])
    assert dist_to_close < dist_to_far

    # Batched version: a batch with one row like above and another row reversed
    enemies_b = torch.zeros(2, 2, 4)
    enemies_b[0, 0] = enemies[0]
    enemies_b[0, 1] = enemies[1]
    enemies_b[1, 0] = enemies[1]
    enemies_b[1, 1] = enemies[0]

    query_b = torch.zeros(2, 4)
    mask = torch.tensor([[1, 1], [1, 1]], dtype=torch.bool)

    pooled_b = pool.forward_batched(enemies_b, query_b, mask)
    # For first batch element, pooled should be closer to enemies_b[0,0]
    assert torch.norm(pooled_b[0] - enemies_b[0, 0]) < torch.norm(pooled_b[0] - enemies_b[0, 1])
    # For second batch element, pooled should be closer to the close enemy at index 1
    assert torch.norm(pooled_b[1] - enemies_b[1, 1]) < torch.norm(pooled_b[1] - enemies_b[1, 0])


def test_attention_single_vs_batched_consistency():
    # small deterministic config as above
    pool = AttentionPool(input_dim=4, embed_dim=4)
    pool.key.weight.data.zero_(); pool.key.bias.data.zero_()
    pool.value.weight.data.copy_(torch.eye(4)); pool.value.bias.data.zero_()

    enemies = torch.tensor([[2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]])
    q = torch.zeros(4)

    out_single = pool.forward(enemies, q)

    # batched (1 element) via pad_and_stack
    enemies_b, mask = pad_and_stack([enemies])
    out_batched = pool.forward_batched(enemies_b, q.unsqueeze(0), mask)
    assert torch.allclose(out_single, out_batched[0], atol=1e-6)


def test_actor_policy_various_input_sizes_and_batched_list():
    torch.manual_seed(2)
    state_dim = 6
    feats_dim = 4
    embed_dim = 8
    max_action = 5.0

    actor = Actor(state_dim, feats_dim, embed_dim, max_action)

    # 0 enemies
    out0 = actor(torch.zeros(0, feats_dim), torch.zeros(state_dim))
    assert out0.shape == (2,)
    assert (out0.abs() <= max_action + 1e-6).all()

    # 1 enemy
    out1 = actor(torch.randn(1, feats_dim), torch.randn(state_dim))
    assert out1.shape == (2,)
    assert (out1.abs() <= max_action + 1e-6).all()

    # many enemies
    out_many = actor(torch.randn(10, feats_dim), torch.randn(state_dim))
    assert out_many.shape == (2,)
    assert (out_many.abs() <= max_action + 1e-6).all()

    # batched list with heterogeneous lengths (including empty)
    enemy_list = [torch.zeros(0, feats_dim), torch.randn(1, feats_dim), torch.randn(5, feats_dim)]
    states_b = torch.randn(3, state_dim)
    out_b = actor(enemy_list, states_b)
    assert out_b.shape == (3, 2)
    assert (out_b.abs() <= max_action + 1e-6).all()
