"""
Small example demonstrating training with the attention-based Actor and Critic
from `rl_classes.py`.

This script creates a MADDPGAgent and trains it on synthetic data so you can
see the batched attention codepaths exercised (actor accepts lists of enemy
feature arrays of variable lengths). It's intentionally small and dependency-
free (no pygame) so you can run it quickly.

Run: python3 example_rl_attention.py
"""
import random
import numpy as np
import torch
import torch.nn.functional as F
from rl_classes import MADDPGAgent


class SimpleReplayBuffer:
    def __init__(self, max_size=10000):
        self.buffer = []
        self.max_size = max_size

    def add(self, enemy_feats, state, action, reward, next_enemy_feats, next_state, done):
        if len(self.buffer) >= self.max_size:
            self.buffer.pop(0)
        self.buffer.append((enemy_feats, state, action, reward, next_enemy_feats, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(batch_size, len(self.buffer)))
        enemy_feats, states, actions, rewards, next_enemy_feats, next_states, dones = zip(*batch)
        return list(enemy_feats), np.array(states, dtype=np.float32), np.array(actions, dtype=np.float32), np.array(rewards, dtype=np.float32), list(next_enemy_feats), np.array(next_states, dtype=np.float32), np.array(dones, dtype=np.float32)

    def __len__(self):
        return len(self.buffer)


def make_random_enemy_feats(max_enemies=6):
    # variable-length lists of enemy feature arrays (N_i, 4)
    N = random.randint(0, max_enemies)
    if N == 0:
        return np.zeros((0, 4), dtype=np.float32)
    feats = np.random.randn(N, 4).astype(np.float32) * 10.0
    return feats


def main():
    seed = 1
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # match dimensions used in the project
    state_dim = 6
    feats_dim_1d = 4
    action_dim = 2
    max_action = 50.0

    agent = MADDPGAgent(state_dim, feats_dim_1d, action_dim, max_action, epsilon=0.2, lr=1e-3)

    buffer = SimpleReplayBuffer(max_size=5000)

    # populate with random transitions
    for _ in range(2000):
        s = np.random.randn(state_dim).astype(np.float32)
        enemy = make_random_enemy_feats()
        a = np.tanh(np.random.randn(action_dim).astype(np.float32)) * max_action
        reward = float(np.random.randn() * 0.1)
        next_s = np.random.randn(state_dim).astype(np.float32)
        next_enemy = make_random_enemy_feats()
        done = False
        buffer.add(enemy, s, a, reward, next_enemy, next_s, done)

    # small training run to exercise actor/critic attention and batched actor call
    batch_size = 64
    iters = 2000
    print("Starting synthetic training run (this is a short demo)...")
    for i in range(iters):
        if len(buffer) < batch_size:
            continue

        enemy_feats_batch, states, actions, rewards, next_enemy_feats_batch, next_states, dones = buffer.sample(batch_size)

        states_t = torch.FloatTensor(states)
        actions_t = torch.FloatTensor(actions)
        rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
        next_states_t = torch.FloatTensor(next_states)
        dones_t = torch.FloatTensor(dones).unsqueeze(1)

        enemy_feats_t = [torch.FloatTensor(e) for e in enemy_feats_batch]
        next_enemy_feats_t = [torch.FloatTensor(e) for e in next_enemy_feats_batch]

        with torch.no_grad():
            next_actions = agent.actor_target(next_enemy_feats_t, next_states_t)
            target_q = agent.critic_target(next_states_t, next_actions)
            target_q = rewards_t + (1.0 - dones_t) * agent.gamma * target_q

        current_q = agent.critic(states_t, actions_t)
        critic_loss = F.mse_loss(current_q, target_q)

        agent.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 1.0)
        agent.critic_optimizer.step()

        # actor update: pass a list of enemy_feats and batched states
    # --- diagnostics: mean Q and gradient norms ---
        actor_actions = agent.actor(enemy_feats_t, states_t)
        actor_loss = -agent.critic(states_t, actor_actions).mean()

        agent.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 1.0)
        agent.actor_optimizer.step()

        # soft updates
        agent.soft_update(agent.actor_target, agent.actor)
        agent.soft_update(agent.critic_target, agent.critic)

        if (i + 1) % 50 == 0:
            # compute mean Q for the actor actions (no grad needed)
            with torch.no_grad():
                mean_q = agent.critic(states_t, actor_actions).mean().item()

            def grad_norm(params):
                total = 0.0
                count = 0
                for p in params:
                    if p.grad is not None:
                        total += p.grad.detach().norm().item() ** 2
                        count += 1
                return (total ** 0.5) if count > 0 else 0.0

            critic_gn = grad_norm(agent.critic.parameters())
            actor_gn = grad_norm(agent.actor.parameters())

            # parameter norms (useful to catch blow-ups)
            def param_norm(params):
                total = 0.0
                count = 0
                for p in params:
                    total += p.detach().norm().item() ** 2
                    count += 1
                return (total ** 0.5) if count > 0 else 0.0

            critic_pn = param_norm(agent.critic.parameters())
            actor_pn = param_norm(agent.actor.parameters())

            print(f"Iter {i+1}/{iters} | Critic Loss: {critic_loss.item():.6f} | Actor Loss: {actor_loss.item():.6f} | mean_Q: {mean_q:.6f} | c_gn: {critic_gn:.6f} | a_gn: {actor_gn:.6f} | c_pn: {critic_pn:.3f} | a_pn: {actor_pn:.3f}")


    # Quick sanity test: forward pass single sample with empty enemy list
    s = torch.FloatTensor(np.zeros(state_dim, dtype=np.float32)).unsqueeze(0)
    empty_enemy = [torch.zeros((0, feats_dim_1d), dtype=torch.float32)]
    try:
        action_out = agent.actor(empty_enemy, s)
        print("Sanity check passed: actor produced action for empty enemy list ->", action_out.shape)
    except Exception as e:
        print("Sanity check FAILED:", e)

    print("Demo finished. If you want to integrate this with the pygame environment, run the real training loop in `main.py`.")


if __name__ == '__main__':
    main()
