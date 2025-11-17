import pygame,random
from pygame.locals import *
import numpy as np
import math
from sea_environment import *
from rl_classes import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, enemy_feats, state, action, reward, next_enemy_feats, next_state, done):
        self.buffer.append(
            (enemy_feats, state, action, reward, next_enemy_feats, next_state, done)
        )

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        enemy_feats_per_agent, states, actions, rewards, next_enemy_feats_per_agent, next_states, dones = zip(*batch)

        # enemy_feats and next_enemy_feats stay lists of arrays!
        return (
            list(enemy_feats_per_agent),
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.float32),
            np.array(rewards, dtype=np.float32),
            list(next_enemy_feats_per_agent),
            np.array(next_states, dtype=np.float32),
            np.array(dones),
        )

    def __len__(self):
        return len(self.buffer)

def load_weights(flag):
    if flag:
        checkpoint = torch.load("models/weights.pt", weights_only=False)
        params = checkpoint['hyperparameters']
        maddpg_agent = MADDPGAgent( params['state_dim'], params['feats_dim_1d'], params['action_dim'], params['max_action'], epsilon, gamma=params['gamma'], tau=params['tau'])
        maddpg_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        maddpg_agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        maddpg_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        if checkpoint['critic_state_dict'] and maddpg_agent is not None:
            total_state_dim = state_dim
            total_action_dim = action_dim
            maddpg_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            maddpg_agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            maddpg_agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    else:
        maddpg_agent = MADDPGAgent(state_dim, feats_dim_1d, action_dim, max_action, epsilon, lr=lr, gamma=gamma, tau=0.01)
        total_state_dim = state_dim
        total_action_dim = action_dim
        maddpg_agent.critic_target.load_state_dict(maddpg_agent.critic.state_dict())

    return maddpg_agent

# hyperparameters
state_dim = 6
feats_dim_1d = 4
action_dim = 2
max_action = 50
batch_size = 200
epsilon = 0.5
epsilon_decay = 0.995
epsilon_min = 0.001
gamma = 0.95
lr = 1e-3

# defaults
default_num_agents = 10
default_num_enemies = 15
default_num_barriers = 0
default_episodes = 1000
default_ticks = 400
decision_making_ans = None

# outside stuff
load_weights_ans = 0
show_sim_ans = 0
save_weights_ans = 0
num_agents_ans = input("\nHow Many Agents? (int): ")
num_enemies_ans = input("How Many Enemies? (int): ")
num_barriers_ans = input("How Many Barriers? (int): ")
num_episodes_ans = input("How Many Episodes? (int): ")
num_ticks_ans = input("How many Ticks? (int): ")
decision_making_ans = input("Decision Making? (random/escape/static/mixed): ")
mesh_ans = input("Mesh? (y/n): ")
print("")

while load_weights_ans != "y" and load_weights_ans != "n":
    load_weights_ans = input("Load Weights? (y/n): ")
while show_sim_ans != "y" and show_sim_ans != "n":
    show_sim_ans = input("Show Sim? (y/n): ")

if show_sim_ans == "y":
    show_sim = True
else:
    show_sim = False
if num_agents_ans == "":
    num_agents = default_num_agents
else:
    num_agents = int(num_agents_ans)
if num_enemies_ans == "":
    num_enemies = default_num_enemies
else:
    num_enemies = int(num_enemies_ans)
if num_barriers_ans == "":
    num_barriers = default_num_barriers
else:
    num_barriers = int(num_barriers_ans)
if num_episodes_ans == "":
    episodes = default_episodes
else:
    episodes = int(num_episodes_ans)
if num_ticks_ans == "":
    max_ticks = default_ticks
else:
    max_ticks = int(num_ticks_ans)
if decision_making_ans == "":
    decision_making = "static"
else:
    decision_making = decision_making_ans
if mesh_ans == "y":
    mesh_ans = True
else:
    mesh_ans = False
if load_weights_ans == "y":
    epsilon = 0.001
    maddpg_agent = load_weights(True)
else:
    maddpg_agent = load_weights(False)

print(f"")
if load_weights_ans == "y":
    print(f"Loaded Weights")
else:
    print(f"Did Not Load Weights")
print(f"Using {num_agents} agents")
print(f"Using {num_enemies} enemies")
print(f"Using {episodes} episodes")
print(f"")

# Pygame stuff
pygame.init()
start_width = 500
start_height = 500
screen = pygame.display.set_mode((start_width, start_height), pygame.RESIZABLE)
black = (0,0,0)
white = (255, 255, 255)
blue = (50,50,160)
red = (160, 50, 50)
yellow = (100, 100, 10)
color_var = white
clock = pygame.time.Clock()
pygame.display.set_caption("UxS MARL SWARMS")

# sim / env
my_world = World(screen)

# time stuff
update_after = 1500
my_world.use_policy_after = 10  # policy & training is used this many ticks (right now 5x per second)
my_world.mesh_ans = mesh_ans

# replay buffer
replay_buffer = ReplayBuffer(max_size=100000)

# diagnostic print limiter for actor actions
ATT_DIAG_PRINTS = 0

# training loop
episode_rewards = []
actor_losses = []
critic_losses = []

exit_flag = False
for episode in range(episodes):
    my_world.reset()

    for i in range(num_agents):
        my_world.add_swarm_uuv_random(blue, maddpg_agent)

    for i in range(num_enemies):
        if decision_making_ans == "mixed":
            if episode <= 4/5 * episodes:
                decision_making = "random"
            else:
                decision_making = "escape"
        my_world.add_enemy_uuv_random(yellow, decision_making)

    for i in range(num_barriers):
        my_world.add_barrier_random(100, (50,50,50))

    episode_reward = 0
    episode_actor_loss = []
    episode_critic_loss = []

    # game loop
    for tick in range(max_ticks):

        # pygame / user interaction stuff
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                ...
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    my_world.add_controllable_uuv(mouse_x, mouse_y, np.array([0,1]), (200, 200, 200))
                if event.key == pygame.K_e:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    my_world.add_enemy_uuv(mouse_x, mouse_y, yellow, decision_making)
                if event.key == pygame.K_p:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    my_world.add_particle(mouse_x, mouse_y, 7, white)
                    for uuv in my_world.uuvs:
                        if isinstance(uuv, SwarmUUV):
                            uuv.add_waypoint((mouse_x, mouse_y))
                if event.key == pygame.K_c:
                    my_world.reset()
            if event.type == pygame.VIDEORESIZE:
                width, height = event.w, event.h
                screen = pygame.display.set_mode((width, height), pygame.RESIZABLE)
            if event.type == QUIT:
                exit_flag = True
        if exit_flag:
            break
        if my_world.controllable_uuv != None:
            if keys[pygame.K_w]:
                my_world.controllable_uuv.increase_throttle()
            if keys[pygame.K_s]:
                my_world.controllable_uuv.decrease_throttle()
            if keys[pygame.K_a]:
                my_world.controllable_uuv.turn_left()
            if keys[pygame.K_d]:
                my_world.controllable_uuv.turn_right()
        if keys[pygame.K_v]:
            mouse_x, mouse_y = pygame.mouse.get_pos()
            dir = np.array([random.uniform(-1,1),random.uniform(-1,1)])
            unit_vec_dir = dir / np.linalg.norm(dir)
            my_world.add_swarm_uuv(mouse_x, mouse_y, unit_vec_dir, blue, maddpg_agent)

        # set previous states
        if tick % my_world.use_policy_after == 0:
            prev_states = []
            prev_actions = []
            prev_agent_ids = []
            prev_enemy_feats_per_agent = []
            for uuv in my_world.uuvs:
                if isinstance(uuv, SwarmUUV):
                    prev_states.append(uuv.get_state())
                    prev_actions.append(uuv.current_action)
                    prev_agent_ids.append(uuv.id)
                    prev_enemy_feats_per_agent.append(uuv.get_enemies())

        # step / tick environment
        screen.fill(black)
        my_world.tick()

        if tick % my_world.use_policy_after == 0:
            # collect transitions
            next_states = []
            rewards = []
            dones = []
            current_ids = {u.id for u in my_world.uuvs if isinstance(u, SwarmUUV)}
            surviving_agents = [u for u in my_world.uuvs if isinstance(u, SwarmUUV)]

            for i, agent_id in enumerate(prev_agent_ids):
                if agent_id in current_ids:
                    uuv = next(u for u in surviving_agents if u.id == agent_id)
                    next_state = uuv.get_state()
                    reward = my_world.get_and_clear_reward(agent_id)

                    if prev_actions[i] is not None:
                        prev_enemy_feats = prev_enemy_feats_per_agent[i]
                        next_enemy_feats = uuv.get_enemies()

                        replay_buffer.add(prev_enemy_feats, prev_states[i], prev_actions[i],
                                        reward, next_enemy_feats, next_state, False)
                        episode_reward += reward
                else:
                    reward = my_world.get_and_clear_reward(agent_id)

                    if prev_actions[i] is not None:
                        # use prev_state as next_state for terminal, agent is done
                        prev_enemy_feats = prev_enemy_feats_per_agent[i]

                        replay_buffer.add(prev_enemy_feats, prev_states[i], prev_actions[i],
                                        reward, prev_enemy_feats, prev_states[i], True)
                        episode_reward += reward

            # training step
            if len(replay_buffer) > update_after:
                enemy_feats_per_agent, states, actions, rewards, next_enemy_feats_per_agent, next_states, dones = replay_buffer.sample(batch_size)

                states_t       = torch.FloatTensor(states)  # (B, state_dim)
                actions_t      = torch.FloatTensor(actions)  # (B, action_dim)
                rewards_t      = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_t  = torch.FloatTensor(next_states)
                dones_t        = torch.FloatTensor(dones).unsqueeze(1)

                # Convert to tensor lists (avoid redundant conversions)
                enemy_feats_t = [torch.FloatTensor(e) for e in enemy_feats_per_agent]
                next_enemy_feats_t = [torch.FloatTensor(e) for e in next_enemy_feats_per_agent]

                with torch.no_grad():
                    # Batched forward pass for target actor
                    next_actions_t = maddpg_agent.actor_target(next_enemy_feats_t, next_states_t)  # (B, action_dim)

                    target_q = maddpg_agent.critic_target(next_states_t, next_actions_t)
                    target_q = rewards_t + (1 - dones_t) * maddpg_agent.gamma * target_q

                current_q = maddpg_agent.critic(states_t, actions_t)
                critic_loss = F.mse_loss(current_q, target_q)

                maddpg_agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                # gradient clipping to stabilize training
                try:
                    torch.nn.utils.clip_grad_norm_(maddpg_agent.critic.parameters(), 1.0)
                except Exception:
                    pass
                maddpg_agent.critic_optimizer.step()
                episode_critic_loss.append(critic_loss.item())


                # update actor (batched)
                actor_actions_t = maddpg_agent.actor(enemy_feats_t, states_t)

                # diagnostic printing for actor outputs when attention debug is enabled
                try:
                    if 'DEBUG_ATTENTION' in globals() and DEBUG_ATTENTION and ATT_DIAG_PRINTS < 10:
                        ATT_DIAG_PRINTS += 1
                        print(f"[ACTOR] actions mean: {actor_actions_t.mean().item():.4f}, std: {actor_actions_t.std().item():.4f}")
                        sample = actor_actions_t[:5].detach().cpu().numpy()
                        print(f"[ACTOR] sample actions (first 5): {sample}")
                except Exception:
                    pass

                actor_loss = -maddpg_agent.critic(states_t, actor_actions_t).mean()

                maddpg_agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                try:
                    torch.nn.utils.clip_grad_norm_(maddpg_agent.actor.parameters(), 1.0)
                except Exception:
                    pass
                maddpg_agent.actor_optimizer.step()
                episode_actor_loss.append(actor_loss.item())

                # soft update target networks
                maddpg_agent.soft_update(maddpg_agent.actor_target, maddpg_agent.actor)
                maddpg_agent.soft_update(maddpg_agent.critic_target, maddpg_agent.critic)

        if show_sim and episode % 5 == 0:
            pygame.display.flip()
            clock.tick(200)

    if exit_flag:
        break

    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    maddpg_agent.epsilon = epsilon

    episode_rewards.append(episode_reward)
    if episode_actor_loss:
        actor_losses.append(np.mean(episode_actor_loss))
    if episode_critic_loss:
        critic_losses.append(np.mean(episode_critic_loss))

    if episode % 5 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode:04d} | Avg Reward: {avg_reward:.2f} | Epsilon: {epsilon:.3f} | Buffer: {len(replay_buffer)}")
        if actor_losses:
            print(f"  Actor Loss: {actor_losses[-1]:.4f} | Critic Loss: {critic_losses[-1]:.4f}")

pygame.quit()

def save_weights():
    print("Training Done")
    save_path = "models/weights.pt"
    torch.save({
        'actor_state_dict': maddpg_agent.actor.state_dict(),
        'actor_target_state_dict': maddpg_agent.actor_target.state_dict(),
        'critic_state_dict': maddpg_agent.critic.state_dict() if maddpg_agent.critic else None,
        'critic_target_state_dict': maddpg_agent.critic_target.state_dict() if maddpg_agent.critic_target else None,
        'actor_optimizer_state_dict': maddpg_agent.actor_optimizer.state_dict(),
        'critic_optimizer_state_dict': maddpg_agent.critic_optimizer.state_dict() if maddpg_agent.critic_optimizer else None,
        'episode': episodes,
        'episode_rewards': episode_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses,
        'hyperparameters': {
            'state_dim': state_dim,
            'feats_dim_1d': feats_dim_1d,
            'action_dim': action_dim,
            'max_action': max_action,
            'gamma': maddpg_agent.gamma,
            'tau': maddpg_agent.tau,
            'lr_actor': lr,
            'lr_critic': lr
        }
    }, save_path)
    print(f"\nModel weights saved to {save_path}")

    import json
    metrics_path = "models/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses
        }, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

save_weights()
