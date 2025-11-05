import pygame,random
from pygame.locals import *
import numpy as np
import math
from sea_environment import *
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = deque(maxlen=max_size)

    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), np.array(rewards),
                np.array(next_states), np.array(dones))

    def __len__(self):
        return len(self.buffer)

class MADDPGAgent:
    def __init__(self, state_dim, action_dim, max_action, lr_actor=1e-3, lr_critic=1e-3,
                 gamma=0.95, tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.max_action = max_action
        self.gamma = gamma
        self.tau = tau

        # Actor networks
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # Critic networks (centralized - takes all agents' states and actions)
        self.critic = None  # Will be set externally
        self.critic_target = None
        self.critic_optimizer = None

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        action = self.actor(state).detach().cpu().numpy()[0]
        if noise > 0:
            action += np.random.normal(0, noise, size=self.action_dim)
            action = np.clip(action, -self.max_action, self.max_action)
        return action

    def soft_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.out = nn.Linear(128, action_dim)
        self.max_action = max_action

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.out(x))
        return x * self.max_action

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

def load_weights(flag):
    if flag:
        checkpoint = torch.load("maddpg_weights.pth", weights_only=False)
        params = checkpoint['hyperparameters']
        maddpg_agent = MADDPGAgent(
            params['state_dim'],
            params['action_dim'],
            params['max_action'],
            gamma=params['gamma'],
            tau=params['tau']
        )
        maddpg_agent.actor.load_state_dict(checkpoint['actor_state_dict'])
        maddpg_agent.actor_target.load_state_dict(checkpoint['actor_target_state_dict'])
        maddpg_agent.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])

        if checkpoint['critic_state_dict'] and maddpg_agent is not None:
            total_state_dim = state_dim
            total_action_dim = action_dim
            maddpg_agent.critic = Critic(total_state_dim, total_action_dim)
            maddpg_agent.critic_target = Critic(total_state_dim, total_action_dim)
            maddpg_agent.critic.load_state_dict(checkpoint['critic_state_dict'])
            maddpg_agent.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])
            maddpg_agent.critic_optimizer = optim.Adam(maddpg_agent.critic.parameters(), lr=lr)
            maddpg_agent.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])
    else:
        maddpg_agent = MADDPGAgent(state_dim, action_dim, max_action,
                                lr_actor=lr, lr_critic=lr,
                                gamma=gamma, tau=0.01)
        total_state_dim = state_dim
        total_action_dim = action_dim
        maddpg_agent.critic = Critic(total_state_dim, total_action_dim)
        maddpg_agent.critic_target = Critic(total_state_dim, total_action_dim)
        maddpg_agent.critic_target.load_state_dict(maddpg_agent.critic.state_dict())
        maddpg_agent.critic_optimizer = optim.Adam(maddpg_agent.critic.parameters(),
                                                lr=lr)

    return maddpg_agent

# Pygame stuff
pygame.init()
start_width = 500
start_height = 500
screen = pygame.display.set_mode((start_width, start_height), pygame.RESIZABLE)
black = (0,0,0)
white = (255, 255, 255)
blue = (50,50,160)
red = (160, 50, 50)
yellow = (160, 160, 50)
color_var = white
clock = pygame.time.Clock()
pygame.display.set_caption("Particle Sim")

# Sim / env
my_world = World(screen)

# Hyperparameters
state_dim = 9
action_dim = 2
max_action = 50.0
batch_size = 200
noise = 0.3
noise_decay = 0.9995
noise_min = 0.01

gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3

# Time stuff
update_after = 1000
max_ticks = 400
my_world.use_policy_after = 10  # policy & training is used this many ticks (right now 5x per second)
episodes = 1000

# Outside stuff
maddpg_agent = load_weights(False)
show_sim = False
maddpg_agent = load_weights(True)
show_sim = True

# Replay buffer
replay_buffer = ReplayBuffer(max_size=100000)

# Training loop
episode_rewards = []
actor_losses = []
critic_losses = []

exit_flag = False
for episode in range(episodes):
    my_world.reset()

    num_agents = 3
    for i in range(num_agents):
        x = random.randint(50, 450)
        y = random.randint(50, 450)
        dir = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        unit_vec_dir = dir / np.linalg.norm(dir)
        my_world.add_vatn_uuv(x, y, unit_vec_dir, 5, blue, maddpg_agent)

    num_enemies = 5
    for i in range(num_agents):
        enemy_x = random.randint(100, 400)
        enemy_y = random.randint(100, 400)
        my_world.add_controllable_uuv(enemy_x, enemy_y, np.array([1,0]), 5, yellow, 0)

    episode_reward = 0
    episode_actor_loss = []
    episode_critic_loss = []


    for tick in range(max_ticks):
        keys = pygame.key.get_pressed()

        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                color = blue
                dir = np.array([random.uniform(-1,1),random.uniform(-1,1)])
                unit_vec_dir = dir / np.linalg.norm(dir)
                my_world.add_vatn_uuv(mouse_x, mouse_y, unit_vec_dir, 5, color, maddpg_agent)
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_g:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    color = yellow
                    my_world.add_controllable_uuv(mouse_x, mouse_y, np.array([0,1]), 5, color, id)
                if event.key == pygame.K_p:
                    mouse_x, mouse_y = pygame.mouse.get_pos()
                    color = white
                    my_world.add_particle(mouse_x, mouse_y, 7, color)
                    for uuv in my_world.uuvs:
                        if isinstance(uuv, VatnUUV):
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

        #####
        #####
        #####
        if tick % my_world.use_policy_after == 0:
            prev_states = []
            prev_actions = []
            for uuv in my_world.uuvs:
                if isinstance(uuv, VatnUUV):
                    prev_states.append(uuv.get_state())

        # Step environment
        screen.fill(black)
        my_world.tick()

        if tick % my_world.use_policy_after == 0:
            # Collect transitions
            next_states = []
            rewards = []
            dones = []

            for i, uuv in enumerate([u for u in my_world.uuvs if isinstance(u, VatnUUV)]):
                if i < len(prev_states):
                    next_states.append(uuv.get_state())
                    rewards.append(uuv.current_reward)
                    dones.append(False)  # Episode ends by max_ticks

                    if uuv.current_action is not None:
                        prev_actions.append(uuv.current_action)

            # Add to replay buffer
            if len(prev_states) == len(next_states) == len(prev_actions):
                for i in range(len(prev_states)):
                    replay_buffer.add(prev_states[i], prev_actions[i],
                                    rewards[i], next_states[i], dones[i])
                    episode_reward += rewards[i]

            # Training step
            if len(replay_buffer) > update_after:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                states_t = torch.FloatTensor(states)
                actions_t = torch.FloatTensor(actions)
                rewards_t = torch.FloatTensor(rewards).unsqueeze(1)
                next_states_t = torch.FloatTensor(next_states)
                dones_t = torch.FloatTensor(dones).unsqueeze(1)

                with torch.no_grad():
                    next_actions = maddpg_agent.actor_target(next_states_t)
                    target_q = maddpg_agent.critic_target(next_states_t, next_actions)
                    target_q = rewards_t + (1 - dones_t) * maddpg_agent.gamma * target_q

                current_q = maddpg_agent.critic(states_t, actions_t)
                critic_loss = F.mse_loss(current_q, target_q)

                maddpg_agent.critic_optimizer.zero_grad()
                critic_loss.backward()
                maddpg_agent.critic_optimizer.step()
                episode_critic_loss.append(critic_loss.item())


                # Update Actor
                actor_actions = maddpg_agent.actor(states_t)
                actor_loss = -maddpg_agent.critic(states_t, actor_actions).mean()

                maddpg_agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                maddpg_agent.actor_optimizer.step()
                episode_actor_loss.append(actor_loss.item())

                # Soft update target networks
                maddpg_agent.soft_update(maddpg_agent.actor_target, maddpg_agent.actor)
                maddpg_agent.soft_update(maddpg_agent.critic_target, maddpg_agent.critic)

        if episode == episodes:
            print("Almost over - demoing now")

        if show_sim or episode == episodes:
            pygame.display.flip()
            clock.tick(50)

    if exit_flag:
        break

    noise = max(noise_min, noise * noise_decay)

    episode_rewards.append(episode_reward)
    if episode_actor_loss:
        actor_losses.append(np.mean(episode_actor_loss))
    if episode_critic_loss:
        critic_losses.append(np.mean(episode_critic_loss))

    if episode % 10 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode:04d} | Avg Reward: {avg_reward:.2f} | Noise: {noise:.3f} | Buffer: {len(replay_buffer)}")
        if actor_losses:
            print(f"  Actor Loss: {actor_losses[-1]:.4f} | Critic Loss: {critic_losses[-1]:.4f}")
    #####
    #####
    #####

pygame.quit()

# Saving Weights
print("Training Done")
save_path = "maddpg_weights.pth"
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
        'action_dim': action_dim,
        'max_action': max_action,
        'gamma': maddpg_agent.gamma,
        'tau': maddpg_agent.tau,
        'lr_actor': lr,
        'lr_critic': lr
    }
}, save_path)
print(f"\nModel weights saved to {save_path}")

# Also save training metrics separately
import json
metrics_path = "training_metrics.json"
with open(metrics_path, 'w') as f:
    json.dump({
        'episode_rewards': episode_rewards,
        'actor_losses': actor_losses,
        'critic_losses': critic_losses
    }, f, indent=2)
print(f"Training metrics saved to {metrics_path}")
