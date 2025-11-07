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

        # actor networks (homogeneous)
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr_actor)

        # critic networks (centralized)
        self.critic = None
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
        checkpoint = torch.load("models/maddpg_weights.pt", weights_only=False)
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
yellow = (100, 100, 10)
color_var = white
clock = pygame.time.Clock()
pygame.display.set_caption("Particle Sim")

# sim / env
my_world = World(screen)
default_num_agents = 5
default_num_enemies = 7

# hyperparameters
state_dim = 12
action_dim = 2
max_action = 20
batch_size = 200
noise = 0.5
noise_decay = 0.995
noise_min = 0.001
gamma = 0.95
lr = 1e-3

# time stuff
update_after = 1000
max_ticks = 400
my_world.use_policy_after = 10  # policy & training is used this many ticks (right now 5x per second)
episodes = 1000

# outside stuff
load_weights_ans = 0
show_sim_ans = 0
save_weights_ans = 0
num_agents_ans = input("How Many Agents? (int): ")
num_agents_ans = input("How Many Enemies? (int): ")

while load_weights_ans != "y" and load_weights_ans != "n":
    load_weights_ans = input("Load Weights? (y/n): ")
while show_sim_ans != "y" and load_weights_ans != "n":
    show_sim_ans = input("Show Sim? (y/n): ")

if load_weights_ans == "y":
    maddpg_agent = load_weights(True)
else:
    maddpg_agent = load_weights(False)
if show_sim_ans == "y":
    show_sim = True
else:
    show_sim = False
if num_agents_ans == "":
    num_agents = default_num_agents
if num_agents_ans == "":
    num_enemies = default_num_enemies

# replay buffer
replay_buffer = ReplayBuffer(max_size=100000)

# training loop
episode_rewards = []
actor_losses = []
critic_losses = []

exit_flag = False
for episode in range(episodes):
    my_world.reset()

    for i in range(num_agents):
        x = random.randint(50, 450)
        y = random.randint(50, 450)
        dir = np.array([random.uniform(-1,1), random.uniform(-1,1)])
        unit_vec_dir = dir / np.linalg.norm(dir)
        my_world.add_swarm_uuv(x, y, unit_vec_dir, 5, blue, maddpg_agent)

    for i in range(num_enemies):
        enemy_x = random.randint(100, 400)
        enemy_y = random.randint(100, 400)
        my_world.add_controllable_uuv(enemy_x, enemy_y, np.array([1,0]), 5, yellow, 0)

    episode_reward = 0
    episode_actor_loss = []
    episode_critic_loss = []

    # game loop
    for tick in range(max_ticks):

        # pygame / user interaction stuff
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_x, mouse_y = pygame.mouse.get_pos()
                color = blue
                dir = np.array([random.uniform(-1,1),random.uniform(-1,1)])
                unit_vec_dir = dir / np.linalg.norm(dir)
                my_world.add_swarm_uuv(mouse_x, mouse_y, unit_vec_dir, 5, color, maddpg_agent)
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
            while save_weights_ans != "y" and save_weights_ans != "n":
                save_weights_ans = input("Save Weights? (y/n): ")
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

        # set previous states
        if tick % my_world.use_policy_after == 0:
            prev_states = []
            prev_actions = []
            prev_agent_ids = []
            for uuv in my_world.uuvs:
                if isinstance(uuv, SwarmUUV):
                    prev_states.append(uuv.get_state())
                    prev_actions.append(uuv.current_action)
                    prev_agent_ids.append(uuv.id)

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
                        replay_buffer.add(prev_states[i], prev_actions[i],
                                        reward, next_state, False)
                        episode_reward += reward
                else:
                    reward = my_world.get_and_clear_reward(agent_id)

                    if prev_actions[i] is not None:
                        # use prev_state as next_state for terminal, agent is done
                        replay_buffer.add(prev_states[i], prev_actions[i],
                                        reward, prev_states[i], True)
                        episode_reward += reward

            # training step
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


                # update actor
                actor_actions = maddpg_agent.actor(states_t)
                actor_loss = -maddpg_agent.critic(states_t, actor_actions).mean()

                maddpg_agent.actor_optimizer.zero_grad()
                actor_loss.backward()
                maddpg_agent.actor_optimizer.step()
                episode_actor_loss.append(actor_loss.item())

                # soft update target networks
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

    if episode % 5 == 0:
        avg_reward = np.mean(episode_rewards[-10:])
        print(f"Episode {episode:04d} | Avg Reward: {avg_reward:.2f} | Noise: {noise:.3f} | Buffer: {len(replay_buffer)}")
        if actor_losses:
            print(f"  Actor Loss: {actor_losses[-1]:.4f} | Critic Loss: {critic_losses[-1]:.4f}")

pygame.quit()

def save_weights():
    print("Training Done")
    save_path = "models/maddpg_weights.pt"
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

    import json
    metrics_path = "models/training_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump({
            'episode_rewards': episode_rewards,
            'actor_losses': actor_losses,
            'critic_losses': critic_losses
        }, f, indent=2)
    print(f"Training metrics saved to {metrics_path}")

if save_weights_ans == "y":
    save_weights()
else:
    print("Did not save weights.")
