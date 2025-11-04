import numpy as np
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# --- Simple Grid Environment ---
class GridEnv:
    def __init__(self, size=5):
        self.size = size
        self.goal = (size-1, size-1)
        self.reset()

    def reset(self):
        self.agent = (random.randint(0, self.size-1), random.randint(0, self.size-1))
        return self._get_state()

    def _get_state(self):
        # Flattened one-hot encoding of agent position
        state = np.zeros((self.size, self.size), dtype=np.float32)
        state[self.agent] = 1.0
        return state.flatten()

    def step(self, action):
        x, y = self.agent
        if action == 0 and x > 0: x -= 1          # up
        elif action == 1 and x < self.size-1: x += 1  # down
        elif action == 2 and y > 0: y -= 1          # left
        elif action == 3 and y < self.size-1: y += 1  # right

        self.agent = (x, y)
        done = (self.agent == self.goal)
        reward = 1.0 if done else -0.1
        return self._get_state(), reward, done

# --- DQN Network ---
class DQN(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.out = nn.Linear(64, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

# --- Hyperparameters ---
env = GridEnv(size=5)
state_dim = env.size * env.size
action_dim = 4
gamma = 0.95
epsilon = 1.0
epsilon_min = 0.05
epsilon_decay = 0.995
lr = 1e-3

# --- Init ---
model = DQN(state_dim, action_dim)
optimizer = optim.Adam(model.parameters(), lr=lr)
replay = []

def select_action(state):
    if random.random() < epsilon:
        return random.randrange(action_dim)
    with torch.no_grad():
        q_values = model(torch.tensor(state).unsqueeze(0))
        return q_values.argmax().item()

def replay_train(batch_size=32):
    if len(replay) < batch_size:
        return
    batch = random.sample(replay, batch_size)
    states, actions, rewards, next_states, dones = zip(*batch)

    states = torch.tensor(states)
    actions = torch.tensor(actions)
    rewards = torch.tensor(rewards)
    next_states = torch.tensor(next_states)
    dones = torch.tensor(dones, dtype=torch.float32)

    q_values = model(states).gather(1, actions.unsqueeze(1)).squeeze()
    next_q_values = model(next_states).max(1)[0]
    target = rewards + gamma * next_q_values * (1 - dones)
    loss = F.mse_loss(q_values, target.detach())

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

# --- Training Loop ---
episodes = 500
for ep in range(episodes):
    state = env.reset()
    total_reward = 0
    for t in range(50):
        action = select_action(state)
        next_state, reward, done = env.step(action)
        replay.append((state, action, reward, next_state, done))
        if len(replay) > 10000:
            replay.pop(0)
        replay_train()
        state = next_state
        total_reward += reward
        if done:
            break
    epsilon = max(epsilon * epsilon_decay, epsilon_min)
    print(f"Episode {ep:03d} | Reward: {total_reward:.2f}")

print("Training complete.")
