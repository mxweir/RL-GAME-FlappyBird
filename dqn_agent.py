# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import namedtuple
import math

# Hyperparameter
LR = 1e-5  # Weiter gesenkte Lernrate
GAMMA = 0.99
BATCH_SIZE = 32  # Kleinere Batch-Größe
MEMORY_SIZE = 100000
TARGET_UPDATE = 1000
EPS_START = 1.0
EPS_END = 0.01
EPS_DECAY = 1000
CLIP_GRADIENT = 0.5  # Kleinere Clip-Werte zur Stabilisierung
ALPHA = 0.6  # Prioritätseinstellung für PER
BETA_START = 0.4
BETA_FRAMES = 100000  # Anzahl der Frames, über die Beta annealed wird

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'reward', 'next_state', 'done'))

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=ALPHA):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.pos = 0
        self.priorities = np.zeros((capacity,), dtype=np.float32)
    
    def push(self, *args):
        max_priority = self.priorities.max() if self.buffer else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append(Transition(*args))
        else:
            self.buffer[self.pos] = Transition(*args)
        self.priorities[self.pos] = max_priority
        self.pos = (self.pos + 1) % self.capacity
    
    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity:
            priorities = self.priorities
        else:
            priorities = self.priorities[:self.pos]
        
        probs = priorities ** self.alpha
        probs /= probs.sum()
        
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[idx] for idx in indices]
        
        total = len(self.buffer)
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        weights = np.array(weights, dtype=np.float32)
        
        batch = Transition(*zip(*samples))
        return batch, indices, torch.FloatTensor(weights).unsqueeze(1).to(device)
    
    def update_priorities(self, batch_indices, batch_priorities):
        for idx, priority in zip(batch_indices, batch_priorities):
            self.priorities[idx] = priority

    def __len__(self):
        return len(self.buffer)

class DQNNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)  # Erhöhte Neuronenanzahl
        self.fc2 = nn.Linear(256, 256)        # Erhöhte Neuronenanzahl
        self.fc3 = nn.Linear(256, 256)        # Weitere Schicht
        self.fc_value = nn.Linear(256, 1)
        self.fc_advantage = nn.Linear(256, output_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        
        value = self.fc_value(x)
        advantage = self.fc_advantage(x)
        
        q_vals = value + advantage - advantage.mean()
        return q_vals

class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        self.policy_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net = DQNNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10000, gamma=0.5)
        self.memory = PrioritizedReplayBuffer(MEMORY_SIZE, alpha=ALPHA)
        self.steps_done = 0
        
        self.epsilon = EPS_START
        self.epsilon_decay = EPS_DECAY
        self.epsilon_end = EPS_END
        
        self.target_update_steps = TARGET_UPDATE
        self.loss_fn = nn.MSELoss()
        
        self.beta_start = BETA_START
        self.beta_frames = BETA_FRAMES
    
    def select_action(self, state):
        self.steps_done += 1
        self.epsilon = self.epsilon_end + (EPS_START - self.epsilon_end) * \
            math.exp(-1. * self.steps_done / self.epsilon_decay)
        
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(device)
                q_values = self.policy_net(state)
                return q_values.argmax().item()
    
    def store_transition(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)
    
    def update(self):
        if len(self.memory) < BATCH_SIZE:
            return None
        
        beta = min(1.0, self.beta_start + self.steps_done * (1.0 - self.beta_start) / self.beta_frames)
        batch, indices, weights = self.memory.sample(BATCH_SIZE, beta=beta)
        
        state_batch = torch.FloatTensor(np.array(batch.state)).to(device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(device)
        reward_batch = torch.FloatTensor(batch.reward).unsqueeze(1).to(device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(device)
        done_batch = torch.FloatTensor(batch.done).unsqueeze(1).to(device)
        
        # Aktuelle Q-Werte
        current_q = self.policy_net(state_batch).gather(1, action_batch)
        
        # Double DQN: Wählt Aktion basierend auf policy_net, Q-Werte basierend auf target_net
        next_actions = self.policy_net(next_state_batch).argmax(1, keepdim=True)
        next_q = self.target_net(next_state_batch).gather(1, next_actions).detach()
        
        # Ziel Q-Werte
        target_q = reward_batch + (1 - done_batch) * GAMMA * next_q
        
        # Verlust berechnen
        loss = (self.loss_fn(current_q, target_q) * weights).mean()
        
        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), CLIP_GRADIENT)
        self.optimizer.step()
        self.scheduler.step()  # Aktualisiere den Scheduler
        
        # Update Prioritäten
        priorities = (current_q - target_q).abs().detach().cpu().numpy() + 1e-6
        self.memory.update_priorities(indices, priorities.flatten())
        
        return loss.item()
    
    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_model(self, path):
        torch.save(self.policy_net.state_dict(), path)
    
    def load_model(self, path):
        self.policy_net.load_state_dict(torch.load(path, map_location=device))
        self.target_net.load_state_dict(self.policy_net.state_dict())
