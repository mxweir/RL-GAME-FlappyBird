# dqn_agent.py
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

class DuelingDQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DuelingDQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        
        # Wert-Stream
        self.value_fc = nn.Linear(256, 128)
        self.value = nn.Linear(128, 1)
        
        # Vorteil-Stream
        self.advantage_fc = nn.Linear(256, 128)
        self.advantage = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        
        value = torch.relu(self.value_fc(x))
        value = self.value(value)
        
        advantage = torch.relu(self.advantage_fc(x))
        advantage = self.advantage(advantage)
        
        # Kombiniere Wert und Vorteil
        q_vals = value + advantage - advantage.mean(dim=1, keepdim=True)
        return q_vals

class DQNAgent:
    def __init__(self, state_dim, action_dim, lr=1e-4, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=2000):
        self.state_dim = state_dim
        self.action_dim = action_dim

        self.gamma = gamma

        self.epsilon = epsilon_start
        self.epsilon_initial = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.policy_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net = DuelingDQN(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.criteria = nn.SmoothL1Loss()  # Huber Loss

        self.memory = deque(maxlen=10000)
        self.batch_size = 64
        self.steps_done = 0
        self.update_target_steps = 1000

        self.use_double_dqn = True

    def select_action(self, state):
        self.steps_done += 1
        epsilon = self.epsilon_end + (self.epsilon_initial - self.epsilon_end) * \
                  np.exp(-1. * self.steps_done / self.epsilon_decay)
        self.epsilon = epsilon

        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                q_values = self.policy_net(state)
                return q_values.max(1)[1].item()

    def store_transition(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def sample_memory(self):
        batch = random.sample(self.memory, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return np.array(states), actions, rewards, np.array(next_states), dones

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.sample_memory()

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Aktuelle Q-Werte
        q_values = self.policy_net(states).gather(1, actions)

        # Ziel Q-Werte
        with torch.no_grad():
            if self.use_double_dqn:
                # Double DQN: Verwende das Policy-Netzwerk, um die besten Aktionen zu finden
                next_actions = self.policy_net(next_states).max(1)[1].unsqueeze(1)
                # Verwende das Ziel-Netzwerk, um die Q-Werte dieser Aktionen zu berechnen
                next_q_values = self.target_net(next_states).gather(1, next_actions)
            else:
                # Standard DQN
                next_q_values = self.target_net(next_states).max(1)[0].unsqueeze(1)

            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        # Verlust berechnen
        loss = self.criteria(q_values, target_q_values)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        # Zielnetzwerk aktualisieren
        if self.steps_done % self.update_target_steps == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

        return loss.item()  # Rückgabe des Verlusts für Logging
