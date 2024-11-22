# agent.py
import numpy as np
import random

class QLearningAgent:
    def __init__(self, state_space_size, action_space_size, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.995, exploration_min=0.01):
        self.state_space_size = state_space_size
        self.action_space_size = action_space_size
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.exploration_rate = exploration_rate
        self.exploration_decay = exploration_decay
        self.exploration_min = exploration_min
        self.q_table = np.zeros((state_space_size, action_space_size))

    def choose_action(self, state):
        if np.random.rand() < self.exploration_rate:
            return random.randrange(self.action_space_size)
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        old_value = self.q_table[state, action]
        future_estimate = 0 if done else np.max(self.q_table[next_state])
        new_value = (1 - self.learning_rate) * old_value + self.learning_rate * (reward + self.discount_factor * future_estimate)
        self.q_table[state, action] = new_value

        # Reduce exploration rate
        if self.exploration_rate > self.exploration_min:
            self.exploration_rate *= self.exploration_decay

    def update_reward(self, state, action, bird_y, pipes, game_over):
        reward = 0
        pipe_x = pipes[0].x
        pipe_gap_y_start = pipes[0].height
        pipe_gap_y_end = pipes[0].height + pipes[0].gap_size
        gap_center_y = (pipe_gap_y_start + pipe_gap_y_end) / 2

        # Reward for staying alive
        reward += 1

        # Reward for moving towards the gap and staying within it
        distance_to_gap_center = abs(gap_center_y - bird_y)
        if pipe_x > 60:
            reward += max(0, 30 - distance_to_gap_center)  # Increased reward for staying near the center of the gap

        # Additional positive reward for successfully passing a pipe
        if pipe_x < bird_y and not game_over:
            reward += 100  # Increased reward for successfully passing a pipe

        # Significant negative reward for colliding with a pipe or going out of bounds
        if game_over:
            reward -= 200  # Increased penalty to strongly discourage collisions

        # Reward based on relative position to gap center
        if bird_y < gap_center_y:
            reward += 10  # Reward for staying above the center of the gap
        else:
            reward += 5  # Smaller reward for staying below the center of the gap

        return reward

    def save(self, filename):
        np.save(filename, self.q_table)

    def load(self, filename):
        self.q_table = np.load(filename)
