import numpy as np
import pickle
from collections import defaultdict

class QLearningAgent:
    """
    Q-Learning agent with epsilon-greedy exploration.
    Q-table: dict mapping (state, action) → float
    """
    def __init__(self, alpha=0.3, gamma=0.9, epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.9995):
        self.alpha   = alpha          # learning rate
        self.gamma   = gamma          # discount factor
        self.epsilon = epsilon        # exploration rate
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.q_table = defaultdict(float)

    def q(self, state, action):
        return self.q_table[(state, action)]

    def best_action(self, state, actions):
        return max(actions, key=lambda a: self.q(state, a))

    def select_action(self, state, actions):
        if np.random.rand() < self.epsilon:
            return np.random.choice(actions)       # explore
        return self.best_action(state, actions)    # exploit

    def update(self, state, action, reward, next_state, next_actions, done):
        future = 0.0 if (done or not next_actions) else max(self.q(next_state, a) for a in next_actions)
        td_error = reward + self.gamma * future - self.q(state, action)
        self.q_table[(state, action)] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(dict(self.q_table), f)
        print(f"Agent saved to {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            self.q_table = defaultdict(float, pickle.load(f))
        self.epsilon = self.epsilon_min  # inference mode
        print(f"Agent loaded from {path}")
