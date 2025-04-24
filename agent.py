import random
import pickle
from collections import defaultdict

import numpy as np

from config import ALPHA, GAMMA, EPSILON


class QAgent:
    """
    Q-learning agent with an epsilon-greedy policy.
    Uses a dict-based Q-table mapping state-tuples → action-value arrays.
    """

    def __init__(self, actions, alpha=ALPHA, gamma=GAMMA, epsilon=EPSILON):
        """
        :param actions: list of available actions (e.g. [0,1,2] for AWS, GCP, Azure)
        :param alpha:   learning rate
        :param gamma:   discount factor
        :param epsilon: exploration rate (ε-greedy)
        """
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Q-table: state_key (tuple) → np.array of length len(actions)
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)))

    def choose_action(self, state):
        """
        Returns an action index.
        With probability ε: random action (explore)
        Else: best-known action (exploit)
        """
        state_key = tuple(state)
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        return int(np.argmax(self.q_table[state_key]))

    def learn(self, state, action, reward, next_state):
        """
        Update Q-table via the Bellman equation:
        Q(s,a) ← Q(s,a) + α [r + γ max_a' Q(s',a') – Q(s,a)]
        """
        st_key = tuple(state)
        nxt_key = tuple(next_state)

        q_predict = self.q_table[st_key][action]
        q_target = reward + self.gamma * np.max(self.q_table[nxt_key])

        # update
        self.q_table[st_key][action] += self.alpha * (q_target - q_predict)

    def save(self, filepath):
        """Save Q-table to disk."""
        with open(filepath, "wb") as f:
            pickle.dump(dict(self.q_table), f)

    def load(self, filepath):
        """Load Q-table from disk."""
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        self.q_table = defaultdict(lambda: np.zeros(len(self.actions)), data)
