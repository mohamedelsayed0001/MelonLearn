import numpy as np


class QLearningAgent:
    """
        Q-Learning (Tabular) — from scratch using NumPy.

        This implementation uses:
        - A discrete state and action space
        - A Q-table to store action-value estimates Q(s, a)
        - An epsilon-greedy policy for exploration
        - The Bellman equation for Q-value updates

        Limitations:
        - Only works for discrete state/action spaces
        - Does not use function approximation (no neural networks)
        - No replay buffer or batch updates
    """
    def __init__(self, state_size, action_size, lr=0.1, gamma=0.99, epsilon=0.1):
        self.state_size = state_size
        self.action_size = action_size
        self.lr = lr              # Learning rate
        self.gamma = gamma        # Discount factor
        self.epsilon = epsilon    # Exploration rate
        self.q_table = np.zeros((state_size, action_size))

    def select_action(self, state):
        """
        Choose action using epsilon-greedy policy.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        return np.argmax(self.q_table[state])

    def update(self, state, action, reward, next_state, done):
        """
        Update Q-table using the Q-learning formula.
        """
        max_q_next = np.max(self.q_table[next_state]) if not done else 0.0
        td_target = reward + self.gamma * max_q_next
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.lr * td_error

    def get_policy(self):
        """
        Return the greedy policy derived from the Q-table.
        """
        return np.argmax(self.q_table, axis=1)

    def decay_epsilon(self, decay_rate=0.99, min_epsilon=0.01):
        """
        Decay exploration rate after each episode.
        """
        self.epsilon = max(min_epsilon, self.epsilon * decay_rate)
