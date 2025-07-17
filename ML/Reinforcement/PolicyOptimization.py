import numpy as np

def _softmax(x):
    x = x - np.max(x)
    exp = np.exp(x)
    return exp / np.sum(exp)


"""
    Proximal Policy Optimization (PPO).

    This implementation uses:
    - A simple 2-layer policy network (actor) with softmax output
    - A 2-layer value network (critic) for state value estimation
    - GAE (Generalized Advantage Estimation) to compute advantages
    - PPO’s clipped surrogate objective for stable policy updates
    - Manual gradient updates using SGD (no deep learning libraries)

    Limitations:
    - Policy gradient is approximated simply (not exact log-prob gradient)
    - Only updates final layer of networks (no full backprop)
    - No entropy regularization or batch training
"""

class _policyNetwork:
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        self.w1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, action_dim) * 0.01
        self.b2 = np.zeros(action_dim)
    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        return _softmax(z2), a1  # probabilities and hidden

    def get_action(self, x):
        probs, _ = self.forward(x)
        return np.random.choice(len(probs), p=probs), probs



class _ValueNetwork:
    def __init__(self, state_dim, hidden_dim=32):
        self.w1 = np.random.randn(state_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)
        self.w2 = np.random.randn(hidden_dim, 1) * 0.01
        self.b2 = np.zeros(1)

    def forward(self, x):
        z1 = np.dot(x, self.w1) + self.b1
        a1 = np.tanh(z1)
        z2 = np.dot(a1, self.w2) + self.b2
        return z2.squeeze(), a1



class PPOAgent:
    def __init__(self, state_dim, action_dim):
        self.policy = _policyNetwork(state_dim, action_dim)
        self.value = _ValueNetwork(state_dim)
        self.lr = 1e-3
        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2

    def compute_gae(self, rewards, values, dones):
        advantages = []
        gae = 0
        values = values + [0]
        for t in reversed(range(len(rewards))):
            delta = rewards[t] + self.gamma * values[t+1] * (1 - dones[t]) - values[t]
            gae = delta + self.gamma * self.lam * (1 - dones[t]) * gae
            advantages.insert(0, gae)
        return advantages

    def update(self, states, actions, old_probs, advantages, returns):
        for _ in range(4):  # small number of epochs
            for i in range(len(states)):
                s = states[i]
                a = actions[i]
                adv = advantages[i]
                ret = returns[i]
                old_p = old_probs[i][a]

                probs, h1 = self.policy.forward(s)
                ratio = probs[a] / (old_p + 1e-8)
                clipped_ratio = np.clip(ratio, 1 - self.eps_clip, 1 + self.eps_clip)
                actor_loss = -min(ratio * adv, clipped_ratio * adv)

                # Update policy weights (simple SGD)
                grad = actor_loss * probs
                self.policy.w2 -= self.lr * np.outer(h1, grad)
                self.policy.b2 -= self.lr * grad

                # Critic loss
                v, h1v = self.value.forward(s)
                critic_loss = (ret - v) ** 2

                grad_v = -2 * (ret - v)
                self.value.w2 -= self.lr * grad_v * h1v[:, None]
                self.value.b2 -= self.lr * grad_v