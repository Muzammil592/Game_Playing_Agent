"""
Session 3: CartPole-v1 solved with discretized Q-Learning
Discretizes the continuous 4D state space into bins so we can use a Q-table.
"""
from pathlib import Path
import numpy as np
import gymnasium as gym
import pickle
import matplotlib.pyplot as plt
from collections import deque


BASE_DIR = Path(__file__).resolve().parent


# ─── Discretization ────────────────────────────────────────────────────────────
# CartPole obs bounds: [pos, vel, angle, angular_vel]
# We clip to meaningful ranges and divide into N bins per dimension
N_BINS    = [6, 6, 12, 12]   # more bins on angle — it matters most
OBS_LOW   = [-2.4, -3.0, -0.25, -3.5]
OBS_HIGH  = [ 2.4,  3.0,  0.25,  3.5]

def discretize(obs):
    """Convert continuous obs array → tuple of bin indices."""
    bins = []
    for i, val in enumerate(obs):
        val   = np.clip(val, OBS_LOW[i], OBS_HIGH[i])
        ratio = (val - OBS_LOW[i]) / (OBS_HIGH[i] - OBS_LOW[i])
        b     = int(ratio * N_BINS[i])
        b     = min(b, N_BINS[i] - 1)  # clamp top edge
        bins.append(b)
    return tuple(bins)


# ─── Agent ─────────────────────────────────────────────────────────────────────
class CartPoleQAgent:
    def __init__(self, n_actions=2,
                 alpha=0.1, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995):
        self.n_actions     = n_actions
        self.alpha         = alpha
        self.gamma         = gamma
        self.epsilon       = epsilon
        self.epsilon_min   = epsilon_min
        self.epsilon_decay = epsilon_decay

        # Q-table: shape = N_BINS + [n_actions]
        self.q_table = np.zeros(N_BINS + [n_actions])

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_actions)
        return int(np.argmax(self.q_table[state]))

    def update(self, state, action, reward, next_state, done):
        future = 0.0 if done else np.max(self.q_table[next_state])
        td_error = reward + self.gamma * future - self.q_table[state][action]
        self.q_table[state][action] += self.alpha * td_error
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump({'q_table': self.q_table, 'epsilon': self.epsilon}, f)
        print(f"Agent saved → {path}")

    def load(self, path):
        with open(path, 'rb') as f:
            d = pickle.load(f)
        self.q_table = d['q_table']
        self.epsilon = self.epsilon_min
        print(f"Agent loaded ← {path}")


# ─── Training ──────────────────────────────────────────────────────────────────
def train(episodes=5_000, log_every=200):
    env    = gym.make('CartPole-v1')
    agent  = CartPoleQAgent()
    scores = []
    recent = deque(maxlen=log_every)

    for ep in range(1, episodes + 1):
        obs, _ = env.reset()
        state  = discretize(obs)
        total  = 0

        for step in range(500):          # CartPole max = 500
            action              = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done                = terminated or truncated
            next_state          = discretize(next_obs)

            # Penalize falling — don't reward the last step
            r = reward if not terminated else -10.0

            agent.update(state, action, r, next_state, done)
            state  = next_state
            total += 1
            if done:
                break

        scores.append(total)
        recent.append(total)

        if ep % log_every == 0:
            avg = np.mean(recent)
            print(f"Ep {ep:>5} | avg score: {avg:>6.1f} | ε={agent.epsilon:.3f}"
                  f"  {'✓ SOLVED' if avg >= 475 else ''}")
            if avg >= 475:
                print(f"\nSolved at episode {ep}!")
                break

    env.close()
    agent.save(BASE_DIR / "cartpole_agent.pkl")

    # Plot training curve
    fig, ax = plt.subplots(figsize=(10, 4))
    fig.patch.set_facecolor('#0f1117')
    ax.set_facecolor('#1a1d2e')
    ax.plot(scores, color='#7F77DD', lw=0.8, alpha=0.5, label='score per episode')
    # Rolling average
    roll = np.convolve(scores, np.ones(100)/100, mode='valid')
    ax.plot(range(99, len(scores)), roll, color='#1D9E75', lw=2, label='100-ep average')
    ax.axhline(475, color='#EF9F27', lw=1.5, linestyle='--', label='solved threshold (475)')
    ax.set_xlabel('Episode', color='#c8cad4')
    ax.set_ylabel('Score (steps balanced)', color='#c8cad4')
    ax.set_title('CartPole-v1 — Q-Learning Training Curve', color='#c8cad4')
    ax.tick_params(colors='#c8cad4')
    ax.legend(facecolor='#1a1d2e', labelcolor='#c8cad4', edgecolor='#2a2d3e')
    for sp in ax.spines.values(): sp.set_edgecolor('#2a2d3e')
    plt.tight_layout()
    plt.savefig(BASE_DIR / 'cartpole_training.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("Plot saved → cartpole_training.png")
    return agent, scores


# ─── Evaluate (no rendering — works on all systems) ────────────────────────────
def evaluate(n_episodes=10):
    env   = gym.make('CartPole-v1')
    agent = CartPoleQAgent()
    agent.load(BASE_DIR / "cartpole_agent.pkl")
    scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state  = discretize(obs)
        total  = 0
        for _ in range(500):
            action = agent.select_action(state)
            obs, _, terminated, truncated, _ = env.step(action)
            state  = discretize(obs)
            total += 1
            if terminated or truncated:
                break
        scores.append(total)
        print(f"  Episode {ep+1}: {total} steps")

    env.close()
    print(f"\nAverage: {np.mean(scores):.1f} / 500")


if __name__ == "__main__":
    print("=" * 50)
    print("SESSION 3: CartPole Q-Learning")
    print("=" * 50)
    print("\n[ TRAINING ]")
    train()
    print("\n[ EVALUATION ]")
    evaluate()
