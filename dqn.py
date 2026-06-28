"""
Session 4: Deep Q-Network (DQN) for CartPole-v1
Components:
  - Neural network to approximate Q(s,a)
  - Replay buffer to break correlation between samples
  - Target network for stable training
"""
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque


# ─── Device ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ─── Q-Network ─────────────────────────────────────────────────────────────────
class QNetwork(nn.Module):
    """
    3-layer MLP: obs(4) → 128 → 128 → actions(2)
    Takes raw continuous state, outputs Q-value per action.
    """
    def __init__(self, state_dim=4, action_dim=2, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )

    def forward(self, x):
        return self.net(x)


# ─── Replay Buffer ─────────────────────────────────────────────────────────────
class ReplayBuffer:
    """
    Stores (s, a, r, s', done) transitions.
    Randomly sampling breaks the temporal correlation that destabilizes training.
    """
    def __init__(self, capacity=10_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(device),
            torch.LongTensor(a).to(device),
            torch.FloatTensor(r).to(device),
            torch.FloatTensor(np.array(ns)).to(device),
            torch.FloatTensor(d).to(device),
        )

    def __len__(self):
        return len(self.buffer)


# ─── DQN Agent ─────────────────────────────────────────────────────────────────
class DQNAgent:
    def __init__(self,
                 state_dim=4, action_dim=2,
                 lr=1e-3, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=64, target_update_freq=10):

        self.action_dim       = action_dim
        self.gamma            = gamma
        self.epsilon          = epsilon
        self.epsilon_min      = epsilon_min
        self.epsilon_decay    = epsilon_decay
        self.batch_size       = batch_size
        self.target_update_freq = target_update_freq
        self.steps            = 0

        # Online network  — trained every step
        self.online_net = QNetwork(state_dim, action_dim).to(device)
        # Target network  — copied from online every N episodes, provides stable targets
        self.target_net = QNetwork(state_dim, action_dim).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.online_net.parameters(), lr=lr)
        self.loss_fn   = nn.MSELoss()
        self.buffer    = ReplayBuffer()

    def select_action(self, state):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.action_dim)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(device)
            return int(self.online_net(s).argmax().item())

    def learn(self):
        if len(self.buffer) < self.batch_size:
            return None

        s, a, r, ns, done = self.buffer.sample(self.batch_size)

        # Current Q-values from online net
        q_values = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Target Q-values from target net (no gradient)
        with torch.no_grad():
            max_next_q = self.target_net(ns).max(1)[0]
            targets    = r + self.gamma * max_next_q * (1 - done)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        # Gradient clipping — prevents exploding gradients
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({'online': self.online_net.state_dict(),
                    'target': self.target_net.state_dict()}, path)
        print(f"DQN saved → {path}")

    def load(self, path):
        d = torch.load(path, map_location=device)
        self.online_net.load_state_dict(d['online'])
        self.target_net.load_state_dict(d['target'])
        self.epsilon = self.epsilon_min
        print(f"DQN loaded ← {path}")


# ─── Training ──────────────────────────────────────────────────────────────────
def train(episodes=600, log_every=50, solve_threshold=475):
    env   = gym.make('CartPole-v1')
    agent = DQNAgent()

    scores, losses, epsilons = [], [], []
    recent = deque(maxlen=100)
    solved_at = None

    for ep in range(1, episodes + 1):
        obs, _  = env.reset()
        state   = np.array(obs, dtype=np.float32)
        total   = 0
        ep_loss = []

        for _ in range(500):
            action                          = agent.select_action(state)
            next_obs, reward, term, trunc, _ = env.step(action)
            done                            = term or trunc
            next_state                      = np.array(next_obs, dtype=np.float32)

            agent.buffer.push(state, action, reward, next_state, float(done))
            loss = agent.learn()
            if loss: ep_loss.append(loss)

            state  = next_state
            total += 1
            if done: break

        agent.decay_epsilon()
        if ep % agent.target_update_freq == 0:
            agent.update_target()

        scores.append(total)
        recent.append(total)
        losses.append(np.mean(ep_loss) if ep_loss else 0)
        epsilons.append(agent.epsilon)

        if ep % log_every == 0:
            avg = np.mean(recent)
            print(f"Ep {ep:>4} | score: {total:>4} | avg100: {avg:>6.1f} "
                  f"| ε={agent.epsilon:.3f} | loss={losses[-1]:.4f}"
                  f"  {'★ SOLVED' if avg >= solve_threshold else ''}")

            if avg >= solve_threshold and solved_at is None:
                solved_at = ep
                print(f"\n  ★ CartPole SOLVED at episode {ep}! avg={avg:.1f}\n")

    env.close()
    agent.save("dqn_cartpole.pth")
    _plot(scores, losses, epsilons, solved_at)
    return agent, scores


def _plot(scores, losses, epsilons, solved_at=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.patch.set_facecolor('#0f1117')

    colors = {'score':'#1D9E75','loss':'#E24B4A','eps':'#7F77DD',
              'avg':'#EF9F27','bg':'#1a1d2e','grid':'#2a2d3e','text':'#c8cad4'}

    def style(ax, title):
        ax.set_facecolor(colors['bg'])
        ax.tick_params(colors=colors['text'], labelsize=9)
        ax.set_title(title, color=colors['text'], fontsize=11)
        ax.grid(True, color=colors['grid'], lw=0.5, alpha=0.7)
        for sp in ax.spines.values(): sp.set_edgecolor(colors['grid'])
        ax.xaxis.label.set_color(colors['text'])
        ax.yaxis.label.set_color(colors['text'])

    x = np.arange(1, len(scores)+1)

    # Score
    axes[0].plot(x, scores, color=colors['score'], lw=0.7, alpha=0.4)
    if len(scores) >= 100:
        roll = np.convolve(scores, np.ones(100)/100, mode='valid')
        axes[0].plot(np.arange(100, len(scores)+1), roll, color=colors['avg'], lw=2)
    axes[0].axhline(475, color='white', lw=1, ls='--', alpha=0.5)
    if solved_at:
        axes[0].axvline(solved_at, color='white', lw=1, ls=':', alpha=0.7)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Steps balanced')
    style(axes[0], 'Score per episode')

    # Loss
    axes[1].plot(x, losses, color=colors['loss'], lw=1)
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('MSE Loss')
    style(axes[1], 'Training loss')

    # Epsilon
    axes[2].plot(x, epsilons, color=colors['eps'], lw=2)
    axes[2].fill_between(x, epsilons, alpha=0.15, color=colors['eps'])
    axes[2].set_xlabel('Episode'); axes[2].set_ylabel('ε')
    style(axes[2], 'Epsilon decay')

    plt.suptitle('DQN — CartPole-v1', color=colors['text'], fontsize=13)
    plt.tight_layout()
    plt.savefig('dqn_training.png', dpi=150, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    print("Plot saved → dqn_training.png")


# ─── Evaluate ──────────────────────────────────────────────────────────────────
def evaluate(n_episodes=10):
    env   = gym.make('CartPole-v1')
    agent = DQNAgent()
    agent.load("dqn_cartpole.pth")
    scores = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        state  = np.array(obs, dtype=np.float32)
        total  = 0
        for _ in range(500):
            action = agent.select_action(state)
            obs, _, term, trunc, _ = env.step(action)
            state  = np.array(obs, dtype=np.float32)
            total += 1
            if term or trunc: break
        scores.append(total)
        print(f"  Episode {ep+1}: {total} steps")

    env.close()
    print(f"\nAverage: {np.mean(scores):.1f} / 500")


if __name__ == "__main__":
    print("=" * 50)
    print("SESSION 4: DQN — CartPole-v1")
    print("=" * 50)
    print("\n[ TRAINING ]")
    train(episodes=1500)
    print("\n[ EVALUATION ]")
    evaluate()
