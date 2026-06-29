"""
Session 6: Vanilla DQN vs Double DQN vs Dueling DQN
Same LunarLander-v3 environment, three agents, one comparison plot.
"""
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt
from collections import deque

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# ═══════════════════════════════════════════════════════════════════
# NETWORKS
# ═══════════════════════════════════════════════════════════════════

class VanillaQNetwork(nn.Module):
    """Standard DQN — directly estimates Q(s,a)"""
    def __init__(self, state_dim=8, action_dim=4, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    def forward(self, x):
        return self.net(x)


class DuelingQNetwork(nn.Module):
    """
    Dueling DQN — splits into V(s) and A(s,a) streams.
    Q(s,a) = V(s) + A(s,a) - mean(A(s,a))
    Better at identifying which states are valuable
    regardless of which action is taken.
    """
    def __init__(self, state_dim=8, action_dim=4, hidden=256):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden),   nn.ReLU()
        )
        self.value_stream     = nn.Linear(hidden, 1)
        self.advantage_stream = nn.Linear(hidden, action_dim)

    def forward(self, x):
        shared = self.shared(x)
        V = self.value_stream(shared)               # shape: (batch, 1)
        A = self.advantage_stream(shared)           # shape: (batch, 4)
        return V + A - A.mean(dim=1, keepdim=True)  # Q = V + A - mean(A)


# ═══════════════════════════════════════════════════════════════════
# REPLAY BUFFER (shared by all agents)
# ═══════════════════════════════════════════════════════════════════

class ReplayBuffer:
    def __init__(self, capacity=50_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, s, a, r, ns, done):
        self.buffer.append((s, a, r, ns, done))

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


# ═══════════════════════════════════════════════════════════════════
# AGENT — mode flag controls which variant runs
# ═══════════════════════════════════════════════════════════════════

class Agent:
    """
    mode = 'vanilla'  → standard DQN
    mode = 'double'   → Double DQN (2-line fix for overestimation)
    mode = 'dueling'  → Dueling DQN (new network architecture)
    """
    def __init__(self, mode='vanilla', state_dim=8, action_dim=4,
                 lr=5e-4, gamma=0.99,
                 epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995,
                 batch_size=128, target_update_freq=10):

        self.mode             = mode
        self.action_dim       = action_dim
        self.gamma            = gamma
        self.epsilon          = epsilon
        self.epsilon_min      = epsilon_min
        self.epsilon_decay    = epsilon_decay
        self.batch_size       = batch_size
        self.target_update_freq = target_update_freq

        # Dueling uses different network; vanilla + double use same network
        NetClass = DuelingQNetwork if mode == 'dueling' else VanillaQNetwork

        self.online_net = NetClass(state_dim, action_dim).to(device)
        self.target_net = NetClass(state_dim, action_dim).to(device)
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
        q_values = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            if self.mode == 'double' or self.mode == 'dueling':
                # ── Double DQN fix ──────────────────────────────────────
                # Online net selects the action (WHICH action is best)
                # Target net evaluates its value (HOW GOOD is that action)
                # This prevents the same network from both picking AND scoring
                next_actions = self.online_net(ns).argmax(1, keepdim=True)
                max_next_q   = self.target_net(ns).gather(1, next_actions).squeeze(1)
            else:
                # ── Vanilla DQN ─────────────────────────────────────────
                # Target net both picks and scores — can overestimate
                max_next_q = self.target_net(ns).max(1)[0]

            targets = r + self.gamma * max_next_q * (1 - done)

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_net.parameters(), 1.0)
        self.optimizer.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path):
        torch.save({'online': self.online_net.state_dict(),
                    'target': self.target_net.state_dict(),
                    'mode':   self.mode}, path)
        print(f"  Saved → {path}")

    def load(self, path):
        d = torch.load(path, map_location=device)
        self.online_net.load_state_dict(d['online'])
        self.target_net.load_state_dict(d['target'])
        self.epsilon = self.epsilon_min


# ═══════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════

def train_agent(mode, episodes=800, log_every=50):
    label = {'vanilla': 'Vanilla DQN',
             'double':  'Double DQN',
             'dueling': 'Dueling DQN'}[mode]

    print(f"\n{'='*60}")
    print(f"  Training: {label}")
    print(f"{'='*60}")
    print(f"{'Ep':>6} | {'Score':>8} | {'Avg100':>8} | {'ε':>6} | Status")
    print("-" * 50)

    env    = gym.make('LunarLander-v3')
    agent  = Agent(mode=mode)
    scores = []
    recent = deque(maxlen=100)
    solved_at = None

    for ep in range(1, episodes + 1):
        obs, _  = env.reset()
        state   = np.array(obs, dtype=np.float32)
        total   = 0.0

        for _ in range(1000):
            action                           = agent.select_action(state)
            next_obs, reward, term, trunc, _ = env.step(action)
            done                             = term or trunc
            next_state                       = np.array(next_obs, dtype=np.float32)

            agent.buffer.push(state, action, reward, next_state, float(done))
            agent.learn()
            state  = next_state
            total += reward
            if done: break

        agent.decay_epsilon()
        if ep % agent.target_update_freq == 0:
            agent.update_target()

        scores.append(total)
        recent.append(total)

        if ep % log_every == 0:
            avg = np.mean(recent)
            if avg >= 200 and solved_at is None:
                solved_at = ep
                status = f"★ SOLVED at ep {ep}!"
            elif avg >= 100: status = "landing consistently"
            elif avg >= 0:   status = "learning"
            else:            status = "still crashing"
            print(f"{ep:>6} | {total:>8.1f} | {avg:>8.1f} | "
                  f"{agent.epsilon:>6.3f} | {status}")

    env.close()
    agent.save(f"{mode}_lunarlander.pth")
    print(f"\n  {label} done. Solved at ep: {solved_at or 'not yet'}")
    return scores, solved_at


# ═══════════════════════════════════════════════════════════════════
# COMPARISON PLOT
# ═══════════════════════════════════════════════════════════════════

def plot_comparison(results):
    """
    results = {
        'vanilla': (scores_list, solved_at),
        'double':  (scores_list, solved_at),
        'dueling': (scores_list, solved_at),
    }
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor('#0f1117')

    palette = {
        'vanilla': '#7F77DD',
        'double':  '#1D9E75',
        'dueling': '#EF9F27',
    }
    labels = {
        'vanilla': 'Vanilla DQN',
        'double':  'Double DQN',
        'dueling': 'Dueling DQN',
    }
    c = {'bg':'#1a1d2e','grid':'#2a2d3e','text':'#c8cad4'}

    def style(ax, title):
        ax.set_facecolor(c['bg'])
        ax.tick_params(colors=c['text'], labelsize=9)
        ax.set_title(title, color=c['text'], fontsize=12)
        ax.grid(True, color=c['grid'], lw=0.5, alpha=0.7)
        for sp in ax.spines.values(): sp.set_edgecolor(c['grid'])
        ax.xaxis.label.set_color(c['text'])
        ax.yaxis.label.set_color(c['text'])

    # Left: raw scores (faint) + 100-ep rolling avg (bold)
    for mode, (scores, solved_at) in results.items():
        col = palette[mode]
        x   = np.arange(1, len(scores) + 1)
        axes[0].plot(x, scores, color=col, lw=0.5, alpha=0.2)
        if len(scores) >= 100:
            roll = np.convolve(scores, np.ones(100)/100, mode='valid')
            axes[0].plot(np.arange(100, len(scores)+1), roll,
                         color=col, lw=2.2, label=labels[mode])
        if solved_at:
            axes[0].axvline(solved_at, color=col, lw=1,
                            ls='--', alpha=0.6)

    axes[0].axhline(200, color='white', lw=1, ls=':', alpha=0.5, label='solve threshold')
    axes[0].axhline(0,   color='white', lw=0.5, ls=':', alpha=0.2)
    axes[0].set_xlabel('Episode')
    axes[0].set_ylabel('Score (100-ep avg)')
    axes[0].legend(facecolor=c['bg'], labelcolor=c['text'],
                   edgecolor=c['grid'], fontsize=9)
    style(axes[0], 'Score Comparison — All Three Agents')

    # Right: final avg100 bar chart
    final_avgs = []
    for mode, (scores, _) in results.items():
        avg = np.mean(scores[-100:]) if len(scores) >= 100 else np.mean(scores)
        final_avgs.append((labels[mode], avg, palette[mode]))

    names  = [x[0] for x in final_avgs]
    avgs   = [x[1] for x in final_avgs]
    colors = [x[2] for x in final_avgs]

    bars = axes[1].bar(names, avgs, color=colors, width=0.5, edgecolor='#2a2d3e')
    axes[1].axhline(200, color='white', lw=1, ls='--', alpha=0.5, label='solve (200)')
    axes[1].axhline(0,   color='white', lw=0.5, ls=':', alpha=0.3)

    for bar, avg in zip(bars, avgs):
        axes[1].text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + 3,
                     f'{avg:.1f}', ha='center', va='bottom',
                     color=c['text'], fontsize=10, fontweight='bold')

    axes[1].set_ylabel('Final Avg Score (last 100 ep)')
    axes[1].legend(facecolor=c['bg'], labelcolor=c['text'],
                   edgecolor=c['grid'], fontsize=9)
    style(axes[1], 'Final Performance Comparison')

    plt.suptitle('DQN Variants — LunarLander-v3 Comparison',
                 color=c['text'], fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('session6_comparison.png', dpi=150,
                bbox_inches='tight', facecolor=fig.get_facecolor())
    print("\nComparison plot saved → session6_comparison.png")


# ═══════════════════════════════════════════════════════════════════
# EVALUATE ALL THREE
# ═══════════════════════════════════════════════════════════════════

def evaluate_all():
    print("\n" + "="*60)
    print("  FINAL EVALUATION — 10 episodes each")
    print("="*60)

    modes = ['vanilla', 'double', 'dueling']
    labels = {'vanilla':'Vanilla DQN','double':'Double DQN','dueling':'Dueling DQN'}

    for mode in modes:
        try:
            env   = gym.make('LunarLander-v3')
            agent = Agent(mode=mode)
            agent.load(f"{mode}_lunarlander.pth")
            scores = []

            print(f"\n  {labels[mode]}:")
            for ep in range(10):
                obs, _ = env.reset()
                state  = np.array(obs, dtype=np.float32)
                total  = 0.0
                for _ in range(1000):
                    action = agent.select_action(state)
                    obs, r, term, trunc, _ = env.step(action)
                    state  = np.array(obs, dtype=np.float32)
                    total += r
                    if term or trunc: break
                scores.append(total)
                status = "LANDED ✓" if total >= 200 else ("crashed ✗" if total < 0 else "ok")
                print(f"    Ep {ep+1:>2}: {total:>8.2f}  {status}")

            env.close()
            print(f"    Average: {np.mean(scores):.2f}")
        except FileNotFoundError:
            print(f"    {mode}_lunarlander.pth not found — run training first")


# ═══════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("\n" + "="*60)
    print("  SESSION 6: DQN vs Double DQN vs Dueling DQN")
    print("  LunarLander-v3 — ~25-30 mins total on CPU")
    print("="*60)

    results = {}

    # Train all three — 800 episodes each
    for mode in ['vanilla', 'double', 'dueling']:
        scores, solved_at = train_agent(mode, episodes=800, log_every=50)
        results[mode] = (scores, solved_at)

    # Comparison plot
    plot_comparison(results)

    # Final eval
    evaluate_all()

    print("\nSession 6 complete!")
    print("Check session6_comparison.png for the full comparison.")