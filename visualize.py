"""
Session 2: Hyperparameter tuning + training curve visualization
Saves plots as PNG files
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from collections import deque
from env import TicTacToeEnv
from agent import QLearningAgent


def run_experiment(alpha, gamma, epsilon_decay, episodes=50_000, window=500):
    env = TicTacToeEnv()
    x_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)
    o_agent = QLearningAgent(alpha=alpha, gamma=gamma, epsilon_decay=epsilon_decay)

    win_rates, draw_rates, epsilons = [], [], []
    recent = deque(maxlen=window)

    for ep in range(1, episodes + 1):
        state = env.reset()
        x_exp = o_exp = None

        while True:
            agent = x_agent if env.current_player == 1 else o_agent
            actions = env.available_actions()
            action = agent.select_action(state, actions)
            next_state, reward, done = env.step(action)
            next_actions = env.available_actions() if not done else []

            if env.current_player == 1 or done:
                if x_exp:
                    x_agent.update(*x_exp, 0.0, next_state, next_actions, False)
                if done:
                    x_agent.update(state, action, reward, next_state, [], True)
                    if o_exp:
                        o_agent.update(*o_exp, -reward if reward == 1 else reward, next_state, [], True)
                else:
                    x_exp = (state, action)
            else:
                if o_exp:
                    o_agent.update(*o_exp, 0.0, next_state, next_actions, False)
                o_exp = (state, action)

            state = next_state
            if done:
                recent.append(1 if reward == 1.0 else (0.5 if reward == 0.5 else 0))
                break

        if ep % window == 0:
            arr = list(recent)
            win_rates.append(arr.count(1) / len(arr))
            draw_rates.append(arr.count(0.5) / len(arr))
            epsilons.append(x_agent.epsilon)

    return win_rates, draw_rates, epsilons


def plot_single_run():
    print("Running single training experiment...")
    episodes = 60_000
    window = 500
    env = TicTacToeEnv()
    x_agent = QLearningAgent(alpha=0.3, gamma=0.9, epsilon=1.0,
                              epsilon_min=0.05, epsilon_decay=0.9995)
    o_agent = QLearningAgent(alpha=0.3, gamma=0.9, epsilon=1.0,
                              epsilon_min=0.05, epsilon_decay=0.9995)

    win_rates, draw_rates, loss_rates, epsilons, qtable_sizes = [], [], [], [], []
    recent = deque(maxlen=window)

    for ep in range(1, episodes + 1):
        state = env.reset()
        x_exp = o_exp = None
        while True:
            agent = x_agent if env.current_player == 1 else o_agent
            actions = env.available_actions()
            action = agent.select_action(state, actions)
            next_state, reward, done = env.step(action)
            next_actions = env.available_actions() if not done else []
            if env.current_player == 1 or done:
                if x_exp:
                    x_agent.update(*x_exp, 0.0, next_state, next_actions, False)
                if done:
                    x_agent.update(state, action, reward, next_state, [], True)
                    if o_exp:
                        o_agent.update(*o_exp, -reward if reward == 1 else reward, next_state, [], True)
                else:
                    x_exp = (state, action)
            else:
                if o_exp:
                    o_agent.update(*o_exp, 0.0, next_state, next_actions, False)
                o_exp = (state, action)
            state = next_state
            if done:
                recent.append(reward)
                break

        if ep % window == 0:
            arr = list(recent)
            win_rates.append(sum(1 for r in arr if r == 1.0) / len(arr))
            draw_rates.append(sum(1 for r in arr if r == 0.5) / len(arr))
            loss_rates.append(sum(1 for r in arr if r == 0.0) / len(arr))
            epsilons.append(x_agent.epsilon)
            qtable_sizes.append(len(x_agent.q_table))

    x = np.arange(1, len(win_rates) + 1) * window

    fig = plt.figure(figsize=(14, 10))
    fig.patch.set_facecolor('#0f1117')
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

    ax_colors = {
        'bg': '#0f1117', 'panel': '#1a1d2e', 'grid': '#2a2d3e',
        'win': '#1D9E75', 'draw': '#EF9F27', 'loss': '#E24B4A',
        'eps': '#7F77DD', 'qt': '#5DCAA5', 'text': '#c8cad4'
    }

    def style_ax(ax, title):
        ax.set_facecolor(ax_colors['panel'])
        ax.tick_params(colors=ax_colors['text'], labelsize=9)
        ax.set_title(title, color=ax_colors['text'], fontsize=11, pad=8)
        ax.grid(True, color=ax_colors['grid'], linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(ax_colors['grid'])
        ax.xaxis.label.set_color(ax_colors['text'])
        ax.yaxis.label.set_color(ax_colors['text'])

    # Plot 1: Win/Draw/Loss rates
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.plot(x, win_rates,  color=ax_colors['win'],  lw=2, label='X wins')
    ax1.plot(x, draw_rates, color=ax_colors['draw'], lw=2, label='Draws')
    ax1.plot(x, loss_rates, color=ax_colors['loss'], lw=2, label='O wins')
    ax1.set_xlabel('Episode'); ax1.set_ylabel('Rate')
    ax1.legend(facecolor=ax_colors['panel'], labelcolor=ax_colors['text'],
               edgecolor=ax_colors['grid'], fontsize=9)
    style_ax(ax1, 'Win / Draw / Loss rates')

    # Plot 2: Epsilon decay
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(x, epsilons, color=ax_colors['eps'], lw=2)
    ax2.fill_between(x, epsilons, alpha=0.15, color=ax_colors['eps'])
    ax2.set_xlabel('Episode'); ax2.set_ylabel('ε (exploration rate)')
    style_ax(ax2, 'Epsilon decay — explore → exploit')

    # Plot 3: Q-table growth
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.plot(x, qtable_sizes, color=ax_colors['qt'], lw=2)
    ax3.fill_between(x, qtable_sizes, alpha=0.15, color=ax_colors['qt'])
    ax3.set_xlabel('Episode'); ax3.set_ylabel('Unique (state, action) pairs')
    style_ax(ax3, 'Q-table growth')

    # Plot 4: Smoothed win rate
    ax4 = fig.add_subplot(gs[1, 1])
    smooth = np.convolve(win_rates, np.ones(10)/10, mode='valid')
    ax4.plot(x[:len(smooth)], smooth, color=ax_colors['win'], lw=2.5)
    ax4.axhline(y=0.95, color=ax_colors['draw'], lw=1, linestyle='--', alpha=0.7)
    ax4.set_xlabel('Episode'); ax4.set_ylabel('Win rate (smoothed)')
    style_ax(ax4, 'Smoothed win rate (convergence)')

    plt.suptitle('Q-Learning Tic-Tac-Toe — Training Curves',
                 color=ax_colors['text'], fontsize=14, y=1.01)

    plt.savefig('/mnt/user-data/outputs/training_curves.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Saved: training_curves.png")


def plot_hyperparam_sweep():
    print("Running hyperparameter sweep (this takes ~1 min)...")
    configs = [
        {'alpha': 0.1,  'gamma': 0.9,  'epsilon_decay': 0.9995, 'label': 'α=0.1 (slow learner)'},
        {'alpha': 0.3,  'gamma': 0.9,  'epsilon_decay': 0.9995, 'label': 'α=0.3 (default)'},
        {'alpha': 0.7,  'gamma': 0.9,  'epsilon_decay': 0.9995, 'label': 'α=0.7 (fast learner)'},
        {'alpha': 0.3,  'gamma': 0.5,  'epsilon_decay': 0.9995, 'label': 'γ=0.5 (short horizon)'},
        {'alpha': 0.3,  'gamma': 0.99, 'epsilon_decay': 0.9995, 'label': 'γ=0.99 (long horizon)'},
        {'alpha': 0.3,  'gamma': 0.9,  'epsilon_decay': 0.999,  'label': 'fast ε-decay'},
    ]

    ax_colors = {
        'bg': '#0f1117', 'panel': '#1a1d2e', 'grid': '#2a2d3e', 'text': '#c8cad4'
    }
    palette = ['#1D9E75','#7F77DD','#EF9F27','#E24B4A','#5DCAA5','#D85A30']

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.patch.set_facecolor(ax_colors['bg'])

    for ax in axes:
        ax.set_facecolor(ax_colors['panel'])
        ax.tick_params(colors=ax_colors['text'], labelsize=9)
        ax.grid(True, color=ax_colors['grid'], linewidth=0.5, alpha=0.7)
        for spine in ax.spines.values():
            spine.set_edgecolor(ax_colors['grid'])
        ax.xaxis.label.set_color(ax_colors['text'])
        ax.yaxis.label.set_color(ax_colors['text'])

    episodes = 40_000
    window = 500
    x = np.arange(1, episodes // window + 1) * window

    for i, cfg in enumerate(configs):
        wr, dr, eps = run_experiment(cfg['alpha'], cfg['gamma'],
                                     cfg['epsilon_decay'], episodes, window)
        color = palette[i % len(palette)]
        axes[0].plot(x, wr,  color=color, lw=1.8, label=cfg['label'])
        axes[1].plot(x, eps, color=color, lw=1.8, label=cfg['label'])

    axes[0].set_title('Win rate by config', color=ax_colors['text'], fontsize=11)
    axes[0].set_xlabel('Episode'); axes[0].set_ylabel('Win rate')
    axes[0].legend(facecolor=ax_colors['panel'], labelcolor=ax_colors['text'],
                   edgecolor=ax_colors['grid'], fontsize=8)

    axes[1].set_title('Epsilon decay by config', color=ax_colors['text'], fontsize=11)
    axes[1].set_xlabel('Episode'); axes[1].set_ylabel('ε')
    axes[1].legend(facecolor=ax_colors['panel'], labelcolor=ax_colors['text'],
                   edgecolor=ax_colors['grid'], fontsize=8)

    plt.suptitle('Hyperparameter sweep', color=ax_colors['text'], fontsize=13, y=1.02)
    plt.tight_layout()
    plt.savefig('/mnt/user-data/outputs/hyperparam_sweep.png',
                dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
    print("Saved: hyperparam_sweep.png")


if __name__ == "__main__":
    plot_single_run()
    plot_hyperparam_sweep()
    print("All plots saved!")
