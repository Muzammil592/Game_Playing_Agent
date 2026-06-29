"""
Fixed self-play training with proper reward signals for both agents.
"""
from pathlib import Path
import numpy as np
from collections import deque
from env import TicTacToeEnv
from agent import QLearningAgent


BASE_DIR = Path(__file__).resolve().parent


def train(episodes=300_000, log_every=10_000):
    env = TicTacToeEnv()
    x_agent = QLearningAgent(alpha=0.5, gamma=0.95, epsilon=1.0,
                              epsilon_min=0.02, epsilon_decay=0.9997)
    o_agent = QLearningAgent(alpha=0.5, gamma=0.95, epsilon=1.0,
                              epsilon_min=0.02, epsilon_decay=0.9997)

    stats = deque(maxlen=log_every)

    for ep in range(1, episodes + 1):
        state = env.reset()
        history = []   # list of (agent, state, action)

        while True:
            agent = x_agent if env.current_player == 1 else o_agent
            actions = env.available_actions()
            action  = agent.select_action(state, actions)
            next_state, reward, done = env.step(action)
            history.append((agent, state, action))

            if done:
                # Assign rewards to ALL moves in the game
                for i, (ag, s, a) in enumerate(history):
                    is_last = (i == len(history) - 1)
                    if is_last:
                        # The agent who just moved caused the terminal
                        if reward == 1.0:
                            ag.update(s, a, 1.0, next_state, [], True)   # winner
                        else:
                            ag.update(s, a, 0.5, next_state, [], True)   # draw
                    elif i == len(history) - 2:
                        # The OTHER agent's last move — they lost or drew
                        if reward == 1.0:
                            ag.update(s, a, -1.0, next_state, [], True)  # loser
                        else:
                            ag.update(s, a, 0.5, next_state, [], True)   # draw
                    else:
                        # Intermediate moves get a small step reward of 0
                        next_s = history[i+1][1]
                        next_acts = env.available_actions() if not done else []
                        ag.update(s, a, 0.0, next_s, list(range(9)), False)

                stats.append(reward)
                break

            state = next_state

        if ep % log_every == 0:
            recent = list(stats)
            wins  = recent.count(1.0) / len(recent) * 100
            draws = recent.count(0.5) / len(recent) * 100
            print(f"Ep {ep:>7} | ε={x_agent.epsilon:.3f} | "
                  f"X wins {wins:.1f}%  Draws {draws:.1f}%  "
                  f"O wins {100-wins-draws:.1f}% | "
                  f"Q-table: {len(x_agent.q_table)}")

    x_agent.save(BASE_DIR / "x_agent.pkl")
    o_agent.save(BASE_DIR / "o_agent.pkl")
    print("\nTraining complete!")
    return x_agent, o_agent


if __name__ == "__main__":
    train()