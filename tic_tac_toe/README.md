# Game_Playing_Agent: Day 1 — Tabular RL & Q-Learning

Welcome to Day 1 of the **AI Playing Video Games** intensive curriculum. This repository documents a structured 3-day roadmap detailing the progression from basic tabular Reinforcement Learning up to Deep Q-Networks (DQNs). 

Day 1 focuses entirely on building an optimal Tic-Tac-Toe agent completely from scratch using Tabular Q-Learning via self-play.

---

## 📂 Project Architecture

The core framework is broken down into four foundational modules:

* **`env.py`** — **The Environment:** Represents the game board as a hashable 9-tuple. The `step()` function takes an action and returns a tuple of `(next_state, reward, done)`. It structures baseline step rewards: `+1` for a win, `+0.5` for a draw, and `0` for ongoing moves.
* **`agent.py`** — **The Q-Agent:** Implements a tabular action-value framework using a Python `defaultdict` as the Q-table. Handles action selection using an $\epsilon$-greedy exploration schedule and computes the temporal difference update.
* **`train.py`** — **The Self-Play Loop:** Orchestrates training by putting Agent X and Agent O against each other. This symmetric self-play setup builds a dynamic curriculum where both players continuously improve, yielding a significantly more robust policy than training against random opponents.
* **`play.py`** — **The Human Interface:** Provides an interactive terminal script allowing human players to play against the trained, optimized agent.

---

## 🧠 Core Reinforcement Learning Concepts

This implementation directly applies the core pillars of Reinforcement Learning:

1. **Markov Decision Process (MDP):** Mapping the game state spaces down to a formal sequence of States ($s$), Actions ($a$), Rewards ($r$), and Next States ($s'$).
2. **The Bellman Equation:** Driving value convergence by adjusting action-value pairs iteratively based on experiential feedback:
   $$Q(s,a) \leftarrow Q(s,a) + \alpha \cdot \left[ r + \gamma \cdot \max_{a'} Q(s',a') - Q(s,a) \right]$$
3. **$\epsilon$-Greedy Exploration:** Beginning with fully random exploratory steps and decaying the exploration parameter ($\epsilon$) systematically to gradually shift toward exploiting high-value policies.
4. **Self-Play Dynamics:** Emulating the training paradigm of AlphaGo. Training an agent against an evolving version of itself forces the discovery of deep defensive and offensive counter-strategies.
5. **Deferred Reward Signals:** Ensuring that critical end-game penalties (like losing) map cleanly back onto the final decisive transition instead of leaving the agent with unassigned feedback.

---

## 🛠️ Debugging and Hyperparameter Tuning

### The Reward Assignment Bug
During early deployment, a critical behavioral edge case was caught: the AI would aggressively assemble an offensive sequence but entirely ignore an opponent's winning threat. It was failing to play defensively.

* **The Cause:** A classical reward-assignment flaw in the step loop. The losing agent's last move wasn't receiving the negative reinforcement signal (`-1`), causing it to remain completely indifferent to terminal defeats.
* **The Fix:** Migrated from a live, rolling step-by-step update to an **Episodic History Approach**. The system records the full history of the game into a buffer and replays the trajectory at termination, assigning the correct relative terminal rewards (`+1` / `-1` / `+0.5`) back to each agent's final state-action pairs. Following this fix, the agent correctly identifies high-risk states and assigns clear negative utility to moves that allow an opponent to win.

### Empirical Analysis & Training Insights
* **Exploration Schedules:** Win rates scale sharply from an initial ~60% (caused by the first-mover advantage inherent to Tic-Tac-Toe) to an elite ~97% as exploration ($\epsilon$) decays.
* **Q-Table Saturation:** The state-action table stabilizes at approximately **10,700 entries**, confirming that the agent has effectively exposed and mapped out the entire valid game tree.
* **Learning Rate ($\alpha$):** Higher learning rates ($\alpha=0.7$) yield rapid initial convergence but introduce noise and policy oscillation. Lower values ($\alpha=0.1$) offer exceptionally smooth, stable convergence, although requiring slightly longer training histories.
* **Discount Factor ($\gamma$):** Variations in $\gamma$ have negligible impact due to the short horizon of the game tree (5 to 9 plies max), meaning immediate and delayed rewards carry highly similar urgency.

---

## 💡 The Real Lesson of Day 1

The ultimate engineering takeaway went far beyond managing boilerplate logic. The absolute key to mastering Reinforcement Learning is **interactive behavior debugging**. Relying strictly on high-level training curves can disguise fatal structural flaws. By manually testing against the agent, diagnosing faulty behavioral patterns, tracking down the anomaly in the reward function, and refactoring the environment replay, we gain true operational insight into how an agent values its world.
