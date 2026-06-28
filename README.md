# Game_Playing_Agent: Continuous Control & Deep Q-Networks (DQN)

Welcome to the **AI Playing Video Games** intensive curriculum repository. This project documents a structured progression from basic tabular Reinforcement Learning up to Deep Q-Networks (DQNs).

---

## 🗺️ The Journey: Day 1 vs. Day 2

### Day 1: Tabular Q-Learning (Tic-Tac-Toe)
* **The Environment:** Discrete state space (9 board positions, finite combinations).
* **The Brain:** A standard Q-table (`defaultdict`) storing explicit state-action pairs.
* **The Limit:** Perfect for small games, but completely hits a wall when state variables become continuous decimal numbers.

### Day 2: Deep Q-Networks (CartPole-v1)
* **The Environment:** Continuous state space (4 raw physics floats: cart position, velocity, pole angle, angular velocity). 
* **The Brain:** A Deep Neural Network (PyTorch MLP) that takes continuous floats and approximates Q-values for unseen states.
* **The Result:** Shattered the "discretization ceiling" where traditional tables fail, setting up the framework to achieve a perfect 500/500 control score.

---

## 📂 Project Architecture (Day 2 Updates)

The repository has evolved to include production-grade Deep RL components under `dqn.py`:

* **`QNetwork`** — **The Brain:** A 3-layer PyTorch Multi-Layer Perceptron (MLP). It takes the raw 4-float continuous state vector directly as input and outputs a 2-float vector representing the predicted Q-values for each discrete action (Move Left / Move Right). No blurry discretization buckets needed.
* **`ReplayBuffer`** — **The Memory Bank:** A cyclic buffer holding up to 10,000 empirical transitions $(s, a, r, s', \text{done})$. By sampling random mini-batches (size 64) for training, it breaks the temporal correlation of sequential driving frames and stabilizes gradient updates.
* **`Target Network`** — **The Frozen Instructor:** A structural clone of the online policy network. By keeping its weights frozen and updating them only every 10 episodes, it prevents the classic "moving target" problem, ensuring the Bellman loss minimization doesn't diverge.

---

## 🧠 Core Reinforcement Learning Concepts Applied

1. **Continuous State Generalization:** Unlike tables that need to visit every exact coordinate, the neural network acts as a global function approximator—learning patterns from a position like `0.017` and successfully generalizing to `0.018`.
2. **The Deep Bellman Objective:** Driving optimization using Mean Squared Error (MSE) loss between the online network's prediction and the Target Network's evaluation:
   $$Q(s,a) \approx r + \gamma \max_{a'} Q_{\text{target}}(s', a')$$
3. **$\epsilon$-Greedy Decay Exploration:** Moving from 100% random exploration ($\epsilon=1.0$) down to a highly focused exploitation mode ($\epsilon=0.05$) via an exponential decay rate of `0.995` per episode.

---

## 📊 Day 2 Empirical Analysis & Metrics

Running the agent on CPU (`Using device: cpu`) for an initial test flight of 600 episodes yields the following progression:

* **Episodes 1–100 (Random Flailing):** High exploration ($\epsilon \approx 0.77$). The agent relies heavily on random actions, averaging a baseline score of ~24-34 steps.
* **Episodes 200–300 (Pattern Discovery):** Average score jumps to `79.0`, hitting peak episode spikes of `147`. The network begins mapping how cart velocity affects pole balance.
* **Episodes 500–600 (Policy Convergence):** The model stabilizes. Evaluation runs across 10 test episodes with **zero exploration ($\epsilon=0$)** yield highly consistent behavior:
  * *Test Performance:* Consistently holding steps between `123` and `135`.
  * *Current Baseline:* **130.0 / 500 average**.
