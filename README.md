# Reinforcement Learning Games Collection

A collection of three reinforcement learning projects demonstrating the progression from tabular Q-learning to Deep Q-Networks (DQN) and advanced DQN architectures. Each game introduces increasingly complex environments and showcases how AI agents learn through experience.

---

## 🎮 Game 1 — Tic-Tac-Toe (Day 1)

Classic 3×3 board game where two players take turns placing **X** and **O**. The first player to align three symbols horizontally, vertically, or diagonally wins.

### Environment

* **State Space:** 9 board cells (discrete)
* **Actions:** Any empty cell
* **Agent:** Q-Table (Dictionary)
* **Training Method:** Self-play

### Learning Process

The agent begins with completely random moves and learns entirely through self-play. By playing hundreds of thousands of games against itself, it gradually discovers which actions lead to victories and which result in losses.

After approximately **300,000 training games**, the agent learns nearly every possible board configuration and stores the optimal action for each state.

Since Tic-Tac-Toe contains only around **5,478 legal board states**, the entire game can be solved using a lookup table without requiring neural networks.

### What the AI Learned

* Block opponent winning moves.
* Prioritize the center position.
* Create forks and multiple winning opportunities.
* Recognize forced wins and draws.

**Performance:** Near-perfect play after reward function corrections.

---

## 🎯 Game 2 — CartPole (Day 2)

CartPole is a classic control problem where a pole is balanced on top of a moving cart. The agent must move the cart left or right to prevent the pole from falling.

### Environment

* **State Space:** 4 continuous variables

  * Cart position
  * Cart velocity
  * Pole angle
  * Pole angular velocity
* **Actions:** Left or Right
* **Maximum Score:** 500 steps

### Challenge

Unlike Tic-Tac-Toe, CartPole uses continuous state values, making a traditional Q-table impractical.

Initial attempts using state discretization achieved only around **88/500** average score.

To overcome this limitation, the project implements a **Deep Q-Network (DQN)** using PyTorch. The neural network estimates Q-values for unseen states, allowing the agent to generalize.

### Network Architecture

* Fully connected neural network
* Hidden layers: 128 units
* Experience replay
* Target network updates

### Training Results

* Approximately 1000 training episodes
* Average score exceeding **475/500**

### What the AI Learned

* Detect pole movement before failure occurs.
* Generate corrective momentum.
* Make counter-intuitive movements away from the center to maintain balance.

**Performance:** Near-perfect balancing behavior.

---

## 🚀 Game 3 — LunarLander (Day 3)

LunarLander is a complex physics-based environment where a spacecraft must land safely on a designated landing pad.

### Environment

* **State Space:** 8 continuous variables
* **Actions:**

  * Do nothing
  * Fire left engine
  * Fire main engine
  * Fire right engine

### Reward System

* Safe landing: **+200**
* Crash: **−100**
* Engine usage: Small negative reward (fuel cost)

### Challenges

* High-dimensional continuous state space.
* Delayed rewards.
* Gravity and inertia.
* Fuel management.

### Deep Reinforcement Learning Approaches

#### Vanilla DQN

Standard Deep Q-Network implementation that tends to overestimate action values.

#### Double DQN

Uses one network for action selection and another for evaluation, reducing overestimation bias.

#### Dueling DQN

Separates learning into:

* State value function (V)
* Action advantage function (A)

This architecture improves learning in situations where multiple actions produce similar outcomes.

### Network Improvements

* Hidden layer size increased to 256 units.
* Larger replay buffer.
* Improved stability and convergence.

### What the AI Learned

* Eliminate horizontal velocity first.
* Descend slowly.
* Use short engine bursts.
* Align with the landing pad.
* Land on the legs instead of the body.

**Performance:** Average reward exceeding 200, indicating successful landings.

---

# 📈 Learning Progression

| Feature         | Tic-Tac-Toe        | CartPole               | LunarLander          |
| --------------- | ------------------ | ---------------------- | -------------------- |
| State Space     | 9 cells (discrete) | 4 continuous values    | 8 continuous values  |
| Actions         | 9 possible moves   | 2 actions              | 4 actions            |
| Agent Type      | Q-Table            | Deep Q-Network         | Double/Dueling DQN   |
| Opponent        | Self-play          | Physics environment    | Physics + fuel costs |
| Solved When     | 97% win rate       | 475+/500 score         | 200+ average reward  |
| Main Challenge  | Reward assignment  | Continuous state space | Delayed rewards      |
| Learning Method | Tabular Q-learning | Deep Q-Learning        | Advanced DQN         |

---

# 🧠 Key Concepts Covered

* Reinforcement Learning
* Q-Learning
* Self-Play Training
* Deep Q-Networks (DQN)
* Experience Replay
* Target Networks
* Double DQN
* Dueling DQN
* Reward Engineering
* Continuous State Spaces

---

# 🛠 Technologies Used

* Python
* PyTorch
* NumPy
* OpenAI Gym / Gymnasium
* Matplotlib

---

# 📚 Project Goal

This project demonstrates the evolution of reinforcement learning techniques:

1. **Tabular Q-Learning** for small, discrete environments.
2. **Deep Q-Networks** for continuous environments.
3. **Advanced DQN variants** for complex physics simulations.

The progression illustrates why simple lookup tables fail in large environments and how neural networks enable agents to learn and generalize across infinite state spaces.
