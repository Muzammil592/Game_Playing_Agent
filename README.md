Game 1 — Tic-Tac-Toe (Day 1)
Classic 3x3 board game. Two players take turns placing X and O, first to get 3 in a row wins.
The AI learns by playing against itself hundreds of thousands of times. It starts completely random — placing pieces anywhere — and slowly figures out which moves lead to wins and which lead to losses. After 300,000 games of self-play, it knows every possible board position and the best move in each one. The entire game tree fits in a dictionary (Q-table) because there are only ~5,478 legal board states. No neural network needed — pure lookup table.
What the AI learned: Block opponent threats, take the center, force forks. Beat it if you can — after the reward bug fix, it plays near-perfectly.

Game 2 — CartPole (Day 2)
A pole is balanced on top of a moving cart on a track. The AI controls the cart — push left or push right — to keep the pole from falling. Score = how many steps the pole stays upright. Max is 500.
This is where the Q-table breaks. The cart's position, velocity, pole angle, and angular velocity are all continuous floating point numbers — infinite possible combinations, impossible to enumerate in a dict. Session 3 tried discretizing (binning) the state into a Q-table anyway and hit a ceiling of ~88/500. Session 4 replaced the table with a PyTorch neural network (DQN) that learns to approximate Q-values for any state it's never seen. Training for ~1000 episodes gets it to 475+/500 — essentially perfect balance.
What the AI learned: Feel the pole leaning and correct before it falls. Counter-intuitive moves like moving away from center to create corrective momentum.

Game 3 — LunarLander (Day 3)
A rocket ship must land safely on a landing pad on the moon. The AI controls 4 actions — fire left engine, fire main engine, fire right engine, or do nothing. It gets rewarded for landing on the pad (+200), penalized for crashing (-100), and loses small points every time it fires an engine (fuel costs money).
This is the hardest of the three. 8-dimensional continuous state, delayed rewards, physics simulation with gravity and inertia. Session 5 applied the same DQN from CartPole — just bigger network (256 hidden units instead of 128) and bigger replay buffer. Session 6 ran three variants side by side:

Vanilla DQN — baseline, tends to overestimate how good situations are
Double DQN — 2-line fix, uses one network to pick the action and another to score it, removes overestimation bias
Dueling DQN — splits into two streams, one learning how good the position is (V) and one learning how good each action is (A), better at spotting when any action is roughly equal

What the AI learned: Kill horizontal velocity first, then descend slowly, fire main engine in short bursts, align with pad, touch down with legs not body.

The Full Progression
Tic-Tac-ToeCartPoleLunarLanderState9 cells, discrete4 floats, continuous8 floats, continuousActions9 (any empty cell)2 (left / right)4 (engines)AgentQ-table (dict)Neural net (DQN)Neural net (Double/Dueling DQN)OpponentItself (self-play)PhysicsPhysics + fuel costSolved when97% win rateAvg 475/500 stepsAvg 200+ rewardKey challengeReward assignment bugContinuous state spaceDelayed rewards + fuel penalty
