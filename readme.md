# OBELIX Reinforcement Learning Agent

## Overview

This project implements reinforcement learning agents for the OBELIX warehouse robot task. The objective is to learn policies for locating, attaching, pushing, and unwedgeing a box under partially observable conditions.

## Approach

The project evolves across multiple phases:

* **Phase 1:** Tabular RL methods (Q-learning, SARSA, Dyna-Q)
* **Phase 2:** PPO with LSTM for partial observability
* **Phase 3:** Double DQN (DDQN) with frame stacking
* **Phase 4:** Policy stabilization and evaluation improvements

The final model is a DDQN agent with:

* Dueling architecture
* Frame stacking for temporal context
* NoisyLinear exploration
* Reward shaping and heuristic guidance

## Repository Structure

```
Phase1_Final_Code/
Phase2_Final_Code/
Phase3_Final_Code/
Phase4_Final_Code/
```

Each folder contains:

* `agent.py` → agent logic
* `ddqn.py / ppo_lstm.py` → model implementation
* `weights.pth` → trained model

## Training Details

* Episodes: ~2000
* Batch size: 64
* Replay buffer: 50,000
* Optimizer: Adam
* Discount factor: 0.99

## Evaluation

* Greedy policy (ε = 0)
* Multiple runs for consistency
* Best model selected using moving average reward

## Results

* Level 1: -1983.80
* Level 2: -1979.70
* Level 3: -1439.58
* Final: -1798.59

## Demo

YouTube: https://youtu.be/J-LMIfNvKBs

## Notes

* Frame stacking is used instead of recurrent memory
* Heuristic logic (unwedge, IR guidance) improves stability

---

Author: Gaurav Srivastava
