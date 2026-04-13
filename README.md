# Deep Reinforcement Learning: PPO Actor-Critic Snake Agent

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-orange.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## 📌 Project Overview
This project implements a state-of-the-art Deep Reinforcement Learning agent to solve a custom-built 2D Grid survival game (Snake). **No pre-built reinforcement learning environments (like OpenAI Gym) were used.** The physics, rendering, and state-space mappings were constructed entirely from scratch using Pygame and Numpy.

The agent is trained using **Proximal Policy Optimization (PPO)**, an industry-standard policy gradient method, utilizing an **Actor-Critic** neural network architecture and **Generalized Advantage Estimation (GAE)**.

## 🧠 Key Technical Implementations
* **Custom Environment (`snake_env.py`):** A custom Markov Decision Process (MDP) featuring dynamic food spawning, static obstacles, and poison blocks.
* **Translation-Invariant State Space:** The agent utilizes "Relative Vision," perceiving the world as a 15-value boolean array (Danger Left/Right/Straight, and relative compass directions to targets). This reduces dimensionality and accelerates convergence compared to absolute coordinate mapping.
* **Actor-Critic Architecture:** * **Actor:** Outputs a stochastic probability distribution over the discrete action space (Straight, Left, Right).
  * **Critic:** Learns a baseline Value function to predict expected future returns, reducing the variance of policy updates.
* **PPO Clipping:** Prevents catastrophic forgetting by bounding the policy update ratio, ensuring monotonic policy improvement.
* **Generalized Advantage Estimation (GAE):** Solves the credit assignment problem by exponentially weighting n-step returns to balance bias and variance.

## 📂 Repository Structure
* `snake_env.py` - The custom Pygame environment and relative state-generation logic.
* `train.py` - The main PPO training loop, including the Actor-Critic network, GAE calculation, and auto-checkpointing.
* `play.py` - The deterministic inference script for demonstrating the fully trained agent.
* `snake_checkpoint.pth` - The saved PyTorch model weights (the trained brain).

## ⚙️ Installation & Prerequisites
Ensure you have Python 3 installed. Install the required dependencies using pip:
```bash
pip install torch numpy pygame
