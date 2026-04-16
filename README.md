# Reinforcement Learning for Market Making under Adverse Selection

A research-driven implementation of reinforcement learning (SAC) for market making in a realistic limit order book environment, with a focus on **risk, robustness, and microstructure realism**.

---

## Overview

Market making is not just about maximizing profit — it's about surviving adverse conditions.

This project builds a **high-fidelity limit order book simulator** and trains a reinforcement learning agent to operate within it. The goal is not to “beat the market,” but to answer a deeper question:

> **Can RL learn to manage risk better than classical strategies under adverse selection?**

---

## Key Results

### 1. RL does NOT dominate in raw PnL
- Heuristic strategies (especially imbalance-based skew) often achieve higher mean returns.

### 2. RL significantly improves risk control
- Lower and more stable inventory exposure
- Avoids extreme positions

### 3. RL dominates in tail risk (main result)

Under increasing adverse selection:

| Strategy | Mean PnL | CVaR (10%) |
|----------|---------:|-----------:|
| SAC      | High     | **Best**   |
| SKEW     | Medium   | Weak       |
| FIXED    | Low      | **Negative (blow-ups)** |

> RL learns to **avoid toxic flow**, while heuristics get exploited.

---

## Environment Design (Why This Is Not a Toy Project)

This simulator includes:

- Event-driven limit order book (not time-stepped)
- Price-time priority matching
- Stochastic order flow (limit / market / cancel)
- Regime switching (low vs high activity)
- **Adverse selection modeling (toxicity after fills)**
- Inventory + mark-to-market PnL tracking

Most RL trading projects fail because the environment is too simple.  
This one is designed so that **naive strategies break**.

---

## Reinforcement Learning Setup

- Algorithm: **Soft Actor-Critic (SAC)**
- Continuous action space:
  - bid/ask offsets from mid-price
  - bid/ask sizes

- Observations include:
  - spread
  - mid-price returns
  - volatility
  - order book imbalance
  - inventory
  - time since last fill

### Critical Insight

Without proper reward scaling:

> SAC learns a **degenerate exploit strategy** (extreme actions, inventory blow-up)

After fixing reward scale:

> SAC learns a **risk-aware market making policy**

---

## Stress Testing (Core Contribution)

We evaluate strategies under increasing **adverse selection (toxicity)**:

- Light
- Medium
- Strong

### Main Finding

> RL becomes more valuable as the environment becomes more adversarial.

- Heuristic strategies fail under strong toxicity
- RL adapts dynamically and maintains **positive tail performance**

---

## Project Structure
├── simulator/ # Limit order book + matching engine
├── env/ # RL environment wrapper
├── rl/ # SAC implementation
├── experiments/ # Training + evaluation scripts
├── analysis/ # Figures and metrics (CVaR, inventory)
├── figures/ # Generated plots
└── paper/ # LaTeX research paper


---

## Running the Project

### 1. Train the RL agent
```bash
python train_sac_v3.py
```
### 2. Evaluate across multiple seeds
```bash
python eval_multiseed.py
```

### 3. Run stres test
```bash
python eval_stress_test.py
```

---

## Example Output

=== STRONG TOXICITY ===
SAC   | PnL: +8970 | CVaR: +4593
FIXED | PnL: +4987 | CVaR: -2199
SKEW  | PnL: +5222 | CVaR: +969

---

## Author
Kaitlyn Pak
Carnegie Mellon University  --- Statistics & Machine Learning

## License 
MIT License
Copyright (c) 2026 Kaitlyn Pak
