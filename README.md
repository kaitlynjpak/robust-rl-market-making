# Robust RL Market Making Under Realistic Microstructure Constraints

This repository contains my independent research on reinforcement learning for market making under more realistic market microstructure conditions.

## Overview

The goal of this project is to study whether an RL-based market-making agent can behave robustly when the environment includes practical constraints such as:

- inventory risk
- adverse selection
- latency
- transaction costs
- partial observability
- changing market regimes

A major focus of this research is not just profitability, but also evaluation: designing a testing framework that prevents reward hacking and exposes fragile strategies.

## Research Question

How robust is an RL market maker across different microstructure regimes compared with classical benchmark strategies under realistic trading constraints?

## Project Goals

- Build a realistic market-making environment
- Train and evaluate RL agents
- Compare RL policies against baseline strategies
- Measure risk, not just average PnL
- Develop an evaluation protocol that detects reward-hacking behavior

## Planned Components

- **Simulator** for limit order book dynamics
- **Agent training pipeline** in Python
- **Baseline strategies** for comparison
- **Evaluation suite** for PnL, inventory risk, tail risk, and robustness
- **Experiments** across multiple market regimes

## Current Status

This project is currently in progress.
