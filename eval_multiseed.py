# eval_multiseed.py

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from lob_env import MarketMakingEnv


# =============================================================================
# Networks (same as training)
# =============================================================================

class Actor(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, action_dim)
        self.log_std_head = nn.Linear(hidden_dim, action_dim)
        self.LOG_STD_MIN = -20.0
        self.LOG_STD_MAX = 2.0

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        mean = self.mean_head(x)
        log_std = torch.clamp(self.log_std_head(x), self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        mean_action = torch.tanh(mean)
        log_prob = dist.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        return action, log_prob, mean_action


class Critic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q_head = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.q_head(x)


class DoubleCritic(nn.Module):
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 128):
        super().__init__()
        self.q1 = Critic(state_dim, action_dim, hidden_dim)
        self.q2 = Critic(state_dim, action_dim, hidden_dim)

    def forward(self, state, action):
        return self.q1(state, action), self.q2(state, action)


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.device = device
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        idx = np.random.randint(0, self.size, size=batch_size)
        return (
            torch.tensor(self.states[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.actions[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.rewards[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.next_states[idx], dtype=torch.float32, device=self.device),
            torch.tensor(self.dones[idx], dtype=torch.float32, device=self.device),
        )

    def __len__(self):
        return self.size


class SACAgent:
    def __init__(self, state_dim, action_dim, hidden_dim=128, actor_lr=3e-4, 
                 critic_lr=3e-4, alpha_lr=3e-4, gamma=0.99, tau=0.005, device="cpu"):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

        self.target_entropy = -float(action_dim)
        self.log_alpha = torch.zeros(1, requires_grad=True, device=device)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=alpha_lr)
        self.alpha = self.log_alpha.exp().item()

    def update(self, replay_buffer, batch_size):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        alpha = self.log_alpha.exp()

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q = rewards + (1.0 - dones) * self.gamma * (
                torch.min(target_q1, target_q2) - alpha * next_log_prob
            )

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_prob, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        actor_loss = (alpha * log_prob - torch.min(q1_pi, q2_pi)).mean()
        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        alpha_loss = -(self.log_alpha * (log_prob + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()
        self.alpha = self.log_alpha.exp().item()

        for src, tgt in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt.data.copy_(self.tau * src.data + (1 - self.tau) * tgt.data)


# =============================================================================
# Baseline Policies
# =============================================================================

def fixed_policy(obs):
    """Fixed offset=2, size=2 -> normalized actions"""
    # offset=2 maps from [0,5] -> need normalized value
    # (2 - 0) / 5 * 2 - 1 = 0.8 - 1 = -0.2... let's just use 0
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def skew_policy(obs):
    """Inventory skew: wider on side with inventory"""
    inv = obs[4]  # normalized_inventory in [-1, 1]
    
    # Skew offsets based on inventory
    # Positive inv -> widen bid (buy less), tighten ask (sell more)
    bid_off = 0.0 + 0.5 * inv   # more positive when long
    ask_off = 0.0 - 0.5 * inv   # more negative when long
    
    return np.array([bid_off, ask_off, 0.0, 0.0], dtype=np.float32)


# =============================================================================
# Evaluation Functions
# =============================================================================

def evaluate_policy_single_seed(env, policy_fn, num_episodes=10, device="cpu"):
    """Evaluate a policy on multiple episodes with one env seed."""
    results = []
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        truncated = False
        inv_min, inv_max = 0, 0
        
        while not (done or truncated):
            if callable(policy_fn):
                action = policy_fn(state)
            else:
                # SAC agent
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, _, mean_action = policy_fn.actor.sample(state_t)
                action = mean_action.squeeze(0).cpu().numpy()
            
            state, reward, done, truncated, info = env.step(action)
            inv = info.get("inventory", 0)
            inv_min = min(inv_min, inv)
            inv_max = max(inv_max, inv)
        
        results.append({
            "pnl": info.get("pnl", 0),
            "inv_range": inv_max - inv_min,
            "final_inv": info.get("inventory", 0),
        })
    
    return results


def train_sac_for_seed(train_seed, eval_seeds, device="cpu"):
    """Train SAC on one seed, evaluate on multiple seeds."""
    STATE_DIM, ACTION_DIM = 6, 4
    
    # Training env
    env = MarketMakingEnv(
        seed=train_seed,
        episode_duration=10.0,
        decision_interval=0.1,
        inventory_penalty=0.01,
        regime_mode="high",
        toxicity_drift=0.05,
        rl_max_offset=5,
        rl_max_size=3,
        reward_scale=1000.0,
    )
    
    agent = SACAgent(STATE_DIM, ACTION_DIM, device=device)
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, capacity=100_000, device=device)
    
    # Training loop (condensed)
    state = env.reset()
    total_steps = 10_000
    warmup_steps = 1_000
    batch_size = 128
    
    for step in range(total_steps):
        if step < warmup_steps:
            action = np.random.uniform(-1, 1, size=(4,)).astype(np.float32)
        else:
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _, _ = agent.actor.sample(state_t)
            action = action.squeeze(0).cpu().numpy().astype(np.float32)
        
        next_state, reward, done, truncated, info = env.step(action)
        buffer.add(state, action, reward, next_state, bool(done or truncated))
        
        if step >= warmup_steps and len(buffer) >= batch_size:
            agent.update(buffer, batch_size)
        
        state = next_state
        if done or truncated:
            state = env.reset()
    
    # Evaluate on all eval seeds
    eval_results = []
    for seed in eval_seeds:
        eval_env = MarketMakingEnv(
            seed=seed,
            episode_duration=10.0,
            decision_interval=0.1,
            inventory_penalty=0.01,
            regime_mode="high",
            toxicity_drift=0.05,
            rl_max_offset=5,
            rl_max_size=3,
            reward_scale=1000.0,
        )
        results = evaluate_policy_single_seed(eval_env, agent, num_episodes=5, device=device)
        for r in results:
            r["seed"] = seed
            eval_results.append(r)
    
    return agent, eval_results


def evaluate_baseline_all_seeds(policy_fn, eval_seeds, policy_name):
    """Evaluate a baseline policy on all seeds."""
    all_results = []
    
    for seed in eval_seeds:
        env = MarketMakingEnv(
            seed=seed,
            episode_duration=10.0,
            decision_interval=0.1,
            inventory_penalty=0.01,
            regime_mode="high",
            toxicity_drift=0.05,
            rl_max_offset=5,
            rl_max_size=3,
            reward_scale=1000.0,
        )
        results = evaluate_policy_single_seed(env, policy_fn, num_episodes=5)
        for r in results:
            r["seed"] = seed
            r["policy"] = policy_name
            all_results.append(r)
    
    return all_results


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    DEVICE = "cpu"
    EVAL_SEEDS = list(range(100, 110))  # 10 different eval seeds
    TRAIN_SEED = 42
    
    print("=" * 60)
    print("MULTI-SEED EVALUATION: SAC vs FIXED vs SKEW")
    print("=" * 60)
    print(f"Train seed: {TRAIN_SEED}")
    print(f"Eval seeds: {EVAL_SEEDS}")
    print()
    
    # Train SAC
    print("Training SAC (10k steps)...")
    sac_agent, sac_results = train_sac_for_seed(TRAIN_SEED, EVAL_SEEDS, DEVICE)
    
    # Evaluate baselines
    print("Evaluating FIXED policy...")
    fixed_results = evaluate_baseline_all_seeds(fixed_policy, EVAL_SEEDS, "FIXED")
    
    print("Evaluating SKEW policy...")
    skew_results = evaluate_baseline_all_seeds(skew_policy, EVAL_SEEDS, "SKEW")
    
    # Aggregate results
    print("\n" + "=" * 60)
    print("RESULTS BY SEED")
    print("=" * 60)
    
    def summarize(results, name):
        pnls = [r["pnl"] for r in results]
        inv_ranges = [r["inv_range"] for r in results]
        return {
            "policy": name,
            "pnl_mean": np.mean(pnls),
            "pnl_std": np.std(pnls),
            "inv_range_mean": np.mean(inv_ranges),
            "inv_range_std": np.std(inv_ranges),
        }
    
    # Per-seed breakdown
    print(f"\n{'Seed':<6} {'SAC PnL':>12} {'FIXED PnL':>12} {'SKEW PnL':>12} {'SAC InvR':>10} {'FIXED InvR':>10} {'SKEW InvR':>10}")
    print("-" * 80)
    
    for seed in EVAL_SEEDS:
        sac_seed = [r for r in sac_results if r["seed"] == seed]
        fixed_seed = [r for r in fixed_results if r["seed"] == seed]
        skew_seed = [r for r in skew_results if r["seed"] == seed]
        
        sac_pnl = np.mean([r["pnl"] for r in sac_seed])
        fixed_pnl = np.mean([r["pnl"] for r in fixed_seed])
        skew_pnl = np.mean([r["pnl"] for r in skew_seed])
        
        sac_inv = np.mean([r["inv_range"] for r in sac_seed])
        fixed_inv = np.mean([r["inv_range"] for r in fixed_seed])
        skew_inv = np.mean([r["inv_range"] for r in skew_seed])
        
        print(f"{seed:<6} {sac_pnl:>+12.0f} {fixed_pnl:>+12.0f} {skew_pnl:>+12.0f} {sac_inv:>10.1f} {fixed_inv:>10.1f} {skew_inv:>10.1f}")
    
    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY STATISTICS")
    print("=" * 60)
    
    sac_summary = summarize(sac_results, "SAC")
    fixed_summary = summarize(fixed_results, "FIXED")
    skew_summary = summarize(skew_results, "SKEW")
    
    print(f"\n{'Policy':<8} {'PnL Mean':>12} {'PnL Std':>12} {'InvRange Mean':>14} {'InvRange Std':>14}")
    print("-" * 65)
    for s in [sac_summary, fixed_summary, skew_summary]:
        print(f"{s['policy']:<8} {s['pnl_mean']:>+12.0f} {s['pnl_std']:>12.0f} {s['inv_range_mean']:>14.1f} {s['inv_range_std']:>14.1f}")
    
    # Winner
    print("\n" + "=" * 60)
    policies = [sac_summary, fixed_summary, skew_summary]
    best_pnl = max(policies, key=lambda x: x["pnl_mean"])
    best_risk = min(policies, key=lambda x: x["inv_range_mean"])
    
    print(f"Best PnL: {best_pnl['policy']} ({best_pnl['pnl_mean']:+.0f})")
    print(f"Best Risk Control: {best_risk['policy']} (inv_range={best_risk['inv_range_mean']:.1f})")
    print("=" * 60)