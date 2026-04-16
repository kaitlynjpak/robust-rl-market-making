# eval_stress_test.py

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from lob_env import MarketMakingEnv


# =============================================================================
# Networks (same as before)
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
    return np.array([0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def skew_policy(obs):
    inv = obs[4]
    bid_off = 0.0 + 0.5 * inv
    ask_off = 0.0 - 0.5 * inv
    return np.array([bid_off, ask_off, 0.0, 0.0], dtype=np.float32)


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy(env, policy_fn, num_episodes=5, device="cpu"):
    results = []
    for _ in range(num_episodes):
        state = env.reset()
        done, truncated = False, False
        inv_min, inv_max = 0, 0

        while not (done or truncated):
            if callable(policy_fn):
                action = policy_fn(state)
            else:
                state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
                with torch.no_grad():
                    _, _, mean_action = policy_fn.actor.sample(state_t)
                action = mean_action.squeeze(0).cpu().numpy()

            state, reward, done, truncated, info = env.step(action)
            inv = info.get("inventory", 0)
            inv_min, inv_max = min(inv_min, inv), max(inv_max, inv)

        results.append({
            "pnl": info.get("pnl", 0),
            "inv_range": inv_max - inv_min,
        })
    return results


def train_sac(env_config, total_steps=10_000, device="cpu"):
    STATE_DIM, ACTION_DIM = 6, 4
    env = MarketMakingEnv(**env_config)
    agent = SACAgent(STATE_DIM, ACTION_DIM, device=device)
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, capacity=100_000, device=device)

    state = env.reset()
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

    return agent


def run_experiment(toxicity_config, eval_seeds, device="cpu"):
    """Run full comparison for a toxicity configuration."""
    
    base_config = {
        "seed": 42,
        "episode_duration": 10.0,
        "decision_interval": 0.1,
        "inventory_penalty": 0.01,
        "regime_mode": "high",
        "rl_max_offset": 5,
        "rl_max_size": 3,
        "reward_scale": 1000.0,
    }
    base_config.update(toxicity_config)
    
    # Train SAC
    print(f"  Training SAC...")
    sac_agent = train_sac(base_config, total_steps=10_000, device=device)
    
    # Evaluate all policies
    all_results = {"SAC": [], "FIXED": [], "SKEW": []}
    
    for seed in eval_seeds:
        eval_config = base_config.copy()
        eval_config["seed"] = seed
        env = MarketMakingEnv(**eval_config)
        
        # SAC
        results = evaluate_policy(env, sac_agent, num_episodes=5, device=device)
        for r in results:
            r["seed"] = seed
        all_results["SAC"].extend(results)
        
        # FIXED
        results = evaluate_policy(env, fixed_policy, num_episodes=5)
        for r in results:
            r["seed"] = seed
        all_results["FIXED"].extend(results)
        
        # SKEW
        results = evaluate_policy(env, skew_policy, num_episodes=5)
        for r in results:
            r["seed"] = seed
        all_results["SKEW"].extend(results)
    
    return all_results


def summarize_results(results):
    """Compute summary statistics including CVaR."""
    summary = {}
    for policy, data in results.items():
        pnls = np.array([r["pnl"] for r in data])
        inv_ranges = np.array([r["inv_range"] for r in data])
        
        # CVaR at 10% (average of worst 10% outcomes)
        sorted_pnls = np.sort(pnls)
        cvar_10_idx = max(1, int(len(sorted_pnls) * 0.1))
        cvar_10 = np.mean(sorted_pnls[:cvar_10_idx])
        
        summary[policy] = {
            "pnl_mean": np.mean(pnls),
            "pnl_std": np.std(pnls),
            "pnl_min": np.min(pnls),
            "cvar_10": cvar_10,
            "inv_range_mean": np.mean(inv_ranges),
            "inv_range_std": np.std(inv_ranges),
        }
    return summary


def print_comparison_table(summaries, labels):
    """Print side-by-side comparison of different toxicity levels."""
    
    print("\n" + "=" * 90)
    print("COMPARISON: SAC vs FIXED vs SKEW UNDER VARYING TOXICITY")
    print("=" * 90)
    
    # Header
    header = f"{'Metric':<20}"
    for label in labels:
        header += f" | {label:^20}"
    print(header)
    print("-" * 90)
    
    policies = ["SAC", "FIXED", "SKEW"]
    metrics = [
        ("PnL Mean", "pnl_mean", "+.0f"),
        ("PnL Std", "pnl_std", ".0f"),
        ("PnL Min", "pnl_min", "+.0f"),
        ("CVaR 10%", "cvar_10", "+.0f"),
        ("Inv Range", "inv_range_mean", ".1f"),
    ]
    
    for policy in policies:
        print(f"\n{policy}:")
        for metric_name, metric_key, fmt in metrics:
            row = f"  {metric_name:<18}"
            for summary in summaries:
                val = summary[policy][metric_key]
                row += f" | {val:^20{fmt}}"
            print(row)


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    DEVICE = "cpu"
    EVAL_SEEDS = list(range(100, 110))
    
    print("=" * 70)
    print("STRESS TEST: TOXICITY SENSITIVITY ANALYSIS")
    print("=" * 70)
    
    # Define toxicity levels
    toxicity_levels = [
        {
            "name": "Light",
            "config": {
                "toxicity_prob": 1.0,
                "toxicity_duration": 10,
                "toxicity_strength": 0.15,
                "toxicity_drift": 0.05,
            }
        },
        {
            "name": "Medium",
            "config": {
                "toxicity_prob": 1.0,
                "toxicity_duration": 15,
                "toxicity_strength": 0.25,
                "toxicity_drift": 0.10,
            }
        },
        {
            "name": "Strong",
            "config": {
                "toxicity_prob": 1.0,
                "toxicity_duration": 20,
                "toxicity_strength": 0.35,
                "toxicity_drift": 0.15,
            }
        },
    ]
    
    all_summaries = []
    labels = []
    
    for level in toxicity_levels:
        print(f"\n{'='*70}")
        print(f"Running: {level['name']} Toxicity")
        print(f"  Config: {level['config']}")
        print(f"{'='*70}")
        
        results = run_experiment(level["config"], EVAL_SEEDS, DEVICE)
        summary = summarize_results(results)
        all_summaries.append(summary)
        labels.append(level["name"])
        
        # Print intermediate results
        print(f"\n  Results for {level['name']}:")
        print(f"  {'Policy':<8} {'PnL Mean':>10} {'PnL Std':>10} {'CVaR 10%':>10} {'InvRange':>10}")
        print(f"  {'-'*50}")
        for policy in ["SAC", "FIXED", "SKEW"]:
            s = summary[policy]
            print(f"  {policy:<8} {s['pnl_mean']:>+10.0f} {s['pnl_std']:>10.0f} {s['cvar_10']:>+10.0f} {s['inv_range_mean']:>10.1f}")
    
    # Final comparison
    print_comparison_table(all_summaries, labels)
    
    # Compute degradation
    print("\n" + "=" * 70)
    print("DEGRADATION ANALYSIS (Strong vs Light)")
    print("=" * 70)
    
    light = all_summaries[0]
    strong = all_summaries[2]
    
    print(f"\n{'Policy':<8} {'Light PnL':>12} {'Strong PnL':>12} {'Degradation':>12} {'Light InvR':>12} {'Strong InvR':>12}")
    print("-" * 72)
    
    for policy in ["SAC", "FIXED", "SKEW"]:
        light_pnl = light[policy]["pnl_mean"]
        strong_pnl = strong[policy]["pnl_mean"]
        degradation = (light_pnl - strong_pnl) / abs(light_pnl) * 100 if light_pnl != 0 else 0
        
        light_inv = light[policy]["inv_range_mean"]
        strong_inv = strong[policy]["inv_range_mean"]
        
        print(f"{policy:<8} {light_pnl:>+12.0f} {strong_pnl:>+12.0f} {degradation:>11.1f}% {light_inv:>12.1f} {strong_inv:>12.1f}")
    
    print("\n" + "=" * 70)
    print("CONCLUSION")
    print("=" * 70)
    
    # Find best policy under strong toxicity
    strong_summary = all_summaries[2]
    best_pnl_policy = max(strong_summary.keys(), key=lambda p: strong_summary[p]["pnl_mean"])
    best_risk_policy = min(strong_summary.keys(), key=lambda p: strong_summary[p]["inv_range_mean"])
    best_cvar_policy = max(strong_summary.keys(), key=lambda p: strong_summary[p]["cvar_10"])
    
    print(f"\nUnder STRONG toxicity:")
    print(f"  Best PnL:        {best_pnl_policy} ({strong_summary[best_pnl_policy]['pnl_mean']:+.0f})")
    print(f"  Best Risk:       {best_risk_policy} (inv_range={strong_summary[best_risk_policy]['inv_range_mean']:.1f})")
    print(f"  Best Tail Risk:  {best_cvar_policy} (CVaR 10%={strong_summary[best_cvar_policy]['cvar_10']:+.0f})")
    print("=" * 70)