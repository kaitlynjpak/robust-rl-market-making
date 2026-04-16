# train_sac.py

import numpy as np
import torch
import copy
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

# Import the C++ environment
from lob_env import MarketMakingEnv


# =============================================================================
# Networks
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


# =============================================================================
# Replay Buffer
# =============================================================================

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


# =============================================================================
# SAC Agent
# =============================================================================

class SACAgent:
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dim: int = 128,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        device: str = "cpu",
    ):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.alpha = alpha

        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def update(self, replay_buffer, batch_size: int):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q_min = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * (
                target_q_min - self.alpha * next_log_prob
            )

        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        new_actions, log_prob, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        q_pi_min = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - q_pi_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # Fixed soft update
        for src_param, tgt_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            tgt_param.data.copy_(self.tau * src_param.data + (1.0 - self.tau) * tgt_param.data)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "avg_q": q_pi_min.mean().item(),
            "avg_log_prob": log_prob.mean().item(),
        }


# =============================================================================
# Evaluation
# =============================================================================

def evaluate_policy(env, agent, num_episodes=3, device="cpu"):
    rewards = []
    pnls = []
    final_inventories = []

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        truncated = False
        ep_reward = 0.0
        last_info = {}

        while not (done or truncated):
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                _, _, mean_action = agent.actor.sample(state_tensor)
            action = mean_action.squeeze(0).cpu().numpy()
            next_state, reward, done, truncated, info = env.step(action)
            ep_reward += reward
            state = next_state
            last_info = info

        rewards.append(ep_reward)
        pnls.append(float(last_info.get("pnl", 0.0)))
        final_inventories.append(float(last_info.get("inventory", 0.0)))

    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_pnl_mean": float(np.mean(pnls)),
        "eval_inventory_mean": float(np.mean(final_inventories)),
    }


# =============================================================================
# Training Loop
# =============================================================================

def train_sac(
    env,
    agent,
    replay_buffer,
    total_steps=50_000,
    warmup_steps=5_000,
    batch_size=128,
    eval_interval=2_000,
    device="cpu",
):
    state = env.reset()
    episode_reward = 0.0
    episode_steps = 0
    episode_index = 0

    train_logs = []
    eval_logs = []

    for step in range(total_steps):
        # Choose action
        if step < warmup_steps:
            action = np.random.uniform(-1.0, 1.0, size=(4,)).astype(np.float32)
        else:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            with torch.no_grad():
                sampled_action, _, _ = agent.actor.sample(state_tensor)
            action = sampled_action.squeeze(0).cpu().numpy().astype(np.float32)

        # Step environment
        next_state, reward, done, truncated, info = env.step(action)

        # Store transition
        replay_buffer.add(state, action, reward, next_state, bool(done or truncated))

        episode_reward += reward
        episode_steps += 1

        # SAC update
        update_info = None
        if step >= warmup_steps and len(replay_buffer) >= batch_size:
            update_info = agent.update(replay_buffer, batch_size)

        state = next_state

        # Episode end
        if done or truncated:
            log_entry = {
                "global_step": step,
                "episode": episode_index,
                "episode_reward": episode_reward,
                "episode_steps": episode_steps,
                "pnl": float(info.get("pnl", 0.0)),
                "inventory": float(info.get("inventory", 0.0)),
            }
            if update_info:
                log_entry.update(update_info)

            train_logs.append(log_entry)
            print(f"[train] step={step} ep={episode_index} reward={episode_reward:.2f} pnl={log_entry['pnl']:.2f} inv={log_entry['inventory']:.0f}")

            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0
            episode_index += 1

        # Evaluation
        if (step + 1) % eval_interval == 0:
            eval_info = evaluate_policy(env, agent, num_episodes=3, device=device)
            eval_info["global_step"] = step + 1
            eval_logs.append(eval_info)
            print(f"[eval] step={step+1} reward={eval_info['eval_reward_mean']:.2f} pnl={eval_info['eval_pnl_mean']:.2f} inv={eval_info['eval_inventory_mean']:.1f}")

            state = env.reset()
            episode_reward = 0.0
            episode_steps = 0

            
    return train_logs, eval_logs


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    STATE_DIM = 6
    ACTION_DIM = 4
    DEVICE = "cpu"

    # Short test run
    env = MarketMakingEnv(
        seed=42,
        episode_duration=10.0,
        decision_interval=0.1,
        inventory_penalty=0.01,
        regime_mode="high",
        toxicity_drift=0.05,
    )

    agent = SACAgent(STATE_DIM, ACTION_DIM, device=DEVICE)
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, capacity=100_000, device=DEVICE)

    # Very short test
    train_logs, eval_logs = train_sac(
        env=env,
        agent=agent,
        replay_buffer=buffer,
        total_steps=500,
        warmup_steps=100,
        batch_size=64,
        eval_interval=250,
        device=DEVICE,
    )

    print("\n=== Done ===")
    print(f"Episodes completed: {len(train_logs)}")
    print(f"Eval checkpoints: {len(eval_logs)}")