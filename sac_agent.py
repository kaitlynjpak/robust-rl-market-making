# sac_agent.py

import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np


# =============================================================================
# Networks (from previous step)
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
# Replay Buffer (from previous step)
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

        # Main networks
        self.actor = Actor(state_dim, action_dim, hidden_dim).to(device)
        self.critic = DoubleCritic(state_dim, action_dim, hidden_dim).to(device)

        # Target critics (exact copy)
        self.target_critic = copy.deepcopy(self.critic).to(device)

        # Optimizers
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)

    def select_action(self, state, deterministic: bool = False):
        """Select action for environment interaction."""
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            action, _, mean_action = self.actor.sample(state_t)
            if deterministic:
                return mean_action.squeeze(0).cpu().numpy()
            return action.squeeze(0).cpu().numpy()

    def update(self, replay_buffer, batch_size: int):
        """One SAC update step."""
        
        # 1. Sample batch
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

        # 2. Compute target Q-values
        with torch.no_grad():
            next_actions, next_log_prob, _ = self.actor.sample(next_states)
            target_q1, target_q2 = self.target_critic(next_states, next_actions)
            target_q_min = torch.min(target_q1, target_q2)
            target_q = rewards + (1.0 - dones) * self.gamma * (
                target_q_min - self.alpha * next_log_prob
            )

        # 3. Critic update
        current_q1, current_q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # 4. Actor update
        new_actions, log_prob, _ = self.actor.sample(states)
        q1_pi, q2_pi = self.critic(states, new_actions)
        q_pi_min = torch.min(q1_pi, q2_pi)
        actor_loss = (self.alpha * log_prob - q_pi_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # 5. Soft target update
        self.soft_update(self.critic, self.target_critic, self.tau)

        return {
            "critic_loss": critic_loss.item(),
            "actor_loss": actor_loss.item(),
            "avg_q": q_pi_min.mean().item(),
            "avg_log_prob": log_prob.mean().item(),
        }

    @staticmethod
    def soft_update(source, target, tau: float):
        for src_param, tgt_param in zip(source.parameters(), target.parameters()):
            tgt_param.data.copy_(tau * src_param.data + (1.0 - tgt_param.data) * tgt_param.data)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    STATE_DIM = 6
    ACTION_DIM = 4
    BATCH_SIZE = 128
    CAPACITY = 10000

    # Create buffer and fill with fake data
    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, CAPACITY)
    for _ in range(500):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        done = float(np.random.rand() < 0.01)
        buffer.add(state, action, reward, next_state, done)

    # Create agent
    agent = SACAgent(STATE_DIM, ACTION_DIM)

    # Run a few update steps
    print("=== SAC Dry Run ===")
    for i in range(5):
        info = agent.update(buffer, BATCH_SIZE)
        print(f"Step {i+1}: {info}")

    print("\nSuccess!")