# sac_networks.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal


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
        log_std = self.log_std_head(x)
        log_std = torch.clamp(log_std, self.LOG_STD_MIN, self.LOG_STD_MAX)

        return mean, log_std

    def sample(self, state):
        mean, log_std = self.forward(state)
        std = log_std.exp()

        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        mean_action = torch.tanh(mean)

        # Log prob with tanh correction
        log_prob = dist.log_prob(z)
        log_prob = log_prob - torch.log(1 - action.pow(2) + 1e-6)
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
# Test
# =============================================================================
if __name__ == "__main__":
    STATE_DIM = 6
    ACTION_DIM = 4
    BATCH_SIZE = 128

    actor = Actor(STATE_DIM, ACTION_DIM)
    critic = DoubleCritic(STATE_DIM, ACTION_DIM)

    # Dummy inputs
    states = torch.randn(BATCH_SIZE, STATE_DIM)
    actions = torch.randn(BATCH_SIZE, ACTION_DIM)

    # Actor test
    mean, log_std = actor(states)
    action, log_prob, mean_action = actor.sample(states)

    print("=== Actor ===")
    print(f"mean:        {mean.shape}")
    print(f"log_std:     {log_std.shape}")
    print(f"action:      {action.shape}")
    print(f"log_prob:    {log_prob.shape}")
    print(f"mean_action: {mean_action.shape}")

    # Critic test
    q1, q2 = critic(states, actions)

    print("\n=== Critic ===")
    print(f"q1: {q1.shape}")
    print(f"q2: {q2.shape}")

    print("\nSuccess!")