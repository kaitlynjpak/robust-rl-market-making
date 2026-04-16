# replay_buffer.py

import numpy as np
import torch


class ReplayBuffer:
    def __init__(self, state_dim: int, action_dim: int, capacity: int, device: str = "cpu"):
        """
        Replay buffer for off-policy RL (SAC).

        Stores transitions of the form:
            (state, action, reward, next_state, done)

        Shapes:
            state:      (state_dim,)
            action:     (action_dim,)
            reward:     scalar
            next_state: (state_dim,)
            done:       scalar (0.0 or 1.0)
        """
        self.capacity = capacity
        self.device = device

        # Preallocate arrays for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        # Ring buffer pointer
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        """
        Add one transition to the buffer.

        Args:
            state:      shape (state_dim,)
            action:     shape (action_dim,) -- normalized action in [-1, 1]
            reward:     scalar float
            next_state: shape (state_dim,)
            done:       bool or float
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)

        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int):
        """
        Randomly sample a batch of transitions.
        Returns PyTorch tensors on the chosen device.
        """
        if self.size < batch_size:
            raise ValueError(f"Not enough samples: {self.size} < {batch_size}")

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
# Test
# =============================================================================
if __name__ == "__main__":
    STATE_DIM = 6
    ACTION_DIM = 4
    CAPACITY = 10000
    BATCH_SIZE = 128

    buffer = ReplayBuffer(STATE_DIM, ACTION_DIM, CAPACITY)

    # Add fake transitions
    for i in range(500):
        state = np.random.randn(STATE_DIM).astype(np.float32)
        action = np.random.uniform(-1, 1, ACTION_DIM).astype(np.float32)
        reward = np.random.randn()
        next_state = np.random.randn(STATE_DIM).astype(np.float32)
        done = float(i % 100 == 99)  # done every 100 steps
        buffer.add(state, action, reward, next_state, done)

    print(f"Buffer size: {len(buffer)}")

    # Sample a batch
    states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)

    print(f"states:      {states.shape}")
    print(f"actions:     {actions.shape}")
    print(f"rewards:     {rewards.shape}")
    print(f"next_states: {next_states.shape}")
    print(f"dones:       {dones.shape}")
    print("Success!")
    