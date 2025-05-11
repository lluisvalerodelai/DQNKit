import numpy as np

class ReplayBuffer:

    # all buffers are stored as fixed size numpy arrs
    # shapes:
    #  - states & next_states: (max_size, *state_dim)
    #  - everything else: (max_size,)
    #  - actions & terminals are stored as int8, everything else as float32
    #  - if youre in need of more vram, change the sizes of dtypes in replay buffer, 
    #    as the replay buffer is often the culprit for high usage 
    #    *only when working high dimensional state spaces, otherwise its always the q networks

    def __init__(self, max_size, state_dim, n_actions, min_num_batches=10) -> None:

        # actions are stored as int8s for better gpu vram usage, and scaled as needed
        assert n_actions < 2**8

        self.max_size = max_size
        self.mem_idx = 0
        self.min_num_batches = min_num_batches

        if type(state_dim) == int:
            self.state_dim = (state_dim,)
        else:
            self.state_dim = state_dim

        self.states_buffer = np.zeros((max_size, *self.state_dim), dtype=np.float32)
        self.actions_buffer = np.zeros((max_size), dtype=np.int8)
        self.rewards_buffer = np.zeros((max_size), dtype=np.float32)
        self.next_states_buffer = np.zeros(
            (max_size, *self.state_dim), dtype=np.float32
        )
        self.dones_buffer = np.zeros((max_size), dtype=np.uint8)

    def insert(self, state, action, reward, next_state, done):
        index = self.mem_idx % self.max_size
        self.states_buffer[index] = state
        self.actions_buffer[index] = action
        self.rewards_buffer[index] = reward
        self.next_states_buffer[index] = next_state
        self.dones_buffer[index] = done
        self.mem_idx += 1

    def sample(self, batch_size):
        mem_range = min(self.mem_idx, self.max_size)
        indexes = np.random.choice(mem_range, batch_size, replace=False)
        states = self.states_buffer[indexes]
        actions = self.actions_buffer[indexes]
        rewards = self.rewards_buffer[indexes]
        next_states = self.next_states_buffer[indexes]
        dones = self.dones_buffer[indexes]
        return states, actions, rewards, next_states, dones

    def can_sample(self, batch_size):
        if self.mem_idx < batch_size * self.min_num_batches:
            return False
        return True

    def reset(self):
        self.states_buffer = np.zeros((self.max_size, *self.state_dim), dtype=np.float32)
        self.actions_buffer = np.zeros((self.max_size), dtype=np.int8)
        self.rewards_buffer = np.zeros((self.max_size), dtype=np.float32)
        self.next_states_buffer = np.zeros(
            (self.max_size, *self.state_dim), dtype=np.float32
        )
        self.dones_buffer = np.zeros((self.max_size), dtype=np.uint8)

        self.mem_idx = 0

class PrioritizedReplayBuffer:
    def __init__(self, max_size, state_dim, n_actions, min_num_batches=10) -> None:

        # actions are stored as int8s for better gpu vram usage, and scaled as needed
        assert n_actions < 2**8

        self.max_size = max_size
        self.mem_idx = 0
        self.min_num_batches = min_num_batches

        if type(state_dim) == int:
            self.state_dim = (state_dim,)
        else:
            self.state_dim = state_dim

        self.states_buffer = np.zeros((max_size, *self.state_dim), dtype=np.float32)
        self.actions_buffer = np.zeros((max_size), dtype=np.int8)
        self.rewards_buffer = np.zeros((max_size), dtype=np.float32)
        self.next_states_buffer = np.zeros(
            (max_size, *self.state_dim), dtype=np.float32
        )
        self.dones_buffer = np.zeros((max_size), dtype=np.uint8)
