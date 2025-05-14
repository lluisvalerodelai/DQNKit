import numpy as np
from utils import stratified_sampling

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

class PrioritizedReplayBuffer(ReplayBuffer):
    def __init__(self, max_size, state_dim, n_actions, alpha, epsilon, beta, min_num_batches=10):

        assert n_actions < 2**8
        assert alpha >= 0
        assert beta > 0
        assert epsilon >= 0

        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta

        self.max_size = max_size
        self.mem_idx = 0
        self.min_num_batches = min_num_batches

        if type(state_dim) == int:
            self.state_dim = (state_dim,)
        else:
            self.state_dim = state_dim

        # all the experience buffers
        self.states_buffer = np.zeros((max_size, *self.state_dim), dtype=np.float32)
        self.actions_buffer = np.zeros((max_size), dtype=np.int8)
        self.rewards_buffer = np.zeros((max_size), dtype=np.float32)
        self.next_states_buffer = np.zeros(
            (max_size, *self.state_dim), dtype=np.float32
        )
        self.dones_buffer = np.zeros((max_size), dtype=np.uint8)

        # the actual sum tree array
            # total leaf nodes: max_size (or capacity of our replay buffer)
            # inner nodes: max_size - 1
            # inner nodes range from indices 0 : max_size - 2
            # leaf nodes go from max_size - 1 : 2max_size - 2
        # at the start all transitions start with probability 0 (no epsilon included) so that they are never sampled
        self.sum_tree = np.zeros((2 * max_size - 1), dtype=np.float32)

    def insert(self, state, action, reward, next_state, done):

        index = self.mem_idx % self.max_size


        # --- Store the transition into the replay buffers ---#
        self.states_buffer[index] = state
        self.actions_buffer[index] = action
        self.rewards_buffer[index] = reward
        self.next_states_buffer[index] = next_state
        self.dones_buffer[index] = done


        self.mem_idx += 1

        # --- Store the transition into the sum tree --- #

        # translate index into sum tree index
        # in the sum tree indices 0 : max_size - 1 are reserved for leaf nodes, so we need to add max_size - 1 to the index
        sum_tree_index = index + (self.max_size - 1)

        old_priority = self.sum_tree[sum_tree_index]

        current_max_priority = np.max(self.sum_tree[self.max_size - 1:])
        new_priority = max(current_max_priority, 1)

        self.sum_tree[sum_tree_index] = new_priority

        # propagate priority change up the tree
        priority_change = new_priority - old_priority

        parent_index = (sum_tree_index - 1) // 2

        while parent_index >= 0:
            self.sum_tree[parent_index] += priority_change
            parent_index = (parent_index - 1) // 2

    def sample(self, batch_size): 

        assert self.can_sample(batch_size)
        
        if self.sum_tree[0] == 0:
            raise ValueError("No transitions in the replay buffer")

        # sample a number within our sum range to index sum tree
        chosen_priorities = stratified_sampling(self.sum_tree[0], batch_size)

        sum_tree_indices = []
        replay_buffer_indices = []

        for i in range(batch_size):

            # traverse the tree to find the leaf node that contains the chosen priority
            sum_tree_index = 0
            current_priority = chosen_priorities[i]

            # while we havent landed in our leaf node range, keep going down
            while sum_tree_index < self.max_size - 1:
                left_child_index = 2 * sum_tree_index + 1
                right_child_index = 2 * sum_tree_index + 2

                # if our chosen priority is less than the left child, go left. else, go right and change the range
                if current_priority <= self.sum_tree[left_child_index]:
                    sum_tree_index = left_child_index
                else:
                    # sums in the right tree are relative to the start of the previous segment
                    current_priority -= self.sum_tree[left_child_index]
                    sum_tree_index = right_child_index

            # translate the sum tree index into the index of the replay buffer
            replay_buffer_indices.append(sum_tree_index - (self.max_size - 1))
            sum_tree_indices.append(sum_tree_index)

        # calculate the importance sampling weights
        assert self.mem_idx > 0
        assert self.sum_tree[0] > 0

        # TODO: 2 things -> make sure none of the sampling probs were 0, and make sure that you normalize IS weights to avoid exploding gradients
        sampling_probs = np.divide(self.sum_tree[sum_tree_indices], self.sum_tree[0])
        if np.any(sampling_probs == 0):
            raise ValueError("Sampling probability was 0")

        # normalize IS weights
        IS_weights = (min(self.mem_idx, self.max_size) * sampling_probs) ** -self.beta
        IS_weights = IS_weights / np.max(IS_weights) if np.max(IS_weights) > 0 else np.zeros_like(IS_weights)
        IS_weights[np.isnan(IS_weights)] = 0
        IS_weights[np.isinf(IS_weights)] = 0

        # sample from the replay buffer, return the sum_tree index of the sampled priorities as well for updating
        states = self.states_buffer[replay_buffer_indices]
        actions = self.actions_buffer[replay_buffer_indices]
        rewards = self.rewards_buffer[replay_buffer_indices]
        next_states = self.next_states_buffer[replay_buffer_indices]
        dones = self.dones_buffer[replay_buffer_indices]

        return states, actions, rewards, next_states, dones, sum_tree_indices, IS_weights

    def update_priorities(self, priority_indices, td_errors): # TODO: implement batched updating

        abs_td = np.abs(td_errors) + self.epsilon
        new_priorities = abs_td ** self.alpha

        for priority_index, new_priority in zip(priority_indices, new_priorities):
            delta_priority = new_priority - self.sum_tree[priority_index]
            self.sum_tree[priority_index] = new_priority

        # propagate the change up the tree
            while priority_index > 0:
                # go the parent node
                priority_index = (priority_index - 1) // 2

                self.sum_tree[priority_index] += delta_priority

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

        self.sum_tree = np.zeros((2 * self.max_size - 1), dtype=np.float32)

        self.mem_idx = 0

class SumTree:
    def __init__(self, max_size, alpha, epsilon, beta):

        assert alpha >= 0
        assert beta > 0
        assert epsilon >= 0

        self.alpha = alpha
        self.epsilon = epsilon
        self.beta = beta
        self.max_size = max_size

        self.tree = np.zeros((2 * max_size - 1), dtype=np.float32)

    def insert(self, priority, index):
        sum_tree_index = index + (self.max_size - 1)

        old_priority = self.tree[sum_tree_index]
        current_max_priority = np.max(self.tree[self.max_size - 1:])
        new_priority = max(current_max_priority, 1)

        self.tree[sum_tree_index] = new_priority

        delta_priority = new_priority - old_priority

        parent_index = (sum_tree_index - 1) // 2

        while parent_index >= 0:
            self.tree[parent_index] += delta_priority
            parent_index = (parent_index - 1) // 2

    def sample(self, batch_size):
        # sample a number within our sum range to index sum tree
        # return an index for the replay buffer (not sum tree index)

        chosen_priority = np.random.uniform(0, self.tree[0])

        sum_tree_index = 0

        while sum_tree_index < self.max_size - 1:
            left_child_index = 2 * sum_tree_index + 1
            right_child_index = 2 * sum_tree_index + 2

            if chosen_priority <= self.tree[left_child_index]:
                sum_tree_index = left_child_index
            else:
                chosen_priority -= self.tree[left_child_index]
                sum_tree_index = right_child_index

        return sum_tree_index - (self.max_size - 1)