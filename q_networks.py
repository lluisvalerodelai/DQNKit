import os
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


class DQN_base(nn.Module):
    def __init__(self, input_dims, n_actions, checkpoint_dir) -> None:
        super().__init__()
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.checkpoint_dir = checkpoint_dir

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"using device: {self.device}")

    def checkpoint(self, fname: str):
        print(f"saving model to {fname}")
        # load to state file in checkpoints dir
        save_path = os.path.join(self.checkpoint_dir, fname)
        torch.save(self.state_dict(), save_path)

    def load(self, fname: str):
        # load from state dict file (must be inside checkpoints dir)
        load_path = os.path.join(self.checkpoint_dir, fname)
        state_dict = torch.load(load_path)

        self.load_state_dict(state_dict)


class DQN_2dstates(DQN_base):
    def __init__(self, input_dims: tuple[int, ...], n_actions, checkpoint_dir) -> None:
        super().__init__(input_dims, n_actions, checkpoint_dir)

        self.conv1 = nn.Conv2d(input_dims[0], 64, 8, 4)
        self.conv2 = nn.Conv2d(64, 64, 4, 2)
        self.conv3 = nn.Conv2d(64, 64, 3)

        conv_output_size = self._calculate_conv_output_size(input_dims)
        self.fc1 = nn.Linear(conv_output_size, 512)
        self.fc2 = nn.Linear(512, self.n_actions)

        self.to(self.device)
        print(f"using device {self.device}")
        print(self)

    def forward(self, X):
        X = F.relu(self.conv1(X))
        X = F.relu(self.conv2(X))
        X = F.relu(self.conv3(X))
        X = torch.flatten(X, start_dim=1)
        X = F.relu(self.fc1(X))
        q_values = self.fc2(X)
        return q_values

    def _calculate_conv_output_size(self, input_dims):
        # calculate the number of features needed as input to linear from conv
        dummy = torch.zeros(1, *input_dims)

        out1 = self.conv1(dummy)
        out2 = self.conv2(out1)
        out3 = self.conv3(out2)

        return int(np.prod(out3.size()))


class DQN_1dstates(DQN_base):
    def __init__(self, input_dims, n_actions, checkpoint_dir) -> None:
        assert type(input_dims) == int
        super().__init__(input_dims, n_actions, checkpoint_dir)

        self.model = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, self.n_actions),
        )

        self.to(self.device)
        print(f"using device {self.device}")
        print(self)

    def forward(self, X):
        return self.model.forward(X)


class DuelingDQN_1dstates(DQN_base):
    def __init__(self, input_dims, n_actions, checkpoint_dir) -> None:
        assert type(input_dims) == int
        super().__init__(input_dims, n_actions, checkpoint_dir)

        self.feature_stream = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU()
        )

        self.value_stream = nn.Sequential(
            nn.Linear(128, 1),
        )

        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, n_actions)
        )

    def forward(self, X):
        features = self.feature_stream.forward(X)
        value = self.value_stream.forward(features)
        advantages = self.advantage_stream(features)
        advantage_anchor = torch.mean(advantages, dim=1, keepdim=True)

        return value + (advantages - advantage_anchor)
