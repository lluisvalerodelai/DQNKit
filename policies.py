import torch
import numpy as np
import random

def e_greedy_policy(state, q_network, epsilon, action_space):
    # do not use this policy for batched state processing
    if type(state) == np.ndarray:
        state = torch.tensor(state)

    state = torch.unsqueeze(state, 0)

    state.to(q_network.device)

    if random.random() < epsilon:
        action = random.choice(action_space)
    else:
        q_values = q_network(state)
        q_greedy = torch.argmax(q_values, 1, keepdim=True)
        action = q_greedy.item()

    return action
