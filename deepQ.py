from torch.utils.tensorboard.writer import SummaryWriter
from q_networks import DQN_1dstates, DQN_2dstates
from replay_buffers import ReplayBuffer
from policies import e_greedy_policy
from torch.optim import AdamW
from typing import Callable
from copy import deepcopy
import gymnasium as gym
import torch.nn as nn
import torch
import os

class deepQ_simple:

    # the simplest, no add-ons deepQ
    # target -> ~ E[Q(s, a) - r + \gamma Q(s', max_a)]

    def __init__(self, env_id, 
                 state_shape, n_actions, lr,
                 batch_size, buffer_size,
                 checkpoint_dir, base_name, log_dir,
                 min_num_batches=10,
                 policy : Callable = e_greedy_policy) -> None:

        self.lr = lr
        self.batch_size = batch_size
        self.buffer_size = buffer_size
        self.checkpoint_dir = checkpoint_dir
        self.run_name = base_name
        self.action_space = [action for action in range(n_actions)]
        self.policy = policy
        self.env_id = env_id
        self.env = gym.make(env_id)

        if type(state_shape) == int:
            self.q_network = DQN_1dstates(input_dims=state_shape, 
                                          n_actions=n_actions, 
                                          checkpoint_dir=checkpoint_dir)
        else:
            self.q_network = DQN_2dstates(input_dims=state_shape, 
                                          n_actions=n_actions, 
                                          checkpoint_dir=checkpoint_dir)

        self.target_network = deepcopy(self.q_network).to(self.q_network.device)
        self.target_network.load_state_dict(self.q_network.state_dict())
        self.target_network.eval()

        self.replay_buffer = ReplayBuffer(max_size=self.buffer_size, state_dim=state_shape, 
                                          n_actions=n_actions,       min_num_batches=min_num_batches)
        
        self.optimizer = AdamW(self.q_network.parameters(), lr=self.lr) 
        self.criterion = nn.MSELoss()

        self.log_dir = os.path.join(log_dir, self.run_name)
        
        self.writer = SummaryWriter(log_dir=self.log_dir)

    def train(self, n_episodes, batch_size,
              epsilon_start, epsilon_min, epsilon_decay,
              gamma, target_net_update_freq, polyak_average, 
              tau, checkpoint_freq):

        self.replay_buffer.reset()
        epsilon = epsilon_start
        
        global_frame = 0
        for ep in range(1, n_episodes + 1):
            state, _ = self.env.reset()
            ep_over = False
            ep_step = 0
            running_reward = 0

            while not ep_over:
                action = self.policy(state, self.q_network, epsilon=epsilon, action_space=self.action_space)
                next_state, reward, done, truncated, _ = self.env.step(action)
                self.replay_buffer.insert(state, action, reward, next_state, done)
                state = next_state

                global_frame += 1
                ep_step += 1
                running_reward += reward # pyright: ignore[]

                if done or truncated:
                    ep_over = True

                if self.replay_buffer.can_sample(batch_size):

                    if global_frame % 593 == 0:
                        loss = self.update_q(batch_size, gamma, track_q=True)
                    else:
                        loss = self.update_q(batch_size, gamma, track_q=False)

                    if global_frame % target_net_update_freq == 0:
                        if polyak_average:
                            for target_param, q_param in zip(self.target_network.parameters(), self.q_network.parameters()):
                                target_param.data.copy_(
                                    # we add t % of the main network
                                    tau * q_param.data + (1.0 - tau) * target_param.data
                                )
                        else:
                            self.target_network.load_state_dict(self.q_network.state_dict()) 

                    self.writer.add_scalar("Loss/loss_per_update", loss, global_frame)

                    total_transitions = self.replay_buffer.mem_idx
                    capacity = min(self.replay_buffer.mem_idx, self.replay_buffer.max_size)
                    self.writer.add_scalars("Buffer/buffer_size", { "transitions_stored" : total_transitions, "current_size" : capacity}, global_frame)




            epsilon = max(epsilon_min, epsilon * epsilon_decay)

            if ep % checkpoint_freq == 0:
                self.q_network.checkpoint(f"{self.run_name}_checkpoint{ep}")

            self.writer.add_scalar("EpisodeStats/ep_len", ep_step, ep)
            self.writer.add_scalar("Returns/cumulative_reward", running_reward, ep)
            self.writer.add_scalar("Hparams/epsilon", epsilon, ep)

            print(f"episode {ep} return {running_reward} epsilon {epsilon}")

    def update_q(self, batch_size, gamma, track_q = False):
        # do a single step based on the basic definition of q target
        
        # sample sars_d batch
        state_b, action_b, reward_b, next_state_b, done_b = self.replay_buffer.sample(batch_size)

        target_device = self.q_network.device
        state_b = torch.tensor(state_b, dtype=torch.float32, device=target_device)
        action_b = torch.tensor(action_b, dtype=torch.float32, device=target_device)
        reward_b = torch.tensor(reward_b, dtype=torch.float32, device=target_device)
        next_state_b = torch.tensor(next_state_b, dtype=torch.float32, device=target_device)
        done_b = torch.tensor(done_b, dtype=torch.float32, device=target_device)

        expected_q = self.q_network(state_b)
        expected_q = torch.gather(expected_q, 1, action_b.view(-1, 1).long())

        with torch.no_grad():
            future_q = self.target_network(next_state_b)
            future_q = torch.max(future_q, dim=1, keepdim=True)[0]

        # everything must be shape (1, dim)
        target_q = reward_b.view(-1, 1) + gamma * future_q * (1-done_b.view(-1, 1))

        if track_q:
            with torch.no_grad():
                q_target_mean = torch.mean(target_q).item()
                q_online_mean = torch.mean(expected_q).item()
                self.writer.add_scalars("QValues/mean_q_values", {"target" : q_target_mean, "online" : q_online_mean})

        self.optimizer.zero_grad()
        loss = self.criterion(expected_q, target_q)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def load_checkpoint(self, cpath):
        check_state_dict = torch.load(cpath)
        self.q_network.load_state_dict(check_state_dict)

    def visual_demo(self):
        r_env = gym.make(self.env_id, render_mode='human')

        state, info = r_env.reset()
        r_env.render()
        done = False
        total_r = 0

        while not done:
            action = e_greedy_policy(state, self.q_network, epsilon=0.01, action_space=[0, 1])
            next_state, reward, done, terminated, info = r_env.step(action)
            total_r += reward # pyright: ignore[reportOperatorIssue] weird...
            print(f"reward {reward} done/terminated {done | terminated} info {info}")
            if done | terminated:
                done = True
            state = next_state

        r_env.close()
        print(f"total reward: {total_r}")


    def record_demo(self, video_folder, demo_name=None):
        os.makedirs(video_folder, exist_ok=True)
        if not demo_name:
            demo_name = f"{self.run_name}_demo"

        env = gym.make(self.env_id, render_mode="rgb_array")

        env = gym.wrappers.RecordVideo(
        env,
        video_folder=video_folder,
        name_prefix=demo_name,
        episode_trigger=lambda episode_id: episode_id == 0 # Record the first episode (index 0)
        )

        state, _ = env.reset()
        done = False

        while not done:
            action = self.policy(state, q_network=self.q_network, epsilon=0, action_space=self.action_space)
            next_state, _, done, terminated, _ = env.step(action)

            if done or terminated:
                done = True

            state = next_state

        env.close()
        print(f"video demo saved to {os.path.join(video_folder, demo_name)}")
