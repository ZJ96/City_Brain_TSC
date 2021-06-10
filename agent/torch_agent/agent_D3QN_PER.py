

import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0]
import sys
import math

sys.path.append(path)
import random

import gym
from typing import Dict, List, Tuple

from pathlib import Path
import pickle
import gym

import os
from collections import deque

import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from segment_tree import MinSegmentTree, SumSegmentTree

# class Network(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int):
#         """Initialization."""
#         super(Network, self).__init__()
#
#         self.layers = nn.Sequential(
#             nn.Linear(in_dim, 128),
#             nn.ReLU(),
#             nn.Linear(128, 128),
#             nn.ReLU(),
#             nn.Linear(128, out_dim)
#         )
#
#     def forward(self, x):
#         return self.layers(x)

class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32,n_step: int = 3):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0


    def store(self,obs: np.ndarray,act: np.ndarray,rew: float,next_obs: np.ndarray):

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)


    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs]
                    )

    def __len__(self) -> int:
        return self.size

class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized Replay buffer.

    Attributes:
        max_priority (float): max priority
        tree_ptr (int): next index of tree
        alpha (float): alpha parameter for prioritized replay buffer
        sum_tree (SumSegmentTree): sum tree for prior
        min_tree (MinSegmentTree): min tree for min prior to get max weight

    """

    def __init__(
            self,
            obs_dim: int,
            size: int,
            batch_size: int = 32,
            alpha: float = 0.6
    ):
        """Initialization."""
        assert alpha >= 0

        super(PrioritizedReplayBuffer, self).__init__(obs_dim, size, batch_size)
        self.max_priority, self.tree_ptr = 1.0, 0
        self.alpha = alpha

        # capacity must be positive and a power of 2.
        tree_capacity = 1
        while tree_capacity < self.max_size:
            tree_capacity *= 2

        self.sum_tree = SumSegmentTree(tree_capacity)
        self.min_tree = MinSegmentTree(tree_capacity)

    def store(self,obs: np.ndarray, act: int,rew: float,next_obs: np.ndarray):
        """Store experience and priority."""
        super().store(obs, act, rew, next_obs)

        self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
        self.tree_ptr = (self.tree_ptr + 1) % self.max_size

    def sample_batch(self, beta: float = 0.4) -> Dict[str, np.ndarray]:
        """Sample a batch of experiences."""
        assert len(self) >= self.batch_size
        assert beta > 0

        indices = self._sample_proportional()

        obs = self.obs_buf[indices]
        next_obs = self.next_obs_buf[indices]
        acts = self.acts_buf[indices]
        rews = self.rews_buf[indices]
        weights = np.array([self._calculate_weight(i, beta) for i in indices])

        return dict(
            obs=obs,
            next_obs=next_obs,
            acts=acts,
            rews=rews,
            weights=weights,
            indices=indices,
        )

    def update_priorities(self, indices: List[int], priorities: np.ndarray):
        """Update priorities of sampled transitions."""
        assert len(indices) == len(priorities)

        for idx, priority in zip(indices, priorities):
            assert priority > 0
            assert 0 <= idx < len(self)

            self.sum_tree[idx] = priority ** self.alpha
            self.min_tree[idx] = priority ** self.alpha

            self.max_priority = max(self.max_priority, priority)

    def _sample_proportional(self) -> List[int]:
        """Sample indices based on proportions."""
        indices = []
        p_total = self.sum_tree.sum(0, len(self) - 1)
        segment = p_total / self.batch_size

        for i in range(self.batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            indices.append(idx)

        return indices

    def _calculate_weight(self, idx: int, beta: float):
        """Calculate the weight of the experience at idx."""
        # get max weight
        p_min = self.min_tree.min() / self.sum_tree.sum()
        max_weight = (p_min * len(self)) ** (-beta)

        # calculate weights
        p_sample = self.sum_tree[idx] / self.sum_tree.sum()
        weight = (p_sample * len(self)) ** (-beta)
        weight = weight / max_weight

        return weight

class NoisyLinear(nn.Module):
    """Noisy linear module for NoisyNet.

    Attributes:
        in_features (int): input size of linear module
        out_features (int): output size of linear module
        std_init (float): initial std value
        weight_mu (nn.Parameter): mean value weight parameter
        weight_sigma (nn.Parameter): std value weight parameter
        bias_mu (nn.Parameter): mean value bias parameter
        bias_sigma (nn.Parameter): std value bias parameter

    """

    def __init__(self, in_features: int, out_features: int, std_init: float = 0.5):
        """Initialization."""
        super(NoisyLinear, self).__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.std_init = std_init

        self.weight_mu = nn.Parameter(torch.Tensor(out_features, in_features))
        self.weight_sigma = nn.Parameter(
            torch.Tensor(out_features, in_features)
        )
        self.register_buffer(
            "weight_epsilon", torch.Tensor(out_features, in_features)
        )

        self.bias_mu = nn.Parameter(torch.Tensor(out_features))
        self.bias_sigma = nn.Parameter(torch.Tensor(out_features))
        self.register_buffer("bias_epsilon", torch.Tensor(out_features))

        self.reset_parameters()
        self.reset_noise()

    def reset_parameters(self):
        """Reset trainable network parameters (factorized gaussian noise)."""
        mu_range = 1 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-mu_range, mu_range)
        self.weight_sigma.data.fill_(
            self.std_init / math.sqrt(self.in_features)
        )
        self.bias_mu.data.uniform_(-mu_range, mu_range)
        self.bias_sigma.data.fill_(
            self.std_init / math.sqrt(self.out_features)
        )

    def reset_noise(self):
        """Make new noise."""
        epsilon_in = self.scale_noise(self.in_features)
        epsilon_out = self.scale_noise(self.out_features)

        # outer product
        self.weight_epsilon.copy_(epsilon_out.ger(epsilon_in))
        self.bias_epsilon.copy_(epsilon_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation.

        We don't use separate statements on train / eval mode.
        It doesn't show remarkable difference of performance.
        """
        return F.linear(
            x,
            self.weight_mu + self.weight_sigma * self.weight_epsilon,
            self.bias_mu + self.bias_sigma * self.bias_epsilon,
        )

    @staticmethod
    def scale_noise(size: int) -> torch.Tensor:
        """Set scale to make noise (factorized gaussian noise)."""
        x = torch.FloatTensor(np.random.normal(loc=0.0, scale=1.0, size=size))

        return x.sign().mul(x.abs().sqrt())

# class Network(nn.Module):
#     def __init__(self, in_dim: int, out_dim: int):
#         """Initialization."""
#         super(Network, self).__init__()
#
#         self.feature = nn.Linear(in_dim, 128)
#         self.noisy_layer1 = NoisyLinear(128, 128)
#         self.noisy_layer2 = NoisyLinear(128, out_dim)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Forward method implementation."""
#         feature = F.relu(self.feature(x))
#         hidden = F.relu(self.noisy_layer1(feature))
#         out = self.noisy_layer2(hidden)
#         return out
#
#     def reset_noise(self):
#         """Reset all noisy layers."""
#         self.noisy_layer1.reset_noise()
#         self.noisy_layer2.reset_noise()


class Network(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        """Initialization."""
        super(Network, self).__init__()
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        # set advantage layer
        self.noisy_layer1 = NoisyLinear(128, 64)
        self.noisy_layer2 = NoisyLinear(64, out_dim)

        # set value layer
        self.value_layer = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        feature = self.feature_layer(x)

        value = self.value_layer(feature)
        hidden = F.relu(self.noisy_layer1(feature))
        advantage = self.noisy_layer2(hidden)

        # advantage = self.advantage_layer(feature)
        q = value + advantage - advantage.mean(dim=-1, keepdim=True)

        return q

    def reset_noise(self):
        """Reset all noisy layers."""
        self.noisy_layer1.reset_noise()
        self.noisy_layer2.reset_noise()


class TestAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        #self.memory = deque(maxlen=20)
        self.memory_size = 32
        self.learning_start = self.memory_size
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.1  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.ob_length = 24
        self.action_space = 8
        #self.memory = ReplayBuffer(self.ob_length,size = self.memory_size ,batch_size=self.batch_size)

        # PER
        # In DQN, We used "ReplayBuffer(obs_dim, memory_size, batch_size)"
        # PER parameters
        self.alpha = 0.2
        self.beta = 0.6
        self.prior_eps = 1e-6
        self.memory = PrioritizedReplayBuffer(self.ob_length, self.memory_size, self.batch_size, self.alpha )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = Network(self.ob_length, self.action_space).to(self.device)
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = Network(self.ob_length, self.action_space).to(self.device)
        self.update_target_network()
        self.target_model.eval()

        # optimizer
        self.optimizer = optim.RMSprop(self.model.parameters(),lr=self.learning_rate)

        # self.model = self._build_model()
        # # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # # path = os.path.split(os.path.realpath(__file__))[0]
        # # self.load_model(path, 99)
        # self.target_model = self._build_model()
        # self.update_target_network()

    ################################
    # don't modify this function.
    # agent_list is a list of agent_id
    def load_agent_list(self, agent_list):
        self.agent_list = agent_list
        self.now_phase = dict.fromkeys(self.agent_list, 1)
        self.last_change_step = dict.fromkeys(self.agent_list, 0)

    # intersections[key_id] = {
    #     'have_signal': bool,
    #     'end_roads': list of road_id. Roads that end at this intersection. The order is random.
    #     'start_roads': list of road_id. Roads that start at this intersection. The order is random.
    #     'lanes': list, contains the lane_id in. The order is explained in Docs.
    # }
    # roads[road_id] = {
    #     'start_inter':int. Start intersection_id.
    #     'end_inter':int. End intersection_id.
    #     'length': float. Road length.
    #     'speed_limit': float. Road speed limit.
    #     'num_lanes': int. Number of lanes in this road.
    #     'inverse_road':  Road_id of inverse_road.
    #     'lanes': dict. roads[road_id]['lanes'][lane_id] = list of 3 int value. Contains the Steerability of lanes.
    #               lane_id is road_id*100 + 0/1/2... For example, if road 9 have 3 lanes, then their id are 900, 901, 902
    # }
    # agents[agent_id] = list of length 8. contains the inroad0_id, inroad1_id, inroad2_id,inroad3_id, outroad0_id, outroad1_id, outroad2_id, outroad3_id
    def load_roadnet(self, intersections, roads, agents):
        self.intersections = intersections
        self.roads = roads
        self.agents = agents

    ################################

    def act_(self, observations_for_agent):
        # Instead of override, We use another act_() function for training,
        # while keep the original act() function for evaluation unchanged.

        actions = {}
        for agent_id in self.agent_list:
            action = self.get_action(observations_for_agent[agent_id]['lane'])
            actions[agent_id] = action
        return actions

    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key[key.find('_') + 1:]
            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            observations_for_agent[observations_agent_id][observations_feature] = val[1:]

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions

    def get_action(self, ob):
        # The epsilon-greedy action selector.
        # if np.random.rand() <= self.epsilon:
        #     return np.random.randint(0, self.action_space)
        ob = self._reshape_ob(ob)
        act_values = self.model(torch.FloatTensor(ob).to(self.device)).argmax()
        act_values = act_values.detach().cpu().numpy().tolist()
        # act_values = self.model.predict([ob])
        # return np.argmax(act_values[0])
        return act_values

    # def _build_model(self):
    #
    #     # Neural Net for Deep-Q learning Model
    #
    #     model = Sequential()
    #     model.add(Dense(20, input_dim=self.ob_length, activation='relu'))
    #     # model.add(Dense(20, activation='relu'))
    #     model.add(Dense(self.action_space, activation='linear'))
    #     model.compile(
    #         loss='mse',
    #         optimizer=RMSprop()
    #     )
    #     return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, ob, action, reward, next_ob):
        #self.memory.append((ob, action, reward, next_ob))
        self.memory.store(ob,action,reward,next_ob)

    def replay(self):
        # Update the Q network from the memory buffer.
        # if self.batch_size > len(self.memory):
        #     minibatch = self.memory
        # else:
        #     minibatch = random.sample(self.memory, self.batch_size)
        #
        # obs, actions, rewards, next_obs, = [np.stack(x) for x in np.array(minibatch).T]
        # obs = torch.FloatTensor(obs).to(self.device)
        # actions = torch.LongTensor(actions.reshape(-1, 1)).to(self.device)
        # rewards =torch.FloatTensor(rewards.reshape(-1, 1)).to(self.device)
        # next_obs = torch.FloatTensor(next_obs).to(self.device)
        # samples = self.memory.sample_batch()
        samples = self.memory.sample_batch(self.beta)
        weights = torch.FloatTensor(samples["weights"].reshape(-1, 1)).to(self.device)
        indices = samples["indices"]

        device = self.device  # for shortening the following lines
        obs = torch.FloatTensor(samples["obs"]).to(device)
        next_obs = torch.FloatTensor(samples["next_obs"]).to(device)
        actions = torch.LongTensor(samples["acts"].reshape(-1, 1)).to(device)
        rewards = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)

        curr_q_value = self.model(obs).gather(1,actions)
        # next_q_value = self.target_model(next_obs).max(dim=1, keepdim=True)[0].detach()
        next_q_value = self.target_model(next_obs).gather(1, self.model(next_obs).argmax(dim=1, keepdim=True)).detach()
        target = (rewards + self.gamma * next_q_value).to(self.device)

        # calculate dqn loss
        elementwise_loss = F.smooth_l1_loss(curr_q_value, target,reduction="none")

        loss = torch.mean(elementwise_loss * weights)

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.model.parameters(), 10.0)
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.model.reset_noise()
        self.target_model.reset_noise()
        # self.optimizer.zero_grad()
        # loss.backward()
        # self.optimizer.step()
        #
        # self.model.reset_noise()
        # self.target_model.reset_noise()
        # target = rewards + self.gamma * np.max(self.target_model.predict([next_obs]), axis=1)
        # target_f = self.model.predict([obs])
        # for i, action in enumerate(actions):
        #     target_f[i][action] = target[i]
        # self.model.fit([obs], target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()


    def load_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        print("load from " + model_name)
        self.model.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/dqn", step=0):
        name = "dqn_agent_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        torch.save(self.model.state_dict(),model_name)


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

