import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0]
import sys
import math

sys.path.append(path)
import random

import gym
from typing import Dict, List, Tuple,Deque

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

class ReplayBuffer:
    def __init__(self, obs_dim: int, size: int, batch_size: int = 32, n_step: int = 1,gamma: float = 0.99):
        self.obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.next_obs_buf = np.zeros([size, obs_dim], dtype=np.float32)
        self.acts_buf = np.zeros([size], dtype=np.float32)
        self.rews_buf = np.zeros([size], dtype=np.float32)
        self.max_size, self.batch_size = size, batch_size
        self.ptr, self.size, = 0, 0

        # for N-step Learning
        self.n_step_buffer = deque(maxlen=n_step)
        self.n_step = n_step
        self.gamma = gamma


    def store(self,obs: np.ndarray,act: np.ndarray,rew: float,next_obs: np.ndarray):
        transition = (obs, act, rew, next_obs)
        self.n_step_buffer.append(transition)

        # single step transition is not ready
        if len(self.n_step_buffer) < self.n_step:
            return ()

        # make a n-step transition
        rew, next_obs = self._get_n_step_info(
            self.n_step_buffer, self.gamma
        )
        obs, act = self.n_step_buffer[0][:2]

        self.obs_buf[self.ptr] = obs
        self.next_obs_buf[self.ptr] = next_obs
        self.acts_buf[self.ptr] = act
        self.rews_buf[self.ptr] = rew
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

        return self.n_step_buffer[0]


    def sample_batch(self) -> Dict[str, np.ndarray]:
        idxs = np.random.choice(self.size, size=self.batch_size, replace=False)
        return dict(obs=self.obs_buf[idxs],
                    next_obs=self.next_obs_buf[idxs],
                    acts=self.acts_buf[idxs],
                    rews=self.rews_buf[idxs],
                    # for N-step Learning
                    indices = idxs,
                    )

    def sample_batch_from_idxs( self, idxs: np.ndarray) -> Dict[str, np.ndarray]:
        # for N-step Learning
        return dict(
            obs=self.obs_buf[idxs],
            next_obs=self.next_obs_buf[idxs],
            acts=self.acts_buf[idxs],
            rews=self.rews_buf[idxs],
        )

    def _get_n_step_info(self, n_step_buffer: Deque, gamma: float) -> Tuple[np.int64, np.ndarray, bool]:
        """Return n step rew, next_obs, and done."""
        # info of the last transition
        rew, next_obs = n_step_buffer[-1][-2:]

        for transition in reversed(list(n_step_buffer)[:-1]):
            r, n_o = transition[-2:]

            rew = r + gamma * rew
            #next_obs, done = (n_o, d) if d else (next_obs, done)

        return rew, next_obs

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
            alpha: float = 0.6,
            n_step: int = 1,
            gamma: float = 0.99,
    ):
        """Initialization."""
        assert alpha >= 0
        super(PrioritizedReplayBuffer, self).__init__(
            obs_dim, size, batch_size, n_step, gamma
        )
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
        transition = super().store(obs, act, rew, next_obs)

        if transition:
            self.sum_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.min_tree[self.tree_ptr] = self.max_priority ** self.alpha
            self.tree_ptr = (self.tree_ptr + 1) % self.max_size
        return transition

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

class Network(nn.Module):
    def __init__(
            self,
            in_dim: int,
            out_dim: int,
            atom_size: int,
            support: torch.Tensor
    ):
        """Initialization."""
        super(Network, self).__init__()

        self.support = support
        self.out_dim = out_dim
        self.atom_size = atom_size

        # set common feature layer
        self.feature_layer = nn.Sequential(
            nn.Linear(in_dim, 64),
            nn.ReLU(),
        )

        # set advantage layer
        self.advantage_hidden_layer = NoisyLinear(64, 64)
        self.advantage_layer = NoisyLinear(64, out_dim * atom_size)

        # set value layer
        self.value_hidden_layer = NoisyLinear(64, 64)
        self.value_layer = NoisyLinear(64, atom_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward method implementation."""
        dist = self.dist(x)
        q = torch.sum(dist * self.support, dim=2)

        return q

    def dist(self, x: torch.Tensor) -> torch.Tensor:
        """Get distribution for atoms."""
        feature = self.feature_layer(x)
        adv_hid = F.relu(self.advantage_hidden_layer(feature))
        val_hid = F.relu(self.value_hidden_layer(feature))

        advantage = self.advantage_layer(adv_hid).view(
            -1, self.out_dim, self.atom_size
        )
        value = self.value_layer(val_hid).view(-1, 1, self.atom_size)
        q_atoms = value + advantage - advantage.mean(dim=1, keepdim=True)

        dist = F.softmax(q_atoms, dim=-1)
        dist = dist.clamp(min=1e-3)  # for avoiding nans

        return dist

    def reset_noise(self):
        """Reset all noisy layers."""
        self.advantage_hidden_layer.reset_noise()
        self.advantage_layer.reset_noise()
        self.value_hidden_layer.reset_noise()
        self.value_layer.reset_noise()





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
        self.memory_size = 2000
        self.learning_start = self.memory_size
        self.update_model_freq = 1
        self.update_target_model_freq = 20

        self.gamma = 0.95  # discount rate
        self.epsilon = 0.2  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.95
        self.learning_rate = 0.005
        self.batch_size = 32
        self.ob_length = 24
        self.action_space = 8
        #self.memory = ReplayBuffer(self.ob_length,size = self.memory_size ,batch_size=self.batch_size)
        # N-step Learning
        self.n_step = 3
        # PER
        # PER parameters
        self.alpha = 0.2
        self.beta = 0.6
        self.prior_eps = 1e-6
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.memory = PrioritizedReplayBuffer(self.ob_length, self.memory_size, self.batch_size, alpha=self.alpha)

        self.use_n_step = True if self.n_step > 1 else False
        if self.use_n_step:
            self.memory_n = ReplayBuffer(self.ob_length, self.memory_size, self.batch_size, n_step=self.n_step, gamma=self.gamma)
        # Categorical DQN parameters

        self.v_min = 0.0
        self.v_max = 200.0
        self.atom_size = 51
        self.support = torch.linspace(self.v_min, self.v_max, self.atom_size).to(self.device)

        #self.memory = PrioritizedReplayBuffer(self.ob_length, self.memory_size, self.batch_size, self.alpha )

        self.model = Network(self.ob_length, self.action_space,self.atom_size, self.support).to(self.device)
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = Network(self.ob_length, self.action_space,self.atom_size, self.support).to(self.device)
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


    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def remember(self, ob, action, reward, next_ob):
        #self.memory.append((ob, action, reward, next_ob))
        one_step_transition = self.memory_n.store(ob, action, reward, next_ob)
        if one_step_transition:
            self.memory.store(*one_step_transition)
        #self.memory.store(ob,action,reward,next_ob)

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

        elementwise_loss = self._compute_dqn_loss(samples, self.gamma)
        # PER: importance sampling before average
        loss = torch.mean(elementwise_loss * weights)

        # N-step Learning loss
        # we are gonna combine 1-step loss and n-step loss so as to
        # prevent high-variance. The original rainbow employs n-step loss only.
        if self.use_n_step:
            gamma = self.gamma ** self.n_step
            samples = self.memory_n.sample_batch_from_idxs(indices)
            elementwise_loss_n_loss = self._compute_dqn_loss(samples, gamma)
            elementwise_loss += elementwise_loss_n_loss

            # PER: importance sampling before average
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

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss.item()

    def _compute_dqn_loss(self, samples: Dict[str, np.ndarray], gamma: float) -> torch.Tensor:
        device = self.device  # for shortening the following lines
        state = torch.FloatTensor(samples["obs"]).to(device)
        next_state = torch.FloatTensor(samples["next_obs"]).to(device)
        action = torch.LongTensor(samples["acts"]).to(device)
        reward = torch.FloatTensor(samples["rews"].reshape(-1, 1)).to(device)
        # Categorical DQN algorithm
        delta_z = float(self.v_max - self.v_min) / (self.atom_size - 1)

        with torch.no_grad():
            # Double DQN
            next_action = self.model(next_state).argmax(1)
            next_dist = self.target_model.dist(next_state)
            next_dist = next_dist[range(self.batch_size), next_action]

            t_z = reward + gamma * self.support
            t_z = t_z.clamp(min=self.v_min, max=self.v_max)
            b = (t_z - self.v_min) / delta_z
            l = b.floor().long()
            u = b.ceil().long()

            offset = (
                torch.linspace(
                    0, (self.batch_size - 1) * self.atom_size, self.batch_size
                ).long()
                    .unsqueeze(1)
                    .expand(self.batch_size, self.atom_size)
                    .to(self.device)
            )

            proj_dist = torch.zeros(next_dist.size(), device=self.device)
            proj_dist.view(-1).index_add_(
                0, (l + offset).view(-1), (next_dist * (u.float() - b)).view(-1)
            )
            proj_dist.view(-1).index_add_(
                0, (u + offset).view(-1), (next_dist * (b - l.float())).view(-1)
            )

        dist = self.model.dist(state)
        log_p = torch.log(dist[range(self.batch_size), action])
        elementwise_loss = -(proj_dist * log_p).sum(1)

        return elementwise_loss




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

