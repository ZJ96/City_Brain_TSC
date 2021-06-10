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

import numpy as np

import torch.nn.functional as F
import torch.optim as optim
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical

device = torch.device('cpu')
if (torch.cuda.is_available()):
    device = torch.device('cuda:0')
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
print("============================================================================================")


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()

        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Tanh()
            )
        else:
            self.actor = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 64),
                nn.Tanh(),
                nn.Linear(64, action_dim),
                nn.Softmax(dim=-1)
            )

        # critic
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 64),
            nn.Tanh(),
            nn.Linear(64, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )

    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        return action.detach(), action_logprob.detach()

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)

            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)

            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)

        return action_logprobs, state_values, dist_entropy

class TestAgent():
    def __init__(self):
        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        self.ob_length = 48
        self.action_space = 8
        self.batch_size = 32

        self.has_continuous_action_space = False

        self.lr_actor = 0.001  # learning rate for actor network
        self.lr_critic = 0.002  # learning rate for critic network
        self.gamma = 0.99  # discount factor
        self.eps_clip = 0.2  # clip parameter for PPO
        self.K_epochs = 4

        self.buffer = RolloutBuffer()

        self.policy = ActorCritic(self.ob_length, self.action_space, self.has_continuous_action_space, 0).to(device)
        self.optimizer = torch.optim.AdamW([
            {'params': self.policy.actor.parameters(), 'lr': self.lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': self.lr_critic}
        ])

        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.policy_old = ActorCritic(self.ob_length, self.action_space, self.has_continuous_action_space, 0).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())

        self.MseLoss = nn.MSELoss()

    def remember(self, state, action, logprob, reward,done):
        self.buffer.states.append(torch.FloatTensor(state))
        self.buffer.actions.append(torch.tensor(action))
        self.buffer.logprobs.append(torch.tensor(logprob))
        self.buffer.rewards.append(reward)
        self.buffer.is_terminals.append(done)

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)

        # self.buffer.states.append(state)
        # self.buffer.actions.append(action)
        # self.buffer.logprobs.append(action_logprob)
        return action.item(),action_logprob.item()

    def update(self):
        # Monte Carlo estimate of returns
        #print(self.buffer.states)
        # print(self.buffer.actions[0].dtype)
        # print(self.buffer.logprobs[0].dtype)
        #print(self.buffer.rewards)
        #print(self.buffer.is_terminals)

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)

        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        tmp_loss = 0
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):
            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)

            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
        tmp_loss += sum(loss.detach().numpy())/self.K_epochs

        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())
        # clear buffer
        self.buffer.clear()
        return tmp_loss
    #
    # def save(self, checkpoint_path):
    #     torch.save(self.policy_old.state_dict(), checkpoint_path)
    #
    # def load(self, checkpoint_path):
    #     self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    #     self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))

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
        actions = {}
        actions_prob  ={}
        for agent_id in self.agent_list:
            action,action_prob = self.select_action(observations_for_agent[agent_id]['lane'])
            actions[agent_id] = action
            actions_prob[agent_id] = action_prob
        return actions,actions_prob

    def act(self, obs):
        observations = obs['observations']
        info = obs['info']
        actions = {}

        # Get state
        observations_for_agent = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key.split('_')[1]
            speed_vehicle_num = key.split('_')[2]

            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            #observations_for_agent[observations_agent_id][observations_feature] = val[1:]
            val = val[1:]
            while len(val) < 24:
                val.append(0)
            if speed_vehicle_num == "speed":
                observations_for_agent[observations_agent_id][observations_feature] = val
            else:
                observations_for_agent[observations_agent_id][observations_feature] += val

        # Get actions
        for agent in self.agent_list:
            actions[agent] = self.get_action(observations_for_agent[agent]['lane']) + 1

        return actions

    def get_action(self, ob):
        #ob = self._reshape_ob(ob)
        action, _ = self.select_action(ob)
        # act_values = self.model.predict([ob])
        # return np.argmax(act_values[0])
        return action


    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))


    # def remember(self, ob, action, reward, next_ob):
    #     #self.memory.append((ob, action, reward, next_ob))
    #     self.memory.store(ob,action,reward,next_ob)
    '''
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
        self.optimizer.step()

        # PER: update priorities
        loss_for_prior = elementwise_loss.detach().cpu().numpy()
        new_priorities = loss_for_prior + self.prior_eps
        self.memory.update_priorities(indices, new_priorities)

        self.model.reset_noise()
        self.target_model.reset_noise()

        return loss.item()'''


    def load_model(self, dir="model/ppo", step=0):
        name = "ppo_agent_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        self.policy.load_state_dict(torch.load(model_name))
        self.policy_old.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/ppo", step=0):
        name = "ppo_agent_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        torch.save(self.policy_old.state_dict(),model_name)





scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

