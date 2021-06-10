import os

path = os.path.split(os.path.realpath(__file__))[0]
import sys
import math
sys.path.append(path)
import random

from typing import Dict, List, Tuple
from torch.distributions import Normal, Categorical
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
import torch
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

class Actor(nn.Module):
    def __init__(self,num_state,num_action):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc = nn.Linear(64,64)
        self.action_head = nn.Linear(64, num_action)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc(x))
        action_prob = F.softmax(self.action_head(x), dim=1)
        return action_prob

class Critic(nn.Module):
    def __init__(self,num_state,num_action):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(num_state, 64)
        self.fc = nn.Linear(64,64)
        self.state_value = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc(x))
        value = self.state_value(x)
        return value


class TestAgent():
    def __init__(self):

        # PPO parameters

        self.now_phase = {}
        self.green_sec = 40
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}


        self.batch_size = 32
        self.ob_length = 24
        self.action_space = 8

        self.gamma = 0.99
        self.clip_param = 0.2
        self.max_grad_norm = 1.0
        self.ppo_update_time = 4
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.actor_net = Actor(self.ob_length,self.action_space).to(self.device)
        self.critic_net = Critic(self.ob_length,self.action_space).to(self.device)
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.buffer = []
        self.training_step = 0

        self.actor_optimizer = optim.RMSprop(self.actor_net.parameters(), lr=0.001)
        self.critic_net_optimizer = optim.RMSprop(self.critic_net.parameters(), lr = 0.002)


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


    def act_(self, observations_for_agent):
        actions = {}
        actions_prob = {}
        for agent_id in self.agent_list:
            action,action_prob = self.select_action(observations_for_agent[agent_id]['lane'])
            actions[agent_id] = action
            actions_prob[agent_id] = action_prob
        return actions,actions_prob

    def act(self, obs):
        observations = obs['observations']
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
            actions[agent] = self.get_action(observations_for_agent[agent]['lane_vehicle_num']) + 1

        return actions

    def get_action(self, ob):
        action,_ = self.select_action(ob)
        return action

    def select_action(self, state):
        state = np.array(state)
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        with torch.no_grad():
            action_prob = self.actor_net(state)
        c = Categorical(action_prob)
        action = c.sample()
        return action.item(), action_prob[:,action.item()].item()

    def get_value(self, state):
        state = torch.from_numpy(state).to(self.device)
        with torch.no_grad():
            value = self.critic_net(state)
        return value.item()


    def store_transition(self, transition):
        self.buffer.append(transition)
    def set_buffer(self,all_data):
        self.buffer = all_data

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))


    def update(self):
        state = torch.tensor([t.state for t in self.buffer], dtype=torch.float).to(self.device)
        action = torch.tensor([t.action for t in self.buffer], dtype=torch.long).view(-1, 1).to(self.device)
        reward = [t.reward for t in self.buffer]

        old_action_log_prob = torch.tensor([t.a_log_prob for t in self.buffer], dtype=torch.float).view(-1, 1).to(self.device)

        R = 0
        Gt = []
        for r in reward[::-1]:
            R = r + self.gamma * R
            Gt.insert(0, R)
        Gt = torch.tensor(Gt, dtype=torch.float).to(self.device)

        for i in range(self.ppo_update_time):
            for index in BatchSampler(SubsetRandomSampler(range(len(self.buffer))), self.batch_size, False):
                #with torch.no_grad():
                Gt_index = Gt[index].view(-1, 1)
                V = self.critic_net(state[index])
                delta = Gt_index - V
                advantage = delta.detach()
                # epoch iteration, PPO core!!!
                action_prob = self.actor_net(state[index]).gather(1, action[index]) # new policy

                ratio = (action_prob/old_action_log_prob[index])
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantage

                # update actor network
                action_loss = -torch.min(surr1, surr2).mean()
                self.actor_optimizer.zero_grad()
                action_loss.backward()
                nn.utils.clip_grad_norm_(self.actor_net.parameters(), self.max_grad_norm)
                self.actor_optimizer.step()

                #update critic network
                value_loss = F.mse_loss(Gt_index, V)
                self.critic_net_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic_net.parameters(), self.max_grad_norm)
                self.critic_net_optimizer.step()
                self.training_step += 1

        del self.buffer[:] # clear experience
        return action_loss.item(), value_loss.item()


    def load_model(self, dir="model/ppo", step=0):
        name = "ppo_actor_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        self.actor_net.load_state_dict(torch.load(model_name))

        name = "ppo_critic_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        self.critic_net.load_state_dict(torch.load(model_name))

    def save_model(self, dir="model/ppo", step=0):
        name = "ppo_actor_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        torch.save(self.actor_net.state_dict(),model_name)

        name = "ppo_critic_{}.pth".format(step)
        model_name = os.path.join(dir, name)
        torch.save(self.critic_net.state_dict(), model_name)


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

