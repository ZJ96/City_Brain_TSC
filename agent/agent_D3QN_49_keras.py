import pickle
import os

path = os.path.split(os.path.realpath(__file__))[0]
import sys

sys.path.append(path)
import random

import gym
from pathlib import Path
import tensorflow as tf
import keras
from keras.models import Sequential,Model
from keras.optimizers import Adam, RMSprop, SGD
import os
from collections import deque
from keras.layers.merge import concatenate
import numpy as np
import keras.backend as K
from random import random, randrange

from keras.layers import Input, Dense, Flatten, Reshape, LSTM, Lambda,Add
from PER import *

#  smooth l1 loss
def smooth_L1_loss(y_true, y_pred):
    THRESHOLD = K.variable(1.0)
    mae = K.abs(y_true-y_pred)
    flag = K.greater(mae, THRESHOLD)
    loss = K.mean(K.switch(flag, (mae - 0.5), K.pow(mae, 2)), axis=-1)
    return loss

HUBER_LOSS_DELTA = 2.0
def huber_loss(y_true, y_pred):
    err = y_true - y_pred
    cond = K.abs(err) < HUBER_LOSS_DELTA
    L2 = 0.5 * K.square(err)
    L1 = HUBER_LOSS_DELTA * (K.abs(err) - 0.5 * HUBER_LOSS_DELTA)
    loss = tf.where(cond, L2, L1)   # Keras does not cover where function in tensorflow :-(
    return K.mean(loss)

class TestAgent():
    def __init__(self):

        # DQN parameters

        self.now_phase = {}
        self.green_sec = 30
        self.red_sec = 5
        self.max_phase = 4
        self.last_change_step = {}
        self.agent_list = []
        self.phase_passablelane = {}

        # Instantiate memory
        memory_size = 8000
        self.MEMORY = Memory(memory_size)
        self.memory = deque(maxlen=3000)
        self.learning_start = 20

        self.USE_PER = True
        self.Soft_Update = True
        self.update_model_freq = 1
        self.update_target_model_freq = 20
        self.tau = 0.05

        self.gamma = 0.8  # discount rate
        self.epsilon = 0.5  # exploration rate
        self.epsilon_min = 0.1
        self.epsilon_decay = 0.95
        self.learning_rate = 0.0008
        self.batch_size = 40
        self.ob_length = 49

        self.action_space = 8

        self.input_shape = (self.ob_length,)
        self.dueling = True

        self.model = self._build_model(self.input_shape,self.action_space,self.dueling)

        # Remember to uncomment the following lines when submitting, and submit your model file as well.
        # path = os.path.split(os.path.realpath(__file__))[0]
        # self.load_model(path, 99)
        self.target_model = self._build_model(self.input_shape,self.action_space,self.dueling)
        self.update_target_network()

        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        self.phase_lane_map_out = [[16, 17, 18, 22, 23, 24], [13, 14, 15, 19, 20, 21],
                                   [13, 14, 15, 19, 20, 21], [16, 17, 18, 22, 23, 24],
                                   [16, 17, 18, 19, 20, 21], [19, 20, 21, 22, 23, 24],
                                   [13, 14, 15, 22, 23, 24], [13, 14, 15, 16, 17, 18]]


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
        now_time = {}

        # Get state
        observations_for_agent = {}
        vehicle_obs = {}
        for key, val in observations.items():
            observations_agent_id = int(key.split('_')[0])
            observations_feature = key.split('_')[1]
            speed_vehicle_num = key.split('_')[2]
            if (observations_agent_id not in observations_for_agent.keys()):
                observations_for_agent[observations_agent_id] = {}
            now_time[observations_agent_id] = val[0]
            #observations_for_agent[observations_agent_id][observations_feature] = val[1:]
            if speed_vehicle_num == "speed":
                observations_for_agent[observations_agent_id][observations_feature] = val[1:]
            else:
                observations_for_agent[observations_agent_id][observations_feature] += val[1:]
                vehicle_obs[observations_agent_id] = val
                observations_for_agent[observations_agent_id][observations_feature].append(self.now_phase[observations_agent_id]-1)

        # Get actions
        for agent in self.agent_list:
            self.epsilon = 0
            # actions[agent] = self.get_action(observations_for_agent[agent]['lane']) + 1
            lane_vehicle_num = observations_for_agent[agent]["lane"]
            action = self.get_action(lane_vehicle_num) + 1
            #action = self.get_final_action(vehicle_obs[agent],action)

            step_diff = now_time[agent] - self.last_change_step[agent]
            if (step_diff >= self.green_sec):   #and self.now_phase[agent]!=action:
                self.now_phase[agent] = action
                self.last_change_step[agent] = now_time[agent]

            actions[agent] = self.now_phase[agent]

        return actions

    def get_action(self, ob):
        if np.random.rand() <= self.epsilon:
            return np.random.randint(0, self.action_space)
        ob = self._reshape_ob(ob)#print(ob) [[-2 -2 -2 -1 -1 -1 -2 -2 -2 -2 -2 -2 -2 -2 -2 -1 -1 -1 -2 -2 -2 -2 -2 -2 0  0  0 -1 -1 -1  0  0  0  0  0  0  0  0  0 -1 -1 -1  0  0  0  0  0  0]]
        act_values = self.model.predict(ob)
        #print([ob]) [array([[-2, -2, -2, -1, -1, -1, -2, -2, -2, -2, -2, -2, -2, -2, -2, -1,-1, -1, -2, -2, -2, -2, -2, -2, 0, 0, 0, -1, -1, -1, 0, 0,0, 0, 0, 0, 0, 0, 0, -1, -1, -1, 0, 0, 0, 0, 0, 0]])]
        #print(act_values)  #[[ 0.22203743  0.72292787  0.32405666 -0.6518022   0.05178177  0.274694 0.9647807  -0.43269655]]
        return np.argmax(act_values[0])


    # def _build_model(self):
    #     model = Sequential()
    #     model.add(Dense(40, input_dim=self.ob_length, activation='relu'))
    #     model.add(Dense(40, activation='relu'))
    #     model.add(Dense(self.action_space, activation='linear'))
    #     model.compile(
    #         loss=huber_loss,
    #         optimizer=RMSprop()
    #     )
    #     return model
    def _build_model(self,input_shape, action_space, dueling):
        x_input = Input(shape=input_shape)
        x = x_input
        x = Dense(32, input_shape=input_shape, activation='relu', kernel_initializer='he_uniform')(x)
        #x = Dense(40, activation='relu', kernel_initializer='he_uniform')(x)

        if dueling:
            state_value = Dense(1, kernel_initializer='he_uniform')(x)
            state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)
            action_advantage = Dense(action_space, kernel_initializer='he_uniform')(x)
            action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)
            x = Add()([state_value, action_advantage])
        else:
            x = Dense(action_space, activation="linear", kernel_initializer='he_uniform')(x)

        model = Model(inputs=x_input, outputs=x)
        model.compile(loss=smooth_L1_loss, optimizer=RMSprop(learning_rate=self.learning_rate))
        #model.summary()
        return model

    def _reshape_ob(self, ob):
        return np.reshape(ob, (1, -1))

    def update_target_network(self):
        if self.Soft_Update:
            W = self.model.get_weights()
            tgt_W = self.target_model.get_weights()
            for i in range(len(W)):
                tgt_W[i] = self.tau * W[i] + (1 - self.tau) * tgt_W[i]
            self.target_model.set_weights(tgt_W)
        else:
            weights = self.model.get_weights()
            self.target_model.set_weights(weights)

    def remember(self, ob, action, reward, next_ob):
        # self.memory.append((ob,action,reward,next_ob))
        experience = ob, action, reward, next_ob, False
        if self.USE_PER:
            self.MEMORY.store(experience)
        else:
            self.memory.append((experience))

    def replay(self):
        if self.USE_PER:
            # Sample minibatch from the PER memory
            tree_idx, minibatch = self.MEMORY.sample(self.batch_size)
            obs = np.zeros(shape=(self.batch_size, self.ob_length))
            next_obs = np.zeros(shape=(self.batch_size, self.ob_length))
            actions, rewards, dones = [], [], []
            for i in range(len(minibatch)):
                obs[i] = minibatch[i][0]
                actions.append(minibatch[i][1])
                rewards.append(minibatch[i][2])
                next_obs[i] = minibatch[i][3]
                dones.append(minibatch[i][4])
        else:
            # Randomly sample minibatch from the deque memory
            minibatch = random.sample(self.memory, min(len(self.memory), self.batch_size))
            obs, actions, rewards, next_obs, _ = [np.stack(x) for x in np.array(minibatch).T]
        # target = rewards + self.gamma * np.max(self.target_model.predict([next_obs]), axis=1)
        # target_f = self.model.predict([obs])
        # for i, action in enumerate(actions):
        #     target_f[i][action] = target[i]
        # self.model.fit([obs], target_f, epochs=1, verbose=0)

        #update model
        target = self.model.predict(obs)
        target_old = np.array(target)
        target_next = self.model.predict(next_obs)
        target_val = self.target_model.predict(next_obs)

        for i in range(len(minibatch)):
            a = np.argmax(target_next[i])
            target[i][actions[i]] = rewards[i]+self.gamma*(target_val[i][a])

        if self.USE_PER:
            indices = np.arange(self.batch_size, dtype=np.int32)
            absolute_errors = np.abs(target_old[indices, np.array(actions)] - target[indices, np.array(actions)])
            # Update priority
            self.MEMORY.batch_update(tree_idx, absolute_errors)
        loss = self.model.fit(obs, target, batch_size=self.batch_size, verbose=0)
        loss_val = abs(float(loss.history['loss'][0]))
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        return loss_val

    def load_model(self, dir="model/dqn_keras", step=0):
        name = "dqn_keras_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.load_weights(model_name)

    def save_model(self, dir="model/dqn_keras", step=0):
        name = "dqn_keras_{}.h5".format(step)
        model_name = os.path.join(dir, name)
        self.model.save_weights(model_name)

    ############max pressure action
    def get_final_action(self, lane_vehicle_num, DQN_action):
        pressures = self.get_phase_pressures(lane_vehicle_num)
        avg_pressure = sum(pressures)/8
        if pressures[DQN_action-1]>=avg_pressure:
            return DQN_action

        #max pressure
        unavailable_phases = self.get_unavailable_phases(lane_vehicle_num)
        max_pressure_id = np.argmax(pressures) + 1
        while (max_pressure_id in unavailable_phases):
            pressures[max_pressure_id - 1] -= 10000
            max_pressure_id = np.argmax(pressures) + 1
        # # print(max_pressure_id)
        return max_pressure_id

    def get_phase_pressures(self, lane_vehicle_num):
        pressures = []
        for i in range(8):
            in_lanes = self.phase_lane_map_in[i]
            out_lanes = self.phase_lane_map_out[i]
            pressure = 0
            for in_lane in in_lanes:
                pressure += lane_vehicle_num[in_lane] * 3
            for out_lane in out_lanes:
                pressure -= lane_vehicle_num[out_lane]
            pressures.append(pressure)
        # # print("pressures: ", pressures)
        return pressures
    def get_unavailable_phases(self, lane_vehicle_num):
        self.phase_lane_map_in = [[1, 7], [2, 8], [4, 10], [5, 11], [2, 1], [5, 4], [8, 7], [11, 10]]
        unavailable_phases = []
        not_exist_lanes = []
        for i in range(1, 25):
            if lane_vehicle_num[i] < 0:
                not_exist_lanes.append(i)
        for lane in not_exist_lanes:
            for phase_id, in_lanes in enumerate(self.phase_lane_map_in):
                phase_id += 1
                if lane in in_lanes and phase_id not in unavailable_phases:
                    unavailable_phases.append(phase_id)

        return unavailable_phases


scenario_dirs = [
    "test"
]

agent_specs = dict.fromkeys(scenario_dirs, None)
for i, k in enumerate(scenario_dirs):
    # initialize an AgentSpec instance with configuration
    agent_specs[k] = TestAgent()
    # **important**: assign policy builder to your agent spec
    # NOTE: the policy builder must be a callable function which returns an instance of `AgentPolicy`

