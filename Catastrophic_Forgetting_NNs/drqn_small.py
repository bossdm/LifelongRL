#!/usr/bin/env python
from __future__ import print_function

# import skimage as skimage
# from skimage import transform, color, exposure
# from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.wrappers import TimeDistributed
from keras.layers.core import Dense, Dropout, Activation, Flatten, RepeatVector, Masking
from keras.layers import Convolution2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, merge, \
    Activation, Embedding
from keras.layers.recurrent import LSTM, GRU
from keras.optimizers import Adadelta
from keras import backend as K
from abc import abstractmethod
from overrides import overrides
# from vizdoom import DoomGame, ScreenResolution
# from vizdoom import *
import itertools as it
from time import sleep

import sys,os


from Catastrophic_Forgetting_NNs.CustomNetworks import CustomNetworks
DEBUG_MODE=False

def preprocessImg(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)

    return img

class ReplayMemory(object):
    def __init__(self,buffer_size):
        self.buffer = [None for i in range(buffer_size)]
        self.buffer_size = buffer_size
        self.sp=-1

    def full(self):
        return self.sp + 1 >= self.buffer_size


    @abstractmethod
    def sample(self, batch_size, trace_length):
        pass
    @abstractmethod
    def add(self,experience):
        pass

class NonEpisodicReplayMemory(ReplayMemory):
    def __init__(self,buffer_size=400000):
        """

        :param buffer_size:  equal to the number of experiences
        """
        ReplayMemory.__init__(self,buffer_size)
        self.max_sp=self.sp
        self.max_reached=False
        self.e_sp=0

    def add(self, episode_experience):
        """

        :param episode_experience: episode in case of Episodic, experience in case of NonEpisodic
        :return:
        """
        if self.full(): # circular buffer starts overwriting first element
            self.max_reached=True
            self.sp = 0

        else:
            self.sp+=1
        self.buffer[self.sp]=episode_experience

    def get_samples(self,trace_length,batch_size):
        """
        do not allow traces from before sp and after sp to come together,
        do allow traces to circle from :sp+1 to sp

        --> choose the sample index either larger than sp + tracelength
        """
        if self.max_reached:
            numbers = range(0, self.sp+1) + range(self.sp+trace_length+1, self.buffer_size)
            return random.sample(numbers,batch_size)
        else:
            numbers = range(trace_length,self.sp+1)
            return random.sample(numbers,batch_size)

    def get_trace(self,start,end):
        if start >= 0:
            b =  self.buffer[start:end]
        else:
            b = self.buffer[start:self.buffer_size] + self.buffer[0:end]

        return b
    def get_top_trace(self,trace_length):
        return self.get_trace(self.sp + 1 - trace_length,
              self.sp + 1)
    @overrides
    def sample(self, batch_size, trace_length):

        samples = self.get_samples(trace_length,batch_size)
        sampledTraces = [None for i in range(batch_size)]
        for i in range(batch_size):
            #print(len(episode))
            end=samples[i]
            start = end - trace_length
            sampledTraces[i]=self.get_trace(start,end)
        sampledTraces = np.array(sampledTraces)
        return sampledTraces #,[]

class EpisodicReplayMemory(ReplayMemory):
    """
    Memory Replay Buffer


    intuitively buffer_size=10000 seems reasonable
    this is supported by
        Zhang, S., & Sutton, R. S. (2017). A Deeper Look at Experience Replay. Retrieved from http://arxiv.org/abs/1712.01275

    """

    def __init__(self, buffer_size=400000):
        """

        :param buffer_size: how many episodes (so the number of experiences:buffer_size * episode_length)
        """
        self.max_num_episodes=buffer_size
        self.ep=-1
        self.max_ep=self.ep

        ReplayMemory.__init__(self, buffer_size)
        self.max_sp = self.sp

    def add(self, episode_experience):
        """

        :param episode_experience: episode in case of Episodic, experience in case of NonEpisodic
        :return:
        """
        if self.full(): # circular buffer starts overwriting first element
            # , but first remove entries later after ep (since they are from even more time ago)
            j=1
            while self.buffer[self.ep+j] is not None:
                self.buffer[self.ep+j]=None
                j+=1
            #self.max_sp=self.sp
            self.sp=0
            self.max_ep=self.ep
            self.ep=0

        else:
            self.ep+=1
            self.max_ep=max(self.ep,self.max_ep)
            #self.max_sp=max(self.sp,self.max_sp)

        self.buffer[self.ep]=episode_experience
        self.sp+=len(episode_experience)

        # print("sp="+str(self.sp))
        # print("ep=" + str(self.ep))
    def get_tr(self,ep,start,end):
        return self.buffer[ep][start:end]
    def get_top_trace(self,trace_length):
        return self.buffer[self.ep][-trace_length:]
    @overrides
    def sample(self, batch_size, trace_length):
        sampled_episodes = np.random.choice(self.max_ep+1, batch_size)
        sampledTraces = [None for i in range(batch_size)]
        #terminals=[False for i in range(batch_size)]
        for i in range(batch_size):
            #print(len(episode))
            episode=self.buffer[sampled_episodes[i]]
            point = np.random.randint(0, len(episode) + 1 - trace_length)
            sampledTraces[i]=episode[point:point + trace_length]
            # if point +trace_length == len(episode):
            #     terminals[i]=True

        sampledTraces = np.array(sampledTraces)
        return sampledTraces #,terminals

    def get_trace(self,start,end):
         return self.buffer[start:end]

class MultiGoalNonEpisodicReplayMemory(NonEpisodicReplayMemory):
    def __init__(self,buffer_size=100000):
        """

        :param buffer_size:  equal to the number of experiences
        """
        self.buffer_size=buffer_size
        self.buffers={}
        self.ts={}
    def add_goal(self,goal):
        if goal not in self.buffers:
            self.buffers[goal]=NonEpisodicReplayMemory(self.buffer_size)
            self.ts[goal]=0
        self.current_goal=goal
    @overrides
    def add(self, episode_experience):
        """

        :param episode_experience: episode in case of Episodic, experience in case of NonEpisodic
        :return:
        """
        self.buffers[self.current_goal].add(episode_experience)

    @overrides
    def sample(self, goals, batch_size, trace_length):
        samplegoals=np.random.choice(goals,batch_size)
        data=[self.buffers[goal].sample(1,trace_length) for goal in samplegoals]
        return np.array(data)
    def replay_ready(self, goal,start, batch_size):
        t=self.ts[goal]
        return t >= start and t >= batch_size
    def get_replay_ready_goals(self,start,batch_size):
        gs=[]
        for goal in self.buffers:
            if self.replay_ready(goal,start,batch_size):
                gs.append(goal)

        return gs
class MultiGoalEpisodicReplayMemory(EpisodicReplayMemory):
    def __init__(self,buffer_size=100000):
        """

        :param buffer_size:  equal to the number of experiences
        """
        self.buffer_size=buffer_size
        self.buffers={}
        self.ts={}

    def add_goal(self,goal):
        if goal not in self.buffers:
            self.buffers[goal]=EpisodicReplayMemory(self.buffer_size)
            self.ts[goal]=0
        self.current_goal=goal
    @overrides
    def add(self, episode_experience):
        """

        :param episode_experience: episode in case of Episodic, experience in case of NonEpisodic
        :return:
        """
        self.buffers[self.current_goal].add(episode_experience)

    @overrides
    def sample(self, goals, batch_size, trace_length):
        samplegoals=np.random.choice(len(goals),batch_size)
        data=[self.buffers[goals[i]].sample(1,trace_length)[0] for i in samplegoals]
        return data

    def replay_ready(self, goal, start, batch_size):
        t=self.ts[goal]
        return t >= start and t >= batch_size

    def get_replay_ready_goals(self, start, batch_size):
        gs = []
        for goal in self.buffers:
            if self.replay_ready(goal, start, batch_size):
                gs.append(goal)

        return gs
class DoubleDRQNAgent:
    q=None
    episodic = False
    def __init__(self, state_size, action_size, trace_length, batch_size=None,episodic=True,
                 double=False, init_epsilon=None, final_epsilon=None, learning_rate=0.10):
        """

        :param state_size:  (trace_length, img_rows, img_cols, img_channels)
        :param action_size: number of actions
        :param trace_length:
        """
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # these are hyper parameters for DRQN
        #
        # cf.
        # -hausknecht & stone https://arxiv.org/pdf/1507.06527.pdf
        # -Mnih et al https://arxiv.org/pdf/1312.5602.pdf

        self.gamma = 0.99
        #self.learning_rate = 0.00025  cf drqn small

        self.initial_epsilon = 1.0 if init_epsilon is None else init_epsilon
        self.epsilon = self.initial_epsilon
        self.final_epsilon = 0.10 if final_epsilon is None else final_epsilon
        self.batch_size = 10 if batch_size is None else batch_size
        self.observe = 0
        self.frame_per_action = 4
        self.replay_start_size = 50000./self.frame_per_action   # 50000 frames in atari
        self.exploration_frame = 20*self.replay_start_size # 1M frames in atari
        self.trace_length = trace_length
        self.update_freq = 4 # Number of timesteps between training interval
        self.update_target_freq = 10000 * self.update_freq  # total t will count steps not frames
        self.total_t = 0

        # Create replay memory
        self.episodic=episodic
        self.init_memory(episodic)

        # Create main model and target model
        self.model = None
        self.double=double
        self.target_model = None


    def init_memory(self,episodic):
        if episodic:
            self.memory =EpisodicReplayMemory()
        else:
            self.memory = NonEpisodicReplayMemory()
    def update_target_model(self):
        """
        After some time interval update the target model to be same with model
        """
        self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:

            # Use all traces for RNN
            # q = self.model.predict(state) # 1x8x3
            # action_idx = np.argmax(q[0][-1])

            # Only use last trace for RNN
            q = self.model.predict(state)  # 1x3

            action_idx = np.argmax(q)
        return action_idx

    def shape_reward(self, r_t, misc, prev_misc, t):

        # Check any kill count
        if (misc[0] > prev_misc[0]):
            r_t = r_t + 1

        if (misc[1] < prev_misc[1]):  # Use ammo
            r_t = r_t - 0.1

        if (misc[2] < prev_misc[2]):  # Loss HEALTH
            r_t = r_t - 0.1

        return r_t
    def train_replay(self):
        # Do the training
        training=False
        if self.check_replay_ready() and self.total_t % self.update_freq == 0:
            if DEBUG_MODE:
                print("train")
            training = True
        if not training: return None,None
        return self._train_replay(self.batch_size)
    def check_replay_ready(self):

        return self.total_t >= self.replay_start_size and  self.total_t >= self.batch_size + self.trace_length
    def compute_target(self,update_input,target_update_input,batch_size,action,reward):
        target = self.model.predict(update_input)  # 32x3
        if self.target_model is not None:
            target_val = self.target_model.predict(target_update_input)  # 32x3
            if self.double:
                val = self.model.predict(target_update_input)
            else:
                val = target_val
        else:
            val = self.model.predict(target_update_input)
            target_val = val

        for i in range(batch_size):
            a = np.argmax(val[i])
            # if self.episodic and terminals[i]:
            #     target[i][int(action[i][-1])] = reward[i][-1]
            # else:
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])
        return target
    def compute_output_and_target(self,update_input,target_update_input,batch_size,action,reward):
        target = self.model.predict(update_input)  # 32x3
        if self.target_model is not None:
            target_val = self.target_model.predict(target_update_input)  # 32x3
            if self.double:
                val = self.model.predict(target_update_input)
            else:
                val = target_val
        else:
            val = self.model.predict(target_update_input)
            target_val = val

        for i in range(batch_size):
            a = np.argmax(val[i])
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])
        return val, target


    # pick samples randomly from replay memory (with batch_size)
    def _train_replay(self,batch_size):

        #, terminals
        sample_traces = self.memory.sample(batch_size, self.trace_length)  # 32x8x4

        # Shape (batch_size, trace_length, img_rows, img_cols, color_channels)
        update_input = np.zeros(((batch_size,) + self.state_size))  # 32x8x64x64x3
        target_update_input = np.zeros(((batch_size,) + self.state_size))

        action = np.zeros((batch_size, self.trace_length))  # 32x8
        reward = np.zeros((batch_size, self.trace_length))

        for i in range(batch_size):
            for j in range(self.trace_length):
                update_input[i, j, :] = sample_traces[i][j][0]
                action[i, j] = sample_traces[i][j][1]
                reward[i, j] = sample_traces[i][j][2]
                target_update_input[i, j, :] = sample_traces[i][j][3]

        # Only use the last trace for training
        target=self.compute_target(update_input,target_update_input,batch_size,action,reward) #,terminals
        loss = self.model.train_on_batch(update_input, target)
        # try:
        #     loss = self.model.train_on_batch(update_input, target)
        # except Exception as e:
        #     print(update_input)
        #     print(target)
        #     print(batch_size)
        #     print(e.message)
        return float(np.max(target[-1, -1])), float(loss)

    # load the saved model
    def load(self, name):
        #self.model.load_weights(name)
        self.model = load_model(name+"_network.h5")
        try:
            self.target_model = load_model(name+"_targetnetwork.h5")
        except:
            self.target_model=None
    # save the model which is under training
    def save(self, name):
        #self.model.save_weights(name)
        self.model.save(name+"_network.h5")
        del self.model
        if self.target_model is not None:
            self.target_model.save(name+"_targetnetwork.h5")
            del self.target_model


class FeatureDoubleDRQNAgent(DoubleDRQNAgent):
    q=None
    episodic = False
    def __init__(self, num_features,state_size, action_size, trace_length, batch_size=None,episodic=True,
                 double=False, init_epsilon=None, final_epsilon=None):
        DoubleDRQNAgent.__init__(self,state_size, action_size, trace_length, batch_size=batch_size,
                                 episodic=episodic, double=double, init_epsilon=init_epsilon, final_epsilon=final_epsilon)
        self.num_features=num_features
    def get_action(self, state):
        """
        Get action from model using epsilon-greedy policy
        """
        if np.random.rand() <= self.epsilon:
            action_idx = random.randrange(self.action_size)
        else:

            # Use all traces for RNN
            # q = self.model.predict(state) # 1x8x3
            # action_idx = np.argmax(q[0][-1])

            # Only use last trace for RNN
            q,_ = self.model.predict(state)  # 1x3

            action_idx = np.argmax(q)
        return action_idx

    def compute_target(self,update_input,target_update_input,batch_size,action,reward):
        target, features = self.model.predict(update_input)  # 32x3
        if self.target_model is not None:
            target_val,_ = self.target_model.predict(target_update_input)  # 32x3
            if self.double:
                val,_ = self.model.predict(target_update_input)
            else:
                val = target_val
        else:
            val, _ = self.model.predict(target_update_input)
            target_val = val

        for i in range(batch_size):
            a = np.argmax(val[i])
            # if self.episodic and terminals[i]:
            #     target[i][int(action[i][-1])] = reward[i][-1]
            # else:
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])
        return [target, features]


class HindsightDoubleDRQNAgent(DoubleDRQNAgent):
    def __init__(self, reward_range,train_rfun,state_size, rsa_size,action_size, trace_length, batch_size=None,episodic=True, double=False):
        """

        :param state_size:  (trace_length, img_rows, img_cols, img_channels)
        :param action_size: number of actions
        :param trace_length:
        """
        DoubleDRQNAgent.__init__(self,state_size,action_size,trace_length,batch_size,episodic,double)
        self.train_rfun=train_rfun
        self.rsa_size=rsa_size
        self.reward_fun = Sequential()

        self.reward_fun.add(LSTM(50,input_shape=self.rsa_size))
        self.reward_fun.add(Dense(50))
        self.reward_fun.add(Dense(output_dim=1, activation='linear'))
        self.total_t={}
        ada_delta=Adadelta(lr=0.1, rho=0.95,clipvalue=10.0)
        buffer_size = 40000 # each goal has 40000 time steps
        self.memory=MultiGoalNonEpisodicReplayMemory(buffer_size)
        self.n_actions=action_size
        self.reward_min,self.reward_max=reward_range
        self.reward_fun.compile(loss='mse',optimizer=ada_delta)
    # pick samples randomly from replay memory (with batch_size)
    def normalise_reward(self,r):
        """ normalise the reward to [0,1] """
        return (r - self.reward_min)/float(self.reward_max-self.reward_min)
    def check_replay_ready(self,task):
        return self.total_t[task] >= self.replay_start_size and  self.total_t[task] >= self.batch_size + self.trace_length+1
    def get_goals(self):
        """
        get all known goals which have enough data
        :return:
        """

        goals=[]
        for goal in self.memory.buffers:
            if self.total_t[goal] % self.update_freq == 0 and self.check_replay_ready(goal):
                if DEBUG_MODE:
                    print("train")
                goals.append(goal)
        return goals

    @overrides
    def _train_replay(self,batch_size):

        goals=self.get_goals()
        if not goals: return None,None
        sampled = self.memory.sample(goals, batch_size, self.trace_length+1)  # 32x8x4


        sample_traces=sampled.values()

        # Shape (batch_size, trace_length, img_rows, img_cols, color_channels)
        update_input = np.zeros(((batch_size,) + self.state_size))  # 32x8x64x64x3
        rewardfuninput=np.zeros(((batch_size,) + self.rsa_size))
        target_update_input = np.zeros(((batch_size,) + self.state_size))

        action = np.zeros((batch_size, self.trace_length))  # 32x8
        reward = np.zeros((batch_size, self.trace_length))

        for i in range(batch_size):
            for j in range(self.trace_length+1):
                if j <= self.trace_length:
                    update_input[i, j, :] = sample_traces[i][j][0]
                    action[i, j] = sample_traces[i][j][1]
                    reward[i, j] = sample_traces[i][j][2]
                    target_update_input[i, j, :] = sample_traces[i][j][3]
                if j >=1:
                    rewardfuninput[i,j-1, :] = np.append(self.normalise_reward(reward[i,j-1]),
                                                         [update_input[i,j],action[i,j]/float(self.n_actions)])



        # Only use the last trace for training
        target = self.model.predict(update_input)  # 32x3
        if self.target_model is not None:
            target_val=self.target_model.predict(target_update_input)  # 32x3
            if self.double:
                val = self.model.predict(target_update_input)
            else:
                val = target_val
        else:
            val = self.model.predict(target_update_input)
            target_val=val



        for i in range(batch_size):
            a = np.argmax(val[i])
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])


        loss = self.model.train_on_batch(update_input, target)

        reward_loss = self.model.train_on_batch(rewardfuninput, reward[:,self.trace_length])

        if DEBUG_MODE:
            print("rewardfunction loss=%.5f"%(reward_loss))
        return np.max(target[-1, -1]), loss



class MultiTaskDoubleDRQNAgent(DoubleDRQNAgent):
    def __init__(self, state_size, action_size, trace_length, batch_size=None,episodic=True, double=False):
        """

        :param state_size:  (trace_length, img_rows, img_cols, img_channels)
        :param action_size: number of actions
        :param trace_length:
        """
        DoubleDRQNAgent.__init__(self,state_size,action_size,trace_length,batch_size,episodic,double)


    @overrides
    def init_memory(self,episodic):
        # Create replay memory
        if episodic:
            self.memory = MultiGoalEpisodicReplayMemory()
        else:
            self.memory = MultiGoalNonEpisodicReplayMemory()


    @overrides
    def _train_replay(self,batch_size):
        # Do the training
        goals=self.memory.get_replay_ready_goals(self.replay_start_size,self.batch_size+self.trace_length)
        if not goals: return None,None
        sample_traces = self.memory.sample(goals,batch_size, self.trace_length)  # 32x8x4

        # Shape (batch_size, trace_length, img_rows, img_cols, color_channels)
        update_input = np.zeros(((batch_size,) + self.state_size))  # 32x8x64x64x3
        target_update_input = np.zeros(((batch_size,) + self.state_size))

        action = np.zeros((batch_size, self.trace_length))  # 32x8
        reward = np.zeros((batch_size, self.trace_length))

        for i in range(batch_size):
            for j in range(self.trace_length):
                update_input[i, j, :] = sample_traces[i][j][0]
                action[i, j] = sample_traces[i][j][1]
                reward[i, j] = sample_traces[i][j][2]
                target_update_input[i, j, :] = sample_traces[i][j][3]

        # Only use the last trace for training
        target = self.model.predict(update_input)  # 32x3
        if self.target_model is not None:
            target_val=self.target_model.predict(target_update_input)  # 32x3
            if self.double:
                val = self.model.predict(target_update_input)
            else:
                val = target_val
        else:
            val = self.model.predict(target_update_input)
            target_val=val



        for i in range(batch_size):
            a = np.argmax(val[i])
            target[i][int(action[i][-1])] = reward[i][-1] + self.gamma * (target_val[i][a])


        loss = self.model.train_on_batch(update_input, target)

        return np.max(target[-1, -1]), loss

def main():
    import tensorflow.compat.v1 as tf
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("../../scenarios/defend_the_center.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows, img_cols = 64, 64
    img_channels = 3  # Color channel
    trace_length = 4  # Temporal Dimension

    state_size = (trace_length, img_rows, img_cols, img_channels)
    agent = DoubleDRQNAgent(state_size, action_size, trace_length)

    agent.model = Networks.drqn_small(state_size, action_size, agent.learning_rate)
    agent.target_model = Networks.drqn(state_size, action_size, agent.learning_rate)

    s_t = game_state.screen_buffer  # 480 x 640
    s_t = preprocessImg(s_t, size=(img_rows, img_cols))

    is_terminated = game.is_episode_finished()

    # Start training
    epsilon = agent.initial_epsilon
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)
    life = 0
    episode_buf = []  # Save entire episode

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    while not game.is_episode_finished():

        loss = 0
        Q_max = 0
        r_t = 0
        a_t = np.zeros([action_size])

        # Epsilon Greedy
        if len(episode_buf) > agent.trace_length:
            # 1x8x64x64x3
            state_series = np.array([trace[-1] for trace in episode_buf[-agent.trace_length:]])
            state_series = np.expand_dims(state_series, axis=0)
            action_idx = agent.get_action(state_series)
        else:
            action_idx = random.randrange(agent.action_size)
        a_t[action_idx] = 1

        a_t = a_t.astype(int)
        game.set_action(a_t.tolist())
        skiprate = agent.frame_per_action
        game.advance_action(skiprate)

        game_state = game.get_state()  # Observe again after we take the action
        is_terminated = game.is_episode_finished()

        r_t = game.get_last_reward()  # each frame we get reward of 0.1, so 4 frames will be 0.4

        if (is_terminated):
            if (life > max_life):
                max_life = life
            GAME += 1
            life_buffer.append(life)
            ammo_buffer.append(misc[1])
            kills_buffer.append(misc[0])
            print("Episode Finish ", misc)
            game.new_episode()
            game_state = game.get_state()
            misc = game_state.game_variables
            s_t1 = game_state.screen_buffer

        s_t1 = game_state.screen_buffer
        misc = game_state.game_variables
        s_t1 = preprocessImg(s_t1, size=(img_rows, img_cols))

        r_t = agent.shape_reward(r_t, misc, prev_misc, t)

        if (is_terminated):
            life = 0
        else:
            life += 1

        # update the cache
        prev_misc = misc

        # Update epsilon
        if agent.epsilon > agent.final_epsilon and t > agent.observe:
            agent.epsilon -= (agent.initial_epsilon - agent.final_epsilon) / agent.explore

        # Do the training
        if t > agent.observe:
            Q_max, loss = agent.train_replay()

        # save the sample <s, a, r, s'> to episode buffer
        episode_buf.append([s_t, action_idx, r_t, s_t1])

        if (is_terminated):
            agent.memory.add(episode_buf)
            episode_buf = []  # Reset Episode Buf

        s_t = s_t1
        t += 1

        # save progress every 10000 iterations
        if t % 10000 == 0:
            print("Now we save model")
            agent.model.save_weights("models/drqn.h5", overwrite=True)

        # print info
        state = ""
        if t <= agent.observe:
            state = "observe"
        elif t > agent.observe and t <= agent.observe + agent.explore:
            state = "explore"
        else:
            state = "train"

        if (is_terminated):
            print("TIME", t, "/ GAME", GAME, "/ STATE", state, \
                  "/ EPSILON", agent.epsilon, "/ ACTION", action_idx, "/ REWARD", r_t, \
                  "/ Q_MAX %e" % np.max(Q_max), "/ LIFE", max_life, "/ LOSS", loss)

            # Save Agent's Performance Statistics
            if GAME % agent.stats_window_size == 0 and t > agent.observe:
                print("Update Rolling Statistics")
                agent.mavg_score.append(np.mean(np.array(life_buffer)))
                agent.var_score.append(np.var(np.array(life_buffer)))
                agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                # Reset rolling stats buffer
                life_buffer, ammo_buffer, kills_buffer = [], [], []

                # Write Rolling Statistics to file
                with open("statistics/drqn_stats.txt", "w") as stats_file:
                    stats_file.write('Game: ' + str(GAME) + '\n')
                    stats_file.write('Max Score: ' + str(max_life) + '\n')
                    stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                    stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                    stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                    stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

def episodic_buffer_test():
    mem=EpisodicReplayMemory(100)
    total_t=0
    for e in range(1000000):
        episode_buffer=[]
        for t in range(48):

            episode_buffer.append(total_t)
            total_t += 1
        mem.add(episode_buffer)

    print("")

def nonepisodic_buffer_test():
    mem=NonEpisodicReplayMemory(50)
    for t in range(1000000):
        mem.add(t)
        if t%110==0 and t>0:
            experiences=mem.sample(10,25)
            assert np.all(experiences[i]<experiences[i+1] for i in range(len(experiences)-1))
            print(experiences)
            print()

if __name__ == "__main__":
    nonepisodic_buffer_test()