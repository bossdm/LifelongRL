#!/usr/bin/env python
from __future__ import print_function


import numpy as np
import scipy.signal

import tensorflow.compat.v1 as tf

import keras.backend as K
from keras.models import load_model

from Catastrophic_Forgetting_NNs.CustomNetworks import entropy_regularisation,entropy_bonus
from Catastrophic_Forgetting_NNs.PPO_objective2 import Policy

from overrides import overrides

DEBUG_MODE=False


def preprocessImg(img, size):
    img = np.rollaxis(img, 0, 3)  # It becomes (640, 480, 3)
    img = skimage.transform.resize(img, size)

    return img
class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.mean(x)
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape




class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape, demean=True, destd=True, clip=10.0):
        self.demean = demean
        self.destd = destd
        self.clip = clip

        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        if self.demean:
            x = x - self.rs.mean
        if self.destd:
            x = x / (self.rs.std+1e-8)
        if self.clip:
            x = np.clip(x, -self.clip, self.clip)
        return x
    def output_shape(self, input_space):
        return input_space.shape

class A2CAgent:
    def __init__(self, state_size, action_size, trace_length,episodic):

        self.episodic=episodic
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size
        self.trace_length = trace_length
        self.value_size = 1
        self.observe = 0

        # These are hyper parameters for the Policy Gradient
        self.gamma = 0.99
        self.learning_rate = 0.00025
        self.update_freq = 100
        self.batch_size=34
        self.beta = 0.01 # entropy regularisation
        self.epochs=1
        #self.normalise=ZFilter(shape=(1,),demean=True,destd=True,clip=False)


        # create model for actor critic network
        self.model =  None#Networks.a2c_lstm(state_size, action_size, self.value_size, self.learning_rate)
        self.reset_states()
        self.mean_r=0
        self.var_r=0
        self.N=0
        self.N_v=0
        self.target_model=None




    # load the saved model
    def load(self, name):
        #self.model.load_weights(name)
        self.model = load_model(name+"_network.h5",
                                custom_objects={'entropy_regularisation': entropy_regularisation})


    # save the model which is under training
    def save(self, name):
        #self.model.save_weights(name)
        self.model.save(name+"_network.h5")
        del self.model

    def reset_states(self):
        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

    # using the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        policy = self.model.predict(state)[0].flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0], policy

    # Instead agent uses sample returns for evaluating policy
    # Use TD(1) i.e. Monte Carlo updates
    # def discount_rewards(self, rewards):
    #     discounted_rewards = np.zeros_like(rewards)
    #     running_add = 0
    #     for t in reversed(range(0, len(rewards))):
    #         if rewards[t] != 0:
    #             running_add = 0
    #         running_add = running_add * self.gamma + rewards[t]
    #         discounted_rewards[t] = running_add
    #     return discounted_rewards
    def get_final_value(self,values,terminal):
        if terminal:
            return 0
        else:
            return values[-1]
    def discount_rewards(self,rewards,terminal,values):
        discounted_rewards = np.zeros(len(rewards)+1)

        discounted_rewards[-1]=self.get_final_value(values,terminal)


        for t in range(len(rewards)-1,-1,-1):
            discounted_rewards[t] = rewards[t] + self.gamma*discounted_rewards[t+1]


        discounted_rewards=discounted_rewards #self.normalise(


        return discounted_rewards[0:-1]



    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)
    # def update_mean_std(self,rewards):
    #     """
    #     note: assume each episode is equally long
    #     :param rewards:
    #     :return:
    #     """
    #     self.N+=len(rewards)
    #     self.N_v+=len(rewards)-1
    #     mean=np.mean(rewards)
    #     var=np.var(rewards)
    #     alpha=len(rewards)/float(self.N)
    #     self.mean_r=alpha*mean + (1-alpha)*self.mean_r
    #     alpha_v=(len(rewards)-1)/float(self.N_v)
    #     self.var_r = alpha_v*var  + (1-alpha_v)*self.var_r


    def get_advantages(self,terminal,values,episode_length):
        discounted_rewards = self.discount_rewards(self.rewards,terminal,values)

        # # Standardized discounted rewards
        # discounted_rewards -= np.mean(discounted_rewards)
        # if np.std(discounted_rewards):
        #     discounted_rewards /= np.std(discounted_rewards)
        # else:
        #     self.states, self.actions, self.rewards = [], [], []
        #     print ('std = 0!')
        #     return 0
        # Similar to one-hot target but the "1" is replaced by Advantage Function i.e. discounted_rewards R_t - Value
        advantages = np.zeros((episode_length-1, self.action_size))

        for i in range(episode_length-1):
            advantages[i][self.actions[i]] = (discounted_rewards[i] - values[i])

        return advantages,discounted_rewards



    def get_target_values(self,state_inputs):
        # Prediction of state values for each state appears in the episode
        values = self.model.predict(state_inputs)[1]

        return values
    # update policy network every N steps
    def train_model(self,terminal):

        episode_length = len(self.states)
        #print("train model, episode length ", episode_length)
        #print("train model, terminal ", terminal)
        state_inputs = np.zeros(((episode_length,) + self.state_size))  # Episode_lengthx4x64x64x3

        # Episode length is like the minibatch size in DQN
        for i in range(episode_length):
            if self.ppo.recurrent:
                state_inputs[i, :, :] = self.states[i]
            else:
                state_inputs[i, :] = self.states[i]


        values=self.get_target_values(state_inputs)

        advantages,discounted_rewards=self.get_advantages(terminal,values.flatten(),episode_length)


        loss=self.update(state_inputs[0:-1],advantages,discounted_rewards)

        self.reset_states()

        return loss


    def update(self,state_inputs,advantages,discounted_rewards):
        loss = self.model.fit(state_inputs, [advantages, discounted_rewards], epochs=self.epochs,
                              batch_size=self.batch_size,verbose=0)
        return loss.history['loss']





class PPO_Agent(A2CAgent):
    def __init__(self, state_size, action_size, trace_length, episodic,params, large_scale,recurrent):
        A2CAgent.__init__(self,state_size,action_size,trace_length,episodic)
        self.update_freq=1000
        self.lbda=.95
        self.epochs = 3
        self.learning_rate=params['learning_rate']
        self.large_scale=large_scale
        self.init_PPO(recurrent=recurrent)
    def init_PPO(self,filename=None,w=None,recurrent=True):
        self.ppo=Policy(self.state_size,self.action_size,neurons=80,learning_rate=self.learning_rate,
                        clipping=0.10,epochs=self.epochs,c1=1.0,c2=self.beta,filename=filename,
                        w=w,large_scale=self.large_scale,recurrent=recurrent)


    def get_advantages(self,terminal,values,episode_length):
        """ Add generalized advantage estimator.
        https://arxiv.org/pdf/1506.02438.pdf

        """

        # temporal differences

        next_vals= np.append(values[1:-1], self.gamma*self.get_final_value(values,terminal))
        #print("reward",len(self.rewards))
        #print("next_vals", len(next_vals))
        #print("values", len(values[0:-1]))
        tds = np.array(self.rewards) + next_vals - values[0:-1]# r_t + V(s_{t+1}) - V(s_t)  (target - value)
        advantages = self.discount(tds, self.gamma * self.lbda)

        discounted_rewards=self.discount_rewards(self.rewards,terminal,values)
        return advantages,discounted_rewards
    def one_hot_actions(self,indices):
        acts = np.zeros((len(indices), self.action_size))

        for i in range(len(indices)):
            acts[i][self.actions[indices[i]]] = 1
        return acts


    # @overrides
    # def update(self,state_inputs,advantages,discounted_rewards):
    #     p_old=self.model.predict(state_inputs)[0]
    #     advantages=np.expand_dims(advantages,axis=1)
    #
    #     loss=0
    #
    #     end=len(self.rewards)
    #     for epoch in range(self.epochs):
    #         i=0
    #         while i < end:
    #             batch_end=min(end,i+self.batch_size)
    #             self.p_old=p_old[i:batch_end]
    #             self.advantages=advantages[i:batch_end]
    #             acts=self.one_hot_actions(range(i,batch_end))
    #             self.compile_model()
    #             loss = self.model.train_on_batch(state_inputs[i:batch_end], [acts, discounted_rewards[i:batch_end]])
    #             i+=self.batch_size
    #             if DEBUG_MODE:
    #                 print("loss="+str(loss))
    #     return loss
    def update(self,state_inputs,advantages,discounted_rewards):
        end=len(self.rewards)
        for e in range(self.epochs):
            i=0
            while i < end:
                batch_end=min(end,i+self.batch_size)
                advs=advantages[i:batch_end]
                acts=self.one_hot_actions(range(i,batch_end))
                states=state_inputs[i:batch_end]
                rs=np.expand_dims(discounted_rewards[i:batch_end],axis=1)
                loss=self.ppo.update_batch(states, acts, advs, rs)
                if DEBUG_MODE:
                    print("epoch %d, batch index %d" % (e,i))
                    print(loss)
                i += self.batch_size

    def discount(self,x, gamma):
        """ Calculate discounted forward sum of a sequence at each point """
        return scipy.signal.lfilter([1.0], [1.0, -gamma], x[::-1])[::-1]




    # using the output of policy network, pick action stochastically (Stochastic Policy)
    def get_action(self, state):
        policy = self.ppo.get_probability(state).flatten()
        if np.isnan(policy).any():
            print("state",state)
            print("policy:",policy)
            print(self.ppo.get_all_weights_list())
        return np.random.choice(self.action_size, 1, p=policy)[0], policy


    def get_target_values(self,state_inputs):
        values= self.ppo.get_values(state_inputs)
        return values


    # load the saved model
    def load(self, name):
        #self.model.load_weights(name)
        self.init_PPO(name+'_session')

    # save the model which is under training
    def save(self, name):
        self.ppo.save(name)
        del self.ppo




    # def objective(self,y_true,y_pred):
    #     """
    #
    #     y_true is indicator, 1 for the chosen action
    #
    #     y_pred are the probabilities
    #
    #     p_old are the old probabilities
    #
    #     """
    #
    #     p_new=-K.categorical_crossentropy(y_true,y_pred) # ---> log(p_new(a))
    #     p_old=-K.categorical_crossentropy(y_true,y_pred) # ---> log(p_old(a))
    #     r=K.exp(p_new - p_old)  # exp(log(p_new(a)/p_old(a)))=exp(log(p_new(a)) - log(p_old(a)))
    #     return self.loss_clip(r,self.advantages)  + entropy_bonus(y_pred)
    # def loss_clip(self,r,advantage):
    #     v=r*advantage
    #     clipped=K.clip(r, 1 - self.clipping, 1 + self.clipping)
    #     clipped_v=clipped*advantage
    #     return - K.minimum(v,clipped_v)
    #
    # def compile_model(self):
    #     """
    #     """
    #
    #     #print('setting up loss with clipping objective')
    #     optimizer = tf.train.AdamOptimizer(self.lr)
    #     self.model.compile(optimizer=optimizer, loss=[self.objective, 'mse'], loss_weights=[1., self.c_1])







