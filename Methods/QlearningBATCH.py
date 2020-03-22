

from random import random, randint

from Methods.Learner import CompleteLearner
from keras.optimizers import RMSprop, Adadelta
import numpy as np
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from overrides import overrides

FREEZE_INTERVAL=1000

class AlphaSchedule(object):
    def __init__(self, decay, times):
        self.decay =decay
        self.times = times
    def updateAlpha(self, alpha, t):
        if t in self.times:
            return alpha*self.decay
        else:
            return alpha
class ReplayBuffer(object):
    def __init__(self, maxSize, batch_size, horizon):
        self.maxSize = maxSize
        self.batch_size = batch_size
        self.horizon = horizon
        self.data = {'laststate': [],'state':[],'action':[],'reward':[]}
    def addElement(self,laststate,state,action,reward):
        if len(self.data) > self.maxSize:
            self.data['laststate'].pop(0)
            self.data['state'].pop(0)
            self.data['action'].pop(0)
            self.data['reward'].pop(0)
        self.data['laststate'].append(laststate)
        self.data['state'].append(state)
        self.data['action'].append(action)
        self.data['reward'].append(reward)
    def getRandomBatch(self):
        i=0
        batch={'laststate':[],'state':[],'action':[],'reward': []}
        while i < self.batch_size:
            r = randint(0,len(self.data['state'])-self.horizon)
            batch['laststate'].append(self.data['laststate'][r:r+self.horizon])
            batch['state'].append(self.data['state'][r:r+self.horizon])
            batch['action'].append(self.data['action'][r+self.horizon-1])
            batch['reward'].append(self.data['reward'][r+self.horizon-1])
            i+=1
        return batch


BATCH_SIZE = 100
REPLAY_MEMORY = 10000
UPDATE_FREQ = 100

default_alpha_schedule = AlphaSchedule(.10,[2*10**6,20*10**6])
class QLambdaLearner(CompleteLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """

    def __init__(self, n_input,actions,file,states=None, epsilon=.05, alpha=0.02, horizon=100,gamma=0.95, qlambda=0.9,network=False):
        self.epsilon=epsilon
        self.horizon=horizon
        self.n_input=n_input
        self.alpha = alpha
        self.gamma = gamma
        self.qlambda = qlambda
        CompleteLearner.__init__(self,actions,file)
        self.numActions=len(self.actions)
        self.laststate=None
        self.state=None
        self.dataset = {'laststate':[],'state':[],'action':[],'reward': []}
        self.nn = None
        self.target_nn=None
        self.t=0
        if network:
            self.batch_size = BATCH_SIZE
            self.replay_memory = REPLAY_MEMORY
            self.update_freq = UPDATE_FREQ
            if self.replay_memory:
                self.replay_buffer = ReplayBuffer(self.replay_memory,batch_size=self.batch_size,horizon=self.horizon)
            self.nn , self.params= self.initNetwork()
            self.target_nn = self.nn
            self.target_params=self.params
        else:
            self.states=states
            self.Q = {}
            for state in self.states:
                self.Q[state]=[]
                for a in range(self.numActions):
                    self.Q[state].append(self.gamma + random())




    def initNetwork(self):
        nn = Sequential()
        nn.add(Dense(self.n_input,batch_input_shape=(self.batch_size,self.horizon,self.n_input)))
        # self.nn.add(Activation('relu'))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you wanna give it a try?
        # nn.add(Dense(20))
        # nn.add(Activation('relu'))
        nn.add(LSTM(20,stateful=False))
        nn.add(Activation('relu'))
        # model.add(Dropout(0.2))
        nn.add(Dense(20))
        nn.add(Activation('relu'))
        nn.add(Dense(output_dim=self.numActions, init='lecun_uniform'))
        nn.add(Activation('linear'))  # linear output so we can have range of real-valued outputs

        #rms = RMSprop(lr=self.alpha)
        adad = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0)  # == default settings
        nn.compile(loss='mse', optimizer=adad)

        layers = nn.layers

        # Grab all the parameters together.
        params = [param
                       for layer in layers
                       for param in layer.trainable_weights]
        return nn, params

    def getAllParams(self):
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(p.get_value())
        return params_value

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            p.set_value(list_of_values[i])
    def _resetQHat(self):
        for i, (param, target_param) in enumerate(zip(self.params, self.target_params)):
            K.set_value(target_param, K.get_value(param))
    def setAction(self): #determine self.chosenAction
        if self.nn and self.t % FREEZE_INTERVAL == 0:
            self._resetQHat()
        if random() < self.epsilon or self.t <= self.horizon:
             self.lastaction = randint(0,self.numActions-1)
        else:
             self.lastaction = np.argmax(self.getValues(np.array(self.dataset['laststate']).reshape(1,self.horizon,self.n_input)))
        self.chosenAction = self.actions[self.lastaction]
    def getValues(self,batch):
        if self.nn:
            return self.nn.predict(batch).flatten()
        else:
            return self.Q[self.laststate]
    def getNextValues(self,next_batch):
        if self.target_nn:
            return self.target_nn.predict(next_batch).flatten()
        else:
            return self.Q[self.state]

    def updateValue(self,dataset, lbda):
        if self.t < self.horizon: return

        qvalues = self.getValues(dataset['laststate'])
        lastaction=dataset['action']
        qvalue = qvalues[lastaction]
        laststate=dataset['laststate']
        lastreward=dataset['lastreward']
        maxnext=self.getNextValues(dataset['state'])
        self.Q[laststate][lastaction]=qvalue + self.alpha * lbda * (lastreward + self.gamma * maxnext - qvalue)
    def updateValuesNN(self,batch):
        # for each element in the batch, get the target Q-value
        qvalues = np.array(self.getValues(np.array(batch['laststate']).reshape((self.batch_size,self.horizon,self.n_input)))).reshape((self.batch_size,self.numActions))
        if qvalues is None: return
        next_qvalues = np.array(self.getNextValues(np.array(batch['state']).reshape((self.batch_size,self.horizon,self.n_input)))).reshape((self.batch_size,self.numActions))

        if self.nn:
            for i in range(self.batch_size):
                max_q_index = np.argmax(next_qvalues[i])
                maxnext = next_qvalues[i][max_q_index]
                update = (batch['reward'][i] + (self.gamma * maxnext))
                lastaction = batch['action'][i]
                qvalues[i][lastaction] = update
        self.nn.fit(np.array(batch['state']).reshape(self.batch_size,self.horizon,self.n_input),qvalues.reshape((self.batch_size,self.numActions)),batch_size=self.batch_size,nb_epoch=1,verbose=0)
    def updateValueBatch(self):
        if self.t > 0 and self.t % self.update_freq != 0:
            return
        if self.replay_memory:
            batch = self.replay_buffer.getRandomBatch()
        else:
            batch=self.dataset
        self.updateValuesNN(batch)



    def Qlambda_learn(self):

        states = self.dataset['state']
        if (len(states) < self.horizon):
            return
        assert len(states)==self.horizon
        actions = self.dataset['action']
        rewards = self.dataset['reward']

        for i in range(self.horizon - 1, 0, -1):
            lbda = self.qlambda ** (self.horizon - 1 - i)
            # if eligibility trace gets too long, break
            if lbda < 0.0001:
                break

            state = states[i]
            laststate = states[i - 1]
            # action = int(actions[i])
            lastaction = int(actions[i - 1])
            lastreward = rewards[i - 1]
            self.updateValue(laststate, state, lastaction, lastreward, lbda)
    def Qlearn(self):
        if self.nn:
            self.updateValueBatch()
        else:
            self.updateValue(self.dataset)

    @overrides
    def learn(self):
        if self.qlambda > 0:
            self.Qlambda_learn()
        else:
            if self.t >= self.horizon + self.batch_size:
                self.Qlearn()
    @overrides
    def setObservation(self,agent,environment):
        environment.setObservation(agent)
        self.t = environment.t
        self.laststate = self.state
        self.state = tuple(self.observation + [self.r])
        if len(self.dataset['state']) >= self.horizon:
            self.dataset['laststate']=self.dataset['laststate'][1:]
            self.dataset['action'] = self.dataset['action'][1:]
            self.dataset['reward'] = self.dataset['reward'][1:]
            self.dataset['state'] = self.dataset['state'][1:]

        print(self.t)
        if self.laststate:
            self.dataset['laststate'].append(self.laststate)
            self.dataset['state'].append(self.state)
            self.dataset['reward'].append(self.r)
            self.dataset['action'].append(self.lastaction)
            if self.replay_memory:
                self.replay_buffer.addElement(self.laststate,self.state,self.lastaction,self.r)
            # if self.t < self.horizon and self.replay_buffer:
            #     assert self.replay_buffer.data['laststate'] == self.dataset['laststate']
            #     assert self.replay_buffer.data['state'] == self.dataset['state']
            #     assert self.replay_buffer.data['reward'] == self.dataset['reward']
            #     assert self.replay_buffer.data['action'] == self.dataset['action']


    @overrides
    def cycle(self,agent,environment):
        self.setObservation(agent,environment)
        if self.laststate != None:
            self.learn()
        self.setAction()
        self.performAction(agent,environment)
        self.setReward(environment)
    @overrides
    def printPolicy(self):
        pass



    def performAction(self, agent, environment):
        self.chosenAction.perform([agent,environment])
        #self.alpha=default_alpha_schedule.updateAlpha(self.alpha,self.t)

    def reset(self):
        pass

    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r

