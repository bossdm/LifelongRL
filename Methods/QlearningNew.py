

from random import random, randint

from Methods.Learner import CompleteLearner

import numpy as np
from keras.models import Sequential
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop
from overrides import overrides

FREEZE_INTERVAL=1000

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
        self.updates=[]
        if network:
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
        from pybrain.structure import *

        n = RecurrentNetwork()
        n.addInputModule(LinearLayer(self.n_input, name='in'))
        n.addModule(LSTMLayer(20, name='lstm'))
        n.addModule(SigmoidLayer(20, name='hidden'))
        n.addOutputModule(LinearLayer(self.numActions, name='out'))
        n.addConnection(FullConnection(n['in'], n['lstm'], name='c1'))
        n.addConnection(FullConnection(n['lstm'], n['hidden'], name='c2'))
        n.addConnection(FullConnection(n['hidden'], n['out'], name='c2'))
        n.addRecurrentConnection(FullConnection(n['lstm'], n['lstm'], name='c3'))

        n.sortModules()

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            p.set_value(list_of_values[i])
    def _resetQHat(self):
        self.target_nn._setParameters(self.nn.params)
    def setAction(self): #determine self.chosenAction
        if self.nn and self.t % FREEZE_INTERVAL == 0:
            self._resetQHat()
        if random() < self.epsilon or self.t <= self.horizon:
             self.lastaction = randint(0,self.numActions-1)
        else:
             self.lastaction = np.argmax(self.getValues())
        self.chosenAction = self.actions[self.lastaction]
    def getValues(self):
        if self.nn:
            return self.nn.predict(self.states, batch_size=1).flatten()
        else:
            return self.Q[self.laststate]
    def getTargetValues(self):
        if self.target_nn:
            return self.target_nn.predict(self.next_states, batch_size=1).flatten()
        else:
            return self.Q[self.state]
    def updateValue(self,laststate, state, lastaction,lastreward,lbda=1):
        if self.t < self.horizon: return
        qvalues = self.getValues()
        if qvalues is None: return
        qvalue = qvalues[lastaction]
        next_qvalues = self.getTargetValues()
        max_q_index = np.argmax(next_qvalues)
        maxnext = next_qvalues[max_q_index]
        if self.nn:
            update = (lastreward + (self.gamma * maxnext))
            qvalues[lastaction] = update
            from pybrain.supervised import RPropMinusTrainer
            trainer = RPropMinusTrainer(self.nn)
            dataset = Sequential()
            trainer.trainOnDataset(dataset)
        else:
            self.Q[laststate][lastaction]=qvalue + self.alpha * lbda * (lastreward + self.gamma * maxnext - qvalue)

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
        laststate = self.states if self.nn else self.laststate
        state = self.next_states if self.nn else self.state
        self.updateValue(laststate, state,self.lastaction, self.r)

    @overrides
    def learn(self):
        print(self.t)
        if self.qlambda > 0:
            self.Qlambda_learn()
        else:
            if self.t >= self.horizon:
                self.Qlearn()
    @overrides
    def setObservation(self,agent,environment):
        environment.setObservation(agent)
        self.t = environment.t


        self.laststate = self.state
        self.state = tuple(self.observation + [self.r])
        if len(self.dataset['state']) >= self.horizon:
            self.dataset['state']=self.dataset['state'][1:]
            self.dataset['action'] = self.dataset['action'][1:]
            self.dataset['reward'] = self.dataset['reward'][1:]
        self.dataset['state'].append(self.state)
        if self.laststate:
            self.dataset['reward'].append(self.r)
            self.dataset['action'].append(self.lastaction)

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


    def reset(self):
        pass

    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r

