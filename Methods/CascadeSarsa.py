




from Learner import CompleteLearner
from overrides import overrides
from random import random,randint
import numpy as np

from CascadeCorrelation import *

BATCH_SIZE=50

class CascadeSarsa(CompleteLearner):
    def __init__(self,actions, n_in, n_out, horizon=100, file=''):
        CompleteLearner.__init__(actions, file)
        self.initNNs(n_in,n_out,horizon)

    def initNNs(self,n_in,n_out,horizon):
        self.nn=[CascadeNet(
                 input_nodes=n_in,
                 output_nodes=n_out,
                 function=tanh,
                 d_function=d_tanh,
                 weight_initialization_func=lambda: (random() - 0.3) * 0.6,
                 num_candidate_nodes=8,
                 horizon=horizon,
                 correlation=False) for i in range(len(self.actions))]
    def updateValue(self,laststate, state, lastaction,action,lastreward,lbda=1):
        if self.t < self.horizon: return
        if self.nn:

            self.nn[lastaction].
        else:
            self.Q[laststate][lastaction] += self.alpha * lbda * (lastreward + self.gamma * self.Q[state][action] - self.Q[laststate][lastaction])

    def SARSA_learn(self):
        laststate = self.states if self.nn else self.laststate
        state = self.next_states if self.nn else self.state
        self.updateValue(laststate, state, self.lastaction, self.r)
    @overrides
    def learn(self):
        if len(self.targets < BATCH_SIZE)
        self.td = self.r + (self.gamma * self.nn[self.action].getValue(self.state))
        self.targets
        self.nn[self.lastaction].train(self, inputs, targets,
              stop_error_threshold=-sys.float_info.max,
              max_hidden_nodes=10,
              mini_batch_size=10,
              max_iterations_per_epoch=12):
        # Train all the connections ending at an output unit with a usual learning algorithm until the error of the net
        # no longer decreases.
    @overrides
    def setObservation(self,agent,environment):
        environment.setObservation(agent)


    @overrides
    def cycle(self,agent,environment):
        self.setObservation(agent,environment)
        self.setAction()
        if self.laststate != None:
            self.learn()
        self.performAction(agent,environment)
        self.setReward(environment)

    @overrides
    def printPolicy(self):
        pass

    def setAction(self): #determine self.chosenAction
        self.lastaction=self.action
        if random() < self.epsilon or self.t <= self.horizon:
             self.action = randint(0,self.numActions-1)
        else:
             self.action = np.argmax(self.getValues())
        self.chosenAction = self.actions[self.action]

    def performAction(self, agent, environment):
        self.chosenAction.perform([agent,environment])
        #self.alpha = self.alpha_schedule.updateAlpha(self.alpha, self.t)

    def reset(self):
        pass

    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r

    def printR(self):
        self.Rfile.write(str(self.R) + "\n")
        self.Rfile.flush()

    def setReward(self,reward):
        self.r = reward
        self.R +=reward

    def getOutputData(self): #can be overriden, but not required
        pass
