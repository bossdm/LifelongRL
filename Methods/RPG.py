""" Implementation of the RPG algorithm

"""




from random import random, randint

from Methods.Learner import CompleteLearner

import numpy as np

from pyrl.policygradient import PolicyGradient
from overrides import overrides



class RPG(CompleteLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """

    def __init__(self, n_input,actions,file,states=None, alpha=0.02, horizon=100,gamma=0.95):
        #self.epsilon=epsilon
        self.horizon=horizon
        self.n_input=n_input
        self.alpha = alpha
        self.gamma = gamma
        CompleteLearner.__init__(self,actions,file)
        self.t=0
        self.rpg=PolicyGradient(self, Task, config_or_savefile, seed, dt=None, load='best')
    @overrides
    def learn(self):
        self.rpg.
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
        if self.nn and self.t >= self.horizon :
            self.states=np.array(self.dataset['state'][:-1]).reshape(1,self.horizon-1,self.n_input)
            self.next_states=np.array(self.dataset['state'][1:]).reshape(1,self.horizon-1,self.n_input)

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
        #self.alpha = self.alpha_schedule.updateAlpha(self.alpha, self.t)

    def reset(self):
        pass

    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r

