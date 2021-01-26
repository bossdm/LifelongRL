from random import randint

from Actions.SpecialActions import *

from Methods.Learner import CompleteLearner


class RandomLearner(CompleteLearner):
    def __init__(self,actions,filename):
        #self.actions=actions
        super(RandomLearner,self).__init__(actions,filename)
    def setAction(self): #determine self.chosenAction
        index = randint(0,len(self.actions)-1)
        self.chosenAction = self.actions[index]
        #print('action= ' + self.chosenAction.function.__name__)

    def learn(self):
        pass
    def setObservation(self,agent,environment):
        pass

    def setReward(self, num):
        self.r = num
        self.R += self.r
    def cycle(self,agent,environment):
        self.setAction()
        self.performAction(agent,environment)
    def atari_cycle(self, observation):
        self.setAction()
    def performAction(self, agent, environment):
        if isinstance(agent.learner.chosenAction,ExternalAction):
            agent.learner.chosenAction.perform([agent,environment])
    def reset(self):
        pass
    def printPolicy(self):
        pass