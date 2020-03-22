
from pybrain.rl.learners.valuebased.qlambda import QLambda
from pybrain.rl.learners.valuebased.nfq import NFQ
from pybrain.rl.explorers import *
from pybrain.rl.learners.learner import  *

from pybrain.rl.agents.learning import LearningAgent, LoggingAgent
from Methods.Learner import CompleteLearner
from pybrain.rl.learners.valuebased.interface import ActionValueNetwork

class NFQLearner(CompleteLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """

    def __init__(self, n_input,actions, alpha=0.5, gamma=0.99, qlambda=0.9,explorer=EpsilonGreedyExplorer(epsilon=0.20,decay=1)):
        CompleteLearner.__init__(self,actions)
        controller = ActionValueNetwork(dimState=n_input, numActions=len(actions))
        learner = NFQ()
        learner.explorer = explorer
        self.learning_agent = LearningAgent(controller, learner)

    def setAction(self): #determine self.chosenAction
        self.chosenAction = self.actions[int(round(self.learning_agent.getAction()))]



    def learn(self):
        print('epsilon:' + str(self.learning_agent.learner.explorer.epsilon))
        if(self.t > 0 and self.t % 1000 == 0):
            self.learning_agent.learner.learn()
            self.learning_agent.lastobs = None
            self.learning_agent.lastaction = None
            self.learning_agent.lastreward = None
            self.learning_agent.history.clear()

    def setObservation(self,agent,environment):
        environment.setObservation(agent)
        self.t = environment.t
        self.learning_agent.integrateObservation(self.observation)

        print("observation =" + str(self.observation))

    def cycle(self,agent,environment):

        self.setObservation(agent,environment)
        self.setAction()
        self.performAction(agent,environment)
        self.setReward(environment)
        self.learn()




    def performAction(self, agent, environment):
        print('chosenAction='+str(self.chosenAction.function.__name__))
        self.chosenAction.perform([agent,environment])


    def reset(self):
        pass

    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r
        self.learning_agent.giveReward(self.r)


class DeepQLearner(CompleteLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """

    def __init__(self, n_input,actions, alpha=0.5, gamma=0.99, qlambda=0.9,explorer=EpsilonGreedyExplorer(epsilon=0.20,decay=1)):
        CompleteLearner.__init__(self,actions)
        controller = ActionValueNetwork(dimState=n_input, numActions=len(actions))
        learner = NFQ()
        learner.explorer = explorer
        self.learning_agent = LearningAgent(controller, learner)

    def setAction(self): #determine self.chosenAction
        self.chosenAction = self.actions[int(round(self.learning_agent.getAction()))]



    def learn(self):
        print('epsilon:' + str(self.learning_agent.learner.explorer.epsilon))
        if(self.t > 0 and self.t % 1000 == 0):
            self.learning_agent.learner.learn()
            self.learning_agent.lastobs = None
            self.learning_agent.lastaction = None
            self.learning_agent.lastreward = None
            self.learning_agent.history.clear()

    def setObservation(self,agent,environment):
        environment.setObservation(agent)
        self.t = environment.t
        self.learning_agent.integrateObservation(self.observation)

        print("observation =" + str(self.observation))

    def cycle(self,agent,environment):

        self.setObservation(agent,environment)
        self.setAction()
        self.performAction(agent,environment)
        self.setReward(environment)
        self.learn()




    def performAction(self, agent, environment):
        print('chosenAction='+str(self.chosenAction.function.__name__))
        self.chosenAction.perform([agent,environment])


    def reset(self):
        pass

    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r
        self.learning_agent.giveReward(self.r)
