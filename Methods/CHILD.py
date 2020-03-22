
from Methods.Learner import CompleteLearner
import numpy as np
import random
from overrides import overrides

ETA = .25 #learning rate
ETA_L = .09 #learning rate multiplier when there is a higher-level unit
SIGMA = .30 # updating weight for the lta and ltmad
GAMMA = .91 #discount factor for q-learning
THETA = .56 # use for decision to build new unit
EPSILON = .11 #use for decision to build new unit
DELTA_T = 2.1 #decrement of the boltzmann temperature

# high-level algorithm here


class CHILD(CompleteLearner):

    def __init__(self, actions, n_inputs, file=''):
        CompleteLearner.__init__(actions,file)
        self.n_inputs=n_inputs
        self.n_actions=len(actions)
        self.actions=actions
        self.units=[0.0 for i in self.sensoryIndexes()+self.actionIndexes()]
    def sensoryIndexes(self):
        return range(self.n_inputs)
    def actionIndexes(self):
        return range(self.n_inputs+self.n_actions)
    def highlevelIndexes(self):
        return range(self.n_inputs+self.n_actions,len(self.units))
    def nonInputIndexes(self):
        return self.actionIndexes()+self.highlevelIndexes()
    def getContext(self,i):
        if i in self.highlevelIndexes():
            return self.units[i][0] # l[0] --> return xy association, l[1] returns the activation
        else:
            raise Exception()
    def tau(self,i):
        if i in self.actionIndexes():
            return 0
        else:
            x=self.getContext(i)[0]
            return 1+self.tau(x)
    def initializeValues(self):
        for i in range(len(self.units)):
            self.units[i]=0
            self.delta[i]=0
    def unitFor(self,i,j):
        for k in self.highlevelIndexes():
            if self.units[k][0]==(i,j):
                return self.units[k][1]
        return 0.0
    def buildUnitFor(self,i,j):
        self.units.append(((i,j),0.0)) # higher-level unit associating (i,j) and current value 0.0
    def propagate(self):
        for i in self.nonInputIndexes(): #non-inputs=outputs because there are no hidden units
            for j in self.sensoryIndexes():
                l = self.unitFor(i,j)
                self.n[i] += self.units[j]*(l + self.w[i][j])
    def addToPrevious(self):
        for j in self.sensoryIndexes():
            self.previous[j].append(self.units[j])
    def setTarget(self):
        self.target = self.previousQs
        # update the target Q-value for action unit i corresponding to the chosen action
        self.target[self.chosenActionIndex] = self.r + GAMMA*max(self.Qs)


    def update(self,i,j,deltaw):
        n = self.indexFor(i,j)
        if n == -1: #l_ij^n does not exist ----> update statistics
            self.weight[i][j] -= ETA*deltaw # weight update
            self.lta[i][j] =  SIGMA*deltaw + (1-SIGMA)*self.lta[i][j]  # long term average weight update
            self.ltmad[i][j] = SIGMA*abs(deltaw) + (1-SIGMA)*self.ltmad[i][j] #long term mean absolute deviation
            if (self.ltmad[i][j] > THETA * abs(self.lta(i, j)) + EPSILON):
                self.buildUnitFor(i,j)  # build unit
                for k in self.sensoryIndexes():  # reset statistics
                    self.lta[i][k] = -1.0;
                    self.ltmad[i][k] = 0.0;
        else: #l_ij^n does exist
            self.delta[n] = deltaw
            self.weight[i][j] -= ETA_L*ETA * deltaw  # weight update with ETA_L*ETA as learning rate



    @overrides
    def setAction(self):  # determine self.chosenAction
        probs=[]
        self.Qs=self.propagate()
        for Q in self.Qs:
            probs.append(np.exp(Q / self.temperature))  # calculate numerators

        # numpy matrix element-wise division for denominator (sum of numerators)
        probs = np.true_divide(probs, sum(probs))

        s=0 #sum
        r=random.random()
        for i in range(len(probs)):
            s+=probs[i]
            if r <= s:
                self.chosenActionIndex = i
                self.chosenAction = self.actions[i]
        self.temperature -= 1/float(1+self.t*DELTA_T)

    @overrides
    def learn(self):
        for i in range(self.actionunits):
            self.delta[i] = self.a[i] - self.target[i]
        for i in range(len(self.nonInputIndexes)):
            for j in range(len(self.sensoryIndexes)):
                delta_wij = self.delta[i]*self.previous[j][-self.tau[i]]
                if delta_wij != 0:
                    self.update(i,j,delta_wij)


    @overrides
    def setObservation(self, agent, environment):

        self.t=environment.t
        environment.setObservation(agent)
        self.s=self.observation
        self.addToPrevious()

    @overrides
    def cycle(self, agent, environment):
        self.initializeValues()
        self.setObservation()
        self.propagate()
        self.setAction()
        self.setTarget()
        self.learn()




    @overrides
    def performAction(self, agent, environment):
        self.chosenAction.perform([agent,environment])
    @overrides
    def reset(self):
        """Whenever the agent begins
        a maze, the learning algorithm is first
        reset, clearing its short-term memory. activations=0, erase previous network inputs
        """
        self.units=[0 for _ in range(len(self.units))]
        self.previous=[[0 for j in range(len(self.nonInputIndexes()))] for i in range(len(self.sensoryIndexes))]

    @overrides
    def printPolicy(self):
        pass

    def printR(self):
        self.Rfile.write(str(self.R) + "\n")
        self.Rfile.flush()

    def setReward(self, reward):
        self.r = reward
        self.R += reward

    def getOutputData(self):  # can be overriden, but not required
        pass
