
import numpy as np
import os
from ExperimentUtils import dump_incremental
from abc import abstractmethod, ABCMeta
from StatsAndVisualisation.LearnerStatistics import LearnerStatistics


class CompleteLearner(object):
    __metaclass__ = ABCMeta
    __globals__ = None
    network_usage=False
    has_internal_actions=False
    testing=False
    def __init__(self,actions, file='',episodic=False):
        self.episodic=episodic
        self.file=file
        self.Rfile = open(file+"R.txt","a")
        self.actions=actions
        self.t=0
        self.r = 0
        self.R = 0
        self.observation=[]
        self.chosenAction=None
        self.initStats()
        self.random_steps = 0

    def continue_experiment(self,args):
        pass

    def save(self,filename):
        pass

    def load(self,filename):
        pass
    def print_initialisation(self):
        print("actions : \n")
        for action in self.actions:
            print("%s \n"%(str(action)))

    def setAction(self): #determine self.chosenAction
        pass



    def learn(self):
        pass


    def setObservation(self,agent,environment):
        pass
    @abstractmethod
    def setAtariTerminalObservation(self,obs):
        pass
    @abstractmethod
    def atari_cycle(self, observation, reward):
        pass
    @abstractmethod
    def cycle(self,agent,environment):
        pass


    def performAction(self, agent, environment):
        pass


    def setTerminalObservation(self,agent,environment):
        pass
    def reset(self):
        pass

    @abstractmethod
    def printPolicy(self):
        pass

    def setTime(self,t):
        self.t = t
    def new_elementary_task(self):
        pass
    def new_task(self,feature):
        pass
    def end_task(self):
        pass
    def writeRtofile(self):
        self.Rfile.write("%.2f \n" % (self.R))
        self.Rfile.flush()
        os.fsync(self.Rfile)
    def printR(self):
        if self.Rfile is not None:
            self.writeRtofile()
        self.stats.R_overTime.append(self.R)
    def printDevelopment(self):
        self.printR()
    def printDevelopmentAtari(self,frames):
        if self.Rfile is not None:
            self.Rfile.write("%.2f \t %d \n" % (self.R, frames))
            self.Rfile.flush()
            os.fsync(self.Rfile)
        self.stats.R_overTime.append(self.R)
    def printStatistics(self):
        pass
    def initStats(self):
        self.stats=LearnerStatistics(self)
    def setReward(self,reward):
        self.r = reward
        self.R +=reward

    def getOutputData(self): #can be overriden, but not required
        pass

    def save_stats(self,filename):
        dump_incremental(filename + '_stats_object',self.stats)

class SupervisedLearner(object):
    def __init__(self,file):
        self.file = file
        self.initStats()
        #self.test_accuracyfile = {} # track test accuracy
    def initStats(self):
        self.stats=None
    def save(self, filename):
        pass

    def load(self, filename):
        pass

    @abstractmethod
    def train(self,x_train,y_train,x_val,y_val):
        """
        train on dataset
        :param x_train:
        :param y_train:
        :param x_val:
        :param y_val:
        :return:
        """
        pass
    @abstractmethod
    def test(self,x_test,y_test):
        """
        test the learner on unseen dataset
        :param x_test:
        :param y_test:
        :return:
        """
        pass
    def new_task(self,feature):
        pass

    def save_stats(self,filename):
        if self.stats is not None:
            dump_incremental(filename + '_stats_object',self.stats)