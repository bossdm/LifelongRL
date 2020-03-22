from abc import abstractmethod
from StatsAndVisualisation.metrics import getDifferential
import numpy as np



class LearnerStatistics(object):
    def __init__(self,learner):
        self.R_overTime = []
        self.correctness = []
        self.correlation = {}  # generic: correlation between two properties (supply couple-name as key)
        self.problem_specific = {} # e.g., best path, worst path, sd path
        self.result=None

    @abstractmethod
    def update(self,learner):
        pass


    def initialise_statistics(self):
        """common for every method: reward speed"""
        self.finalstats={}
        self.developstats={}
        self.summary={}
        self.development={}

        self.developstats['correctness'] = self.correctness
        self.developstats['V']=[]
        self.developstats['R']=[self.R_overTime[0]]


    # def task_specific_development(self,task_markers,Vstep,opt_speeds):


    def development_statistics(self,total_samples,Vstep,opt_speed=1.):
        for j in range(0, total_samples, Vstep):
            self.development_statistics_iteration(j,Vstep,opt_speed)


        V = self.developstats['V']
        rews = self.R_overTime
        self.finalstats['avgVforRun'] = sum(V) / len(V)
        # non-normalised speeds
        initial_speed=rews[0 + Vstep] - rews[0]
        final_speed = rews[-1] - rews[-1 - Vstep]
        diff_speed = final_speed - initial_speed
        self.finalstats['initial_speed'] = initial_speed
        self.finalstats['final_speed'] = final_speed
        self.finalstats['diff_speed'] = diff_speed
        self.finalstats['prop_diffspeed'] = diff_speed / (initial_speed + .00000001)
        self.finalstats['avg_speed'] = sum(rews) / len(rews)



    def development_statistics_iteration(self,j,Vstep,opt_speed=1.):
        speed = self.R_overTime[j + Vstep] - self.R_overTime[j]
        proportional_speed = speed / opt_speed
        self.developstats['V'].append(proportional_speed)
        self.developstats['R'].append(self.R_overTime[j+Vstep])

    def pretty_print(self):
        for stat,numbers in self.developstats.items():
            print(stat + ": \n" + str(numbers))

        for stat,numbers in self.finalstats.items():
            print(stat + ": \n" + str(numbers))





