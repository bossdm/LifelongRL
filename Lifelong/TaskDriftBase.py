
from abc import abstractmethod
import numpy as np
class VarStat(object):
    def __init__(self):
        self._n = 0
        self._M = 0
        self._S = 0
    def push(self, x):
        x = np.mean(x)
        self._n += 1
        if self._n == 1:
            self._M = x
        else:
            oldM = self._M
            self._M = oldM + (x - oldM)/self._n
            self._S = self._S + (x - oldM)*(x - self._M)
    @property
    def mean(self):
        return self._M
    @property
    def n(self):
        return self._n
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)


class TaskDriftBase(object):

    def __init__(self,episodic_performance):
        self.velocity={}
        self.episodic_performance = episodic_performance
        self.num_episodes = 0
        self.var_stat = VarStat()
    @abstractmethod
    def create_similar_policy(self):
        """
        create a new policy with the same parameters and policy
        :return:
        """
        pass

    def update_task_time(self,increment):
        self.t+=increment
    def update_task_reward(self,reward):
        self.r=reward
        self.R+=reward
    def get_task_t(self,feature):
        return self.t
    def get_task_R(self,feature):
        return self.R
    def add_to_ignored(self,ignored_t,ignored_R):
        ignored_t+=self.t
        ignored_R+=self.R
        return ignored_t, ignored_R
    def get_avg_velocity(self,current_feature=None):
        task_t=self.get_task_t(current_feature) if not self.episodic_performance else self.num_episodes
        if task_t == 0:
            return -float("inf") # since no time has passed, any policy that has some experience will have better velocity
        task_R=self.get_task_R(current_feature)
        self.velocity[current_feature]=float(task_R)/float(task_t) # used like this in BlockStackSSAs (they assume R and t are synchronised to the current task)
        return self.velocity[current_feature]
    def get_UCB(self,n,current_feature=None):
        V_max = 200 # for cartpole, max reward per episode is 200; modify if using this elsewhere
        self.var_stat.push(self.velocity[current_feature]/V_max) # --> [0,1]
        V = self.var_stat.var + np.sqrt(2*np.log(n)/self.var_stat.n)
        UCB = self.var_stat.mean + np.sqrt(np.log(n)/self.var_stat.n * min(V,1/4))
        print("UCB ",UCB, "mean", self.var_stat.mean, "V", V, "var ", self.var_stat.var, "ln(n)/n_j", np.log(n)/self.var_stat.n)
        return UCB

    def end_pol(self):
        """
        call before switching policy
        not necessarily overridden
        :return:
        """
        if self.episodic_performance:
            self.num_episodes+=1

    def set_tasks(self,occurrence_weights):
        self.total_num_tasks=len(occurrence_weights)
        for task in occurrence_weights:
            self.velocity[task]=None

    def check_policy_variables(self):
        """

        :return:
        """

        pass


