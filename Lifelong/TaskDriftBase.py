
from abc import abstractmethod

class TaskDriftBase(object):

    def __init__(self,episodic_performance):
        self.velocity={}
        self.episodic_performance = episodic_performance
        self.num_episodes = 0
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


    def end_pol(self):
        """
        call before switching policy
        not necessarily overridden
        :return:
        """
        if self.episodic_performance:
            self.num_episodes+=1

    def set_tasks(self,occurence_weights):
        self.total_num_tasks=len(occurence_weights)
        for task in occurence_weights:
            self.velocity[task]=None

    def check_policy_variables(self):
        """

        :return:
        """

        pass


