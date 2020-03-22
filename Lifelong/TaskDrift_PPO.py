



import numpy as np
from Catastrophic_Forgetting_NNs.A2C_Learner import PPO_Learner
from copy import deepcopy
from overrides import overrides
from TaskDriftBase import TaskDriftBase
from keras.models import Model, load_model, clone_model

class TaskDriftPPO(PPO_Learner,TaskDriftBase):
    epsilon_change=False
    def __init__(self,PPO_Params):
        PPO_Learner.__init__(self,**deepcopy(PPO_Params))
        TaskDriftBase.__init__(self)
        self.PPO_Params=PPO_Params
        self.task_t={}
        self.task_R={}
    def get_single_array(self):
        return self.agent.ppo.get_all_weights()
    @classmethod
    def get_diversity(cls,PPO_instances):
        # policy_vectors=[instance.get_single_array() for instance in PPO_instances]
        # M=np.max(policy_vectors)
        # m=np.min(policy_vectors)
        # return np.array(policy_vectors).std()/float(M-m)
        return 0.0
    @overrides
    def create_similar_policy(self):
        """
        create a new policy with the same parameters and policy
        :return:
        """

        new_instance = TaskDriftPPO(self.PPO_Params)
        new_instance.set_policy(self.get_policy_copy())
        return new_instance
    @overrides
    def new_task(self,feature):
        self.current_feature=feature
    @overrides
    def update_task_time(self,increment):
        self.task_t[self.current_feature] += increment

        self.total_t+=increment
        self.t += increment

    @overrides
    def update_task_reward(self,reward):
        self.task_R[self.current_feature] += reward
        self.r =reward
        self.R += reward
        self.add_sample()
    @overrides
    def add_to_ignored(self,ignored_t,ignored_R):
        for feature in self.task_R:
            ignored_t+=self.task_t[feature]
            ignored_R+=self.task_R[feature]
        return ignored_t,ignored_R

    @overrides
    def set_tasks(self, occurrence_weights):
        TaskDriftBase.set_tasks(self, occurrence_weights)
        for task in occurrence_weights:
            self.task_t[task] = 0
            self.task_R[task] = 0

    @overrides
    def get_task_R(self, feature):
        return self.task_R[feature]

    @overrides
    def get_task_t(self, feature):
        return self.task_t[feature]
