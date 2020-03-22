



import numpy as np
from Catastrophic_Forgetting_NNs.A2C_Learner import A2C_Learner
from copy import deepcopy
from overrides import overrides
from TaskDriftBase import TaskDriftBase
from keras.models import Model, load_model, clone_model

class TaskDriftA2C(A2C_Learner,TaskDriftBase):
    epsilon_change=False
    def __init__(self,A2C_Params):
        A2C_Learner.__init__(self,**deepcopy(A2C_Params))
        TaskDriftBase.__init__(self)
        self.A2C_Params=A2C_Params
        self.task_t={}
        self.task_R={}
    def get_single_array(self):
        weights=np.array([])
        for layer in self.agent.model.layers:
            layer_weights = layer.get_weights()
            for W in layer_weights:
                weights=np.append(weights,W)
        return weights
    @classmethod
    def get_diversity(cls,A2C_instances):
        policy_vectors=[instance.get_single_array() for instance in A2C_instances]
        M=np.max(policy_vectors)
        m=np.min(policy_vectors)
        return np.array(policy_vectors).std()/float(M-m)
    @overrides
    def create_similar_policy(self):
        """
        create a new policy with the same parameters and policy
        :return:
        """

        new_instance = TaskDriftA2C(self.A2C_Params)
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
