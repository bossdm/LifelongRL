
import numpy as np
from TaskSpecificSSABaselineSimple_Sorting import Lifelong_SSA_gradientQ
from SSA_gradientQ import SSA_gradientQ
from copy import deepcopy
from overrides import overrides
from TaskDriftBase import TaskDriftBase
from keras.models import Model, load_model, clone_model

class TaskDriftLifelongAAP(Lifelong_SSA_gradientQ,TaskDriftBase):
    epsilon_change=False
    def __init__(self,SSAgradientQ_Params):
        Lifelong_SSA_gradientQ.__init__(self,**deepcopy(SSAgradientQ_Params))
        TaskDriftBase.__init__(self)
        self.SSA_gradientQ_Params=SSAgradientQ_Params
        self.task_t={}
        self.task_R={}
    def get_single_array(self):
        weights=np.array([])
        for Qlearner in self.Qlearners:
            for layer in Qlearner.model.layers:
                layer_weights = layer.get_weights()
                for W in layer_weights:
                    weights=np.append(weights,W)
        return weights
    @classmethod
    def get_diversity(cls,DRQN_instances):
        policy_vectors=[instance.get_single_array() for instance in DRQN_instances]
        M=np.max(policy_vectors)
        m=np.min(policy_vectors)
        return np.array(policy_vectors).std()/float(M-m)
    @overrides
    def create_similar_policy(self):
        """
        create a new policy with the same parameters and policy
        :return:
        """

        new_instance = Lifelong_SSA_gradientQ(self.SSA_gradientQ_Params)
        new_instance.set_policy(self.get_policy_copy())
        return new_instance
    @overrides
    def update_task_time(self,increment):
        self.task_t[self.current_feature] += increment
        new_t=self.t+increment
        self.setTime(new_t)

    @overrides
    def update_task_reward(self,reward):
        self.task_R[self.current_feature] += reward
        self.setReward(reward)
    @overrides
    def set_tasks(self, occurrence_weights):
        TaskDriftBase.set_tasks(self, occurrence_weights)
        SSA_gradientQ.set_tasks(self,occurrence_weights)
        for task in occurrence_weights:
            self.task_t[task] = 0
            self.task_R[task] = 0
    # @overrides
    # def new_task(self,feature):
    #     self.new_task()
    @overrides
    def update_task_time(self,increment):
        self.task_t[self.current_feature] += increment
        new_t=self.t+increment
        self.setTime(new_t)

    @overrides
    def update_task_reward(self,reward):
        self.task_R[self.current_feature] += reward
        self.setReward(reward)

    @overrides
    def get_task_R(self, feature):
        return self.task_R[feature]

    @overrides
    def get_task_t(self, feature):
        return self.task_t[feature]
    @overrides
    def add_to_ignored(self,ignored_t,ignored_R):
        for feature in self.task_R:
            ignored_t+=self.task_t[feature]
            ignored_R+=self.task_R[feature]
        return ignored_t, ignored_R

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
