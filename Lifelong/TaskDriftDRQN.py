
import numpy as np
from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
from copy import deepcopy
from overrides import overrides
from Lifelong.TaskDriftBase import TaskDriftBase
from keras.models import Model, load_model, clone_model

class TaskDriftDRQN(DRQN_Learner,TaskDriftBase):
    epsilon_change=False
    def __init__(self,DRQN_Params,episodic_performance=False):
        DRQN_Learner.__init__(self,**deepcopy(DRQN_Params))
        TaskDriftBase.__init__(self, episodic_performance)
        self.DRQN_Params=DRQN_Params
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
    def get_output_diversity(cls,DRQN_instances,datapoints,metric_type="std"):
        assert metric_type=="totalvar"
        trace_length=DRQN_instances[0].agent.trace_length
        N = len(datapoints)-trace_length
        spread=0
        for i in range(0,N):
            if trace_length==1:
                input=datapoints[i]
            else:
                input = datapoints[i:i+trace_length]
            predictions = [instance.process_datapoint(input) for instance in DRQN_instances]
            if metric_type=="totalvar":
                #M = np.mean(predictions,axis=0)   # compute total variation distance of all pols to the mean pol
                mv=0.0
                count=0.0
                for i in range(len(DRQN_instances)):
                    for j in range(len(DRQN_instances)):
                        if i==j:
                            continue
                        temp=np.abs(predictions[i]-predictions[j])
                        mv+=sum(temp)
                        count+=1.0
                spread += 0.5*mv/count
            else:
                assert False, "dont want to use this one right now"
                spread += np.mean(np.std(predictions,axis=0))  # std across policies, average over actions
        spread/=float(N)
        return spread


    @classmethod
    def get_diversity(cls,DRQN_instances):
        policy_vectors=[instance.get_single_array() for instance in DRQN_instances]
        between_policies = np.std(policy_vectors,axis=0)
        return np.mean(between_policies)
    @overrides
    def create_similar_policy(self):
        """
        create a new policy with the same parameters and policy
        :return:
        """

        new_instance = TaskDriftDRQN(self.DRQN_Params)
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
        self.agent.total_t = self.total_t

    @overrides
    def update_task_reward(self,reward):
        self.task_R[self.current_feature] += reward
        self.r =reward
        self.R += reward
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
