import sys, os
sys.path.insert(0, str(os.environ['HOME']) + '/VizDoom-Keras-RL/')

import numpy as np
import random

from ExperimentUtils import read_incremental,dump_incremental
from overrides import overrides

from Methods.Learner import CompleteLearner

from Catastrophic_Forgetting_NNs.drqn_small import *
from Catastrophic_Forgetting_NNs.CustomNetworks import CustomNetworks

from keras.models import clone_model

from Catastrophic_Forgetting_NNs.EWC import *

def clone_kerasmodel(model):
    copy_m = clone_model(model)
    copy_m.set_weights(model.get_weights())
    return model

DEBUG_MODE=False


class DRQN_Learner(CompleteLearner):
    loss=None
    recorded_Qs={}
    recorded_loss={}
    recorded_targets={}
    intervals=[]
    exploration_schedule={}
    Q_max = None

    def __init__(self,task_features,use_task_bias,use_task_gain,n_inputs,trace_length,actions,file,episodic,loss=None,
                 target_model=False,num_neurons=80,epsilon_change=False,init_epsilon=None,final_epsilon=None,
                 agent=None,intervals=[],num_features=0,learning_rate=0.10,recurrent=True,multigoal=False,buffer_size=None):
        CompleteLearner.__init__(self,actions,file,episodic)
        self.init_variables(epsilon_change)


        if agent is None:
            self.agent=DRQN_Learner.init_agent(n_inputs,actions,trace_length,episodic,
                                               task_features, use_task_bias, use_task_gain, num_neurons,
                                                   target_model,init_epsilon,final_epsilon,
                                               num_features,learning_rate,
                                               recurrent,multigoal,buffer_size)
        self.state_size = DRQN_Learner.set_state_size(n_inputs, trace_length,recurrent)
        self.continue_experiment(intervals)
        # self.model_objective = EWC_objective(lbda_task,learning_rate,batch_size,model,n_in, n_out,lbda,output_type=OutputType.linear,epochs=200,
        #          objective_type=ObjectiveType.EWC,task_weights=None)
        # self.target_model_objective = EWC_objective_linoutput(out,lbda)
        # agent.model.compile(loss=self.model_objective.objective)
        #
        # agent.target_model.compile()

        print(self.__dict__)
        print(self.agent.__dict__)
    @classmethod
    def init_agent(cls,n_inputs,actions,trace_length,episodic,task_features, use_task_bias, use_task_gain,
                   num_neurons, target_model,init_epsilon=None,final_epsilon=None,num_features=0,
                   learning_rate=0.10,
                   recurrent=True,
                   multigoal=False,
                   buffer_size=None):
        action_size = len(actions)
        state_size = DRQN_Learner.set_state_size(n_inputs, trace_length, recurrent)
        if multigoal:
            agent = MultiTaskDoubleDRQNAgent(buffer_size,state_size, action_size, trace_length,
                                    episodic=episodic,init_epsilon=init_epsilon,
                                    final_epsilon=final_epsilon)
        else:
            if num_features > 0:
                agent = FeatureDoubleDRQNAgent(num_features, state_size, action_size, trace_length,
                                        episodic=episodic, init_epsilon=init_epsilon,
                                        final_epsilon=final_epsilon)
            else:
                agent = DoubleDRQNAgent(state_size, action_size, trace_length,
                                    episodic=episodic,init_epsilon=init_epsilon,
                                    final_epsilon=final_epsilon)


        #agent.epsilon=.05
        agent.recurrent=recurrent
        input_shape = (None,) + state_size
        if isinstance(n_inputs,tuple): # use convolution
            if num_features > 0:
                # input_shape, action_size, learning_rate, task_features, use_task_bias, use_task_gain
                agent.model = CustomNetworks.feature_drqn(num_features,state_size, action_size, task_features,
                                                  use_task_bias, use_task_gain, num_neurons,learning_rate=learning_rate)
                if target_model:
                    agent.target_model = CustomNetworks.feature_drqn(num_features,state_size, action_size,
                                                             task_features, use_task_bias, use_task_gain,
                                                             num_neurons,learning_rate=learning_rate)
            else:
                # input_shape, action_size, learning_rate, task_features, use_task_bias, use_task_gain
                agent.model = CustomNetworks.drqn(input_shape, action_size, learning_rate,task_features,use_task_bias,use_task_gain)
                if target_model:
                    agent.target_model = CustomNetworks.drqn(input_shape, action_size, learning_rate,task_features,use_task_bias,use_task_gain)
        else:
            #input_shape, action_size, learning_rate, task_features, use_task_bias, use_task_gain
            agent.model = CustomNetworks.small_scale_drqn(input_shape, action_size, task_features,
                                                          use_task_bias, use_task_gain,num_neurons,
                                                          learning_rate=learning_rate,
                                                          recurrent=recurrent)
            if target_model :
                agent.target_model = CustomNetworks.small_scale_drqn(input_shape, action_size,
                                                                     task_features, use_task_bias,
                                                                     use_task_gain,num_neurons,
                                                                     learning_rate=learning_rate,
                                                                     recurrent=recurrent)
        return agent

    def fill_eps(self,num_acts,maxindex,eps):
        vec=np.zeros(num_acts)
        for i in range(num_acts):
            if i==maxindex:
                vec[i]=1-eps + eps/float(num_acts)
            else:
                vec[i]=eps/float(num_acts)
        return vec

    def process_datapoint(self,datapoint):
        """

        :param datapoint: last trace_length observationsd
        :return:
        """
        eps=self.agent.epsilon
        self.agent.epsilon=0 # to be able to get the best output all the time

        self.action_idx = self.agent.get_action(np.expand_dims(datapoint,0))
        probs = self.fill_eps(self.agent.action_size,self.action_idx,eps)
        self.agent.epsilon=eps
        return probs
    @classmethod
    def set_state_size(cls,n_inputs,trace_length,recurrent):
        if recurrent:
            if isinstance(n_inputs,tuple):
                # is a shape
                state_size=(trace_length,)+n_inputs

            else:
                state_size=(trace_length,n_inputs,)
        else:
            state_size=(n_inputs,)
        return state_size
    def init_variables(self,epsilon_change):
        self.epsilon_change=epsilon_change
        self.action_idx=0

        self.total_t = 0
        self.episode_buf = []
    def zero_padding(self):
        trace_length=self.agent.trace_length
        while len(self.episode_buf) < trace_length:
            self.episode_buf.insert(0,self.null_experience())
    def null_state(self,n_inputs):
        return np.zeros(n_inputs)-1
    def null_experience(self):
        _, n_inputs= self.state_size
        return [self.null_state(n_inputs), 0., 0., self.null_state(n_inputs)]
    def get_policy_copy(self):
        return clone_kerasmodel(self.agent.model)

    def set_policy(self,pol):
        self.agent.model=pol
    @overrides
    def printPolicy(self):
        pass
    @overrides
    def reset(self):
        if self.episodic:
            if self.t > 0:
                self.zero_padding()
                self.agent.memory.add(self.episode_buf)
            # if DEBUG_MODE:
            #     print("episode buffer size %d  "%(len(self.episode_buf)))
    @overrides
    def new_elementary_task(self):
        if self.episodic :
            self.episode_buf = []  # Reset Episode Buf
            self.t = 0

    def track_q(self,old_location,location,intervals):
        """

        :return:
        """


        for min, max in intervals:

            if min <= self.total_t < max:
                self.record_qs(old_location,location)

            if self.total_t == max:
                self.terminate_qs(min,max)
    def record_qs(self,old_location,location):
        if old_location not in self.recorded_Qs:
            self.recorded_Qs[old_location] = []
            self.recorded_targets[old_location] = []
            self.recorded_loss[old_location] = []
        s, a, r, s_ = self.agent.memory.buffer[self.agent.memory.sp]
        output, target = self.agent.compute_output_and_target(self.old_input, self.new_input, 1, [[a]],
                                                              [[r]])
        self.recorded_Qs[old_location].append(output)
        self.recorded_targets[old_location].append(target)
        self.recorded_loss[old_location].append(np.mean((output - target) ** 2))
    def terminate_qs(self,min,max):
        del self.intervals[0]
        self.save_recordings(min, max)
        self.recorded_Qs = {}
        self.recorded_loss = {}
        self.recorded_targets = {}
    def save_recordings(self,min,max,folder=''):
        recorded_stuff = {'Qs': self.recorded_Qs, 'targets': self.recorded_targets,
                          'loss': self.recorded_loss}

        dump_incremental(folder+self.file + '(%d,%d)_recordings' % (min, max), recorded_stuff)

    @overrides
    def new_task(self,feature):
        """
        when new feature arrives, need to switch to task-specific (do nothing except when experience matching)
        :param feature:
        :return:
        """
        self.agent.new_task(feature)
    @overrides
    def setObservation(self,agent,environment):
        self.agent.total_t = self.total_t
        environment.setObservation(agent)
        self.observation=agent.learner.observation # in case of task drif

        self.s_t1 = np.array(self.observation)
    @overrides
    def setTerminalObservation(self,agent,environment):
        self.setObservation(agent,environment)
        self.add_experience()
    @overrides
    def setAtariTerminalObservation(self,obs):
        self.observation = obs  # in case of task drif
        self.s_t1 = np.array(self.observation)
        self.add_experience()
    def get_input(self):
        if self.episodic:
            if len(self.episode_buf) > self.agent.trace_length:
               buffer=self.episode_buf
               state_series = np.array([trace[-1] for trace in buffer[-self.agent.trace_length:]])
               if self.agent.recurrent:
                    state_series = np.expand_dims(state_series, axis=0)
               return state_series
            else:
                return None
        else:
            if self.agent.memory.max_reached or self.agent.memory.sp > self.agent.trace_length:
                buffer = self.agent.memory.get_trace(self.agent.memory.sp + 1 - self.agent.trace_length,
                                                        self.agent.memory.sp + 1)
                state_series = np.array([trace[-1] for trace in buffer])
                state_series = np.expand_dims(state_series, axis=0)
                return state_series
            else:
                return None


    @overrides
    def setAction(self):
        self.new_input=self.get_input()
        if self.new_input is not None:


            self.action_idx = self.agent.get_action(self.new_input)

        else:
            self.action_idx = random.randrange(self.agent.action_size)

        self.chosenAction=self.actions[self.action_idx]

        #print(self.chosenAction)
        self.s_t = self.s_t1
        self.old_input = self.new_input


    def add_experience(self):
        experience = [self.s_t, self.action_idx, self.r, self.s_t1]
        self.add_single_experience(experience)

    def add_single_experience(self,experience):
        if self.episodic:
            # save the sample <s, a, r, s'> to episode buffer
            self.episode_buf.append(experience)
            if DEBUG_MODE:
                print("this experience:" + str(self.episode_buf[-1]))
        else:
            self.agent.memory.add(experience)

    def setTime(self,t):
        increment= t - self.total_t
        self.t += increment
        self.total_t += increment
        self.agent.total_t = self.total_t


    def update_target(self):
        if self.total_t % self.agent.update_target_freq == 0:
            self.agent.update_target_model()
    @overrides
    def continue_experiment(self,intervals):
        self.intervals=intervals
        if self.intervals:
            self.loss_file=open(self.file+"_loss",mode="wb")

    @overrides
    def learn(self):
        # if DEBUG_MODE:
        #     print("epsilon=%.4f"%(self.agent.epsilon))
        # # Update epsilon
        if self.testing:
            return
        #print(self.epsilon_change)
        if self.epsilon_change and not self.exploration_schedule:
            if self.agent.epsilon > self.agent.final_epsilon:
                self.agent.epsilon -= (self.agent.initial_epsilon - self.agent.final_epsilon) / float(self.agent.exploration_frame)



        self.Q_max, self.loss = self.agent.train_replay()


        if self.t > 0:
            self.add_experience()


        # # Update the target model to be same with model
        if self.agent.target_model is not None:
            self.update_target()




        if DEBUG_MODE:
            print("epsilon:"+str(self.agent.epsilon))
            print("initial epsilon:" + str(self.agent.initial_epsilon))
            print("final epsilon:" + str(self.agent.final_epsilon))
            print("Qmax:"+str(self.Q_max))
            print("loss:"+str(self.loss))
            print("t=" + str(self.total_t))
            print("R="+str(self.R))

    @overrides
    def save(self,filename):
        self.agent.save(filename)
    @overrides
    def load(self,filename):
        self.agent.load(filename)
        print(self.__dict__)
        print(self.agent.__dict__)

    @overrides
    def performAction(self, agent, environment):
        self.chosenAction.perform([agent,environment])
        #self.t = environment.t
    @overrides
    def atari_cycle(self,observation):
        self.observation = observation  # in case of task drif
        self.s_t1 = np.array(self.observation)
        self.learn()
        self.setAction()
        # ??? agent.learner.chosenAction = self.chosenAction  # cf. task drift (Homeostatic Pols)
    @overrides
    def cycle(self,agent,environment):
        if self.exploration_schedule:
            self.agent.epsilon=self.exploration_schedule[(agent.x,agent.y)]
        self.setObservation(agent, environment)
        self.learn()
        self.setAction()
        agent.learner.chosenAction = self.chosenAction # cf. task drift (Homeostatic Pols)
        self.performAction(agent,environment)


class EWC_Learner(DRQN_Learner):
    def __init__(self,timesteps,DRQN_opts,loss="likelihood"):
        DRQN_Learner.__init__(self,**DRQN_opts)
        self.ewc = EWC_objective(lbda_task={},
                                 learning_rate=0.10,
                                 batch_size=None,
                                 model=self.agent.model,
                                 n_in=None, n_out=len(self.actions))
        self.previous_t = 0
        self.agent.memory.stop_replay=timesteps
        self.loss = loss


    @overrides
    def new_task(self,feature):
        """
        when new feature arrives, need to switch to task-specific (do nothing except when experience matching)

        "To apply EWC, we compute the Fisher information matrix at each task switch."
        :param feature:
        :return:
        """
        all_tasks = self.agent.memory.get_replay_ready_nostop_goals(self.agent.replay_start_size, self.agent.batch_size)
        replay_goals = self.agent.memory.get_replay_ready_goals(self.agent.replay_start_size, self.agent.batch_size)
        print("replay ready tasks ", replay_goals)
        print("all tasks (including converged) ", all_tasks)
        delta_t = self.total_t - self.previous_t
        self.previous_t = self.total_t
        self.ewc.end_task(delta_t,self.agent.memory.stop_replay)
        self.agent.new_task(feature)
        self.ewc.start_task(feature)
        batches = []
        if self.total_t>0:

            for i in range(100):
                samples,terminals = self.agent.memory.sample(self.agent.batch_size,self.agent.trace_length,all_tasks)
                x,y=self.agent.get_xy(self.agent.batch_size,samples,terminals)
                batches.append((x,y))
        self.agent.model = self.ewc.compile_EWC(batches,self.agent.model,self.loss)





class ExplorationDRQN_Learner(DRQN_Learner):
    def __init__(self,recordingsfile,task_features,use_task_bias,use_task_gain,n_inputs,trace_length,actions,file,episodic,loss=None,
                 target_model=False,num_neurons=80,epsilon_change=False,agent=None,intervals=[]):
        recordings=read_incremental(recordingsfile)
        self.exploration_schedule=recordings['epsilon']
        DRQN_Learner.__init__(self,task_features,use_task_bias,use_task_gain,n_inputs,trace_length,actions,file,episodic,loss,
                 target_model,num_neurons,epsilon_change,agent,intervals)

    @overrides
    def setObservation(self,agent,environment):
        # set epsilon based on the location in the map
        self.agent.epsilon=self.exploration_schedule[(agent.x,agent.y)]
        DRQN_Learner.setObservation(self,agent,environment)
