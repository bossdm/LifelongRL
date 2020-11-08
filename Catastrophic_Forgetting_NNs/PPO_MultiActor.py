from Catastrophic_Forgetting_NNs.A2C_Learner2 import *


class PPO_MultiActor(PPO_Learner):   # the same as PPO_Learner excapt train_model is done outside the learner
    def __init__(self,PPO_args):
        PPO_Learner.__init__(self,**PPO_args) # initialise 1 PPO policy
        self.update_time=False
    @overrides
    def setObservation(self, agent, environment):
        self.agent.total_t = self.total_t
        environment.setObservation(agent)
        self.observation = agent.learner.observation  # in case of task drif
        obs = np.expand_dims(self.observation, axis=0)
        self.s_t = np.append(self.s_t[1:, :], obs, axis=0)
        # self.mazefile.write("x=%.2f,y=%.2f\n"%(agent.x,agent.y))
        # self.mazefile.flush()
        # os.fsync(self.mazefile)
        if len(self.agent.rewards) == self.agent.update_freq:

            if self.testing:
                return
            self.agent.states.append(self.s_t)  # add final state
            self.update_time=True
        #     print("t=",self.t)
        #     print("total_t=",self.total_t)
        #     print("time to update (obs)")
        #     print("states in setObs: ", len(self.agent.states))
        #     # loss = self.agent.train_model(terminal=False)
        #     # if DEBUG_MODE:
        #     #     print("loss=" + str(loss))
        # else:
        #     print("t=",self.t)
        #     print("total_t=",self.total_t)
        #     print("number of rewards: ",len(self.agent.rewards))
        #     print("number of states in setObs: ", len(self.agent.states))

    @overrides
    def reset(self):
        if len(self.agent.rewards)>=1:
            if self.testing:
                return
            self.update_time=True
            #print("t=", self.t)
            #print("total_t=", self.total_t)
            #print("time to update (reset)")
            #print("states in reset: ", len(self.agent.states))
            #print("states in reset:",self.agent.states)
            # loss=self.agent.train_model(terminal=self.terminal_states_known)
            # if DEBUG_MODE:
            #     print("loss="+str(loss))
    @overrides
    def setTerminalObservation(self,agent,environment):
        #print("setting terminal observation")
        self.setObservation(agent,environment)
        if self.t >= self.agent.trace_length and len(self.agent.rewards) >= 1:
            #print("appending state")
            self.agent.states.append(self.s_t) # add final state
            self.did_terminal = True

    def time_to_update(self):
        return self.update_time
    def update_model(self,terminal):
        if terminal:
            #print("terminal states known="+str(self.terminal_states_known))
            loss=self.agent.train_model(terminal=self.terminal_states_known)
        else:
            #print("terminal state NO=" + str(self.terminal_states_known))
            loss=self.agent.train_model(terminal=False)

    def update_states(self):
        #print("update states=\n",self.agent.states)
        return (self.agent.states,self.agent.actions,self.agent.rewards)
    def process_update_states(self,args):
        self.agent.states, self.agent.actions, self.agent.rewards = args
    def get_weights(self):
        return self.agent.ppo.get_all_weights_list()
    def set_weights(self,w):
        #print("will set w:")
        #for ww in w:
         #   print(ww.shape)
        #self.agent.init_PPO(w=w)
        self.agent.ppo.set_all_weights(w)
        #print("just set the weights:")
        #print(self.agent.ppo.get_all_weights())