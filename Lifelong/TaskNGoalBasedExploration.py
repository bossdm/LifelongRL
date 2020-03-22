import numpy as np
import os
from IS.SSA import SSA_with_WM
from IS.Stack import Stack
import IS.StackEntries
import keras.backend as K
from overrides import overrides
from Actions.SpecialActions import PLA
from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
from Lifelong.LifelongSSA_WM import Lifelong_SSA_with_WM
from copy import deepcopy

DEBUG_MODE = False
DO_CHECKS = True

"""
these are handy tools to be used as a supplement to DRL methods


NOTE: if you change one NetworkOptimisation code, change the other one similarly
"""

class RunningStat(object):
    def __init__(self, shape):
        self._n = 0
        self._M = np.zeros(shape)
        self._S = np.zeros(shape)
    def push(self, x):
        x = np.mean(x)
        self._n += 1
        if self._n == 1:
            self._M[...] = x
        else:
            oldM = self._M.copy()
            self._M[...] = oldM + (x - oldM)/self._n
            self._S[...] = self._S + (x - oldM)*(x - self._M)
    @property
    def n(self):
        return self._n
    @property
    def mean(self):
        return self._M
    @property
    def var(self):
        return self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape




class ZFilter(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape):


        self.rs = RunningStat(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        return x
    def output_shape(self, input_space):
        return input_space.shape

class NetworkOptimisationBase():
    lr_changed = False
    eps_changed = False
    num_param_indexes=1
    def __init__(self):
        self.a_lr, self.b_lr = (10**(-6), 0.20)
        self.a_eps, self.b_eps = (0.00, 0.40)

        self.stats.lr_overTime = []
        self.stats.eps_overTime = []
        self.stats.lrdeviation_overTime = []
        self.stats.epsdeviation_overTime = []
        self.stats.num_lr_changesOverTime = []
        self.stats.num_eps_changesOverTime = []
        self.lr=0.10
        self.eps=0.20
        self.num_lr_changes = 0
        self.num_eps_changes = 0
        self.refresh_tracking()
        print("initialised networkoptimisation with parameters:")
        print("LR-init=(%.3f,%.3f)" % (self.a_lr, self.b_lr))
        print("EPS-init=(%.3f,%.3f)" % (self.a_eps, self.b_eps))
    def get_lr(self):
        if self.num_param_indexes > 1:
            index = self.get_slice_indexPROG(self.IP - self.ProgramStart)
            return self.lr[index]
        else:
            return self.lr
    def get_eps(self):
        if self.num_param_indexes > 1:
            index=self.get_slice_indexPROG(self.IP-self.ProgramStart)
            return self.eps[index]
        else:
            return self.eps
    def init_slices(self,num_slices):
        self.lr=np.zeros(num_slices)+0.10
        self.eps=np.zeros(num_slices)+0.20
        self.num_param_indexes=num_slices
        self.slice_size = self.n_ops/self.num_param_indexes

    def refresh_tracking(self):
        self.track_lr=ZFilter((self.num_param_indexes,))
        self.track_eps=ZFilter((self.num_param_indexes,))
        self.lr_changed=False
        self.eps_changed=False
    def printLR(self):
        return "lr: "+str(self.lr)
    def printEPS(self):
        return "eps: " + str(self.eps)
    def parameter_development(self):
        self.stats.num_lr_changesOverTime.append(self.track_lr.rs.n)
        if self.num_param_indexes==1:
            m= self.track_lr.rs.mean[0] if self.track_lr.rs.n>0 else self.lr
            s=self.track_eps.rs.std[0]
        else:
            m = self.track_lr.rs.mean if self.track_lr.rs.n > 0 else self.lr
            s=self.track_eps.rs.std
        self.stats.lr_overTime.append(m)
        self.stats.lrdeviation_overTime.append(s)

        self.stats.num_eps_changesOverTime.append(self.track_eps.rs.n)
        if self.num_param_indexes==1:
            m= self.track_eps.rs.mean[0] if self.track_eps.rs.n>0 else self.eps
            s=self.track_eps.rs.std[0]
        else:
            m = self.track_eps.rs.mean if self.track_eps.rs.n > 0 else self.eps
            s = self.track_eps.rs.std
        self.stats.eps_overTime.append(m)
        self.stats.epsdeviation_overTime.append(s)

        self.refresh_tracking()

    def add_track_lr(self):
        if self.lr_changed:
            self.track_lr(np.array([self.lr]))
            self.lr_changed=False
    def add_track_eps(self):
        if self.eps_changed:
            self.track_eps(np.array([self.eps]))
            self.eps_changed=False
    def set_lr(self,a1):
        new_lr=self.a_lr +  a1*(self.b_lr - self.a_lr)/float(self.n_ops - 1)
        self.lr = new_lr
        self.lr_changed=True
        # tracking is later (since lr only effective periodically)
        if DEBUG_MODE:
            print("lr="+str(self.lr))

    def set_eps(self,a1):
        new_eps=self.a_eps + a1 * (self.b_eps - self.a_eps) / float(self.n_ops - 1)
        self.eps = new_eps
        self.eps_changed=True
        if DEBUG_MODE:
            print("eps="+str(self.eps))

    def move_proportional(self,current,argument,min,max):
        direction=argument - self.n_ops / 2.
        proportion = argument/float(10*self.n_ops)

        if direction == 0:
            return current
        else:

            if direction < 0:
                direction = proportion * (min - current)
            else:
                direction = proportion *(max - current)
        return current+direction
    def prepareParameterChange(self,param_type,index):
        if not self.time_passed_modification() or self.disablePLA: #time needs to have passed for evaluation
            if DEBUG_MODE:
                print("no time passed, not changing")
            return False

        self.addParameterChange(param_type,index)
        return True
    def restore(self, entry):
        if isinstance(entry, IS.StackEntries.StackEntry):
            if (entry.oldP is not None):
                if isinstance(entry.address,tuple):
                    type,index = entry.address
                    self.__dict__[type][index] = entry.oldP
                else:
                    self.Pol[entry.address] = entry.oldP  # to restore old policy
            else:
                raise Exception()
        elif isinstance(entry, IS.StackEntries.PolicyStackEntry):
            self.Pol=entry.pol
        elif isinstance(entry, IS.StackEntries.StackEntryPredMod):
            if (entry.oldMeans is not None):
                self.predictiveMod.means[entry.address - self.ProgramStart] = entry.oldMeans  # to restore old policy
            else:
                raise Exception()
        else:
            raise Exception()
    def addParameterChange(self,type,index):
        # change polic and return the old policy
        self.polChanged = True
        oldP = self.__dict__[type][index]
        first = self.getPointerToFirstModification(len(self.Stack))  # pointer to the index of the first modification of this SMS on the stack (take into account the new entry so size-1+1)
        newEntry = self.add_entry(t=self.t, R=self.R, oldP=oldP, address=(type,index), first=first)
        self.Stack.push(newEntry)

        if DEBUG_MODE:
            print("New Stack Entry: " + str(newEntry))
            self.writeStack()

        return oldP
    def get_slice_index(self,a2):
        slice_size = self.n_ops/self.num_param_indexes
        index = a2/slice_size
        return index
    def get_slice_indexPROG(self,IP):
        slice_size = self.m / self.num_param_indexes
        index = IP/slice_size
        return index
    def inc_lr(self,a1,a2):
        if DEBUG_MODE:
            print("agent" + str(self.task))
            print("before inc_a_lr"+self.printLR())
            return
        index = self.get_slice_index(a2)
        if not self.prepareParameterChange(param_type="eps",index=index):
            return
        self.lr[index]=self.move_proportional(self.lr[index],a1,self.a_lr,self.b_lr)
        if DEBUG_MODE:
            print("after inc_a_lr" + self.printLR())



    def inc_eps(self, a1,a2):
        if DEBUG_MODE:
            print("agent" + str(self.task))
            print("before inc_a_eps" + self.printEPS())
        index = self.get_slice_index(a2)
        if not self.prepareParameterChange(param_type="eps",index=index):
            return
        self.eps[index] = self.move_proportional(self.eps[index], a1, self.a_eps, self.b_eps)
        if DEBUG_MODE:
            print("after inc_a_eps" + self.printEPS())



class NetworkOptimisationSSA(NetworkOptimisationBase, SSA_with_WM):
    """
    exploration SSA: simplified IS
    """
    slices=False
    def __init__(self, task, SSA_Params):
        self.task = task
        SSA_with_WM.__init__(self, **SSA_Params)

        NetworkOptimisationBase.__init__(self)

        self.Rfile = None

        if self.slices:
            self.init_slices(num_slices=4)


    def parseInternalActions(self, list):
        actions = []
        for action in list.keys():
            if action in ['inc_lr', 'inc_eps']:
                n_args = list[action]
                function = getattr(self, action)
                actions.append(PLA(function, n_args))
                list.pop(action)
                self.slices=True
        return actions + SSA_with_WM.parseInternalActions(self, list)

    @overrides
    def printDevelopment(self):
        SSA_with_WM.printDevelopment(self)
        self.parameter_development()

    # @overrides  # (CompleteLearner)
    # def setObservation(self, agent, environment):
    #     """
    #     key difference to IS: observation is the agent's exploration observation (TD_error,loss,reward_variance)
    #     :param agent:
    #     :param environment:
    #     :return:
    #     """
    #
    #     self.set_wm_internal_vars()

    @overrides
    def performAction(self, agent, environment):
        """
        difference: don't set agent.chosenAction, also don't check for external actions
        :param agent:
        :param environment:
        :return:
        """
        argument_list = self.currentInstruction[1:]

        self.chosenAction.perform(argument_list)  # record result for statistics
        #
        # if DEBUG_MODE:
        #     print("action="+str(self.chosenAction))


class NetworkOptimisationLifelongSSA(NetworkOptimisationBase, Lifelong_SSA_with_WM):
    """
    exploration SSA: simplified IS
    """
    slices=False
    def __init__(self, task, SSA_Params):
        self.task = task

        Lifelong_SSA_with_WM.__init__(self, **SSA_Params)
        NetworkOptimisationBase.__init__(self)
        self.Rfile = None
        if self.slices:
            self.init_slices(num_slices=4)

    def parseInternalActions(self, list):
        actions = []
        for action in list.keys():
            if action in ['inc_lr', 'inc_eps']:
                n_args = list[action]
                function = getattr(self, action)
                actions.append(PLA(function, n_args))
                list.pop(action)
                self.slices=True
        return actions + SSA_with_WM.parseInternalActions(self, list)

    @overrides
    def printDevelopment(self):
        Lifelong_SSA_with_WM.printDevelopment(self)
        self.parameter_development()

    # @overrides  # (CompleteLearner)
    # def setObservation(self, agent, environment):
    #     """
    #     key difference to IS: observation is the agent's exploration observation (TD_error,loss,reward_variance)
    #     :param agent:
    #     :param environment:
    #     :return:
    #     """
    #
    #     self.set_wm_internal_vars()

    @overrides
    def performAction(self, agent, environment):
        """
        difference: don't set agent.chosenAction, also don't check for external actions
        :param agent:
        :param environment:
        :return:
        """
        argument_list = self.currentInstruction[1:]

        self.chosenAction.perform(argument_list)  # record result for statistics

        # if DEBUG_MODE:
        #     print("action="+str(self.chosenAction))


class LearningType(object):
    none = 0


    taskbased = 1
    lifetime = 2
    lifetime_taskspecific = 3
    lifetime_taskspecificrelative = 4

    single_lifetime = 5
    single_lifetimetaskspecific = 6
    single_lifetimetaskspecificrelative=7
    @classmethod
    def has_single_ssa(cls,learning_type):
        return learning_type>=5

    @classmethod
    def has_no_ssa(cls,learning_type):
        return learning_type==LearningType.none


    @classmethod
    def has_multiple_ssa(cls,learning_type):
        return 1<=learning_type<=4

class RewardType(object):
    reward = 0
    loss = 1


class TaskBasedSMP_DRQN(DRQN_Learner):
    """
    optimise two parameters a,b which determine a sampling of a parameter in the log-scale
     for each task at each episode;


    """
    reward_type = RewardType.reward
    num_internals=10
    def __init__(self, learning_type, SSA_Params, DRQN_Params, reward_type="reward"):


        self.learning_type = learning_type
        if reward_type=="loss":
            self.reward_type = RewardType.loss
        elif reward_type=="reward":
            self.reward_type = RewardType.reward
        else:
            raise Exception("reward type must be either 'loss' or 'reward'")

        self.cumulative_nega_loss = 0
        DRQN_Learner.__init__(self, **DRQN_Params)

        self.stats.negativelossOverTime = []
        self.SSA_Params = SSA_Params

        self.last_R = None
        self.last_t = None
        self.ssa_agent = None
        self.seen_tasks = set([])
        self.loss_file = open(self.file + "_loss", mode="wb")

        self.optimisation_started=False
        print("USING REWARDTYPE: "+str(reward_type))


    def set_tasks(self, tasks):
        self.task_ts = {task: 0 for task in tasks}
        self.task_Rs = {task: 0.0 for task in tasks}
        if LearningType.has_no_ssa(self.learning_type):
            return
        if LearningType.has_single_ssa(self.learning_type):
            if self.learning_type==LearningType.single_lifetime:
                self.ssa_agent = NetworkOptimisationSSA(None, deepcopy(self.SSA_Params))
            else:
                self.ssa_agent = NetworkOptimisationLifelongSSA(None, deepcopy(self.SSA_Params))
            return

        self.ssa_agents = {}
        for task in tasks:
            if self.learning_type == LearningType.taskbased:
                self.ssa_agents[task] = NetworkOptimisationSSA(tuple(task), deepcopy(self.SSA_Params))
            elif self.learning_type == LearningType.lifetime:
                self.ssa_agents[task] = NetworkOptimisationSSA(tuple(task), deepcopy(self.SSA_Params))
            elif self.learning_type == LearningType.lifetime_taskspecific:
                self.ssa_agents[task] = NetworkOptimisationLifelongSSA(tuple(task), deepcopy(self.SSA_Params))
            elif self.learning_type == LearningType.lifetime_taskspecificrelative:
                self.ssa_agents[task] = NetworkOptimisationLifelongSSA(tuple(task), deepcopy(self.SSA_Params))
            else:
                raise Exception("learningtype %s not found" % (str(self.learning_type)))
            #self.ssa_agents[task].init_slices(4)
        del self.SSA_Params

    @overrides
    def setTime(self, t):
        increment = t - self.total_t
        if self.optimisation_started:
            self.task_ts[self.current_feature] += increment
        DRQN_Learner.setTime(self, t)

    def set_reward(self):
        """
        set the reward for the instruction module
        :return:
        """
        if self.optimisation_started:
            if self.loss is not None:
                self.cumulative_nega_loss += - self.loss
            if self.reward_type == RewardType.loss:
                self.task_Rs[self.current_feature] = self.cumulative_nega_loss
                    # assert np.isclose(self.cumulative_nega_loss,sum(self.task_Rs.values())),"%.3"
            else:
                self.task_Rs[self.current_feature] += self.r



    @overrides
    def setReward(self, reward):

        DRQN_Learner.setReward(self, reward)
        self.set_reward()

    @overrides
    def writeRtofile(self):
        DRQN_Learner.writeRtofile(self)
        self.loss_file.write("%.2f \n" % (self.cumulative_nega_loss))
        self.loss_file.flush()
        os.fsync(self.Rfile)

    @overrides
    def printDevelopment(self):
        DRQN_Learner.printDevelopment(self)
        if LearningType.has_multiple_ssa(self.learning_type):
            for agent in self.ssa_agents.values():
                agent.printDevelopment()
        else:
            if LearningType.has_single_ssa(self.learning_type):
                self.ssa_agent.printDevelopment()
        self.stats.negativelossOverTime.append(self.cumulative_nega_loss)
        if self.total_t > 0:
            if DO_CHECKS:
                self.ssa_agent.printLR()
                self.ssa_agent.printEPS()

    def add_task_dataLifelongSSA(self, agent, feature, set_time=False):

        agent.current_feature = feature  # because current_feature is used inside setTime & setReward
        agent.add_task_to_data(feature)
        if set_time:
            self.add_task_t_R(agent, feature)

    def share_baselines(self, agent, F,other_agent):
        other_agent.Stack.running_stats[F]=agent.Stack.running_stats[F]


    def add_lifelongdata_condition(self):
        return self.learning_type in [LearningType.lifetime_taskspecific, LearningType.lifetime_taskspecificrelative]
    def single_lifelong_condition(self):
        return self.learning_type in [LearningType.single_lifetimetaskspecific, LearningType.single_lifetimetaskspecificrelative]
    @overrides
    def new_task(self, task):
        #if self.optimisation_started :
        #    old_feature = self.current_feature
        #    if self.add_lifelongdata_condition():


        new_task = tuple(task)
        if self.learning_type != LearningType.none:
            if LearningType.has_multiple_ssa(self.learning_type):


                if self.add_lifelongdata_condition():
                    # baselines might have changed: need to check again, but first set polChanged so it has effect
                    # evaluation --> modification needed to avoid modification evaluation gap (do not set disablePLA=True)
                    if self.seen_tasks:
                        for F, agent in self.ssa_agents.items():
                            if F != self.current_feature:  # update timers and baselines of the other agents not
                                self.add_task_dataLifelongSSA(agent, self.current_feature, set_time=True)
                                self.share_baselines(self.ssa_agent, self.current_feature, agent)
                            if F==new_task: # update timers and baselines of the other agents
                                agent.polChanged = True
                                agent.params_changed = True  # evaluate all previous entries of the new ssa agent !
                                agent.finish_current_evaluation()


                    self.ssa_agent = self.ssa_agents[new_task]
                    self.add_task_dataLifelongSSA(self.ssa_agent, new_task)
                else:
                    self.ssa_agent = self.ssa_agents[new_task]
                    self.ssa_agent.current_feature = new_task
            else:

                if self.single_lifelong_condition():
                    if self.seen_tasks:
                        self.ssa_agent.polChanged=True
                    self.ssa_agent.new_task(new_task)
                else:
                    self.ssa_agent.current_feature = new_task
        self.current_feature = new_task
        self.seen_tasks.add(new_task)

    @overrides
    def end_task(self):
        if self.learning_type != LearningType.none:
            self.time_feedback()

    @overrides
    def save_stats(self, filename):
        DRQN_Learner.save_stats(self, filename)
        if LearningType.has_multiple_ssa(self.learning_type):
            for task, agent in self.ssa_agents.items():
                f1,f2,f3=task
                agent.save_stats(filename + 'SSAagent(%d,%d,%d)' % (f1,f2,f3))
        else:
            if LearningType.has_single_ssa(self.learning_type):
                self.ssa_agent.save_stats(filename+ 'SSAagent')
    def get_total_time(self):
        """
        get the total time after optimisation started
        :return:
        """
        return sum(self.task_ts.values())

    def get_total_reward(self):
        """
        get the lifetime cumulative reward after optimisation started
        :return:
        """
        return sum(self.task_Rs.values())

    def check_time_rewards(self):
        total_t = self.get_total_time()
        if self.learning_type == LearningType.taskbased:
            agentsum = sum([agent.R for agent in self.ssa_agents.values()])
            agentsumt = sum([agent.t for agent in self.ssa_agents.values()])

            assert np.isclose(self.get_total_reward(), agentsum), "%.5f   %.5f" % (self.get_total_reward(),agentsum)
            assert total_t == agentsumt, "%d %d" % (total_t, agentsumt)
        elif self.learning_type == LearningType.lifetime or self.learning_type == LearningType.single_lifetime:
            assert self.ssa_agent.t == total_t
            assert np.isclose(self.ssa_agent.R, self.get_total_reward())
        elif self.add_lifelongdata_condition() or self.single_lifelong_condition():
            assert self.ssa_agent.t == total_t
            assert np.isclose(self.ssa_agent.R, self.get_total_reward())
            assert sum(self.ssa_agent.Stack.task_ts.values()) == total_t
            assert np.isclose(sum(self.ssa_agent.Stack.task_Rs.values()),
                              self.get_total_reward()), "%.5f   %.5f" % (
            sum(self.ssa_agent.Stack.task_Rs.values()), self.get_total_reward())
        else:
            raise Exception("learning type="+str(self.learning_type))


    def feedback_condition(self):
        started = self.Q_max is not None
        learning_t = self.learning_type != LearningType.none  # update was done
        if not self.optimisation_started and started:
            self.optimisation_started=started
        return started and learning_t
    @overrides
    def cycle(self, agent, environment):
        """
        after each update to the main DRL method, call this one
        :return:
        """
        # print(self.total_t)
        if self.optimisation_started:
            K.set_value(self.agent.model.optimizer.lr, self.ssa_agent.get_lr())
            self.agent.epsilon=self.ssa_agent.get_eps()
            self.ssa_agent.add_track_eps()

        DRQN_Learner.cycle(self, agent, environment)
        if self.feedback_condition():  # update was done
            self.ssa_agent.add_track_lr()  # track the used learning rate only if training was done
            self.provide_feedback()

        if self.optimisation_started and self.learning_type != LearningType.none:
            for step in range(self.num_internals):
                self.ssa_agent.cycle(agent, environment)
                environment.observation_set = True

    def time_feedback(self):
        if self.learning_type == LearningType.lifetime or LearningType.has_single_ssa(self.learning_type):
            # updates can proceed based on the global time and R
            self.add_lifetime_t_R()
        else:
            # updates
            self.add_task_t_R(self.ssa_agent, self.current_feature)

    def scale_loss(self, loss):
        return np.clip(loss * self.ssa_agent.MaxInt, -self.ssa_agent.MaxInt, self.ssa_agent.MaxInt)

    def provide_feedback(self):
        self.time_feedback()

        # self.ssa_agent.observation = np.array(
        #     [int(np.clip(self.Q_max * 10., -self.ssa_agent.MaxInt, self.ssa_agent.MaxInt)),
        #      int(self.scale_loss(self.loss))])  # cf task drift
        # self.ssa_agent.set_wm_external_vars(self.ssa_agent.observation)
        # self.ssa_agent.c[self.ssa_agent.Min + 3] = int(self.r)

        if DO_CHECKS:
            self.check_time_rewards()

    def add_task_t_R(self, agent, task):
        """
        add the task specific timing for the agent;
        note: taskbased method --> update only for its own task
           vs lifetime_taskspecific --> update for all other tasks as well
        :param task:
        :return:
        """

        if self.learning_type == LearningType.taskbased:
            added_task_R = self.task_Rs[task] - agent.R  # R==task_R
            t = self.task_ts[task]
        elif self.learning_type in [LearningType.lifetime_taskspecific, LearningType.lifetime_taskspecificrelative]:
            added_task_R = self.task_Rs[task] - agent.Stack.task_Rs[task]  # task_Rs need to be tracked
            t = self.get_total_time()
        else:
            raise Exception("should not be called for none/single/lifetime learning types")
        agent.setTime(t)
        agent.setReward(added_task_R)

    def add_lifetime_t_R(self):
        added_R = self.get_total_reward() - self.ssa_agent.R
        self.ssa_agent.setTime(self.get_total_time())
        self.ssa_agent.setReward(added_R)




