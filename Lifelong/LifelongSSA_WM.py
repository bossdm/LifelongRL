
"""
NOTE: a copy of the SSA_gradientQ one, just SSA_gradientQ replaced with SSA
"""



from IS.SSA import SSA_with_WM

import numpy as np

from overrides import overrides
from IS.SSA import SSAimplementor

from ExperimentUtils import dump_incremental
import random


from LifelongSSA_Utils import *
DEBUG_MODE = False






class Lifelong_SSA_with_WM(SSA_with_WM):
    delta_tolerance=5*10**(-5)
    max_num_velocities = 3
    min_updates=5
    def __init__(self, weighting_type,absolute,SSA_with_WMparams):

        SSA_with_WM.__init__(self, **SSA_with_WMparams)
        self.current_feature = None
        self.seen_tasks=set()
        self.absolute=absolute
        self.V_old_second=float("inf")


        if weighting_type=="fixed":
            self.weighting_type=WeightingType.fixed
        elif weighting_type=="time_consumption":
            self.weighting_type=WeightingType.time_consumption
        else:
            raise Exception("unknown weighting type")


        self.stats.baselineOverTime={}
        self.stats.num_checks = 0
        self.stats.num_checksOverTime = []
        self.stats.num_evaluations = 0
        self.stats.num_evaluationsOverTime = []
        self.stats.num_backchecks = 0
        self.stats.num_backchecksOverTime = []
        self.stats.num_backtracking = 0
        self.stats.num_backtrackingOverTime = []
        self.stats.num_earlystops=0
        self.stats.num_earlystopsOverTime = []
        self.stats.num_prestops = 0
        self.stats.num_prestopsOverTime = []

        self.max_V={}
        #self.t_M={}
        self.first_task_instance=False
        self.set_sorted_firsts([0])

        print("initialised taskspecific with weighting type %s and absolute=%s criterion"%(weighting_type,str(absolute)))

    def set_sorted_firsts(self,list):

        self.sorted_firsts=list
        if DEBUG_MODE:
            print("sorted firsts:" + str(self.sorted_firsts))
    def new_task(self, feature):
        if DEBUG_MODE:
            print("new task :" + str(feature))

        self.finish_current_evaluation()
        self.add_task_to_data(feature)


    def add_task_to_data(self,feature):
        F = tuple(feature)
        if F not in self.seen_tasks:
            self.first_task_instance=True  # will evaluate all entries
            self.stats.baselineOverTime[F] = []
            self.seen_tasks.add(F)
            self.max_V[F]=-float("inf")
        # else:

            # if F!=self.current_feature:  # will use the best estimate of V_old
            #     self.V_old_task, self.V_old_second = self.getLastSpeeds(self.Stack.Stack.second_first(),F)

        self.task_time = 0
        self.current_feature = F
        self.Stack.new_task(F,self.absolute)

        if self.weighting_type == WeightingType.fixed:
            self.weights = {task:1. for task in self.seen_tasks}
        else:
            self.weights = self.get_task_weight_timeconsumption()


    def create_selfmod(self):
        index = np.random.randint(self.ProgramStart, self.Max + 1)
        self.indexToBeModified=index
        j = np.random.randint(0, self.n_ops)
        oldPol=self.addPolicyChange()
        self.Pol[index][j] += np.random.random() * .001
        C = sum(self.Pol[self.indexToBeModified])
        for o in range(self.maxInstr):
            self.Pol[self.indexToBeModified][o] /= C
            if (self.Pol[self.indexToBeModified][o] < self.minP):
                self.Pol[self.indexToBeModified] = oldPol
                return
        C_again=sum(self.Pol[self.indexToBeModified])
        for o in range(self.maxInstr):
            self.Pol[self.indexToBeModified][o] /= C_again # due to numerical precision need to do again
        self.remove_evaluate_and_add()
        assert self.Stack.top().t == self.t  # successful modification after evaluation

    def finish_current_evaluation(self):
        if self.polChanged:
            self._finish_current_evaluation()

    def _finish_current_evaluation(self):
        if self.disablePLA:
            self.evaluate_and_reset()
        else:
            if self.wait_eval_until_next_mod:
                self.wait_modification = True
                while self.wait_modification:
                    self.create_selfmod()





        # otherwise assumed that the success story is satisfied

    def at_task_boundary(self):
        if self.Stack.sp == 0:
            return False
        return not self.Stack.top().eval.get(self.current_feature, False)


    def setTime(self, t):
        increment = t - self.t
        SSA_with_WM.setTime(self, t)

        self.Stack.task_ts[self.current_feature] += increment


    def get_task_weight_timeconsumption(self):
        weights = {task:1. + self.Stack.task_ts[task]/float(WEIGHT_FREQUENCY) for task in self.seen_tasks}
        C=weights.values()
        weights={task:weights[task]/float(C) for task in self.seen_tasks}
        return weights

    def check_baseline_condition(self):
        return   self.Stack.task_ts[self.current_feature] % INIT_BASELINE_FREQUENCY == 0 and self.Stack.num_baseline_updates(self.current_feature) < self.min_updates or \
                self.Stack.task_ts[self.current_feature] % BASELINE_FREQUENCY == 0 and self.min_updates <= self.Stack.num_baseline_updates(self.current_feature) <= NUM_BASE_LINE_UPDATES

    def setReward(self, reward):
        SSA_with_WM.setReward(self,reward)
        self.Stack.task_Rs[self.current_feature] += reward

        if self.check_baseline_condition():
            t=self.get_time_passed()
            R=self.get_reward_passed()
            if t > 0:
                self.Stack.compute_baseline(R,t)
                self.after_estimation()
        if self.weighting_type == WeightingType.time_consumption and \
                self.Stack.task_ts[self.current_feature] % WEIGHT_FREQUENCY == 0:

            self.weights = self.get_task_weight_timeconsumption()
            self.after_estimation()


    @overrides
    def initStack(self):
        self.Stack = MultiStack()


    @overrides
    def printDevelopment(self):


        SSA_with_WM.printDevelopment(self)
        self.stats.num_checksOverTime.append(self.stats.num_checks)
        self.stats.num_evaluationsOverTime.append(self.stats.num_evaluations)
        self.stats.num_backchecksOverTime.append(self.stats.num_backchecks)
        self.stats.num_backtrackingOverTime.append(self.stats.num_backtracking)
        self.stats.num_earlystopsOverTime.append(self.stats.num_earlystops)
        self.stats.num_prestopsOverTime.append(self.stats.num_prestops)
        self.stats.num_checks = 0
        self.stats.num_evaluations = 0
        self.stats.num_backchecks=0
        self.stats.num_backtracking=0
        self.stats.num_earlystops=0
        self.stats.num_prestops=0
        if self.t > 0:  # add two additional to verify it stopped
            if len(self.stats.baselineOverTime[self.current_feature])  < NUM_BASE_LINE_TRACKS:
                self.stats.baselineOverTime[self.current_feature].append(self.Stack.get_base(self.current_feature))

    # @overrides
    # def popNumberVelocities(self, until, num_velocities):
    #     firsts, Vs = self.get_top_SMS_velocities(until=until, num_velocities=num_velocities)
    #
    #     popped = self.forceConservativeSSC(Vs, firsts)
    #
    #     return popped

    # def forceConservativeSSC(self, Vs, firsts):
    #     maintain_index = self.call_ConservativeSSC(Vs)
    #
    #     stop = firsts[maintain_index]
    #
    #     popped = False
    #     # note that we already pushed the new entry, we now decide if we have to pop it
    #     while True:
    #         # get the index until which the conservative SSC is valid
    #         sp = self.Stack.top().first
    #         if sp == stop or sp == 0:
    #             break
    #         assert sp in firsts
    #         self.popBackUntil(sp)
    #         popped = True
    #
    #     return popped


    # @overrides
    # def popUntil(self):
    #     # pop the stack until SSC satisfied
    #     if self.conservative:
    #
    #         # if DEBUG_MODE:
    #         #
    #         #     print("popUntil: popback=%d"%(until))
    #         #     print("for task"+str(self.current_feature))
    #         popped = self.popUntilConservativeSSC()
    #         # if self.Stack.sp ==0 or F == self.Stack.top().F or not popped:
    #         #     break
    #         # print("doing again ?")
    #
    #         if DEBUG_MODE:
    #             ts, Vs = self.get_top_SMS_velocities()
    #             self.check_conservativeSSC(Vs)
    #
    #
    #     else:
    #         raise Exception("not implemented")
    #         self.popUntilSSC()

    def velocity_compared_to_B(self,first,task,weight):

        return weight*(self.getTaskSpeed(first,task) - self.Stack.get_base(task))

    def get_time_passed(self):
        return (self.Stack.task_ts[self.current_feature] - self.Stack.last_baseline_t[self.current_feature])
    def get_reward_passed(self):
        return (self.Stack.task_Rs[self.current_feature] - self.Stack.last_baseline_R[self.current_feature])

    def after_estimation(self):
        self.Stack.last_baseline_t[self.current_feature] = self.Stack.task_ts[self.current_feature]
        self.Stack.last_baseline_R[self.current_feature] = self.Stack.task_Rs[self.current_feature]
        self.after_parameter_change()
    def get_Z_score_velocity(self,first,task,weight):
        difference = self.velocity_compared_to_B(first, task,weight)
        if difference == 0:
            return 0  # baseline performance or no time passed on the task
        S = self.Stack.get_std(task)
        return difference/S



    @overrides
    def popUntil(self):
        """
        here we will need to ensure comparisons are done
        :return:
        """

        if self.Stack.sp == 0:
            return
        # pop the stack until SSC satisfied
        self.popUntilConservativeSSC()
        if CHECK_VS and np.random.random()<.002:
            print("t=" + str(self.t))
            self.check_conservativeSSC("", "")




    def sort_stack(self):
        """
        sort the stack in ascending order and get the firsts in ascending order

        just for debugging purposes
        :return:
        """

        firsts, Vs = self.get_task_velocities(self.Stack.sp, float("inf"), until=0)


        indexes=np.argsort(Vs) # sort ascending
        #self.set_sorted_firsts([firsts[index] for index in indexes])
        temp_firsts=[firsts[index] for index in indexes]
        #assert firsts==self.sorted_firsts, "firsts %s ,  sorted firsts %s"%(firsts,self.sorted_firsts)
        return temp_firsts

    def remove_sorted_firsts(self):
        """
        after popping,
        remove all firsts no longer in the stack, or the top, because it is assumed to be the best
        :param sp:
        :return:
        """
        sp=self.Stack.top_first()
        if DEBUG_MODE:
            print("remove sorted")
        self.set_sorted_firsts([first for first in self.sorted_firsts if first < sp])

    def add_top(self):
        self.sorted_firsts.append(self.Stack.top_first())
    # def get_sorted_topfirst(self):
    #     """
    #     get the sorted topfirst (use for debugging: should be equal to self.Stack.top())
    #     :param sp:
    #     :return:
    #     """
    #
    #     return self.sorted_firsts[-1]
    def get_sorted_topsecond(self):
        """
        get the sorted topsecond, for comparing the best two velocities
        :return:
        """
        return self.sorted_firsts[-1]
    def after_parameter_change(self):
        """
        after a weight update, a baseline change, or a standard deviation change,
        sort the stack, pop back until the maximal velocity, remove sorted firsts
        :return:
        """


        self.params_changed=True

    # if CHECK_SORTING_ORDER:
    #     if self.evaluationcondition():
    #         self.sort_and_evaluate()
    # def sort_and_evaluate(self):
    #     temp_firsts=self.sort_stack()
    #     self.popUntilMaintainIndex(temp_firsts[-1])
    #     self.remove_sorted_firsts()
    #     self.add_top()
    #     self.resetFlowVariables()

    def get_tolerance(self):
        # if self.absolute:
        #     tol=self.delta_tolerance
        # else :
        #     sd=self.Stack.get_sd(self.current_feature)
        #     tol=self.delta_tolerance/float(sd)
        #return self.weights[self.current_feature]*self.delta_tolerance/self.Stack.get_sd(self.current_feature,absolute=self.absolute)
        return self.delta_tolerance
    def get_max_tolerance(self):
        if self.absolute:
            tol=self.delta_tolerance
        else :
            SD=self.Stack.get_std(self.current_feature)
            tol=self.delta_tolerance/SD
        return tol


    # def add_maxV_estimate(self,above_task_V,time_passed):
    #     # checks whether the observed V > maxV, if so maxV=Vobserved
    #     total_time=self.t_M[self.current_feature] + time_passed
    #     self.max_V[self.current_feature] = self.t_M[self.current_feature]/float(total_time)*self.max_V[self.current_feature] + \
    #         time_passed/float(total_time)*above_task_V
    #     self.t_M[self.current_feature] = total_time
    # def reset_maxV(self,current_t):
    #     Vs=[self.getTaskSpeed(first, self.current_feature) for first in self.sorted_firsts]
    #     maxind=np.argmax(Vs)
    #     maxfirst=self.sorted_firsts[maxind]
    #     self.max_V[self.current_feature]=Vs[maxind]
    #     self.t_M[self.current_feature]=current_t - self.Stack[maxfirst].task_ts.get(self.current_feature,0)
    # def set_maxV(self,time_passed,Vtop):
    #     self.max_V[self.current_feature]=Vtop
    #     self.t_M[self.current_feature]=time_passed
    def popBackUntilTaskTimePassed(self):
        """

        check the sorted firsts

        each of the components is monotonically increasing, therefore no sorting required

        if the top one has larger task-specific velocity, return

        else continue checking entries' global velocities until its task-specific velocity is greater
        :return:
        """
        top_first = self.Stack.top_first()
        second_first = self.sorted_firsts[-1]
        task_t = self.Stack.task_ts.get(self.current_feature, 0)
        topt = self.Stack[top_first].task_ts.get(self.current_feature,0)
        above_task_V,max=self.getLastSpeeds(top_first,self.current_feature)
        second_task_V,second_Vtilde=self.getLastSpeeds(second_first,self.current_feature)
        # if above_task_V>self.max_V[self.current_feature]:
        #     self.reset_maxV(task_t - topt, above_task_V)
        # else:
        # add to prior estimate


        if self.params_changed or self.first_task_instance:
            self.popNumberVelocities(until=0,num_velocities=float("inf"))
            self.remove_sorted_firsts()
            self.params_changed=False
            self.first_task_instance=False
            #self.V_old_task, self.V_old_second = self.getLastSpeeds(self.Stack.top_first(), self.current_feature)
            #
            # self.reset_maxV()
            self.add_top()
            return

        self.stats.num_evaluations+=1
        tol=self.get_tolerance()

        if above_task_V>self.max_V[self.current_feature]:
            self.max_V[self.current_feature]=above_task_V


        self.stats.num_checks+=2
        if above_task_V<=second_task_V:

            # delta_V_second  = second_Vtilde - self.V_old_second
            #
            # delta_V_task = second_task_V - self.V_old_task
            # abs_delta_V = -delta_V_task
            # S=self.Stack.get_std(self.current_feature,absolute=self.absolute)
            # factor=self.weights[self.current_feature]/S
            # assert np.isclose(delta_V_second,delta_V_task*factor),"%.6f %.6f %.6f, at time t=%d"%(delta_V_second,delta_V_task,factor,self.t)
            max=second_Vtilde
            #max_task_V =second_task_V
            max_first = self.sorted_firsts[-1]

            Mhat=self.max_V[self.current_feature]
            e_task_t = self.Stack[max_first].task_ts.get(self.current_feature,0)
            abs_delta_V=(task_t-topt)/ float(task_t - e_task_t) *(Mhat - above_task_V)

            if abs_delta_V >= tol:  # task-specific velocity smaller
                self.stats.num_backtracking+=1
                for i in range(len(self.sorted_firsts)-2,-1,-1): # go over all sp - 1 sorted firsts until you hit an evaluated entry
                    first=self.sorted_firsts[i]
                    e = self.Stack[first]
                    e_task_t = e.task_ts.get(self.current_feature,0)

                    vt,V =self.getLastSpeeds(first,self.current_feature)
                    self.stats.num_checks += 1
                    self.stats.num_backchecks+=1

                    if V >= max:
                        max=V
                        max_first=first
                        #max_task_V =vt
                    is_negative=vt>above_task_V
                    if is_negative:
                        abs_delta_V = (task_t-topt)/ float(task_t - e_task_t) *(Mhat - above_task_V)
                        if abs_delta_V < tol:
                            self.stats.num_earlystops+=1
                            break
            else:
                self.stats.num_prestops+=1

            if DEBUG_MODE:
                print("num backtracking=" + str(self.stats.num_backtracking))
                print("num backchecks=" + str(self.stats.num_backchecks))
                print("maxV=" + str(self.max_V[self.current_feature]))
                print("num early stops=" + str(self.stats.num_earlystops))
                print("delta="+str(abs_delta_V))
                print("num prestops=" + str(self.stats.num_prestops))
                print("num prestops=" + str(self.stats.num_prestopsOverTime))

        else:

            max_first = self.Stack.top_first()
            #max_task_V = above_task_V






        #self.V_old_task=max_task_V
        #self.V_old_second = max  # regardless of what happens, always refresh the V_old_second
        if max_first < self.Stack.top_first():
            if DEBUG_MODE:
                print("popping; maintain "+str(max_first))
            self.popUntilMaintainIndex(max_first)
            self.remove_sorted_firsts()

        self.add_top()


    # def popBackUntilTaskTimePassed(self):
    #     """
    #
    #     check the sorted firsts
    #
    #     each of the components is monotonically increasing, therefore no sorting required
    #
    #     if the top one has larger task-specific velocity, return
    #
    #     else continue checking entries' global velocities until its task-specific velocity is greater
    #     :return:
    #     """
    #     if self.params_changed:
    #         self.popNumberVelocities(until=0,num_velocities=float("inf"))
    #         self.remove_sorted_firsts()
    #         self.params_changed=False
    #         _,self.V_old_second = self.getLastSpeeds(self.Stack.top_first(),self.current_feature)
    #         return True
    #     self.stats.num_evaluations+=1
    #     tol=self.get_tolerance()
    #     top_first= self.Stack.top_first()
    #
    #     above_task_V,max=self.getLastSpeeds(top_first,self.current_feature)
    #     second_first = self.sorted_firsts[-1]
    #
    #     second_task_V,second_Vtilde=self.getLastSpeeds(second_first,self.current_feature)
    #     self.stats.num_checks+=1
    #     if above_task_V<=second_task_V:
    #         top_task_t = self.Stack.task_ts.get(self.current_feature,0)
    #         second_task_t = self.Stack[second_first].task_ts.get(self.current_feature,0)
    #         delta_V_second  = second_Vtilde - self.V_old_second
    #         max=second_Vtilde
    #         max_first = self.sorted_firsts[-1]
    #         if -delta_V_second >= tol:
    #
    #             delta_V_temp=delta_V_second * (top_task_t - second_task_t)
    #             for i in range(len(self.sorted_firsts)-2,-1,-1): # go over all sp - 1 sorted firsts until you hit an evaluated entry
    #                 first=self.sorted_firsts[i]
    #                 e = self.Stack[first]
    #                 e_task_t = e.task_ts.get(self.current_feature,0)
    #
    #                 _,V =self.getLastSpeeds(first,self.current_feature)
    #                 self.stats.num_checks += 1
    #
    #                 if V >= max:
    #                     max=V
    #                     max_first=first
    #                 delta_t = top_task_t - e_task_t
    #                 delta_V_est = delta_V_temp / float(delta_t)
    #                 if -delta_V_est < tol:
    #                         break
    #     else:
    #         max_first = self.Stack.top_first()
    #
    #     self.V_old_second = max  # regardless of what happens, always refresh the V_old_second
    #     if max_first < self.Stack.top_first():
    #         if DEBUG_MODE:
    #             print("popping; maintain "+str(max_first))
    #         self.popUntilMaintainIndex(max_first)
    #         self.remove_sorted_firsts()
    #
    #         return True
    #     else:
    #         return False
    #
    # @overrides
    def popUntilConservativeSSC(self):
        """
        go over the sorted_indexes until you hit one with time passed for the current task
        then compare the velocity to the top one

        return True if popped, False otherwise
        :return:
        """
        if DEBUG_MODE:
            print("pop until conservative SSC")



        self.popBackUntilTaskTimePassed()



    def getLastSpeeds(self,sp,F):
        first = self.Stack[sp].first





        #weights=[1 for task in self.seen_tasks]
        if self.absolute:
            Vs={task:self.velocity_compared_to_B(first,task,self.weights[task]) for task in self.seen_tasks}
        else:
            Vs ={task:self.get_Z_score_velocity(first,task,self.weights[task]) for task in self.seen_tasks}
        #Vs = [(self.getTaskSpeed(first, task) - self.Stack.get_base(task)) for task in [self.current_feature]]
        V_tilde=sum(Vs.values())
        return self.getTaskSpeed(first,F),V_tilde
    @overrides
    def getLastSpeed(self, sp):
        return self.getLastSpeeds(sp,self.current_feature)[1]

    def getTaskSpeed(self, first, F):


        R=self.Stack.current_R(F)
        t=self.Stack.current_t(F)
        assert t!=0, self.Stack.printFirsts()
        topR,topt=self.Stack.get_R_and_t(F, first)
        if topt==t: # no task time has passed; need to estimate it
            return self.in_between_velocity(first,F)
        return (R - topR) / float(t - topt)



    def in_between_velocity(self,first,F):

        sp=self.Stack.get_last_eval(first,F)


        return self.getTaskSpeed(sp,F)

    @overrides
    def add_entry(self, t, R, oldP, address, first):
        return self.Stack.create_stack_entry(tuple(self.current_feature), t, R, oldP, address, first)




    @overrides
    def time_passed_evaluation(self):
        if self.Stack.num_baseline_updates(self.current_feature)< self.min_updates : #
            return False
        if self.stall_evaluation:
            R, t = self.Stack.get_R_and_t(self.current_feature, self.Stack.top().first)
            return self.Stack.current_t(self.current_feature) > t
        else:
            R, t = self.Stack.get_R_and_t(self.current_feature, self.Stack.sp)
            return self.Stack.current_t(self.current_feature) > t



    @overrides
    def time_passed_modification(self):
        if self.stall_evaluation:
            return True
        else:
            top=self.Stack.top()
            t = top.task_ts.get(self.current_feature, 0)
            return self.Stack.current_t(self.current_feature) > t
    @overrides
    def get_t_V(self, sp, R, t):

        entry = self.Stack[sp]
        R0 = entry.task_Rs.get(self.current_feature, 0)
        t0 = entry.task_ts.get(self.current_feature, 0)

        V = self.get_speed(R0, t0, R, t)
        return t0, V


    def estimating_task(self,task):
        return self.Stack.num_baseline_updates(task) < NUM_BASE_LINE_UPDATES
    def is_estimating(self):
        for task in self.seen_tasks:
            if self.estimating_task(task):
                return True
        return False
    def get_task_velocities(self,sp,num_velocities,until):
        firsts = []
        Vs = []
        minimum = float("inf")
        total_velocities = num_velocities
        while sp >= until:
            first = self.Stack[sp].first
            V = self.getTaskSpeed(first,self.current_feature)
            firsts.insert(0, first)
            Vs.insert(0, V)
            sp = first - 1
            if V < minimum:
                minimum = V
            if len(Vs) == total_velocities:
                if Vs[0] != minimum:
                    total_velocities += num_velocities
                else:
                    break

        return firsts, Vs


    def check_conservativeSSC(self, Vs, message=''):
        """
        given some velocities, check the conservative success story criterion
        :param Vs:
        :return:
        """
        print(type(self))
        print(self.current_feature)
        print(" stack length = " + str(len(self.Stack)))

        i = 0

        firsts2, Vs2 = self.get_num_velocities(self.Stack.sp,until=0,num_velocities=float('inf'))
        if DEBUG_MODE:
            self.Stack.print_running_stats(absolute=self.absolute)
            print(str(Vs2))
        assert np.isclose(np.max(Vs2),Vs2[-1], atol=self.get_max_tolerance(),rtol=0.0), ' \n' + message  +  str(self.current_feature) + str(Vs2) + "\n" \
                                     +str(self.sorted_firsts) + "\n" + \
                                                     self.Stack.printFirsts()+"\n"

        print("Success Story Satisfied")
    # @overrides
    def stop_checking_evaluation_modification_gap(self, V0, V1,first0,first1):
        return V0 < V1 and self.Stack[first0].task_ts.get(self.current_feature,0)<self.Stack[first1].task_ts.get(self.current_feature,0)
    @overrides
    def learn(self):
        self.evaluate(None, None)


    def update_pointers(self):
        self.Stack.set_first(self.current_feature)
        # the top feature may have changed --> previous block pointer is way before where it was
        # F=self.Stack.top().F
        # self.Stack.update_prior_task_sp(self.Stack.sp, F)

        # self.Stack.max_popback=self.Stack.get_task_popback(self.current_feature)

    @overrides
    def popNoRestoreUntil(self, pop_back_until):

        SSAimplementor.popNoRestoreUntil(self, pop_back_until)
        self.update_pointers()

    @overrides
    def popBackUntil(self, pop_back_until):
        if DEBUG_MODE and STACK_PRINTING:
            dump_incremental("StackBEFORE" + str(self.t) + str(self.current_feature), self.Stack)
        SSAimplementor.popBackUntil(self, pop_back_until)
        self.update_pointers()

        if DEBUG_MODE and STACK_PRINTING:
            dump_incremental("Stack" + str(self.t) + str(self.current_feature), self.Stack)

