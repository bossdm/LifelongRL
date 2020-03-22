

import numpy as np
from IS.Stack import  Stack
from IS.StackEntries import GlobalStackEntrySimple

from enum import Enum
DEBUG_MODE = False
INIT_BASELINE_FREQUENCY=20  # first three estimates at frequency 20
BASELINE_FREQUENCY=50000 # should get rapid estimate at first presentation of a task
NUM_BASE_LINE_UPDATES=float("inf")# after 40000 steps estimates should no longer change anyway


WEIGHT_FREQUENCY=50000

NUM_BASE_LINE_TRACKS=float("inf") # print 15 baseline calculations (each 10000 time steps; last 5 should be equal)
CHECK_VS=True
STACK_PRINTING=False
CHECK_SORTING_ORDER=False
class RunningStatSSA(object):
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
        # we bias it to return 1/(n-1) if S still 0--> progressively get smaller variance but never zero

        var= self._S/(self._n - 1) if self._n > 1 else np.square(self._M)
        if var == 0:
            return np.zeros(self.shape) + 1. / float(self._n - 1)
        else:
            return var
    @property
    def std(self):
        return np.sqrt(self.var)
    @property
    def shape(self):
        return self._M.shape




class ZFilterSSA(object):
    """
    y = (x-mean)/std
    using running estimates of mean,std
    """

    def __init__(self, shape):


        self.rs = RunningStatSSA(shape)

    def __call__(self, x, update=True):
        if update: self.rs.push(x)
        return x
    def output_shape(self, input_space):
        return input_space.shape

class WeightingType(Enum):
    fixed=0
    time_consumption=1


class MultiStack(object):





    def __init__(self):

        # feature, max_size, initial_entry
        self.Stack = Stack(5000, initial_entry=GlobalStackEntrySimple({}, {}, None, 0, 0, "start"))
        self.task_ts = {}
        self.task_Rs = {}
        self.task_sp = {}
        # self.prior_task_sp = {}
        self.max_popback = 1
        self.baseline={}
        self.running_stats={}
        # self.var_V={}
        # self.S = {}
        #self.num_baseline_updates={}
        self.last_baseline_t = {}
        self.last_baseline_R = {}
        self.task_popback={}


        self.tasks_since_last_eval=[]
        self.sp = self.Stack.sp



    def __getitem__(self, item):
        return self.Stack[item]
    def __setitem__(self, key,value):
        self.Stack[key] = value

    def __len__(self):
        return self.Stack.sp + 1

    def new_task(self, F, absolute=True):
        if F not in self.task_ts:
            self.task_ts[F] = 0
            self.task_Rs[F] = 0
            self.baseline[F] = 0.
            self.running_stats[F]=ZFilterSSA(1,) # mean and sd
                # self.var_V[F] =0.0
                # self.S[F]  = 0.0
            #self.num_baseline_updates[F] = 0
            self.last_baseline_t[F]=0
            self.last_baseline_R[F]=0
        #     self.set_task_sp(F, 0)
        #     self.prior_task_sp[F]=0
        # else:
        #     self.prior_task_sp[F]=self.task_sp[F]

        self.current_feature = F

        print("new task")

    def current_R(self, task):
        return self.task_Rs.get(task, 0.)

    def current_t(self, task):
        return self.task_ts.get(task, 0)

    def get_R_and_t(self, task, sp):
        top = self.Stack[sp]
        return top.task_Rs.get(task, 0.), top.task_ts.get(task, 0)


    # def get_alpha(self,N):
    #     return 1/float(N)
    # def adjust_base(self,alpha,newV,B):
    #
    #     return (1-alpha)*B+ alpha*newV
    # def adjust_S(self,newV,newB,oldB,task,weight=1.):
    #     old_s = self.get_S(task)
    #
    #     new_S = old_s + weight*(newV-newB)*(newV-oldB)
    #
    #     return new_S
    # def adjust_var(self,S,N):
    #     return S/float(N)
    def num_baseline_updates(self,F):
        return self.running_stats[F].rs.n
    # def compute_baseline(self,R,t,absolute=True):
    #     self.num_baseline_updates[self.current_feature] += 1
    #     N=self.num_baseline_updates[self.current_feature]
    #     alpha=self.get_alpha(N)
    #     newV = R / float(t)
    #
    #     B = self.get_base(self.current_feature)
    #
    #
    #     newB=self.adjust_base(alpha,newV,B)
    #
    #     self.baseline[self.current_feature]=newB
    #     if not absolute:
    #         self.S[self.current_feature]=newS=self.adjust_S(newV,newB,B,self.current_feature)
    #         self.var_V[self.current_feature]=self.adjust_var(newS,N)
    #
    #     if DEBUG_MODE:
    #         print(" got reward R=%.2f, time=%d"%(R,t))
    #         print("F="+str(self.current_feature))
    #         print("baseline is now B_i=%.4f"%(self.baseline[self.current_feature]))
    #
    #     return
    def compute_baseline(self,R,t):
        #self.num_baseline_updates[self.current_feature] += 1
        V=R/float(t)
        self.running_stats[self.current_feature](V)
    def printFirsts(self):
        return self.Stack.printFirsts()

    def get_base(self,F):
        if F not in self.running_stats:
            return 0.
        return self.running_stats[F].rs.mean[0]

    def print_running_stats(self,absolute):

        for task in self.running_stats:
            print("task:"+str(task))
            print("B="+str(self.get_base(task)))
            print("S="+str(self.get_std(task,absolute)))
    # def get_var(self,F):
    #     if F not in self.running_stats:
    #         return 0.
    #     return self.running_stats[F].rs.
    def get_std(self,F,absolute=False):
        if absolute:
            return 1.0
        if F not in self.running_stats:
            return 1.0
        return self.running_stats[F].rs.std[0]
    # def get_S(self,F):
    #     """
    #     helper to calculate the variance incrementally
    #     :param F:
    #     :return:
    #     """
    #     return self.S.get(F,0)

    # def get_max_popback(self,sp, task):
    #     """
    #
    #     :param task:
    #     :return:
    #     """
    #     pb=self.get_popback(sp,task)
    #     assert self.Stack[pb].F!=task
    #     return max(1,pb)
    #
    #
    #     # if task != self.Stack.top().F:
    # def get_popback(self,sp,task):
    #     if sp==0:
    #         return 0
    #     prev=self.Stack[sp].first - 1
    #     while True:
    #         first=self.Stack[prev].first
    #         if self.Stack[first].F != task:
    #             return first
    #         prev=first-1
    #
    # def get_task_popback(self,task):
    #
    #     """
    #     for debugging purposes define the task popback
    #
    #     to see if each SSC is satisfied until task popback
    #     :param task:
    #     :return:
    #     """
    #
    #
    #
    #     sp=self.task_sp.get(task,0)
    #
    #     self.assert_task_sp(task)
    #     if sp==0:
    #         return 1
    #     entry=self.Stack[sp]
    #     while entry.F == task:
    #         first=self.Stack[entry.first - 1].first
    #         if first<=1:
    #             return 1
    #         entry=self.Stack[first]
    #     return entry.first
    def get_top_firsts(self):
        """
        assuming at_task_boundary() is true, get the top two firsts

        (assert that they are not equal to the current task)
        :return:
        """
        e_top=None
        e_second=None
        sp=self.Stack.sp
        while sp > 0:
            first =self.Stack[sp].first
            print(first)
            e = self.Stack[first]
            sp = first - 1
            if e.F == self.current_feature:
                return e_top,e_second
            eval_entry=self.get_evaluated_entry(first,self.current_feature)
            if eval_entry is not None:
                continue # already evaluated
            if e_top is None:
                e_top=e
            else:
                return e_top, e

        return e_top,e_second
    def get_last_eval(self, sp, F):
        """
        because unevaluated entries have unknown velocities

        the algorithm may only compare to the first evaluated entry


        note: this is obtained by starting from the back and returning just before the first non-evaluated index
        :param: just_after: if requesting the firsteval just after a particular entry
        :return:
        """
        if sp==0:
            return 0
        t = self.Stack[sp].task_ts.get(F,0)
        if t == 0:
            return None
        sp-=1
        while sp >= 0:
            first = self.Stack[sp].first
            entry=self.Stack[first]
            if entry.task_ts.get(F,0) < t:
                return entry.first
            sp -= 1
        #return self.Stack[self.task_sp[F]].first


    def get_first_eval(self, sp, task):
        """
        because unevaluated entries have unknown velocities

        the algorithm may only compare to the first evaluated entry


        note: this is obtained by starting from the back and returning just before the first non-evaluated index
        :param: just_after: if requesting the firsteval just after a particular entry
        :return:
        """

        if sp == 0:
            return 1
        max_pop_back = sp
        while sp > 0:
            first = self.Stack[sp].first
            e = self.Stack[sp]
            sp = first - 1

            if e.F == task:
                max_pop_back = first
                continue
            eval_entry = self.get_evaluated_entry(first, task)
            if eval_entry is not None:
                max_pop_back = first
                continue  # already evaluated
            else:
                return max_pop_back


        return max_pop_back

    def get_not_evaluated_entry(self, sp, F):
        """
        return the entry only if it is NOT evaluated
        :param sp:
        :param F:
        :return:
        """
        if sp == 0:  # always evaluated
            return None
        e = self.Stack[sp]
        if e.eval.get(F, False):
            return None
        return e

    def get_evaluated_entry(self, sp, F):
        """
                return the entry only if it is evaluated
                :param sp:
                :param F:
                :return:
        """
        e = self.Stack[sp]
        if sp == 0 or e.eval.get(F, False):  # always evaluated
            return e
        return None

    #     else:
    #         self.max_pop_back = self.get_block_start(task)

    def get_block_start(self, task, sp):
        """
        get the start of the block with this task
        :param sp: entry known to be in that task
        :return:
        """
        entry = self.Stack[sp]
        if entry.F != task:
            return None
        while entry.F == task and sp > 1:
            sp = self.Stack[sp - 1].first
            entry = self.Stack[sp]

        return sp

    # def set_task_popback(self,task,sp):
    #
    #     start=self.get_block_start(task,sp)
    #     first=self.Stack[start-1].first
    #     self.task_popback[task]=first




    def get_task_stack(self, F, sp, first_match=False, add_init=True, until=0):
        """
        get the stack entries introduced in task F
        :param task:
        :return:
        """
        if not first_match:
            if add_init:
                stack = [self.Stack[0]]
                start = 1
            else:
                stack = []
                start = 0
        sp = self.Stack[sp].first
        while sp >= until:
            entry = self.Stack[sp]
            evaluated_entry = self.get_evaluated_entry(sp, F)
            if evaluated_entry is not None:
                if first_match:
                    return sp
                stack.insert(start, entry)
            sp = self.Stack[sp - 1].first
        if first_match:
            return None
        else:
            return stack

    def push(self, stack_entry):
        """
        add entry to the current stack, and add pointers to this entry in the other stacks
        :param stack_entry:
        :param task:
        :return:
        """
        self.Stack.push(stack_entry)
        self.sp += 1
        F = self.current_feature
        self.set_task_sp(F,self.sp)
        self.update_popback(F)

    def update_popback(self,F):
        if self.task_popback.get(F,None) is None:
            self.task_popback[F] = self.Stack[self.sp].first # lowest index seen with e.taskt > 0
            if self.task_popback[F]==1:
                self.task_popback[F]=0


    def set_first(self,F):
        if self.task_popback.get(F,None) is None:
            return
        if self.Stack.sp < self.task_popback[F]:
            self.task_popback[F]=None

    def top(self):
        return self.Stack.top()

    def create_stack_entry(self, task, t, R, oldP, address, first):
        return GlobalStackEntrySimple(dict(self.task_ts), dict(self.task_Rs), task, t, R, oldP, address, first)

    def top_first(self):
        return self.Stack.top_first()

    def set_task_sp(self, F, task_sp):
        self.task_sp[F] = task_sp
        if DEBUG_MODE:
            self.assert_task_sp(F)

    def assert_task_sp(self, F):
        sp = self.task_sp.get(F, 0)
        if sp == 0:
            return
        F1 = self.Stack[self.task_sp[F]].F
        assert F1 == F, str(self.task_sp[F]) + " stacklength=" + str(self.sp) + str(F1) + str(F)

    def update_task_sp(self, F):
        """

        :param F:
        :return:
        """

        task_sp = self.Stack.sp
        prev = self.Stack[task_sp]  # get the top entry

        while prev.F != F:

            if task_sp == 0:
                break
            task_sp = self.Stack[task_sp].first - 1  # gget the last entry of the previous SMS
            prev = self.Stack[task_sp]
        self.set_task_sp(F, task_sp)
        # self.update_prior_task_sp(task_sp,F)
        if DEBUG_MODE:
            print("task sp " + str(self.task_sp[F]))
            print(self.Stack.sp)

    def get_prior_task_sps(self, sp, num):
        """
        get 'num' earlier first entries which were evaluated for the current task
        :param sp:
        :param num:
        :return:
        """
        F = self.current_feature
        first = self.Stack[sp].first
        if first == 0:
            return [first]
        sps = []
        while True:
            entry = self.get_evaluated_entry(first, F)
            if entry is not None:
                sps.insert(0, first)

            first = self.Stack[first - 1].first
            if len(sps) == num or first == 0:
                break
        if 0 not in sps:
            sps.insert(0, 0)
        return sps

    def pop(self):
        """
        pop the entry and return it
        :return:
        """
        entry = self.Stack.top()
        self.Stack.pop()
        self.sp -= 1
        # update the task's sp
        self.update_task_sp(self.current_feature)
        if entry.F != self.current_feature:
            self.update_task_sp(entry.F)

        # assert np.all(sp for sp in self.last_sps.values())
        return entry

    def update_task_time(self, task, increment):
        self.task_ts[task] += increment

    def update_task_R(self, task, increment):
        self.task_Rs[task] += increment

    def get_time_passed(self,sp,F):
        return (self.task_ts.get(F,0) - self.Stack[sp].task_ts.get(F,0))

    def __str__(self):
        return str(self.Stack)


