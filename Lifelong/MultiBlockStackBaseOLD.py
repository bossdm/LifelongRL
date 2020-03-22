
from IS.Stack import Stack, TaskSpecificStack
import IS.StackEntries
from IS.SSA import SSAimplementor

from abc import abstractmethod
from overrides import overrides

DEBUG_MODE=False

class BlockedStack(Stack):
    """
    recall that each
    each policy has a blocked stack, an object containing task-specific stacks across different task intervals. For example,

    S=[ F1: SMS1 , F2: SMS2, SMS3, F1: None]
    at the second instance of F1, before any modifications, V(SMS3) > V(SMS1) is tested. if that did not work out SMS3 is removed.
    then V(SMS2) > V(SMS1) is tested
    (Note that SMS1 serves as reference since it is the last verified on task F1)
    ---> success story is maintained for a single policy over various tasks !
    """
    def __init__(self,max_blocks,block_size):
        self.block_size=block_size
        #self.last_block={}
        Stack.__init__(self, max_blocks, initial_entry=None)

    def total_t_and_R_consumption(self,features):
        """

        :return:
        """
        tt=0
        RR=0
        for feature in features:
            t,R=self.get_final_t_and_R( feature)
            if t is None or R is None:
                continue
            tt+=t
            RR+=R
        return tt,RR

    def new_stack_block(self,current_feature,initial_entry):
        """
        at the start of a new task class, need to create a new task-specific stack
        :param initial_entry:
        :return:
        """
        self.push(TaskSpecificStack(current_feature,self.block_size,initial_entry))


    def set_velocity(self,min_time_evaluation):
        """
        get the velocity of for a specific task (namely, the current one)
        (because
        :return:
        """
        block = self.top()
        block.set_velocity(min_time_evaluation)
    # def set_last_block(self,feature):
    #     """
    #     keep track of the last verified block of this feature
    #     :param feature:
    #     :return:
    #     """
    #     if feature is None:
    #         return
    #     self.last_block[feature]=self.sp
    def get_first_block_index(self,feature):
        """
        return the index of the very first block with a given feature (starting from 0)
        :param feature:
        :return:
        """
        sp=0
        while sp <=self.sp:
            block=self[sp]
            if block.feature==feature:
                return sp
            sp+=1
        return None


    def get_last_block(self,feature):
        """
        return last block with a given feature (starting from previous index)
        :param feature:
        :return:
        """
        sp=self.sp-1
        while sp >=0:
            block=self[sp]
            if block.feature==feature:
                return sp,block
            sp-=1
        return None,None
    def get_final_block(self,feature):
        """
        return the final block index with a given feature (starts from current index)
        :param feature:
        :return:
        """
        sp=self.sp
        while sp >=0:
            block=self[sp]
            if block.feature==feature:
                return block
            sp-=1
        return None

    def get_final_t_and_R(self, feature):
        """

        :param feature:
        :return:
        """
        block=self.get_final_block(feature)
        if block is None:
            return None,None
        return block.t,block.R
    def get_next_block(self,feature,sp):
        sp=sp+1
        while sp <=self.sp:
            block=self[sp]
            if block.feature==feature:
                return block
            sp-=1
        return None
    def get_previous_other_block(self,feature,sp):
        """
        get previous block if it is NOT equal to a feature
        :param feature:
        :param sp:
        :return:
        """
        sp=sp-1
        block=self[sp]
        if block.feature!=feature:
                return block
        else:
            return None

    def get_previous_block(self,sp):
        """
        get previous block
        :param feature:
        :param sp:
        :return:
        """
        sp=sp-1
        if sp >= 0:
            return self[sp]
        return None

    def is_empty(self,sp):
        block=self[sp]
        if len(block) == 1:
            if self.get_first_block_index(block.feature)==sp:
                return False
            else:
                return True
        elif len(block) > 1:
            return False
        else:
            raise Exception()

    def get_last_entry(self,feature):
        """
        use this to get the top entry of another feature
        :param feature:
        :return:
        """

        block=self.get_last_block(feature)
        if block is not None:
            return block.top()
        return None
    def second_top(self):
        return self[self.sp-1]
    def top_feature(self):
        if self.sp < 0:
            return None
        return self.top().feature

    def get_all_topfirst_entries(self,sp):
        block= self[sp]
        firsts=[]
        index = block[sp].top_first()
        first=index
        while True:
            firsts.insert(first,0)
            index = first - 1
            first=block[sp][index].first
            if first==0:
                return firsts

    def __str__(self):
        string=""
        for i in range(self.sp+1):
            string+="F="+str(self[i].feature)+" :\n"
            string+=str(self[i])+"\n"
        return string
    __repr__ = __str__


class MultiBlockStackBase(object):
    """
    a simpler multistack base, use for homeostatic pols
    (uses task-policy specific
    """
    min_time_evaluation = 0
    required_rewards = 0
    def __init__(self,total_num_tasks):
        self.total_num_tasks=total_num_tasks
        #self.current_feature = None


    def initial_entry(self):
        index,last_block=self.BlockStack.get_last_block(self.current_feature)
        if last_block is not None:
            # handy to keep track of start of the block

            return IS.StackEntries.PolicyStackEntry(t=last_block.t, R=last_block.R, pol=None, first=0)  # just the simplest entry imaginable
        # current_block=self.BlockStack.top()
        # if current_block is not None:
        #     return IS.StackEntries.PolicyStackEntry(t=current_block.t,R=current_block.R,pol=None,first=0)
        # else:
        return IS.StackEntries.PolicyStackEntry(t=0, R=0, pol=None, first=0)  # just the simplest entry imaginable


    def previous_other_block_to_check(self):
        """

        :return:
        """
        sp=self.BlockStack.sp-1
        while True:
            previous_block=self.BlockStack.get_previous_block(sp)
            # if previous block is none, the previous block is the same feature
            if previous_block is None:
                return None
            # if previous block is empty, need to go further
            if not self.BlockStack.is_empty(sp-1):
                return previous_block

            # if previous block has the same feature, al
            sp-=1

    def previous_same_block_to_check(self):
        """

        :return:
        """
        sp = self.BlockStack.sp - 1
        while True:
            previous_block = self.BlockStack.get_last_block(self.current_feature)
            # if previous block is none, the previous block is the same feature
            if previous_block is None:
                return None
            # if previous block is empty, need to go further
            if not self.BlockStack.is_empty(sp - 1):
                return previous_block

            # if previous block has the same feature, al
            sp -= 1
    def finaliseStack(self):
        self.BlockStack[self.BlockStack.sp]=self.Stack
    def finish_SMS(self):
        #if not self.conservative:
        self.endSelfMod()  # finalise the current SMS and evaluate

        self.evaluate(self.Stack.t, self.Stack.R)


    def renew_stack(self,new_feature):
        # create new block in stack if required

        old_feature=self.BlockStack.top_feature()
        self.current_feature = new_feature
        if old_feature != self.current_feature:
            init_entry=self.initial_entry()

            self.BlockStack.new_stack_block(self.current_feature,initial_entry=init_entry)
            self.Stack=self.BlockStack.top()
            self.Stack.t=init_entry.t
            self.Stack.R=init_entry.R
            self.decide_test()
        self.t = self.Stack.t
        self.R = self.Stack.R
        self.polChanged = self.Stack.polChanged
    def prepare_test_of_previous_block(self,testing):
        """
        the previous block is tested by not changing the policy for a while,
        then assessing whether the reward velocity has speed up
        :return:
        """
        if testing:
            self.polChanged = True
            self.disablePLA = True
            self.numR = self.required_rewards
            self.beginOfSMS =False
            self.check_other_task_stack = True
        else:
            #if self.polChanged and self.disablePLA: # conditional prevents checking already tested stack parts
            self.check_same_task_stack = True
    def end_test_of_previous_block(self):
        """
        the previous block is tested by not changing the policy for a while,
        then assessing whether the reward velocity has speed up
        :return:
        """

        self.check_other_task_stack=False

    #
    # @overrides
    # def printR(self):
    #     if self.Rfile is not None:
    #         self.Rfile.write("%.2f \n"%(self.totalR))
    #         self.Rfile.flush()
    #     self.stats.R_overTime.append(self.totalR)

    def popAndRestore_CustomStack(self, sp,stack):  # evaluate if condition is met
        entry = stack[sp]
        self.restore(entry)
        if DEBUG_MODE:
            print("*custom* popping stack entry " + str(stack[sp]))
        #del self.Stack[sp]  # pop final element off the stack
        stack.pop()
    def popBackOther(self):

        stack=self.previous_other_block_to_check()
        if stack is None:
            self.end_test_of_previous_block()
            return
        pop_back_until=stack.top_first()
        #last_block=self.BlockStack.get_last_block(self.current_feature)
        sp=stack.sp
        while (sp >= pop_back_until):
            if DEBUG_MODE:
                print(sp)
            self.popAndRestore_CustomStack(sp,stack)
            sp-=1




        if (stack.sp == 0):
            raise Exception("popped too much it seems; never pop the first entry of a block")

        if stack.sp == 1:
            self.prepare_test_of_previous_block(True)  # prepare another test, for the previous SMS

        return sp

    def popBackSame(self):

        stack = self.previous_same_block_to_check()
        if stack is None:
            self.end_test_of_previous_block()
            return
        pop_back_until = stack.top_first()
        # last_block=self.BlockStack.get_last_block(self.current_feature)
        sp = stack.sp
        while (sp >= pop_back_until):
            if DEBUG_MODE:
                print(sp)
            self.popAndRestore_CustomStack(sp, stack)
            sp -= 1

        if (stack.sp == 0):
            raise Exception("popped too much it seems; never pop the first entry of a block")

        if stack.sp == 1:
            self.prepare_test_of_previous_block(False)  # prepare another test, for the previous SMS

        return sp
    # @overrides
    # def popUntilSSC(self):
    #
    #     # note that we already pushed the new entry, we now decide if we have to pop it
    #     while (True):
    #         if self.check_previous_task_stack:
    #             SSC = self.evaluate_and_compare_block_speeds()
    #             if SSC:
    #                 break
    #             self.popBackUntil_CustomStack()
    #         else:
    #
    #             SSC = self.callSSC()
    #             if self.check_previous_task_stack:
    #                 continue
    #             if SSC: # disable PLA means prepEval was called in empty_stack(), waiting to evaluate until velocity is estimated
    #                 break
    #             self.popBackUntil(self.Stack.top_first())
    #
    #     self.check_previous_task_stack=False
    #     if DEBUG_MODE:
    #         print("Stack=\n"+str(self.Stack))
    #         print("BlockStack=\n" + str(self.BlockStack))

    def popOtherSMS(self):



        SSC, index = self.evaluate_and_compare_other_block_speeds()
        if SSC:
            self.check_other_task_stack = False # no need to check anymore
            return
        self.popBackOther(index)
    def popSameSMS(self):


        while True:
            SSC,index = self.evaluate_and_compare_same_block_speeds()
            if SSC:
                self.check_same_task_stack = False # no need to check anymore
                return
            self.popBackSame(index)




    def check_previous_block(self):

        index,last_block = self.BlockStack.get_last_block(self.current_feature)
        return index,last_block is not None
    def decide_test(self):

        index, previous = self.check_previous_block()
        if previous:
            self.prepare_test_of_previous_block(testing=True)
    def empty_stack(self):
        """"""
        # if self.BlockStack.get_last_block(self.current_feature) is None:
        #     return self.Stack.sp == 0
        # else:
        #     if self.Stack.sp == 0:
        #         self.check_previous_task_stack=True  # next loop in the SSA evaluation: start checking previous blocks
        #         return True
        #     return False
        if len(self.Stack)==1:
            self.decide_test()
            return True
        else:
            print("sp=%d"%(self.Stack.sp))
            return False



def test_empty_check():
    from IS.StackEntries import StackEntry
    BlockStack = BlockedStack(10, 50)
    BlockStack.push(TaskSpecificStack((1.0,0.,0.),50,initial_entry=StackEntry(t=0,R=0,oldP=None,address=None,first=0)))
    assert not BlockStack.is_empty(0)

    BlockStack.push(TaskSpecificStack((1.0, 0., 0.), 50, initial_entry=StackEntry(t=5, R=0, oldP=None, address=None, first=0)))
    assert BlockStack.is_empty(1)
    assert not BlockStack.is_empty(0)
    t,R=BlockStack.get_final_t_and_R((1.0,0.,0.))
    assert t==BlockStack.top().t
    assert R==BlockStack.top().R
    BlockStack.push(
        TaskSpecificStack((0.0, 0., 0.), 50, initial_entry=StackEntry(t=10, R=0, oldP=None, address=None, first=0)))
    t, R = BlockStack.get_final_t_and_R((1.0, 0., 0.))
    assert t == BlockStack.second_top().t
    assert R == BlockStack.second_top().R
def test_block_stack():
    test_empty_check()



if __name__ == "__main__":
    test_block_stack()