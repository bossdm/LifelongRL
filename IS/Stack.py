"""
generic stack object, useful for ALL SSA IMPLEMENTATIONS
"""
import IS.StackEntries
class Stack(object):

    def __init__(self,max_size,initial_entry):
        self.max_size=max_size
        self.chunk_size=max_size
        self.data=self.get_chunk(max_size)
        self.sp = -1
        if initial_entry is not None:
            self.push(initial_entry)
    @classmethod
    def listToStack(self,l):
        stack=Stack(5000,None)
        for entry in l:
            stack.push(entry)

        return stack

    def get_chunk(self,size):
        return [None]*size
    def __getitem__(self, item):
        return self.data[item]
    def __setitem__(self, key, value):
        self.data[key]=value
    def __len__(self):
        return self.sp+1
    def top(self):
        """
        get top element without popping
        :return:
        """
        return self.data[self.sp]
    def second_first(self):
        return self.data[self.top_first()-1].first
    def push(self,entry):
        """
        add entry on top
        :param entry:
        :return:
        """
        self.sp+=1
        if self.sp >= self.max_size:
            self.data.extend(self.get_chunk(self.chunk_size))
            self.max_size+=self.chunk_size
        self.data[self.sp]=entry
    def pop(self):
        self.data[self.sp]=None
        self.sp-=1
    def pop_index(self,index):
        self.data[index:self.sp]=self.data[index+1:self.sp+1]
        self.sp-=1
    def top_first(self):
        return self.top().first
    def top_first_entry(self):
        return self[self.top().first]
    def previous_first(self):
        return self[self.top_first()-1].first
    def printFirsts(self):

        s = ""

        first = self.sp
        while first >= 0:
            first = self[first].first
            s += str(self[first]) + " \n"
            first -= 1

        return s
    def __str__(self):
        strin=""
        for i in range(self.sp+1):
            strin+=str(self.data[i]) + "\n"
        return strin
    __repr__=__str__



class TaskSpecificStack(Stack):
    """
    a stack specific to a task

    this implies:
    -computes time relative to its usage
    -computes reward relative to its usage
    """
    def __init__(self,feature,max_size,initial_entry):
        self.polChanged=False
        self.t=0
        self.R=0
        Stack.__init__(self,max_size,initial_entry)
        self.velocity=None
        self.feature=feature

        self.previous_block_index = None
        self.evaluation_marker = None

    def set_velocity(self,min_time_evaluation):
        """
        get the velocity of a specific stack
        (because
        :return:
        """

        sp = self.sp

        while True:

            entry = self[sp]
            t = entry.t
            if self.t - t >= min_time_evaluation:
                break
            if sp == 0:
                self.velocity=None
                return
            sp -= 1
        V= (self.R - entry.R) / float(self.t - t)
        self.velocity=V

    def __str__(self):
        strin=""
        strin += "t,R= " + str((self.t, self.R)) + "\n"
        strin += "eval_marker" + str(self.evaluation_marker) + "\n"
        for i in range(self.sp+1):
            strin+=str(self.data[i]) + "\n"

        return strin