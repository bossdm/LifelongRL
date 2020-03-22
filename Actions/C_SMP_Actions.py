from Action import Action

from enum import Enum
import numpy as np
import copy
from random import random
from abc import abstractmethod
from overrides import overrides
class ArgumentTypes(Enum):

    PREDICTION_POINTER=1
    SUBPOL_POINTER=2
    INSTR_POINTER=3
    INSTR_SET_POINTER=4
    WM_POINTER = 5 # address sets in WM

class ArgumentGenerator(object):
    def __init__(self):
        pass
    @abstractmethod
    def generate_argument(self):
        pass


# ignore for now
class DeterministicArgumentGenerator(ArgumentGenerator):
    def __init__(self,arg_list):
        self.arg_list
        ArgumentGenerator.__init__(self)
    @overrides
    def generate_argument(self):
        return self.arg_list

class StochasticArgumentGenerator(ArgumentGenerator):
    """
    class used to generate a single integer argument between 0 and max-1.
    since these arguments are pointers they can still refer to floats
    """
    def __init__(self,max,p=None):
        self.max=max
        if p is not None:
            self.p=p
        else:
            equiprob=1/float(self.max)
            self.p = [equiprob for _ in range(self.max)] # probabilities for each
        ArgumentGenerator.__init__(self)
    @overrides
    def generate_argument(self):
        """

        :return: integer argument in [0,max-1] based on probability list p
        """
        return np.random.choice(self.max, p=self.p)




class C_SMP_Action(Action):
    def __init__(self,function,maxima):
        """


        :param function: the function performed during action
        :param maxima: list of ints used to specify the range of the parameters

        because in C_SMP the number of pointers is fixed (though not always all of them actually implement something)
        """
        self.maxima=maxima

        Action.__init__(self,function,len(maxima))
        self.arg_generators=[]
        for arg in range(self.action.n_args):
            self.arg_generators.append(ArgumentGenerator(self.action.maxima[arg]))

class C_SMP_InstructionCell(object):
    def __init__(self):
        pass
    @abstractmethod
    def generate_instruction(self):
        pass
class DeterministicInstructionCell(C_SMP_InstructionCell):
    def __init__(self, action):
        """

        :param action: a C_SMP_Action instance
        """

        self.action=action
        C_SMP_InstructionCell.__init__(self)
    @overrides
    def generate_instruction(self):
        arguments=[generator.generate_argument() for generator in self.action.arg_generators]
        return self.action,arguments
    def __str__(self):
        return "DeterministicCSMPCell:"+str(self.action.function.__name__)
    def __eq__(self,other):
        return self.action == other.action
    __repr__ = __str__

class StochasticInstructionCell(C_SMP_InstructionCell):
    def __init__(self,actions, p=None):
        """
        :param actions: list of C_SMP_Action instances
        """
        self.actions=actions
        self.max=len(self.actions)
        if p is not None:
            self.p=p
        else:
            equiprob=1/float(self.max)
            self.p = [equiprob for _ in range(self.max)] # probabilities for each
        C_SMP_InstructionCell.__init__(self)
    @classmethod
    def construct(cls,instance):
        return cls(copy.copy(instance.p),instance.max)
    @overrides
    def generate_instruction(self):
        actionIndex= np.random.choice(self.max, p=self.p)
        action=self.actions[actionIndex]
        arguments = [generator.generate_argument() for generator in action.arg_generators]
        return action,arguments

    def __getitem__(self, key):
         return self.p[key]

    def __setitem__(self,key,item):
        self.p[key] = item
    def __delitem__(self,key):
        del self.p[key]
    def __str__(self):
        action_strings = [action.function.__name__ for action in self.actions]
        return "StochasticCSMPCell: actions %s   probabilities %s"%(str(action_strings),str(self.p))
    def __eq__(self,other):
        return self.p == other.p and self.actions == other.actions and self.max==other.max
    __repr__ = __str__
