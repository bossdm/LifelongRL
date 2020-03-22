from copy import copy, deepcopy
from enum import Enum
from Actions.Action import Action

class PLAResult(Enum):
    error=1
    not_prepared=2
    success=3
    removed_change=4
class PLA(Action):
    modifying=True
    def __init__(self,function=None,n_args=0):
        Action.__init__(self,function,n_args)

class NetworkPLA(Action):
    modifying=True
    def __init__(self,function=None,n_args=0):
        Action.__init__(self,function,n_args)
class NeatAction(Action):
    def __init__(self,function=None,n_args=0):
        Action.__init__(self,function,n_args)
class NeatPLA(NetworkPLA):
    def __init__(self,function=None,n_args=0):
        NetworkPLA.__init__(self,function,n_args)
class MultipleNetworks_Action(Action): #actions that involve creation or deletion of networks
    def __init__(self,function=None,n_args=0):
        Action.__init__(self,function,n_args)
class MultipleNetworks_PLA(NetworkPLA): #actions that involve creation or deletion of networks
    def __init__(self,function=None,n_args=0):
        NetworkPLA.__init__(self,function,n_args)
class ExternalAction(Action):

    def __init__(self,function=None,n_args=0):
        Action.__init__(self,function,n_args)

class VectorAction(ExternalAction):
    "use for vizdoom"
    def __init__(self,idx,vector,function,n_args=0):
        self.idx=idx
        self.vector=vector
        ExternalAction.__init__(self, function, n_args)

    def perform(self,args):
        """performs the action."""
        #print(str(self.function))
        return self.function(*args)


    def __deepcopy__(self, memo):
        cls=self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.function=self.function
        result.n_args=self.n_args
        result.time_cost=self.time_cost
        result.energy_cost=self.energy_cost
        result.modifying=self.modifying
        result.vector=deepcopy(self.vector)
        result.idx=self.idx
        return result

class ClassificationAction(ExternalAction):

    def __init__(self,k=1,function=None,n_args=0):   # function should set chosenClass=k, something like that
        # using k allows to generate different actions for each possible classification
        self.k=k
        ExternalAction.__init__(self,function,n_args)

    def __deepcopy__(self, memo):
        cls = self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.function = deepcopy(self.function)
        result.n_args = self.n_args
        result.time_cost = self.time_cost
        result.energy_cost = self.energy_cost
        result.modifying = self.modifying
        result.k=self.k

        return result

