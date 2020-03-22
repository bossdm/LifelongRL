



# note: seems most handy to keep them in single file, have some troubles with isinstance otherwise

class Action(object):
    modifying=False
    def __init__(self, function=None,n_args=0,time_cost=None,energy_cost=None):
        self.function = function
        self.n_args = n_args # how many arguments required for SSA ?
        self.time_cost = time_cost
        self.energy_cost = energy_cost

    def __str__(self):
        return self.function.__name__
    def perform(self,args):
        """performs the action."""
        #print(str(self.function))
        return self.function(*args)

    def __ne__(self, other):
        return not self == other
    def __eq__(self,other):
        return (self.function.__name__,self.n_args,self.time_cost,self.energy_cost)==(other.function.__name__,self.n_args,self.time_cost,self.energy_cost)
    def __hash__(self,other):
        return hash((self.function.__name__,self.n_args,self.time_cost,self.energy_cost))
    def __deepcopy__(self, memo):
        cls=self.__class__
        result = cls.__new__(cls)
        memo[id(self)] = result
        result.function=self.function
        result.n_args=self.n_args
        result.time_cost=self.time_cost
        result.energy_cost=self.energy_cost
        result.modifying=self.modifying

        return result
    __rep__ = __str__


