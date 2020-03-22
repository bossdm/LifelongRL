

class StackEntry(object):

    def __init__(self,t,R,oldP,address=None,first=0):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.t = t
        self.R = R
        self.address = address
        self.first = first
        self.oldP = oldP

    def __str__(self):
        return '[t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldP) + ']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldP == other.oldP \
               and self.first == other.first

    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.t, self.R, self.address,self.oldP,self.first)) # note oldP's __hash__ is defined elsewhere
    __repr__=__str__

class StackEntryNP(object):

    def __init__(self,t,R,first,oldNP,address):

        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.t = t
        self.R = R
        self.address = address
        self.first = first
        self.oldNP = oldNP

    def __str__(self):
        return '[t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        'oldNP'+str(self.oldNP)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldNP == other.oldNP \
               and self.first == other.first


class GlobalStackEntry(StackEntry):

    def __init__(self,task_ts,task_Rs,F,t,R,oldP,address=None,first=0):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.task_ts=task_ts
        self.task_Rs=task_Rs

        self.F=F
        self.eval={}
        for F in self.task_ts:
            if F != self.F:
                self.eval[F]=False
            else:
                self.eval[F]=True
        StackEntry.__init__(self,t,R,oldP,address,first)

    def __str__(self):
        return '[eval='+str(self.eval)+'[F='+str(self.F)+'t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldP) + '\nts:\n'+str(self.task_ts)+'\nRs:\n'+str(self.task_Rs)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldP == other.oldP \
               and self.first == other.first

    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.t, self.R, self.address,self.oldP,self.first)) # note oldP's __hash__ is defined elsewhere
    __repr__=__str__

class GlobalStackEntryNP(StackEntryNP):

    def __init__(self,task_ts,task_Rs,F,t,R,oldNP,address=None,first=0):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.task_ts=task_ts
        self.task_Rs=task_Rs

        self.F=F
        self.eval={}
        for F in self.task_ts:
            if F != self.F:
                self.eval[F]=False
            else:
                self.eval[F]=True
        StackEntryNP.__init__(self,t=t,R=R,first=first,oldNP=oldNP,address=address)

    def __str__(self):
        return '[eval='+str(self.eval)+'F='+str(self.F)+'t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldNP) + '\nts:\n'+str(self.task_ts)+'\nRs:\n'+str(self.task_Rs)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldNP == other.oldNP \
               and self.first == other.first

    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.t, self.R, self.address,self.oldP,self.first)) # note oldP's __hash__ is defined elsewhere
    __repr__=__str__




class GlobalStackEntrySimple(StackEntry):

    def __init__(self,task_ts,task_Rs,F,t,R,oldP,address=None,first=0):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.task_ts=task_ts
        self.task_Rs=task_Rs

        self.F=F

        StackEntry.__init__(self,t,R,oldP,address,first)

    def __str__(self):
        return '['+str(self.F)+'t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldP) + '\nts:\n'+str(self.task_ts)+'\nRs:\n'+str(self.task_Rs)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldP == other.oldP \
               and self.first == other.first

    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.t, self.R, self.address,self.oldP,self.first)) # note oldP's __hash__ is defined elsewhere
    __repr__=__str__

class GlobalStackEntryNPSimple(StackEntryNP):

    def __init__(self,task_ts,task_Rs,F,t,R,oldNP,address=None,first=0):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.task_ts=task_ts
        self.task_Rs=task_Rs

        self.F=F

        StackEntryNP.__init__(self,t=t,R=R,first=first,oldNP=oldNP,address=address)

    def __str__(self):
        return '[F='+str(self.F)+'t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldNP) + '\nts:\n'+str(self.task_ts)+'\nRs:\n'+str(self.task_Rs)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldNP == other.oldNP \
               and self.first == other.first

    def __ne__(self, other):
        return not self == other
    def __hash__(self):
        return hash((self.t, self.R, self.address,self.oldP,self.first)) # note oldP's __hash__ is defined elsewhere
    __repr__=__str__




class TaskStackEntry(StackEntry):
    def __init__(self, t, R, oldP, address=None, first=0,previous_block_marker=None,previous_block_index=None):
        StackEntry.__init__(self,t,R,oldP,address,first)
        self.previous_block_marker = previous_block_marker
        self.previous_block_index = previous_block_index

    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldP == other.oldP \
               and self.first == other.first and self.previous_block_index==other.previous_block_start_index \
                                    and self.previous_block_marker==other.previous_block_marker


class StackEntryLSTMSSA(object):

    def __init__(self,t,R,oldP,address=None,first=0,type=None):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.t = t
        self.R = R
        self.address = address
        self.first = first
        self.oldP = oldP
        self.type=type

    def __str__(self):
        return '[t='+str(self.t)+',R='+str(self.R)+',address='+str(self.address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldP) + ',type='+str(self.type)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldP == other.oldP \
               and self.first == other.first and self.type==self.type

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        # note: assuming oldP is not huge array
        return hash((self.t, self.R, self.address,str(self.oldP),self.first,self.type))

    __repr__ = __str__

class StackEntryTransfer(object):
    def __init__(self,t,R,oldP,IP_address=None,transfer_address=None,first=0):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.t = t
        self.R = R
        self.IP_address = IP_address
        self.transfer_address = transfer_address
        self.first = first
        self.oldP = oldP

    def __str__(self):
        return '[t='+str(self.t)+',R='+str(self.R)+',IPaddress='+str(self.IPaddress)+',transfer_address='\
               +str(self.transfer_address)+',first='+str(self.first)+\
        ',oldP='+ repr(self.oldP) + ']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.address==other.address and self.oldP == other.oldP \
               and self.first == other.first

    __repr__ = __str__
class PolicyStackEntry(object):
    """
    when an 'initial policy' is used, this is added on top of the current stack
    note: "first" should always be this very pointer
    """
    def __init__(self,t,R,pol,first):
        self.t=t
        self.R=R
        self.pol=pol
        self.first=first
    def __str__(self):
        return '[t='+str(self.t)+',R='+str(self.R)+',first='+str(self.first)+\
        ',oldP='+ repr(self.pol)+']'
    def __eq__(self,other):
        return self.t==other.t and self.R==other.R and self.pol == other.pol \
               and self.first == other.first

    def __ne__(self, other):
        return not self == other

    def __hash__(self):
        # note: assuming oldP is not huge array
        raise Exception()


class StackEntryPredMod(object):
    def __init__(self, t, R, oldMeans, address, first):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.t = t
        self.R = R
        self.address = address
        self.first = first
        self.oldMeans = oldMeans

    def __str__(self):
        return '[t=' + str(self.t) + ',R=' + str(self.R) + ',address=' + str(self.address) + ',first=' + str(
            self.first) + \
               ',oldMeans=' + repr(self.oldMeans) + ']'

class MultiStackEntryNP(object):

    def __init__(self, t, R, first, oldNP, old_net_key):
        # t is a large integer representing time
        # R is a floating point representing cumulative reward
        # first is the index at the start of the PMP
        # oldP is the old policy
        self.t = t
        self.R = R
        self.first = first
        self.oldNP = oldNP
        self.old_net_key = old_net_key

    def __str__(self):
        return '[t=' + str(self.t) + ',R=' + str(self.R) + ',first=' + str(self.first) + 'oldNP' + str(
                self.oldNP) + ']'
# # stack entry for a change in network policy
# class StackEntryNP(object):
# 	def __init__(self, t, R, oldNetworkP):
# 		# t is a large integer representing time
# 		# R is a floating point representing cumulative reward
# 		# oldNetworkP is the old policy
# 		# (e.g., [network1,network2] --> [network1]--> [network1,network3]
# 		# will have stack entries: [ E(t1,R1, [network1,network2]),E(t2,R2,[network1])]
# 		# [network1a] --> [network1b] has stack entry [E(t1,R1,network1a)]
# 		self.t = t
# 		self.R = R
# 		self.oldNetworkP = oldNetworkP
#
# 	def __str__(self):
# 		return '[t=' + str(self.t) + ',R=' + str(self.R) + ',oldNetworkP='+ str(self.oldNetworkP)+']'