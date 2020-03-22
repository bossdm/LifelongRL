

import IS.StackEntries


#from StatsAndVisualisation.analyse_utils import heatmap, PLT

from IS.Stack import Stack

from IS.ArgumentConversions import narrowing_conversion,conversion

from Actions.Action import Action
from Actions.SpecialActions import PLA, ExternalAction, PLAResult
#from IS.EvaluationPolicy import SupervisedSelfModel
from Methods.Learner import CompleteLearner
from IS.Policy import *
from ExperimentUtils import dump_incremental
import random



from IS.IS_LearnerStatistics import IS_LearnerStatistics

def arity(unboundmethod):
    return unboundmethod.func_code.co_argcount - 1 # remove self
from overrides import overrides

# # Success Story Algorithm (SSA), also called Environment Independent Reward Acceleration (EIRA)
# # Implementation by David Bossens

DEBUG_MODE=False
STACK_PRINTING=False
REAL_EXPERIMENT=True
CHECK_Vs=True
RANDOM_STEPS=15


class SSAimplementor(CompleteLearner):
    wait_eval_until_next_mod=True
    stall_evaluation=False
    jump_at_reset=False
    has_internal_actions=True
    conservative=False
    max_num_velocities=3
    r_steps=RANDOM_STEPS
    external_actions=[]

    def __init__(self,actions,filename,prepEvalParam=1,maxTime=None,conservative=False,jump_at_reset=False,stall_evaluation=False,
                 episodic=False):
        CompleteLearner.__init__(self, actions,filename,episodic=episodic)
        self.prepEvalParam = prepEvalParam # affects how early prepEval will come
        self.maxTime = maxTime # if numR stays low, evaluate eventually after maxtime ticksfevaluat passed
        self.n_ops = len(actions)
        # flow control parameters
        self.conservative=conservative
        self.disablePLA=False
        self.beginOfSMS=True
        self.currentSMS=0
        self.currentInstruction=None
        self.disablePLAticks=0
        self.polChanged=False
        self.wait_modification=False
        self.jump_at_reset=jump_at_reset
        self.stall_evaluation=stall_evaluation
        self.t = 0
        self.Pol = "emptyPol"  # the current policy = program cells
        self.indexToBeModified=-1

        self.action_is_set=False
        self.initStack()
        self.polfile = open(self.file + 'policy.txt', "w")
        #self.SMSfile = open(self.file + 'SMS.txt', "w")
        self.Stackfile=open(self.file + 'Stack.txt',"w")
        print('initialStack'+self.printStack())
        self.numR=0

        self.num_external_actions,self.external_actions=SSAimplementor.count_external_actions(self.actions)
    @classmethod
    def count_external_actions(cls,actions):
        external_actions=[]
        num_external_actions=0
        for action in actions:
            if isinstance(action,ExternalAction):
                num_external_actions+=1
                external_actions.append(action)
        return num_external_actions,external_actions
    @overrides
    def initStats(self):
        self.stats=IS_LearnerStatistics(self)

    def initStack(self):
        self.Stack = Stack(5000,StackEntries.StackEntry(self.t, self.R, self.Pol)) # initialize stack with initial stack entry
    def printStack(self):
        stack_size = len(self.Stack)
        strin=""
        for i in range(stack_size):
            strin+="t=%d \t" % (self.t,) + str(self.Stack[i]) + "\n"
            strin+="t=%d \t" % (self.t,) + str(self.Stack[i]) + "\n"
        return strin
    def writeStack(self):
        stack_size = len(self.Stack)
        self.Stackfile.write("t=%d   length=%d\t" % (self.t,stack_size))
        for i in range(stack_size):
            self.Stackfile.write(str(self.Stack[i]) + ",")
        self.Stackfile.write("\n")
        self.Stackfile.flush()
    @overrides
    def printDevelopment(self):
        CompleteLearner.printDevelopment(self)
        addedModifications = self.stats.numPModifications
        if self.stats.numPModifications:
            addedModifications-=self.stats.numPModificationsOverTime[-1]
        self.stats.numPModificationsOverTime.append(self.stats.numPModifications)
        self.stats.numPModificationErrorsOverTime.append(self.stats.numPModificationErrors)
        if hasattr(self.stats,'stackLengthOverTime'):
            if self.t==0:
                self.stats.stackLengthOverTime.append(0)
            else:
                self.stats.stackLengthOverTime.append(len(self.Stack))
            if addedModifications == 0:
                self.stats.changeSize=np.nan
            else:
                self.stats.changeSize /= float(addedModifications)
            self.stats.changeSizeOverTime.append(self.stats.changeSize)
            self.stats.changeSize=0 # reset (we will increment it)
            self.stats.instruction_frequenciesOverTime.append(self.stats.instruction_frequencies)
            self.stats.init_action_freq(self.actions+self.external_actions) # reset (we will increment it)
            if DEBUG_MODE:
                print("changesize=%.2f "%(self.stats.changeSize))


    @abstractmethod
    def initPol(self):
        pass

    @abstractmethod
    def preparePolicyChange(self,index):
        pass
    def getPointerToFirstModification(self,sp):# pointer to the first entry representing a modification computed by the current SMS

            if (self.beginOfSMS):
                self.beginOfSMS = False
                return sp
            else:

                return self.Stack[sp-1].first
    def count_down_to_evaluation(self):
        if self.r != 0 and self.disablePLA:
            self.numR -= 1
    #virtual void print()
    @overrides
    def setReward(self,reward):
        CompleteLearner.setReward(self,reward)
        self.count_down_to_evaluation()


    @overrides
    def printPolicy(self):
        self.polfile.write("Final Pol: \n")
        self.polfile.write("\t ")
        for j in range(self.n_ops):
            self.polfile.write(str(self.actions[j].function.__name__) + ' (%d)\t' % (j,))
        self.polfile.write("\n")
        for i in range(self.m):
            self.polfile.write("Cell/IP %d \t" % (i,))
            for j in range(self.n_ops):
                self.polfile.write('%04.3f \t' % (self.Pol[i][j],))
            self.polfile.write("\n")
        stack_size = len(self.Stack)

        # self.polfile.write("\n ------------------- \n Final Stack: \n")
        # for i in range(stack_size):
        #     self.polfile.write(str(self.Stack[i]) + "\n")
        self.polfile.write("\n ------------------ \n Num Valid P Modifications %d "%(stack_size-1,))
        self.polfile.write("\n ------------------ \n Num Total P Modifications %d " % (self.stats.numPModifications,))
        self.polfile.flush()


    #
    # def get_probability_matrix_overTime(self,filename):
    #     """
    #     warning: only do after saving
    #     :return:
    #     """
    #
    #
    #     sp=self.Stack.sp
    #     self.plotMatrix(filename)
    #     while sp > 0:
    #         first=self.Stack.top_first()
    #         self.popAndRestore(first)
    #         self.plotMatrix(filename)
    # def plotMatrix(self,filename):
    #
    #     matrix = self.Pol.asMatrix()
    #     t = self.Stack.top().t
    #     # Bilinear interpolation - this will look blurry
    #
    #     scale = self.m/float(self.n_ops) # scale it so the plot is not too huge vertically
    #     fig, ax = PLT.subplots(figsize=(100.,100.*scale))
    #
    #     im, cbar = heatmap(matrix, row_labels=self.Pol.p.keys(), col_labels=[str(a) for a in self.actions], ax=ax,
    #                        cmap="Greys_r", cbarlabel="probability")
    #
    #     # plt.imsave(, im, cbar)
    #     #texts = annotate_heatmap(im, valfmt="{x:.1f} t")
    #
    #     fig.tight_layout()
    #     PLT.savefig(filename + "matrix_t" + str(t) + ".png")

    def printStatistics(self):
        # see number of modifications and their success over time
        self.Stackfile.write("num modifications over time: \n")
        ind= 0
        result=[]
        sum=0
        num_slices=len(self.stats.R_overTime)
        self.stats.numValidPModificationsOverTime=[]

        for tt in range(0,int(self.t),int(self.t)/num_slices):
            time=tt+int(self.t)/num_slices
            # get all entries before
            temp=[]
            for i in range(ind,len(self.Stack)):
                if self.Stack[i].t >= time:
                    ind=i
                    break
                temp.append(self.Stack[i])
             # 1. count the number of valid modifications as time progresses
            self.Stackfile.write(str(len(temp))+"\t")
            sum+=len(temp)
            self.stats.numValidPModificationsOverTime.append(sum)

        self.Stackfile.write("\n")
        self.Stackfile.flush()


        # self.polfile.write("\n ------------------- \n Final Stack: \n")
        # for i in range(stack_size):
        #     self.polfile.write(str(self.Stack[i]) + "\n")

    def get_speed(self,R0,t0,R1,t1):
        """

        :param R0: initial cumulative reward
        :param t0: initial time
        :param R1: later cumulative reward
        :param t1: later time
        :return: the reward speed (float)
        """
        return (R1-R0)/float(t1-t0)

    def getLastSpeed(self,sp):
        return (self.R - self.Stack[self.Stack[sp].first].R) / float(self.t - self.Stack[self.Stack[sp].first].t)

    def getSpeeds(self):


        sp=self.Stack.sp
        v1 = self.getLastSpeed(sp)
        v2 = self.getLastSpeed(self.Stack[sp].first - 1)
        return v1, v2
    def at_max_popback(self):
        return self.Stack.sp == 0
    def callSSC(self):
        """
        get the reward speed of the latest modification sequence and compare it to the paenultimate modification sequence in stack
        :param sp:
        :return:
        """
        #print(self.t)
        # SSA1
        if self.at_max_popback(): # only the initial state --> SSA is satisfied, so return True
            #(time ti,  Re, int addr, Policy pol, size_t firstS, size_t sp)
            #Stack.push_back(StackEntry(t,R,changedadress,oldPol,sp+1))
            #polChanged = False
            return True# 1 will end the current evaluation interval

        ## SSA2: if copy size > 1, there may be some other policy to compare with
        #if (Stack.size() == 2) # if there is only one other entry, compare to the initial one
        #
        #	secondTopTag = tag(0, 0)
        #
        # SSA3
        v1,v2=self.getSpeeds()
        return v1 > v2

    def prepEval(self,instr):
        if self.disablePLA or not self.polChanged:
            return #already preparing evaluation or policy has not changed yet
        self.disablePLA = True
        self.numR = 1+self.prepEvalParam* (instr + 1)
        if DEBUG_MODE:
            print("PREPARE_EVAL: disabled PLAs and waiting for " +str(self.numR) + " more rewards")
    def jumpHome(self):
        self.IP = self.ProgramStart

    def time_passed_evaluation(self):
        if self.stall_evaluation:
            return self.t > self.Stack[self.Stack.top().first].t
        else:
            return self.t > self.Stack.top().t

    def time_passed_modification(self):
        if self.stall_evaluation:
            return True
        else:
            return self.t > self.Stack.top().t

    # PLAs
    def endSelfMod(self):
        if self.polChanged and self.time_passed_evaluation():   #evaluate immediately unless policy not yet changed
            self.disablePLA=True
            self.numR = 0

    def endSelfModAfterNext(self):
        """
        await the next self-modification before applying endSelfMod;
        the next modification will be immediately after evaluation, ensuring the conservative SSA
        :return:
        """
        if self.polChanged and self.time_passed_evaluation():
            self.wait_modification=True
    def remove_evaluate_and_add(self):
        self.removePolicyChange()

        self.evaluate_and_reset()

        self.addPolicyChange()
        self.wait_modification = False
        assert self.Stack.top().t == self.t

    def success_result(self):
        """
        record the success of a PLA
        :return:
        """
        self.stats.result = PLAResult.success
        if self.wait_modification:
            self.remove_evaluate_and_add()


    #     self.wait_CSSA()
    # def wait_CSSA(self):
    #     """
    #     if awaiting modification, ensure that the action was modifying and its modification carried out successfully;
    #     if so, evaluate !
    #
    #     # note: if evaluation condition not yet met, 'wait_modification' is still true --> wait for the next success result
    #     :return:
    #     """
    #     if self.wait_modification:
    #             if DEBUG_MODE:
    #                 print("finally, endSelfMod")
    #             self.endSelfMod() # disable PLA --> make way for evaluation
    #             evaluated=self.evaluate(self.t,self.R)  # evaluate if evaluation conditions met
    #             if evaluated:
    #                 self.wait_modification = False # no longer waiting, evaluation succeeded;
                # else, wait for the next succesful modification after this one
    # def setPseudoR(self):
    #     raise NotImplementedError


    def forceEvaluation(self):

        if (self.disablePLA):

            ++self.disablePLAticks

        else :
            self.disablePLAticks = 0

        if (self.disablePLAticks >= self.maxTime):
            #self.setPseudoR()
            self.numR = 0# --> force evaluation

    def restore(self, entry):
        if isinstance(entry, StackEntries.StackEntry):
            if (entry.oldP is not None):
                self.Pol[entry.address] = entry.oldP  # to restore old policy
            else:
                raise Exception()
        elif isinstance(entry, StackEntries.PolicyStackEntry):
            self.Pol=entry.pol
        elif isinstance(entry, StackEntries.StackEntryPredMod):
            if (entry.oldMeans is not None):
                self.predictiveMod.means[entry.address - self.ProgramStart] = entry.oldMeans  # to restore old policy
            else:
                raise Exception()
        else:
            raise Exception()
    def restoreUntil(self,until):

        sp=self.Stack.sp

        while (sp >= until):
            if DEBUG_MODE:
                print(sp)
            entry=self.Stack[sp]
            self.restore(entry)
            sp -= 1
    def popNoRestoreUntil(self,pop_back_until):
        sp=len(self.Stack) - 1
        while (sp >= pop_back_until):
            if DEBUG_MODE:
                print(sp)
            self.Stack.pop()
            sp -= 1

        if (len(self.Stack) == 0):
            raise Exception("popped too much it seems")

        return sp

    def popAndRestore(self, sp):  # evaluate if condition is met
        entry = self.Stack[sp]
        self.restore(entry)
        if DEBUG_MODE:
            print("popping stack entry " + str(self.Stack[sp]))
        #del self.Stack[sp]  # pop final element off the stack
        self.Stack.pop()


    def popBackUntil(self,pop_back_until):
        sp=len(self.Stack) - 1
        while (sp >= pop_back_until):
            if DEBUG_MODE:
                print(sp)
            self.popAndRestore(sp)
            sp -= 1

        if (len(self.Stack) == 0):
            raise Exception("popped too much it seems")

        return sp

    def popUntilSSC(self):

        # note that we already pushed the new entry, we now decide if we have to pop it
        while (True):
            # getReward(environment,t0, t)#environment will run now from t0 to t to see if any changes during the loop() function
            SSC = self.callSSC()
            if SSC:
                break
            self.popBackUntil(self.Stack.top().first)
        if DEBUG_MODE:
            self.check_SSC()
        if self.wait_eval_until_next_mod and CHECK_Vs:
            if random.random()<.002:
                self.check_conservativeSSC("")

    def get_CSSC_index(self,Vs):
        return len(Vs)-1



    def popNumberVelocities(self,until,num_velocities):
        firsts, Vs = self.get_top_SMS_velocities(until, num_velocities=num_velocities)

        if DEBUG_MODE:
            print(self.Stack.sp)
            print(Vs)

        if DEBUG_MODE and STACK_PRINTING:
            dump_incremental("StackBEFORE" + str(self.t), self.Stack)

        popped = self.forceConservativeSSC(Vs,firsts)

        return popped
    def get_all(self):
        return False

    def getNumVelocitiesandUntil(self):
        if self.get_all():
            velocities=float("inf")
            return velocities,0
        else:
            velocities=self.max_num_velocities
            return velocities,0
    def popUntilConservativeSSC(self,until=0):
     # note: first check top three velocities, then do in pairs as usual
        if DEBUG_MODE:
            print("pop until conservative SSC")


        popped=True
        velocities,until=self.getNumVelocitiesandUntil()

        while popped and self.Stack.sp != 0:
            popped=self.popNumberVelocities(until,num_velocities=velocities)
            if velocities==float("inf") and until==0:
                break



        if DEBUG_MODE and STACK_PRINTING:

            dump_incremental("Stack" + str(self.t),self.Stack)

        if CHECK_Vs:
            if random.random()<.002:
                self.check_conservativeSSC("")

        return popped

    def popUntilMaintainIndex(self,stop):
        popped = False
        # note that we already pushed the new entry, we now decide if we have to pop it
        while True:
            # get the index until which the conservative SSC is valid
            sp = self.Stack.top().first
            if sp == stop or sp == 0:
                break
            self.popBackUntil(sp)
            popped = True

        return popped

    def forceConservativeSSC(self,Vs,firsts):
        maintain_index = self.call_ConservativeSSC(Vs)
        stop=firsts[maintain_index]
        self.popUntilMaintainIndex(stop)


    def resetFlowVariables(self):

        # reset flow variables after evaluation
        self.disablePLA = False  # enable PLAs again not
        self.polChanged = False
        self.beginOfSMS = True
        self.currentSMS += 1

        # if REAL_EXPERIMENT:
        # 	self.SMSfile.write(str(self.t)+"\n")

        if DEBUG_MODE: self.writeStack()
    # def forbiddenAction(self):
    # 	# action is forbidden if
    # 	# 1. PLA and PLA disabled
    # 	# 2. prepEval and PLA already disabled or the policy not yet changed since last evaluation
    # 	return self.disablePLA and isinstance(self.chosenAction, PLA) or (
    # 		(self.chosenAction.function == self.prepEval or self.chosenAction.function == self.endSelfMod) and (not self.polChanged or self.disablePLA))
    def evaluationcondition(self):
        return self.numR <= 0 and self.disablePLA and self.polChanged and self.time_passed_evaluation() #time needs to have passed for evaluation
    def popUntil(self):

        if self.Stack.sp == 0:
            return
        # pop the stack until SSC satisfied
        if self.conservative:
            if self.get_all():
                self.popUntilConservativeSSC()
            elif self.wait_eval_until_next_mod:

                self.popUntilSSC()
            else:

                self.popUntilConservativeSSC()

        else:


            self.popUntilSSC()
    def evaluate_and_reset(self):
        self.popUntil()
        self.resetFlowVariables()

    def evaluate(self, time,  R) : # evaluate if condition is met
        self.forceEvaluation()
        if(self.evaluationcondition()):

            if DEBUG_MODE:
                print( "CALL_SSA : evaluation condition met. ")
                print("NumR = " + str(self.numR))
                print( "disablePLA = " + str(self.disablePLA))
                print("Stack before evaluation: " + self.printStack())
            self.evaluate_and_reset()
            if DEBUG_MODE:
                print("Stack after evaluation: " + self.printStack())
                #self.display_stackblock_velocities(plot=True)
            return True
        return False


    def get_t_V(self,sp,R,t):

        entry = self.Stack[sp]

        R0 = entry.R
        t0 = entry.t

        V = self.get_speed(R0, t0, R,t )
        return entry.t, V

    def get_num_velocities(self,sp,num_velocities,until):
        firsts = []
        Vs = []
        minimum = float("inf")
        total_velocities = num_velocities
        while sp >= until:
            first = self.Stack[sp].first
            V = self.getLastSpeed(first)
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
    def get_top_SMS_velocities(self,until=0,num_velocities=float('inf')):

        sp = self.Stack.sp
        firsts,Vs=self.get_num_velocities(sp,num_velocities,until)
        return firsts, Vs  # no task markers
    def display_stackblock_velocities(self,plot=True):
        """
        for each task: display the velocity for each SMS over time, and assert that these velocities, for each task, grow over time
        :return:
        """

        firsts,Vs,_=self.get_top_SMS_velocities() # get the starting time ("first") and the velocities


        #     if max(Vs)> y_max:
        #         y_max=max(Vs)
        #     if min(Vs) < y_min:
        #         y_min=min(Vs)
        # # normalise
        # y_max=y_min + 1.2*(y_max-y_min)
        if plot:
            import matplotlib as mpl
            from matplotlib import pyplot as PLT

            PLT.scatter(ts,Vs)

            PLT.savefig("SSA.png")
            PLT.gcf().clear()

        self.check_conservativeSSC()
    def call_ConservativeSSC(self,Vs):
        Vs=np.array(Vs)
        index = np.argmax(Vs)
        if isinstance(index,list):
            return index[0]
        else:
            return index

    def check_SSC(self,Vs):
        """
        given some velocities, check the success story criterion
        :param Vs:
        :return:
        """
        assert len(Vs) <= 1 or Vs[-1] > Vs[-2], str(Vs)
    def check_conservativeSSC(self, message=''):
        """
        given some velocities, check the conservative success story criterion
        :param Vs:
        :return:
        """
        print(type(self))
        print(" stack length = " + str(len(self.Stack)))
        i = 0

        firsts2, Vs2 = self.get_num_velocities(self.Stack.sp,num_velocities=float("inf"),until=0)


        print(Vs2)

        assert sorted(Vs2)==Vs2, ' \n' + message + str(Vs2) + str(self.t) + str(self.R)+'\n'+self.Stack.printFirsts()
        #assert len(Vs2) <= 1 or Vs2[-1] == max(Vs2), ' \n' + message + str(Vs2) + str(self.t) + str(self.R)+'\n'+self.Stack.printFirsts()

        print("Success Story Satisfied")
    @abstractmethod
    def addPolicyChange(self):
        # change polic and return the old policy
        pass

class SSA(SSAimplementor):
    evalPol=None
    max_arguments = 0
    def __init__(self, num_program, actions,internal_actionsSSA,filename,maxLoop=10,freezingThresh=.50,prepEvalParam=2,
                 maxTime=10000,incprobGamma=.15,conservative=False,jump_at_reset=False,episodic=False,
                 separate_arguments=False):
        ####
        self.separate_arguments=separate_arguments
        self.minP = 0.0005

        print(internal_actionsSSA)

        for key,value in internal_actionsSSA.items():
            function = getattr(self, key)
            if key in ['incProb','decProb','searchP']:
                actions.append(PLA(function,value))
            else:
                actions.append(Action(function,value))
            if value > self.max_arguments:
                self.max_arguments=value
            print(actions[-1].function.__name__)

        SSAimplementor.__init__(self,actions,filename,prepEvalParam,maxTime,conservative=conservative,
                                jump_at_reset=jump_at_reset,episodic=episodic)

        # keep track of last modification

        self.incProbGamma = incprobGamma  # increase by 1 percent * instruction
        self.m = num_program  # number of program cells
        self.freezingThresh = freezingThresh
        self.maxLoop = maxLoop
        self.maxLoop = maxLoop


        self.initPol()


    def get_policy_copy(self):
        # raise Exception("need to check the deepcopy")
        return copy.deepcopy(self.Pol)
    def set_policy(self,pol):
        self.Pol=pol

    @overrides
    def initPol(self):
        assert self.n_ops - 1 + (self.n_ops - 1) ** 2 >= self.m - 1 # needed for narrowing conversion
        self.ProgramStart=0
        self.Max = self.m - 1
        self.MaxInt = 1000
        self.IP = 0  # instruction pointer (points to one of the programcells)
        self.maxInstr = self.n_ops
        pol = np.zeros((self.m, self.n_ops))
        for i in range(self.m):
            for j in range(self.n_ops):
                pol = 1. / float(self.n_ops)
        self.Pol = Policy(pol,[StochasticProgramCell])
    @overrides
    def save(self,filename):
        if self.evalPol:
            self.evalPol.save(filename)
    @overrides
    def load(self,filename):
        if self.evalPol:
            self.evalPol.load(filename)

    def initEvaluationPol(self,filename):
        #self.evalPol=EvaluationPolicy(self.Pol,self.n_ops,self.minP,self.ProgramStart,self.Max)
        loss_file=open(filename+"_loss","wb")
        self.evalPol=SupervisedSelfModel(self.Pol,self.n_ops,loss_file=loss_file)

    def incProb(self,instr0,instr1,instr2):
        prepared=self.preparePolicyChange(self.getIndexToBeModified(instr0,instr1))
        if not prepared:
            return

        # if DEBUG_MODE:
        # 	print('incProb:' + str(self.Pol))
        oldPol = copy.copy(self.Pol[self.indexToBeModified])
        if self.policyCondition(self.indexToBeModified):
            # below one has trouble enforcing minP, unless you want to ignore almost every incProb
            # for o in range(0,self.n_ops):
            #
            # 	if (o == instr2):
            # 		self.Pol[self.indexToBeModified][o] += self.incProbGamma
            # 	else:
            # 		self.Pol[self.indexToBeModified][o] -= self.incProbGamma/(self.n_ops-1)
            # 		if self.Pol[self.indexToBeModified][o] < self.minP:
            # 			self.Pol=oldPol
            # 			break
            self.Pol[self.indexToBeModified][instr2] += self.incProbGamma
            C = sum(self.Pol[self.indexToBeModified])
            for o in range(self.maxInstr):
                self.Pol[self.indexToBeModified][o] /= C
                if(self.Pol[self.indexToBeModified][o] < self.minP):
                    self.Pol[self.indexToBeModified] = oldPol
                    self.removePolicyChange()
                    self.stats.result=PLAResult.removed_change
                    break
        if DEBUG_MODE:
            print('--->' + str(self.Pol))
        self.success_result()


    def decProb(self, instr0,instr1, instr2):
            prepared = self.preparePolicyChange(self.getIndexToBeModified(instr0,instr1))
            if not prepared:
                return
            # if DEBUG_MODE:
            # 	print('incProb:' + str(self.Pol))
            oldPol = copy.copy(self.Pol[self.indexToBeModified])
            if self.policyCondition(self.indexToBeModified):
                # below one has trouble enforcing minP, unless you want to ignore almost every incProb
                # for o in range(0,self.n_ops):
                #
                # 	if (o == instr2):
                # 		self.Pol[self.indexToBeModified][o] += self.incProbGamma
                # 	else:
                # 		self.Pol[self.indexToBeModified][o] -= self.incProbGamma/(self.n_ops-1)
                # 		if self.Pol[self.indexToBeModified][o] < self.minP:
                # 			self.Pol=oldPol
                # 			break
                self.Pol[self.indexToBeModified][instr2] -= self.incProbGamma
                C = sum(self.Pol[self.indexToBeModified])
                for o in range(self.maxInstr):
                    self.Pol[self.indexToBeModified][o] /= C
                    if (self.Pol[self.indexToBeModified][o] < self.minP):
                        self.Pol[self.indexToBeModified] = oldPol
                        self.removePolicyChange()
                        self.stats.result=PLAResult.removed_change
                        break
            if DEBUG_MODE:
                print('--->' + str(self.Pol))
            self.success_result()

    def freeze(self,a1, a2,a3):
        cell = narrowing_conversion(a1*self.maxInstr + a2, (0,(self.maxInstr+1)*self.maxInstr-1),(0,self.m))
        progCycles = 1+a3
        if max(self.Pol[cell].p) < self.freezingThresh:
            if DEBUG_MODE: print("freezing thresh %f not met"%(self.freezingThresh,))
            return
        self.Pol[cell].timeUntilUnfrozen = progCycles + 1
        dummyIndex=self.Pol[cell].frozenInstruction()
        dummyAction = self.actions[dummyIndex]
        if  dummyAction == self.freeze: #cannot freeze the freeze action
            if self.jumpHome_at_error: self.jumpHome()
            return
        for i in  range(dummyAction.n_args):
            self.Pol[cell+i].timeUntilUnfrozen = progCycles
    def setIP(self, instr): #instr in [0,n_ops-1]
        if DEBUG_MODE:
            print('SET IP:' + str(self.IP))
        self.IP = round(instr*(self.m-1)/float(self.n_ops-1))
        if DEBUG_MODE:
            print('--->' + str(self.IP))
    def instructionToAction(self):

        return self.actions[self.currentInstruction[0]]
    def instructionToParam(self):
        pass
    def looping(self):
        pass
    def random_action(self):


        self.action_idx = random.randrange(self.num_external_actions)
        self.currentInstruction = [-(self.action_idx + 1)]
        self.chosenAction = self.external_actions[self.action_idx]
        self.random_steps -= 1
    def generateInstruction(self):
        if DEBUG_MODE:
            print("disablePLA="+str(self.disablePLA))
            print("numR="+str(self.numR))



        if self.episodic and self.random_steps > 0:
            self.random_action()

        else:
            self.currentInstruction = []
            self.currentInstruction.append(self.Pol.generateInstruction(self.IP))

            self.chosenAction=self.instructionToAction()
        #print("chosenaction="+str(self.chosenAction.function.__name__))


    def afterInstruction(self):
        self.incrementIP()
        self.N_j = self.chosenAction.n_args

    def incrementIP(self,num=1):
        self.IP += num
    def getIPindex(self):
        return self.Min+2
    def get_IP_addition(self):
        return self.N_j if not self.separate_arguments else self.max_arguments
    def generateParameters(self):
        if DEBUG_MODE: print('N_j'+str(self.N_j))

        self.currentInstruction.extend([self.Pol.generateInstruction(self.IP+i) for i in range(0,self.N_j)])
        self.IP+=self.get_IP_addition()
        self.c[self.getIPindex()]=self.IP
    def afterParameters(self):
        # print(self.currentInstruction)
        # print(self.IP)
        # After successful setting, check whether IP > m - 1 again
        if (self.IP == self.Max+1):
            #print("jumpHome")
            self.jumpHome()  # if so, reset IP for the following loop
        else:
            if(self.IP > self.Max+1):
                raise Exception()
    def resetIP(self):
        if self.IP > self.Max:
            self.jumpHome()

    def getIndexToBeModified(self,instr1,instr2):
        # here instr is a list containing the action and the arguments

        return narrowing_conversion(instr1 * self.maxInstr + instr2,(0,(self.maxInstr-1)*(self.maxInstr+1)), (0,self.m-1))



    def __str__(self):
        str='\t'
        for i in range(0,self.n_ops):
            str+=str(self.actions[i]) + '\t'
        str+='\n'
        for i in range(0,self.m):

            str+='Cell '+str(i)+'\t'
            for j in range(0,self.n_ops):

                str+=" \t "+str(self.Pol[i][j])

            str+="\n"
        return str
    def add_entry(self,t,R,oldP,address,first):
        return StackEntries.StackEntry(t=t, R=R, oldP=oldP, address=address, first=first)
    def addPolicyChange(self):
        # change polic and return the old policy
        self.polChanged = True
        # reward should be modified by the environment (preferably in parallel)
        address = self.indexToBeModified  # address of the modified program cell
        oldP = StochasticProgramCell.construct(self.Pol[self.indexToBeModified])
        first = self.getPointerToFirstModification(len(self.Stack))  # pointer to the index of the first modification of this SMS on the stack (take into account the new entry so size-1+1)
        newEntry = self.add_entry(t=self.t, R=self.R, oldP=oldP, address=address, first=first)

        self.Stack.push(newEntry)

        if DEBUG_MODE:
            print("New Stack Entry: " + str(newEntry))
            self.writeStack()

        return oldP
    def removePolicyChange(self):
        # remove the policy change (restoring the old policy is done outside;
        # use when modification was not reasonable, e.g. p < minP)

        if self.Stack.top().first==self.Stack.sp:
            self.polChanged = False
            self.beginOfSMS=True
        self.Stack.pop()

    #converting instruction argument to reasonable ranges

    def policyCondition(self,index):
        return index >=self.ProgramStart and index <=self.Max

    def preparePolicyChange(self,index):
        if not self.time_passed_modification(): #time needs to have passed for evaluation
            if DEBUG_MODE:
                print("no time passed, not changing")
            return False,None
        if (isinstance(self.chosenAction, PLA) and not self.disablePLA):
            self.indexToBeModified=index
            if self.policyCondition(self.indexToBeModified):
                oldP=self.addPolicyChange()
                return True, oldP
            if DEBUG_MODE:
                print("index does not point to a part of Pol")
        if DEBUG_MODE:
            print("not a PLA or PLAs are disabled")
            if not isinstance(self.chosenAction,PLA):
                raise Exception()
        return False,None

    def preparePredictiveModification(self, index):
        if not self.time_passed_modification():  # time needs to have passed for evaluation
            if DEBUG_MODE:
                print("no time passed, not changing")
            return False
        if (isinstance(self.chosenAction, PLA) and not self.disablePLA):
            self.indexToBeModified = index
            if self.policyCondition(self.indexToBeModified):
                self.addPolicyChange()
                return True
            if DEBUG_MODE:
                print("index does not point to a part of Pol")
        if DEBUG_MODE:
            print("not a PLA or PLAs are disabled")
            if not isinstance(self.chosenAction, PLA):
                raise Exception()
        return False

    def generateParamsEtc(self):
        if not self.policyCondition(self.IP+self.get_IP_addition()-1):
            self.jumpHome()
            return False

        self.generateParameters()
        self.afterParameters()
        if DEBUG_MODE:
            print("currentInstruction =" + str(self.currentInstruction))
        return True




    @overrides
    def setAction(self):
        self.action_is_set=False
        if DEBUG_MODE: print("IP=" + str(self.IP))
        if self.looping():
            self.action_is_set=True
            return
        # step 1: generate instructions anc check for IP and to-be-modified parameters
        self.generateInstruction()
        self.afterInstruction()
        if DEBUG_MODE:
            print("chosenAction =" + str(self.chosenAction.function.__name__))
            print("args="+str(self.currentInstruction))
            print("Nj=" + str(self.N_j))
            print("IP addition"+str(self.get_IP_addition()))
        #self.wait_CSSA()
        if not self.generateParamsEtc():
            return

        self.action_is_set=True


    @overrides#(CompleteLearner)
    def setObservation(self,agent,environment):
        environment.setObservation(agent) # only useful for the SSA NEAT
        self.observation = agent.learner.observation

    @overrides
    def learn(self) :
        self.evaluate(self.t, self.R)

    @overrides
    def setReward(self, reward):
        SSAimplementor.setReward(self,reward)

    @overrides
    def reset(self) :
        pass


    @overrides
    def new_elementary_task(self):
        if self.episodic:
            self.jumpHome()
            self.random_steps=self.r_steps
            self.task_time = 0

    def setTime(self,t):
        increment=t-self.t
        if self.episodic:
            self.task_time+=increment
        CompleteLearner.setTime(self,t)

    @overrides
    def performAction(self, agent, environment):
        if DEBUG_MODE: print("Performing" + str(self.chosenAction))
        argument_list = self.currentInstruction[1:]

        if(isinstance(self.chosenAction,ExternalAction)):
            argument_list.extend([agent,environment])

        self.chosenAction.perform(argument_list)

    @overrides
    def cycle(self, agent, environment):
        self.setObservation(agent, environment)
        #violations = [(key, value) for key, value in self.c.items() if abs(value) > self.MaxInt]
        self.setAction()
        agent.learner.chosenAction = self.chosenAction  # cf. task drift (Homeostatic Pols)
        if self.action_is_set:
            self.performAction(agent, environment)
            #violations = [(key,value) for key,value in self.c.items() if abs(value) > self.MaxInt]

        self.learn()

        if DEBUG_MODE:
            print("t="+str(self.t))
            print("r="+str(self.r))
            print("R="+str(self.R))
            print("End Cycle")



# 1999 version, with Working Memory (some added options)
class SSA_with_WM(SSA):
    def __init__(self, num_program=10, num_inputs=4, wm_cells=100,actions=[], internal_actionsSSA={}, enforce_correctness=False,enhance_PLA=0,filename="",
                eval=False,predictiveSelfMod=False,real_numbers=False, additional_inputs=0, jumpHome_at_error=True, output_cells=0,maxLoop =10,
                 freezingThresh=.50,prepEvalParam=2, maxTime=10000,conservative=False,jump_at_reset=True,
                 episodic=False,separate_arguments=False):
        ####

        self.enforceCorrectness = enforce_correctness
        self.jumpHome_at_error=jumpHome_at_error
        self.enhance_PLA = enhance_PLA
        actions+=self.parseInternalActions(internal_actionsSSA)
        self.m = num_program  # number of program cells
        self.num_inputs = num_inputs
        self.wm_cells =wm_cells
        self.output_cells=output_cells
        assert wm_cells > 4 + self.num_inputs
        self.eval=eval
        self.predictiveSelfMod=predictiveSelfMod
        self.real_numbers=real_numbers
        self.monitor = [None,None]

        SSA.__init__(self,num_program,actions,{},filename,maxLoop=maxLoop,freezingThresh=freezingThresh,
                     prepEvalParam=prepEvalParam, maxTime=maxTime,conservative=conservative,
                     jump_at_reset=jump_at_reset,episodic=episodic,separate_arguments=separate_arguments) #do NOT pass internal actions  + incprobGamma


        print([str(a) for a in self.actions])
        if self.eval:self.initEvaluationPol(filename)
        # if self.predictiveSelfMod:
        #     self.predictiveMod=PredictiveModification(self.n_ops,self.m,self.minP)
        #     self.numPredictiveModifications=0
    @classmethod
    def get_diversity(cls,ssa_instances):
        policy_matrices=np.array([ssa.Pol.asMatrix() for ssa in ssa_instances])
        return np.array(policy_matrices).std()

    def parseInternalActions(self, list):
        actions=[]
        self.update_model_in_instructions = False

        for key, value in list.items():
            function = getattr(self, key)
            if key in ['incP','decP','searchP','sample','inc_mean','dec_mean']:
                for i in range(1+self.enhance_PLA):
                    actions.append(PLA(function, value))
            else:
                actions.append(Action(function, value))
                if key == "updateModel":
                    self.update_model_in_instructions = True
            if value > self.max_arguments:
                self.max_arguments=value
            print(actions[-1].function.__name__)
        return actions

    def init_alternative(self):
        self.num_register_cells = 0
        self.maxInstr = 1000
        self.RegisterStart = 0
        self.ProgramStart = self.RegisterStart + self.num_register_cells

        self.Min = self.ProgramStart - self.wm_cells
        self.Max = self.ProgramStart + self.m - 1  # we will use weak inequality
        self.OutputStart = self.Max + 1 if self.output_cells > 0 else None
        self.MaxInt = max(self.Max, abs(self.Min))  # ensure MaxInt > Max and > abs(Min)
        self.IP = self.ProgramStart  # instruction pointer (points to one of the programcells)
        self.c = {}

        for a in range(self.Min, self.Max + 1):
            if a < self.ProgramStart:
                self.c[a] = randint(-self.MaxInt, self.MaxInt)  # storage initialized to 0
            else:
                self.c[a] = 0
        for i in range(self.output_cells):
            self.c[self.OutputStart + i] = 0

        self.Pol = None
        self.maxArgs = 0
    def init_cells(self,init_random=True):
        for a in range(self.Min,self.Max+1):
            if not init_random:
                self.c[a]=0
            else:
                if a < self.ProgramStart:
                    self.c[a] = randint(-self.MaxInt,self.MaxInt)  # storage initialized to 0
                else:
                    self.c[a]=randint(0,self.n_ops-1)

    def initWM(self):


        self.num_register_cells = self.n_ops
        self.maxInstr = self.n_ops
        self.RegisterStart = 0
        self.ProgramStart = self.RegisterStart + self.num_register_cells

        self.Min = self.ProgramStart - self.wm_cells
        self.Max = self.ProgramStart + self.m - 1  # we will use weak inequality
        self.OutputStart = self.Max+1 if self.output_cells> 0 else None
        self.MaxInt = max(self.Max , abs(self.Min))   # ensure MaxInt > Max and > abs(Min)
        assert self.n_ops < self.MaxInt
        assert self.n_ops < self.wm_cells
        self.IP = self.ProgramStart  # instruction pointer (points to one of the programcells)
        self.c={}

        self.init_cells()

        for i in range(self.output_cells):
            self.c[self.OutputStart+i] = 0


    @overrides
    def initPol(self):
        if self.n_ops == 0:
            # means initialisation without arguments
            self.init_alternative()
        self.initWM()
        # if self.enhance_PLA>0:
        #     self.n_ops += self.enhance_PLA
        #     self.actions+=[PLA(self.incP,3)]*(self.enhance_PLA/2)
        #     self.actions+=[PLA(self.decP,3)]*(self.enhance_PLA/2)

        pol={} # using a map instead of list simplifies things, can use the same index as the storage c
        # for a in range(self.ProgramStart,self.Max+1):
        # 	self.Pol[a]=[1./float(self.n_ops)]*self.n_ops # the actual program cells

        for a in range(self.ProgramStart, self.Max + 1):
            pol[a]=[1./float(self.n_ops)]*self.n_ops # the actual program cells
        self.Pol = MapPolicy(pol,[StochasticProgramCell])

        self.maxArgs=0
        for i in range(self.maxInstr):
            num = self.actions[i].n_args
            if num > self.maxArgs:
                self.maxArgs = num
            # if self.enhance_PLA > 0:
            # 	for i in range(self.n_ops):
            # 		if self.actions[i].function.__name__ in ['incP', 'decP']:
            # 			self.Pol[a][i]*=self.enhance_PLA/2
            # 		C=sum(self.Pol[a])
            # 		self.Pol[a].p=[x / C for x in self.Pol[a]]
            # for i in range(self.n_ops):
            # 	if self.actions[i].function.__name__ in ['incP','decP']:
            # 		self.Pol[a][i] *=150
            # C=sum(self.Pol[a])
            # for i in range(self.n_ops):
            # 	self.Pol[a][i]/=C
        if DEBUG_MODE: print(self.Pol)





    @overrides
    def incrementIP(self,num=1):
        self.IP+=num
        self.c[self.Min+2] = self.IP
    @overrides
    def afterInstruction(self):
        self.c[self.IP] = self.currentInstruction[0]  # write the choice to the appropriate place in storage
        SSA.afterInstruction(self) # then move IP and set args, etc

    @overrides
    def generateParameters(self):
        SSA.generateParameters(self)
        j=1
        startingIP=self.IP - self.get_IP_addition()
        for i in range(startingIP,startingIP+self.N_j):
            self.c[i]=self.currentInstruction[j]
            j+=1
    # def clip(self,c): # clip the content to the bounds
    # 	np.clip(c,-self.MaxInt,+self.MaxInt)
    # def clipRead(self,):
    #
    # def clipIP(self,c):
    # 	np.clip(c,0,self.Max-4) #leave room for 3 arguments
    # def clipWriting(self,c):
    # 	np.clip()
    @overrides
    def printPolicy(self):

        self.polfile.write("\n ----------------- \n Final Input Cells: \n")
        for i in range(self.Min, self.RegisterStart):
            self.polfile.write(str(self.c[i]) + ",")
        self.polfile.write("\n --------------- \n Final Register cells: \n")
        for i in range(self.RegisterStart, self.ProgramStart):
            self.polfile.write(str(self.c[i]) + ",")
        if self.output_cells>0:
            for i in range(self.Max, self.Max+self.output_cells):
                self.polfile.write(str(self.c[i]) + ",")

        stack_size = len(self.Stack)
        if stack_size <= 150:
            self.polfile.write("\n --------------- \n Final Stack: \n")
            for i in range(stack_size):
                self.polfile.write(str(self.Stack[i]) + "\n")
        list=[entry for entry in self.Stack if isinstance(entry,StackEntries.StackEntry)]
        self.polfile.write("\n ------------------ \n Num Valid P-Modifications %d " % (len(list)-1,))
        self.polfile.write("\n ------------------ \n Num Total P-Modifications %d " % (self.stats.numPModifications,))

        if self.predictiveSelfMod:
            list = [entry for entry in self.Stack if isinstance(entry, StackEntries.StackEntryPredMod)]
            self.polfile.write("\n ------------------ \n Num Valid Predictive Modifications %d " % (len(list),))
            self.polfile.write("\n ------------------ \n Num Total Predictive Modifications %d " % (self.stats.numPredictiveModifications,))

        self.polfile.write("\n --------------- \n Final Program Cells: \n")
        self.polfile.write("\t ")
        for j in range(self.n_ops):
            self.polfile.write(str(self.actions[j].function.__name__) + ' (%d)\t' % (j,))
        self.polfile.write("\n")
        for i in range(self.ProgramStart,self.Max+1):
            self.polfile.write("Cell/IP %d \t" % (i,))
            for j in range(self.n_ops):
                self.polfile.write('%04.3f \t' % (self.Pol[i][j],))
            self.polfile.write("\n")
        self.polfile.flush()


        #self.plotMatrix(self.file)        # obj0, obj1, obj2 are created here...



    #simple IP changing operation
    def setIP(self,num):
        self.IP = num
        self.c[self.Min+2] = self.IP
    @overrides
    def jumpHome(self):
        self.setIP(self.ProgramStart)

    def jump(self,a1):
        if self.enforceCorrectness:
            ca1= self.IPcondition(self.c[a1])
        else:
            ca1=int(self.c[a1])
            if not self.relaxedIPcondition(ca1):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.setIP(ca1)

    def loop(self, a1, a2) : #loop following action until cca1 input <= cca2 input
        print("currentIP: %d, maxIP: %d"%(self.IP,self.Max))
        if not self.policyCondition(self.IP + self.get_IP_addition()):
            if self.jumpHome_at_error: self.jumpHome()
            return
        ca1=int(narrowing_conversion(self.c[a1],(-self.MaxInt,self.MaxInt),(self.Min,self.num_inputs)))
        ca2=int(narrowing_conversion(self.c[a2],(-self.MaxInt,self.MaxInt),(self.Min,self.num_inputs)))
        if self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2):
            print("start loop")
            print("monitor cells %d and %d"%(ca1,ca2))
            print("with current values %f and %f"%(self.c[ca1],self.c[ca2]))
            self.monitor=[ca1,ca2]
            self.loopTime = self.maxLoop
        else:
            if self.jumpHome_at_error: self.jumpHome()
    def looping(self):
        if self.monitor != [None,None] and isinstance(self.chosenAction.function,ExternalAction) :
            stopLooping = self.loopTime == 0 or self.c[self.monitor[0]] <= self.c[self.monitor[1]]
            if stopLooping:
                self.monitor=[None,None]
                return False
            print("Looping action %s" % self.chosenAction.function.__name__)
            print("Looping instruction " + str(self.currentInstruction))
            self.loopTime -=1
            return True
        self.loopTime = 0
        return False

    def freeze(self,a1, a2):
        ca1 = int(self.c[a1])

        if not self.relaxedreadcondition(ca1):
            if self.jumpHome_at_error: self.jumpHome()
            return
        cell = int(self.c[ca1])
        if not self.relaxedIPcondition(cell):
            if self.jumpHome_at_error: self.jumpHome()
            return

        progCycles = a2
        if max(self.Pol[cell].p) < self.freezingThresh:
            if DEBUG_MODE: print("freezing thresh %f not met"%(self.freezingThresh,))
            return

        self.Pol[cell].timeUntilUnfrozen = progCycles + 1
        if DEBUG_MODE:
            print("freezing ")
            print("Cell on IP=%d  for %d progCycles" %(cell,progCycles))
            print("with maxInstruction="+str(self.actions[np.argmax(self.Pol[cell].p)]))
        dummyIndex=self.Pol[cell].frozenInstruction()
        dummyAction = self.actions[dummyIndex]
        if  dummyAction == self.freeze: #cannot freeze if instruction may be the freeze action
            if self.jumpHome_at_error: self.jumpHome()
            return


    def getP(self,a1,a2,a3): #introspection: get the probability of Pol[c[a1]][c[a2]] in useful size and store it
         ca1 = int(self.c[a1])
         ca2 = int(self.c[a2])
         ca3 = int(self.c[a3])
         if self.enforceCorrectness:
             ca1 = self.polcondition(ca1)
             ca2 = self.opcondition(ca2)
             ca3 = self.writecondition(ca3)
         else:
             if not (self.relaxedpolcondition(ca1) and self.relaxedopcondition(ca2) and self.relaxedwritecondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return

         temp = round(self.MaxInt*self.Pol[ca1][ca2])
         self.c[ca3] = int(temp)

    # self-modification
    def incP(self, a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1 = self.polcondition(ca1)
            ca2 = self.opcondition(ca2)
            ca3 = self.readcondition(ca3)
            cca3 = self.c[ca3]
            cca3 = self.cca3ProbCondition(cca3)
        else:
            if not (self.relaxedpolcondition(ca1) and self.relaxedopcondition(ca2) and self.relaxedreadcondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                self.stats.result=PLAResult.error
                return
            cca3 = self.c[ca3]
            if not (cca3>=1 and cca3<= 99):
                #print("cca3:"+str(cca3))
                if self.jumpHome_at_error: self.jumpHome()
                self.stats.result=PLAResult.error
                return
        prepared,oldPol = self.preparePolicyChange(ca1)
        if not prepared:
            self.stats.result = PLAResult.not_prepared
            return


        self.Pol[self.indexToBeModified][ca2] = 1 - .01*cca3*(1-self.Pol[self.indexToBeModified][ca2]) #increase
        for i in range(self.n_ops):
            if i != ca2:
                self.Pol[self.indexToBeModified][i]= .01*cca3*self.Pol[self.indexToBeModified][i] #decrease
            if (self.Pol[self.indexToBeModified][i] < self.minP):
                self.Pol[self.indexToBeModified] = oldPol
                self.removePolicyChange()
                self.stats.result = PLAResult.removed_change
                return
        if self.eval and not self.evalPol.accept(self.Pol,self.t,self.R,self.Stack):
            self.Pol[self.indexToBeModified] = oldPol
            self.removePolicyChange()
            self.stats.result = PLAResult.removed_change
            return
        self.stats.changeSize+=cca3*.01
        # due to possible precision errors, we divide by the sum
        C=sum(self.Pol[self.indexToBeModified])
        self.Pol[self.indexToBeModified].p = [x/C for x in self.Pol[self.indexToBeModified]]
        if self.eval:
            #self.evalPol.addEntry(self.Pol[self.indexToBeModified].p,self.t,self.R,self.indexToBeModified)
            self.evalPol.addEntry(self.Pol,self.t,self.R)
        if DEBUG_MODE:
            print('incP --->' + str(self.Pol[self.indexToBeModified]))
        # success !
        self.success_result()
    def incThresh(self,a1):
        current=self.evalPol.threshold
        self.evalPol.setThresh(current+.010*a1/float(self.n_ops))
    def decThresh(self,a1):
        current=self.evalPol.threshold
        self.evalPol.setThresh(current-.010*a1/float(self.n_ops))
    def updateModel(self):
        self.evalPol.updateModel(self.t,self.R)
    def decP(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1 = self.polcondition(ca1)
            ca2 = self.opcondition(ca2)
            ca3 = self.readcondition(ca3)
            cca3 = self.c[ca3]
            cca3 = self.cca3ProbCondition(cca3)
        else:
            if not (self.relaxedpolcondition(ca1) and self.relaxedopcondition(ca2) and self.relaxedreadcondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                self.stats.result = PLAResult.error
                return
            cca3 = self.c[ca3]
            if not (cca3>=1 and cca3<= 99):
                if self.jumpHome_at_error: self.jumpHome()
                self.stats.result = PLAResult.error
                return
        prepared, oldPol = self.preparePolicyChange(ca1)
        if not prepared:
            self.stats.result = PLAResult.not_prepared
            return

        previousProb = self.Pol[self.indexToBeModified][ca2]
        self.Pol[self.indexToBeModified][ca2] = .01 * cca3 * previousProb  # shrink
        for i in range(self.n_ops):
            if i != ca2:
                self.Pol[self.indexToBeModified][i] = self.Pol[self.indexToBeModified][i] * (
                1 - .01 * cca3 * previousProb) / (1 - previousProb)  # increase
            if (self.Pol[self.indexToBeModified][i] < self.minP):
                self.Pol[self.indexToBeModified] = oldPol
                self.removePolicyChange()
                self.stats.result = PLAResult.removed_change
                return
        if self.eval and not self.evalPol.accept(self.Pol,self.t,self.R,self.Stack):
            self.Pol[self.indexToBeModified] = oldPol
            self.removePolicyChange()
            self.stats.result = PLAResult.removed_change
            return
        self.stats.changeSize += cca3 * .01
        # due to possible precision errors, we divide by the sum
        C = sum(self.Pol[self.indexToBeModified])
        violations=[p for p in self.Pol[self.indexToBeModified] if p >= 1]
        self.Pol[self.indexToBeModified].p = [x / C for x in self.Pol[self.indexToBeModified]]
        if self.eval:
            #self.evalPol.addEntry(self.Pol[self.indexToBeModified].p,self.t,self.R,self.indexToBeModified)
            self.evalPol.addEntry(self.Pol,self.t,self.R)
        if DEBUG_MODE:
            print('decP --->' + str(self.Pol[self.indexToBeModified]))
        self.success_result()
    def addPredictiveModification(self,oldMeans):
        self.polChanged = True
        # reward should be modified by the environment (preferably in parallel)
        address = self.indexToBeModified  # address of the modified program cell
        #oldMeans = copy.copy(self.predictiveMod.means[self.indexToBeModified])  # prob distrutions before the modification (we use multiple, in case we want at some point PLAs that change multiple pol_i's at a time)
        first = self.getPointerToFirstModification(len(self.Stack))  # pointer to the index of the first modification of this SMS on the stack (take into account the new entry so size-1+1)
        newEntry = StackEntries.StackEntryPredMod(t=self.t, R=self.R, oldMeans=oldMeans, address=address, first=first)

        self.Stack.push(newEntry)

        if DEBUG_MODE:
                print("New Stack Entry: " + str(newEntry))
                self.writeStack()

    def removePredictiveModification(self):
        self.polChanged=False
        if self.Stack.top().first==self.Stack.sp:
            self.beginOfSMS=True
        self.Stack.pop()
    def searchP(self, a1,a2,a3=None):
        ca1 = int(self.c[a1])
        ca3 = int(self.c[a3]) if a3 is not None else None
        if self.enforceCorrectness:
            ca1 = self.polcondition(ca1)
        else:
            if not (self.relaxedpolcondition(ca1)):
                if self.jumpHome_at_error: self.jumpHome()
                return
            if ca3 is not None:
                if not self.relaxedopcondition(ca3):
                    if self.jumpHome_at_error: self.jumpHome()
                    return
        prepared = self.preparePolicyChange(ca1)
        if not prepared:
            return
        max = self.evalPol.threshold
        maxpol=None
        #print('pol =' + str(self.Pol[self.indexToBeModified]))
        pol=copy.copy(self.Pol[self.indexToBeModified].p)
        # for a given IP address, generate modifications and choose the one with maximal predicted speedup
        for i in range(self.evalPol.search_iterations):

            new_pol,speedup = self.evalPol.generate_modification(copy.copy(pol),ca1,a2,ca3)
            if speedup > max:
                max = speedup
                maxpol = new_pol
        if maxpol is not None:
            self.Pol[ca1].p=maxpol
            if self.evalPol:
                self.addedToBuffer = self.evalPol.addEntry( new_pol,self.t, self.R,
                                                           self.indexToBeModified)
        else:
            self.removePolicyChange()

        #print(self.Pol[ca1].p)

        # print("maxP"+str(max))
        # print('searchP --->' + str(self.Pol[self.indexToBeModified]))
    def update_evalP(self):

        self.evalPol.updateModel(self.t,self.R)

    def inc_mean(self, a1,a2):
        prog = int(self.c[a1])
        if not self.polcondition(prog): return
        prepared = self.preparePredictiveModification(prog)
        if not prepared:
            return

        prog_ind = prog - self.ProgramStart
        oldMeans=self.predictiveMod.inc_mean(prog_ind,a2)
        if not oldMeans:
            # didn't make the change
            return
        self.addPredictiveModification(oldMeans)

    def dec_mean(self, a1,a2):
        prog = int(self.c[a1])
        if not self.polcondition(prog): return
        prepared = self.preparePredictiveModification(prog)
        if not prepared:
            return
        prog_ind = prog - self.ProgramStart
        oldMeans = self.predictiveMod.dec_mean(prog_ind, a2)
        if not oldMeans:
            # didn't make the change
            return
        self.addPredictiveModification(oldMeans)

    def sample(self, a1,a2):
        prog=int(self.c[a1])
        if not self.polcondition(prog): return
        prepared = self.preparePolicyChange(prog)
        if not prepared:
            return
        prog_ind=prog-self.ProgramStart
        sampled_p=self.predictiveMod.sample(prog_ind,a2)
        if not sampled_p:
            self.removePolicyChange()
            return
        self.Pol[self.indexToBeModified].p=sampled_p


    #syntactical correctness (see also polcondition in SSA)
    # def IPcondition(self,ca1):
    # 	return ca1 >= self.ProgramStart and ca1 < self.Max - 3
    # def opcondition(self,ca1):
    # 	return ca1 >=0 and ca1< self.n_ops
    #
    # def readcondition(self,ca1):
    # 	return ca1 >= self.Min and ca1 <= self.Max
    # def writecondition(self,ca3):
    # 	return ca3 >= self.Min and ca3 < self.ProgramStart
    #
    # def clip_division(self,ca1,ca2):
    # 	if ca2==0:
    # 		return np.sign(ca1)*self.MaxInt
    # 	return ca1//ca2
    # def clip_remainder(self,ca1,ca2):
    # 	if ca2==0:
    # 		return ca1
    # 	return ca1%ca2
    # def clip(self,ca1):
    # 	return np.clip(ca1,-self.MaxInt,self.MaxInt)

    # syntactical correctness (see also polcondition in SSA)



    def cca3ProbCondition(self,cca3):
        return narrowing_conversion(cca3,(-self.MaxInt,+self.MaxInt),(1,99))
    def _cca3condition(self,ca1):
        return ca1 >= 1 and ca1<= 99
    def IPcondition(self,ca1):
        return narrowing_conversion(ca1,(-self.MaxInt,+self.MaxInt),(self.ProgramStart,self.Max - self.maxArgs))
    def relaxedIPcondition(self,ca1):
        return ca1 >= self.ProgramStart and ca1<= self.Max-self.maxArgs
    def opcondition(self,ca1):
        return narrowing_conversion(ca1, (-self.MaxInt, +self.MaxInt), (0, self.n_ops-1))
    def relaxedopcondition(self,ca1):
        return ca1 >= 0 and ca1<= self.n_ops-1
    def readcondition(self,ca1):
        return narrowing_conversion(ca1, (-self.MaxInt, +self.MaxInt), (self.Min, self.Max))
    def relaxedreadcondition(self,ca1):
        return ca1 >= self.Min and ca1<= self.Max
    def writecondition(self,ca3):
        return narrowing_conversion(ca3, (-self.MaxInt, +self.MaxInt), (self.Min, self.ProgramStart - 1))
    def relaxedwritecondition(self,ca1):
        return ca1 >= self.Min and ca1<= self.ProgramStart-1
    def polcondition(self,ca1):
        return narrowing_conversion(ca1, (-self.MaxInt, +self.MaxInt), (self.ProgramStart, self.Max))
    def relaxedpolcondition(self,ca1):
        return ca1 >= self.ProgramStart and ca1<= self.Max
    def clip_division(self,ca1,ca2):
        """
        both params in range --> ca1//ca2 will always be in range (no need to clip)
        exception is ca2==0 --> set to maxint
        :param ca1: in range -Maxint,Maxint
        :param ca2: in range -Maxint,Maxint
        :return:
        """
        if ca2==0:
            return np.sign(ca1)*self.MaxInt
        if self.real_numbers:
            return ca1/float(ca2)
        return ca1//ca2
    def clip_remainder(self,ca1,ca2):
        """
        both params in range --> ca1%ca2 will always be in range (no need to clip)
        (experiments did clip but makes no difference essentially)
        exception is ca2==0 --> set to ca1
        :param ca1: in range -Maxint,Maxint
        :param ca2: in range -Maxint,Maxint
        :return:
        """
        if ca2==0:
            return ca1
        #return ca1%ca2
        return self.clip(ca1%ca2)
    def clip(self,ca1):
        return np.clip(ca1,-self.MaxInt,self.MaxInt)

    # changing IP based on cell contents
    def jump_lower2(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:

            ca1 = self.readcondition(ca1)
            ca2 = self.readcondition(ca2)
            ca3 = self.readcondition(ca3)
            #cca3 = self.c[ca3]
            #cca3 = self.IPcondition(cca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedreadcondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
            #cca3 = int(self.c[ca3])
            if not self.relaxedIPcondition(ca3):
                if self.jumpHome_at_error: self.jumpHome()
                return
        if self.c[ca1] < self.c[ca2]:

            self.setIP(ca3)

    def jump_equal2(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1=self.readcondition(ca1)
            ca2=self.readcondition(ca2)
            ca3=self.readcondition(ca3)
            #cca3 = self.c[ca3]
            #cca3 = self.IPcondition(cca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedreadcondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
            #cca3 = int(self.c[ca3])
            # if not self.relaxedIPcondition(cca3):
            #     if self.jumpHome_at_error: self.jumpHome()
            #     return
        if self.c[ca1] == int(self.c[ca2]):
            self.setIP(ca3)

    # changing IP based on cell contents
    def jump_lower(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:

            ca1 = self.readcondition(ca1)
            ca2 = self.readcondition(ca2)
            ca3 = self.readcondition(ca3)
            cca3 = self.c[ca3]
            cca3 = self.IPcondition(cca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedreadcondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
            cca3 = int(self.c[ca3])
            if not self.relaxedIPcondition(cca3):
                if self.jumpHome_at_error: self.jumpHome()
                return
        if self.c[ca1] < self.c[ca2]:

            self.setIP(cca3)

    def jump_equal(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1=self.readcondition(ca1)
            ca2=self.readcondition(ca2)
            ca3=self.readcondition(ca3)
            cca3 = self.c[ca3]
            cca3 = self.IPcondition(cca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedreadcondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
            cca3 = int(self.c[ca3])
            if not self.relaxedIPcondition(cca3):
                if self.jumpHome_at_error: self.jumpHome()
                return
        if self.c[ca1] == int(self.c[ca2]):
            self.setIP(cca3)
    # changing cell contents based on cell contents
    def add(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1=self.readcondition(ca1)
            ca2=self.readcondition(ca2)
            ca3=self.writecondition(ca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedwritecondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca3]=self.clip(self.c[ca1] + self.c[ca2])
    def mult(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1 = self.readcondition(ca1)
            ca2 = self.readcondition(ca2)
            ca3 = self.writecondition(ca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedwritecondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca3]=self.clip(self.c[ca1] * self.c[ca2])
    def sub(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1 = self.readcondition(ca1)
            ca2 = self.readcondition(ca2)
            ca3 = self.writecondition(ca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedwritecondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca3]=self.clip(self.c[ca1] - self.c[ca2])
    def div(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1 = self.readcondition(ca1)
            ca2 = self.readcondition(ca2)
            ca3 = self.writecondition(ca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedwritecondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca3]=self.clip_division(self.c[ca1], self.c[ca2])
    def rem(self,a1,a2,a3):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        ca3 = int(self.c[a3])
        if self.enforceCorrectness:
            ca1 = self.readcondition(ca1)
            ca2 = self.readcondition(ca2)
            ca3 = self.writecondition(ca3)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedreadcondition(ca2) and self.relaxedwritecondition(ca3)):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca3]=self.clip_remainder(self.c[ca1],self.c[ca2])
    def inc(self,a1):
        ca1 = int(self.c[a1])
        if self.enforceCorrectness:
            ca1 = self.writecondition(ca1)
        else:
            if not self.relaxedwritecondition(ca1):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca1] = self.clip(self.c[ca1] +  1)
    def dec(self,a1):
        ca1 = int(self.c[a1])
        if self.enforceCorrectness:
            ca1 = self.writecondition(ca1)
        else:
            if not self.relaxedwritecondition(ca1):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca1] = self.clip(self.c[ca1] - 1)
    def mov(self,a1,a2):
        ca1 = int(self.c[a1])
        ca2 = int(self.c[a2])
        if self.enforceCorrectness:
            ca1 = self.readcondition(ca1)
            ca2 = self.writecondition(ca2)
        else:
            if not (self.relaxedreadcondition(ca1) and self.relaxedwritecondition(ca2)):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[ca2]=int(self.c[ca1])
    def init(self,a1,a2):
        if self.enforceCorrectness:
            a1=self.writecondition(a1)
        else:
            a1 = a1 - self.ProgramStart - 2 # as in paper
            if not self.relaxedwritecondition(a1):
                if self.jumpHome_at_error: self.jumpHome()
                return
        self.c[a1] = int(a2)
    def conditional_jump(self,a1,a2,a3):
        """ active perception: read an input cell, if larger than """
        input_cell=narrowing_conversion(a1,(0,self.n_ops-1),(0,self.num_inputs+4-1))
        comparison=conversion(a2,(0,self.n_ops-1),(-self.MaxInt,self.MaxInt))
        jump_size=conversion(a3,(0,self.n_ops-1),(0,self.m/2))
        input = self.c[self.Min + input_cell]
        if input > comparison:
            new = self.IP+jump_size
            self.setIP(new)
            if self.IP > self.Max:
                self.jumpHome()
    def set_wm_internal_vars(self):
        if self.episodic:
            t=self.task_time
        else:
            t=self.t
        self.c[self.Min] = min(self.Stack.sp, self.MaxInt)  # sp
        self.c[self.Min + 1] = int(t) % self.MaxInt
        self.c[self.Min + 2] = self.IP

    def set_wm_external_vars(self,observation):
        """
        set the reward and observation


        :param observation: sensory variables in range (-1.,1.)
        :return:
        """
        self.c[self.Min + 3] = int(self.r)
        # environment inputs
        # update only if new external action taken, useful to manipulate and use these input values
        for i in range(self.num_inputs):
            self.c[self.Min + 4 + i] = int(observation[i] * self.MaxInt)

        if DEBUG_MODE:
            print([self.c[i] for i in range(self.Min,self.Min+4+self.num_inputs)])
    @overrides  # (CompleteLearner)
    def setObservation(self, agent, environment):

        # in static environments it sufffices to change input cells only when external action is taken
        # in dynamic environments, need to track changes all the time



        self.set_wm_internal_vars()
        if not environment.observation_set:
            environment.setObservation(agent)
            self.observation = agent.learner.observation # cf task drift
            self.set_wm_external_vars(agent.learner.observation)


    @overrides
    def learn(self):
        self.evaluate(self.t, self.R)

        if self.eval and not self.update_model_in_instructions:
            if self.t % self.evalPol.update_freq == 0:
                self.evalPol.updateModel(self.t, self.R)

    @overrides
    def setReward(self, reward):
        SSAimplementor.setReward(self, reward)

    @overrides
    def performAction(self, agent, environment):
        argument_list = self.currentInstruction[1:]

        if (isinstance(self.chosenAction, ExternalAction)):
            argument_list.extend([agent, environment])

        self.chosenAction.perform(argument_list)  # record result for statistics
        agent.learner.chosenAction = self.chosenAction  # cf. homeostatic pols

    @overrides
    def getOutputData(self):
        out=[]
        for i in range(self.output_cells):
            out.append(self.c[self.OutputStart+i])
        return out
    # def restore(self,entry):
    #     if isinstance(entry,StackEntries.StackEntry):
    #         if (entry.oldP != None):
    #             self.Pol[entry.address] = entry.oldP  # to restore old policy
    #         else:
    #             raise Exception()
    #     elif isinstance(entry,StackEntries.StackEntryPredMod):
    #         if (entry.oldMeans != None):
    #             self.predictiveMod.means[entry.address-self.ProgramStart] = entry.oldMeans  # to restore old policy
    #         else:
    #             raise Exception()
    #     else:
    #         raise Exception()
    #
    # @overrides
    # def popAndRestore(self, sp):  # evaluate if condition is met
    #     entry = self.Stack[sp]
    #     self.restore(entry)
    #     if DEBUG_MODE:
    #         print("popping stack entry " + str(self.Stack[sp]))
    #     del self.Stack[sp]  # pop final element off the stack
# 1997 version, with Working Memory, single index
class SSA_SingleIndexWM(SSA):
        def __init__(self, num_program, num_inputs, wm_cells, actions, internal_actionsSSA,
                     enhance_PLA, filename, skip_max_params=False,simple_observation=True, output_cells=0,freezingThresh=.25,prepEvalParam=2, maxTime=10000):
            ####

            print(internal_actionsSSA)
            self.maxParams=0

            for key, value in internal_actionsSSA.items():
                function = getattr(self, key)
                if key == 'incProb' or key == 'decProb':
                    actions.append(PLA(function, value))
                else:
                    actions.append(Action(function, value))
                if value > self.maxParams:
                    self.maxParams=value

                print(actions[-1].function.__name__)
            self.simple_observation = simple_observation
            self.skip_max_params = skip_max_params
            self.m = num_program  # number of program cells
            self.num_inputs = num_inputs
            self.wm_cells = wm_cells
            self.MaxNum = 10**9
            self.output_cells = output_cells
            self.enhance_PLA = enhance_PLA
            SSA.__init__(self, num_program, actions, {}, filename, freezingThresh=freezingThresh,
                         prepEvalParam=prepEvalParam,maxTime=maxTime)  # do NOT pass internal actions  + incprobGamma

        @overrides
        def initPol(self):

            if self.enhance_PLA > 0:

                self.actions += [PLA(self.incProb, 3)] * (self.enhance_PLA)
                self.n_ops += self.enhance_PLA
                self.actions += [PLA(self.decProb, 3)] * (self.enhance_PLA)
                self.n_ops += self.enhance_PLA
            assert self.wm_cells % self.n_ops == 0  # TODO: implementation that does not require this ?
            self.maxInstr = self.wm_cells # e.g. n_ops=2 --> 0 -> 0 1 -> 1 2->0 3->1, helps us to have more wm cells
            self.k = self.maxInstr/self.n_ops

            assert self.wm_cells > 4 + self.num_inputs
            assert (self.maxInstr + 1) * (self.maxInstr-1) >= self.m  # required for narrowing conversion at incProb
            self.ProgramStart = 0
            # note: this is Max of the program
            self.Max = self.ProgramStart + self.m - 1  # we will use weak inequality
            # note: this is Min of the working area
            self.Min = 0
            self.MaxInt = 50000 #used for setting observation (stack size and time) and for divvision by zero
            self.IP = self.ProgramStart  # instruction pointer (points to one of the programcells)
            self.c = np.zeros((self.wm_cells+self.output_cells,))

            for a in range(self.Min, self.Min + self.wm_cells):
                self.c[a] = -10+random.random()*10  # storage initialized to 0
            if self.output_cells > 0:
                self.OutputStart = self.wm_cells - self.output_cells


            self.Pol = [] # using a map instead of list simplifies things, can use the same index as the storage c
            for a in range(self.ProgramStart, self.Max + 1):
                self.Pol.append( [1. / float(self.maxInstr)] * self.maxInstr ) # the actual program cells
            # if self.enhance_PLA > 0:
            # 	for i in range(self.n_ops):
            # 		if self.actions[i].function.__name__ in ['incP', 'decP']:
            # 			self.Pol[a][i]*=self.enhance_PLA/2
            # 		C=sum(self.Pol[a])
            # 		self.Pol[a]=[x / C for x in self.Pol[a]]
            # for i in range(self.n_ops):
            # 	if self.actions[i].function.__name__ in ['incP','decP']:
            # 		self.Pol[a][i] *=150
            # C=sum(self.Pol[a])
            # for i in range(self.n_ops):
            # 	self.Pol[a][i]/=C
            if DEBUG_MODE: print(self.Pol)
        def isIntegerHoldingAddres(self,address):
            return address <= 4 # first 4 indicate variables that may only be integers (IP,t,min(len(stack),MaxInt),
        @overrides
        def incrementIP(self,num=1):
            self.IP += num
            if not self.simple_observation:
                self.c[self.Min] = self.IP

        @overrides
        def afterInstruction(self):
            SSA.afterInstruction(self)  # then move IP and set args, etc



        # def clip(self,c): # clip the content to the bounds
        # 	np.clip(c,-self.MaxInt,+self.MaxInt)
        # def clipRead(self,):
        #
        # def clipIP(self,c):
        # 	np.clip(c,0,self.Max-4) #leave room for 3 arguments
        # def clipWriting(self,c):
        # 	np.clip()
        @overrides
        def printPolicy(self):

            self.polfile.write("\n ----------------- \n Final Working Cells: \n")
            for i in range(self.Min, self.OutputStart):
                self.polfile.write(str(self.c[i]) + ",")

            if self.output_cells > 0:
                self.polfile.write("\n ----------------- \n Final Output Cells: \n")
                for i in range(self.OutputStart, self.OutputStart + self.output_cells):
                    self.polfile.write(str(self.c[i]) + ",")
            self.polfile.write("\n --------------- \n Final Stack: \n")
            stack_size = len(self.Stack)
            if stack_size < 150:
                self.polfile.write("Final Stack")
                for i in range(stack_size):
                    self.polfile.write(str(self.Stack[i]) + "\n")
            list = [entry for entry in self.Stack if isinstance(entry, StackEntries.StackEntry)]
            self.polfile.write("\n ------------------ \n Num Valid P-Modifications %d " % (len(list) - 1,))
            self.polfile.write("\n ------------------ \n Num Total P-Modifications %d " % (self.stats.numPModifications,))
            self.polfile.write("\n --------------- \n Final Program Cells: \n")
            self.polfile.write("\t ")
            for j in range(self.maxInstr):
                self.polfile.write(str(self.actions[j//self.k].function.__name__) + ' (%d)\t' % (j,))
            self.polfile.write("\n")
            for i in range(self.ProgramStart, self.Max + 1):
                self.polfile.write("Cell/IP %d \t" % (i,))
                for j in range(self.maxInstr):
                    self.polfile.write('%04.3f \t' % (self.Pol[i][j],))
                self.polfile.write("\n")

            self.polfile.flush()
            # obj0, obj1, obj2 are created here...
            # obj0, obj1, obj2 are created here...

        # simple IP changing operation

        def clip(self,num):
            return max(-self.MaxNum,min(num,self.MaxNum))
        @overrides
        def jumpHome(self):
            self.setIP(self.ProgramStart)


        @overrides
        def getOutputData(self):
            return self.c[self.OutputStart:]

        def jump(self, a1):
            self.setIP(a1//self.k)

        # changing cell contents based on cell contents
        def add(self, a1, a2, a3):
            self.c[a3] = self.clip(self.c[a1] + self.c[a2])

        def mult(self, a1, a2, a3):
            self.c[a3] = self.clip(self.c[a1] * self.c[a2])

        def sub(self, a1, a2, a3):
            self.c[a3] = self.clip(self.c[a1] - self.c[a2])

        def div(self, a1, a2, a3):
            if self.c[a2] == 0:
                self.c[a3] = np.sign(self.c[a1])*self.MaxNum
                return
            self.c[a3] = self.clip(self.c[a1] / float(self.c[a2]))


        def mov(self, a1, a2):
            self.c[a2] = self.c[a1]



        @overrides  # (CompleteLearner)
        def setObservation(self, agent, environment):

            # in static environments it sufffices to change input cells only when external action is taken
            # in dynamic environments, need to track changes all the time
            oldO = agent.learner.observation
            environment.setObservation(agent)
            self.observation = agent.learner.observation
            if self.simple_observation:
                self.c[self.Min : self.Min + self.num_inputs] = agent.learner.observation
            else:

                # own variables
                self.c[self.Min] = self.IP

                self.c[self.Min + 1] = self.t % self.MaxInt
                self.c[self.Min + 2] = int(self.r)
                # environment inputs
                if agent.learner.observation != oldO:
                    # update only if new observation is seen
                    self.c[self.Min+3 : self.Min + 3 + self.num_inputs] = agent.learner.observation

                self.c[self.Min + 1] = min(self.Stack.sp, self.MaxInt)  # sp


        @overrides
        def learn(self):
            self.evaluate(self.t, self.R)




        @overrides
        def setReward(self, reward):
            SSAimplementor.setReward(self, reward)

        @overrides
        def instructionToAction(self):
            index=self.currentInstruction[0]//self.k
            return self.actions[index]
        @overrides
        def generateParamsEtc(self):
            if not self.policyCondition(self.IP + self.get_IP_addition() - 1):
                self.jumpHome()
                return False

            self.generateParameters()

            if DEBUG_MODE:
                print("currentInstruction =" + str(self.currentInstruction))
            return True

        @overrides
        def setAction(self):
            self.action_is_set=False
            if self.looping():
                self.action_is_set=True
                return
            if DEBUG_MODE: print("IP=" + str(self.IP - self.ProgramStart))
            # step 1: generate instructions anc check for IP and to-be-modified parameters
            self.generateInstruction()
            self.afterInstruction()
            if DEBUG_MODE:
                print("chosenAction =" + str(self.chosenAction.function.__name__))

            if not self.generateParamsEtc():
                return

            self.action_is_set = True

        @overrides
        def performAction(self, agent, environment):
            argument_list = self.currentInstruction[1:]

            if (isinstance(self.chosenAction, ExternalAction)):
                argument_list.extend([agent, environment])

            self.chosenAction.perform(argument_list)





## SSA RNN:
## the neural network strategy described by
#  Schmidhuber, J. H. (1999).
#  A general method for incremental self-improvement a
# nd multi-agent learning. In Evolutionary Computation: Theory and Applications.

# each agent modifies a weight w=(i,j) has its own stack of previous self-modifications
class SSA_RNN(SSAimplementor):
    def __init__(self,n_in,n_out,n_hidden, actions,filename,prepEvalParam,maxTime):

        SSAimplementor.__init__(actions,filename,prepEvalParam,maxTime)
        self.n_in = n_in
        self.n_out=n_out
        self.n_hidden=n_hidden
        self.n_units=self.n_in +self.n_out + self.n_hidden
        self.initPol()
    def setIP(self, instr):
        if (instr > self.IP):
            #forward pass
            self.forward()
        else:
            #backward pass
            self.backward()
        self.IP = instr



    @overrides
    def initPol(self):
        # here Pol is the weight matrix where each neuron may be connected to each other
        self.Pol = np.zeros((self.n_units, self.n_units))
        # activations of the units
        self.o = np.zeros(self.n_units)
        for i in range(self.n_units):
            for j in range(self.n_units):
                self.Pol[i][j] = random.uniform(-2,2)
            self.o[i]=0.0
    @overrides
    def initStack(self):
        self.Stacks=[]
        for i in self.n_units:
            self.Stacks.append()
    @overrides
    def evaluate(self, time, R):  # evaluation done FOR EACH WEIGHT !
        if self.evaluationcondition():  # checkpoint reached : start evaluation not

            if DEBUG_MODE:
                print("CALL_SSA : evaluation condition met. ")
                print("NumR = " + str(self.numR))
                print("disablePLA = " + str(self.disablePLA))
                print("Stack before evaluation: " + self.printStack())
            # pop the stack until SSC satisfied
            self.popUntilSSC(time, R, self.Stack)

            # reset flow variables
            self.resetFlowVariables()



            if DEBUG_MODE:
                print("Stack after evaluation: " + self.printStack())

    def restore(self,entry):
        if (entry.oldP is not None):
            self.Pol[self.current_i][self.current_j] = entry.oldP  # to restore old policy
        else:
            raise Exception()





