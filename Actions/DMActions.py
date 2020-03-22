class Action(object):
    def __init__(self,string):
        self.string = string
class ExternalAction(Action): # e.g. move forward
    def __init__(self,string, vector):
        Action.__init__(self,string, vector)
        self.vector = vector
class LearningAction(Action): # modify the policy
    def __init__(self,string):
        Action.__init__(self,string)
class InternalAction(Action): # all internal actions that do not modify the policy e.g. observe something or change Instruction Pointer
    def __init__(self,string):
        Action.__init__(self,string)

DM_LAB_ACTIONS = [  ExternalAction('look_left',(-20, 0, 0, 0, 0, 0, 0)),
                    ExternalAction('look_right',(20, 0, 0, 0, 0, 0, 0)),
                    ExternalAction('look_up',(0, 10, 0, 0, 0, 0, 0)),
                    ExternalAction('look_down',(0, 10, 0, 0, 0, 0, 0)),
                    ExternalAction('strafe_left',(0,0,-1, 0, 0, 0, 0)),
                    ExternalAction('strafe_right',(0, 0, 1, 0, 0, 0, 0)),
                    ExternalAction('forward'),(0, 0, 0, -1, 0, 0, 0),
                    ExternalAction('backward',(0, 0, 0,1, 0, 0)),
                    ExternalAction('fire',(0, 0, 0, 0, 0, 1, 0)),
                    ExternalAction('jump',(0, 0, 0, 0, 0, 0, 1)),
                    ExternalAction('crouch',(0, 0, 0, 0, 0, 0, 1))]
DM_LAB_ACTIVE_PERCEPTION = [InternalAction('observe')] #only visual

STANDARD_SSA_INTERNAL_ACTIONS = [

                        InternalAction('prepEval'),
                        InternalAction('setIP'),


                         ]
SSA_DL_INTERNAL_ACTIONS=[InternalAction('output'), #output --> connect layer[IP] to the output layer and output the action
                        InternalAction('storeWM'), #store data in working memory (this can be sensory input, or output from any layer !)
                         InternalAction('fetchWM'), #get data from the working memory
                         InternalAction('fetchWM'), #get data from the working memory
                         InternalAction('dropout'), # switch off some units in the current layer
                         InternalAction('reassign') #switch on again
                         ]
SSA_LEARNING_ACTIONS = [LearningAction('incProb'),
                        LearningAction('incEvolutionParam')]
def get_SSA_actions(DL=True,active_perception=False): # Learning actions
   actions=[]
   actions = DM_LAB_ACTIONS
   if(active_perception):
       actions+=DM_LAB_ACTIVE_PERCEPTION
   actions += STANDARD_SSA_INTERNAL_ACTIONS
   if(DL):
       actions+=SSA_DL_INTERNAL_ACTIONS
   else:
       raise NotImplementedError
   actions += SSA_LEARNING_ACTIONS


def isPLA(action):
    return isinstance(action,LearningAction)
def executeAction(agent,):
    if(isinstance(agent.method.chosenAction,ExternalAction)):
        agent.action=agent.method.chosenAction.vector

    else:
        SSA_action(agent.method)
def SSA_action(ssa):
    switch={
        'prepEval': ssa.prepareEvaluation(ssa.currentInstruction[3]),
        'setIP': ssa.setIP(ssa.currentInstruction[1]),
        'output': ssa.output(),
        'forwardPass': ssa.forwardPass(),
        'backwardPass': ssa.backwardPass(),
        'incProb': ssa.incProb(),
        'incEvolutionParam': ssa.incEvolutionParam(),
    }
    func = switch.get(ssa.chosenAction)
    return func()

