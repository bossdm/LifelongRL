from Configs.InstructionSets import *

SSA_WM_Params = {'num_program': 100, 'wm_cells':120,'additional_inputs':0,
                 'internal_actionsSSA':internalActionsSSA_WM, 'eval':False,'predictiveSelfMod':False,'enforce_correctness': False,'jumpHome_at_error':False,'prepEvalParam':1}
SSANeat_Params = {'internal_actionsNEAT': internalActionsNEAT,
                  'maxNetworks':6,'use_setnet': False}



