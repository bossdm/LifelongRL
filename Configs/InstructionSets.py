# all useful named instruction sets here
from copy import deepcopy
searchPset={ 'searchP':3,
             'update_evalP':0}

predictiveSelfMod={'inc_mean':2,'dec_mean':2,'sample':2}

incPset={'incP':3,'decP':3}


#no WM ops disables any WM operations, but retains IP changes that do not make use of WM
internalSSA_NoWMops={'jumpHome':0,
                       'jump':1,
                       'endSelfMod': 0#'endSelfMod':0 #''endSelfMod': 0
                }

internalActionsSSA_WM={'jumpHome':0,
                       'jump_lower':3,'jump_equal':3,
                       'add':3,'mult':3,'sub':3,'div':3,'rem':3,
                       'inc':1,'dec':1,'init':2,'mov':2,
                       'endSelfMod': 0
                      }
internalActionsPredictive={'incThresh':1,'decThresh':1,'updateModel':0}
# internalActionsNEAT={'add_node':2,'add_connection':2,'bias_change':1,'response_change':1,'type_change':1,'weight_change':2,
# 'reset_weight':2,'disable_weight':2,'enable_weight':2,'get_output':1}


internalActionsNEAT={'add_node':2,'add_connection':2,'weight_change':2,
'get_output':0}

internalActionsGradientQ={'train_replay':1,
'getQoutput':0}
internalActionsGradientQsequence={'train_replay':1,
'doQuntil':3,'set_experience':1}

internalActionsGradientQsequencenoeps={'train_replay':1,
'doQuntil':2,'set_experience':1} # note, the third argument is stripped (epsilon always zero)

internalActionsGradientQsequencenoeps={'train_replay':1,
'doQuntil':2,'set_experience':1} # note, the third argument is stripped (epsilon always zero)

internalActionsGradientQsequence_fixedexperience={'train_replay':1,
'doQuntil':3}

internalActionsGradientQsequence_fixedexperience={'train_replay':1,
'doQuntil':3}

internalActionsGradientQsequence_notrainreplay={'doQuntil':3,'set_experience':1} # note, the third argument is stripped (epsilon always zero)

internalActionsNEATdel={'add_node':2,'add_connection':2,'weight_change':2,
'get_output':0,"del_node":1,"del_connection":2}
internalActionsNEATfixed = {'weight_change':2, 'get_output':0}

internalActionsNEATrandom={'r_add_node':0, 'r_add_connection':0,'r_weight_change':0,'get_output':1}

internalActionsMulti={'add_network': 0, 'set_net':1}

internalActions_kNets={'set_net':1}

WM={'add':3,'mult':3,'sub':3,'div':3,'rem':3,
                       'inc':1,'dec':1,'init':2,'mov':2
    }
internalActions_LSTM_SSA={'endSelfMod':0}
delta_ops={'incWeightDelta':5,'incDelta':5,'multDelta':5}
uncertainty_ops={'multUncertainty':2}
PM_ops = {'perceptual_advice':0,'perceptual_modification':5}
internalActions_LSTM_SSA_WM=dict(internalActions_LSTM_SSA,**WM)
internalActions_LSTM_SSA_WM_deltas=dict(internalActions_LSTM_SSA_WM,**delta_ops)
internalActions_LSTM_SSA_WM_deltasuncertainty=dict(internalActions_LSTM_SSA_WM_deltas,**uncertainty_ops)
internalActions_LSTM_SSA_WM_deltasuncertaintyPM=dict(internalActions_LSTM_SSA_WM_deltasuncertainty,**PM_ops)
internalActions_LSTM_SSA_deltas=dict(internalActions_LSTM_SSA,**delta_ops)
internalActions_LSTM_SSA_deltasuncertainty=dict(internalActions_LSTM_SSA_deltas,**uncertainty_ops)

internalActions_LSTM_SSA_deltasuncertaintyPM=dict(internalActions_LSTM_SSA_WM_deltasuncertainty,**PM_ops)
CSMP_SSA_Params = None
CSMP_SSANeat_Params = {'internal_actionsNEAT': {'add_node':2,'add_connection':2,'weight_change':2},
                 'maxNodes':200, 'maxNetworks':6,'use_setnet': False}


# config params user for initial subpols for C_SMP

config_paramsFF={'input_range':[], 'instruction_set':internalActionsNEAT, 'types': ['sigmoid'],
                          'input_type': 'ident',
                          'probabilistic':False, 'feedforward': True,
                          'topology': None, 'cell_type':"argument",'coord_initialisation': "default"}
config_paramsRNN={'input_range':[], 'instruction_set':internalActionsNEAT, 'types': ['sigmoid'],
                          'input_type': 'ident',
                          'probabilistic':False, 'feedforward': False,
                          'topology': None,'cell_type':'argument','coord_initialisation': "default"}
#

SSA_WM_Params = {'num_program': 100, 'wm_cells':120,'additional_inputs':0,
                 'internal_actionsSSA':internalActionsSSA_WM, 'eval':False,'predictiveSelfMod':False,
                 'enforce_correctness': False,'jumpHome_at_error':False,'prepEvalParam':1,
                 'maxTime':10000}


SSANeat_Params = {'internal_actionsNEAT': internalActionsNEAT,
                  'maxNetworks':6,'use_setnet': False}
