# all useful named instruction sets here
from copy import deepcopy


#no WM ops disables any WM operations, but retains IP changes that do not make use of WM

# improving once gives an extra 10 consumptions




homeostatic_params={'stepsize':0.10, 'init_provision':float("inf"), 'consumption':1.,'reward_scale':3.0}
homeostatic_params_USEHOMEO={'stepsize':0.10, 'init_provision':3.0, 'consumption':1.,'reward_scale':3.0}
internalActionsSSA_WM={'jumpHome':0,
                       'jump_lower':3,'jump_equal':3,
                       'add':3,'mult':3,'sub':3,'div':3,'rem':3,
                       'inc':1,'dec':1,'init':2,'mov':2,
                       'endSelfModAfterNext': 0,
                       'getP':3
                      }

internalActionsNEAT={'add_node':2,'add_connection':2,'weight_change':2,
'get_output':1}


internalActionsNEATfixed = {'weight_change':2, 'get_output':1}

internalActionsNEATrandom={'r_add_node':0, 'r_add_connection':0,'r_weight_change':0,'get_output':1}



WM={'add':3,'mult':3,'sub':3,'div':3,'rem':3,
                       'inc':1,'dec':1,'init':2,'mov':2
    }
internalActions_LSTM_SSA={'endSelfModAfterNext':0}
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

SSA_WM_Params = {'num_program': 200, 'wm_cells':200,'additional_inputs':0,
                 'internal_actionsSSA':internalActionsSSA_WM, 'eval':False,'predictiveSelfMod':False,'enforce_correctness': False,'jumpHome_at_error':False,'prepEvalParam':1,
                 'maxTime':10000,'separate_arguments':True}


SSANeat_Params = {'internal_actionsNEAT': internalActionsNEAT,
                  'maxNetworks':6,'use_setnet': False}

# actually used
internalActionsGradientQsequence={'train_replay':1,
'doQuntil':3,'set_experience':1}
internalActionsGradientQsequenceTrainReplay={'train_replay':1,
'doQuntil':3,'set_experience':1,'jump_experience':0,'jump_within':1}

internalActionsGradientQsequenceTrainReplay2={'train_replay':1,
'doQuntil2':3,'set_experience2':1,'jump_experience2':0,'jump_within':1}


internalActionsGradientQsequenceTrainReplay0={'train_replay':1,
'doQuntil':3,'set_experience':1,'jump_within':1}

internalActionsGradientQsequenceTrainReplay3={'train_replay':1,
'doQuntil2':3,'set_experience2':1,'jump_within':1}


internalActionsGradientQsequenceTrainReplayNOJUMPEXP={'train_replay':1,
'doQuntil2':3,'set_experience2':1,'jump_within':1}


# TaskBasedSMP_DRQN_actionset={'inc_a_lr':1,
# 'inc_b_lr':1,'inc_a_eps':1,'inc_b_eps':1}


TaskBasedSMP_DRQN_actionset={'set_lr':1,
'set_eps':1}
TaskBasedSMP_DRQN_actionsetINC={'inc_lr':2,
'inc_eps':2}


TaskBasedSMP_DRQN_NOSTACK_actionset={'inc_a_lr_nostack':1,
'inc_b_lr_nostack':1,'inc_a_eps_nostack':1,'inc_b_eps_nostack':1}