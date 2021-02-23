
from MazeUtils import *

from Environment import *


from ExperimentUtils import dump_incremental
from StatsAndVisualisation.Statistics import POcmanMeltingPotStatistics
from random import randint

from Agents.Agents import PacmanAgent
#from overrides import overrides
from mapobjects import EmptyObject, PacmanFood, PacmanPoison,NormalGhost, ChasedGhost, Power, Obstacle
from MazeUtils import manhattan_dist, directions, directional_dist, opposite_direction, VonNeumannNeighbourhoodPlus, \
    check_toroid_X, check_toroid_Y

from mapobjects import *
#import ray
DEBUG_MODE = False
STOPTIME = 2000000 #just a default
SAMPLING_RATE= 10000
from POcmanEnums import *



from os import getpid



import os
from Configs.InstructionSetsMultiExp import *




def get_SSA_configs(SSA_WM_Params,filename,num_PLAs,episodic):
    SSA_WM_Params['episodic'] = episodic
    enhance_PLA = 20
    SSA_WM_Params['filename'] = filename
    SSA_WM_Params['enhance_PLA'] = enhance_PLA / num_PLAs if not pacmanparams['real_time'] else 0
    SSA_WM_Params['jump_at_reset'] = False

    print("enhance PLA = " + str(SSA_WM_Params['enhance_PLA']))

def get_TaskBasedSSA_configs(SSA_WM_Params,filename,num_PLAs):
    get_SSA_configs(SSA_WM_Params,filename,num_PLAs,episodic=False)
    SSA_WM_Params['internal_actionsSSA'].update(TaskBasedSMP_DRQN_actionset)
    SSA_WM_Params['wm_cells'] = 100
    SSA_WM_Params['num_program'] = 80
    SSA_WM_Params['actions'] = []
    # SSA_WM_Params['num_inputs'] = 11  # loss and Qmax
    SSA_WM_Params['enhance_PLA']=5 # total of 16+2 self-mod + 4 param mods

def get_TaskBasedSSA_configsINC(SSA_WM_Params,filename,num_PLAs):
    get_SSA_configs(SSA_WM_Params,filename,num_PLAs,episodic=False)
    SSA_WM_Params['internal_actionsSSA'].update(TaskBasedSMP_DRQN_actionsetINC)
    SSA_WM_Params['wm_cells'] = 100
    SSA_WM_Params['num_program'] = 80
    SSA_WM_Params['actions'] = []
    # SSA_WM_Params['num_inputs'] = 11  # loss and Qmax
    SSA_WM_Params['enhance_PLA']=5 # total of 16+2 self-mod + 4 param mods

def get_TaskBasedSSA_nostack_configs(SSA_WM_Params,filename,num_PLAs):
    get_SSA_configs(SSA_WM_Params,filename,num_PLAs,episodic=False)
    SSA_WM_Params['internal_actionsSSA'].update(TaskBasedSMP_DRQN_NOSTACK_actionset)
    SSA_WM_Params['wm_cells'] = 100
    SSA_WM_Params['num_program'] = 80
    SSA_WM_Params['actions'] = []
    SSA_WM_Params['num_inputs'] = 2  # loss and Qmax
    SSA_WM_Params['enhance_PLA']=5 # total of 16+2 self-mod + 4 param mods

def get_SSA_Fixed_configs(num_PLAs,filename):
    enhance_PLA = 18
    SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATfixed
    n_input = SSA_WM_Params['num_inputs'] + 4 + SSA_WM_Params['additional_inputs']
    SSA_WM_Params['enhance_PLA'] = enhance_PLA / num_PLAs if not pacmanparams['real_time'] else 0
    print("enhance PLA = " + str(SSA_WM_Params['enhance_PLA']))
    SSA_WM_Params['filename'] = filename
    # del SSA_WM_Params['internal_actionsSSA']["endSelfMod"]
    config = {'input_range': range(n_input), 'instruction_set': SSANeat_Params['instruction_sets'],
              'types': ['sigmoid'],
              'input_type': 'ident',
              'probabilistic': False, 'feedforward': True,
              'topology': [80, 80]}  #approximately same topology as DRQN one
    configs = getIS_NEAT_configs(config, SSANeat_Params)
    return configs


def get_DRQN_configs(inputs,externalActions,filename,episodic):
    d=get_DRQN_agent_configs(inputs, externalActions, filename, episodic)
    d.update({'file': filename, 'loss': None})
    return d
def get_DRQN_agent_configs(inputs,externalActions,filename,episodic):
    if os.environ["tuning_lr"]:
        lr=float(os.environ["tuning_lr"])
    else:
        lr=0.10
    return  {'num_neurons':80,'task_features': [], 'use_task_bias': False,
                   'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 15,
                   'actions': deepcopy(externalActions),'episodic': episodic,
                'target_model':True,'init_epsilon':.20,'final_epsilon':.20,'epsilon_change':False,"learning_rate":lr}
def get_A2C_configs(inputs,externalActions, filename, episodic):
    paramsdict={}
    if os.environ["tuning_lr"]:
        paramsdict["learning_rate"]=float(os.environ["tuning_lr"])
    else:
        paramsdict["learning_rate"]=0.00025
    return {'num_neurons': 80, 'task_features': [], 'use_task_bias': False,
            'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 15,
            'actions': deepcopy(externalActions), 'episodic': episodic,'file':filename, 'params':paramsdict
            }
def get_SMP_DRQN_configs(SMP_DRQN_actions,SSA_WM_Params,num_PLAs,filename,num_inputs,externalActions,episodic):
    enhance_PLA = 18

    SSA_WM_Params['episodic'] = episodic
    SSA_WM_Params['filename'] = filename
    SSA_WM_Params['enhance_PLA'] = enhance_PLA / num_PLAs if not pacmanparams['real_time'] else 0
    print("enhance PLA = " + str(SSA_WM_Params['enhance_PLA']))
    input_addresses = range(4, 4+num_inputs)
    DRQN_params=get_DRQN_agent_configs(num_inputs,externalActions,filename,episodic)
    del DRQN_params['epsilon_change'] # not used in SMP-DRQN
    from IS.SSA_gradientQ import ConversionType
    SSA_WM_Params['internal_actionsSSA'].update(SMP_DRQN_actions)
    fixed_training=False if 'train_replay' in SMP_DRQN_actions else True
    SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATfixed
    # config = {'input_range': [2]+input_addresses, 'instruction_set': SSANeat_Params['instruction_sets'], #2 is for IP
    #           'types': ['sigmoid'],
    #           'input_type': 'ident',
    #           'probabilistic': True, 'feedforward': True,
    #           'topology': [10, 10]}
    # configs = getIS_NEAT_configs(config, SSANeat_Params)
    return {'n_outputs':len(externalActions),'trace_length':DRQN_params['trace_length'],
            'input_addresses':input_addresses, 'conversion_type':ConversionType.double_index,
            'SSA_WM_Params':SSA_WM_Params,'Q_internal':False,
                 'fixed_training':fixed_training,'intervals':loss_intervals,
           'DRQN_params':DRQN_params,'init_with_training_freq': True}
           # 'SSA_NEAT_Params':SSANeat_Params, 'NEAT_configs':configs}




def get_method(methodname,externalActions,filename,num_PLAs,pacmanparams,inputs,conservative,episodic=True):

    print("EPISODIC="+str(episodic))
    if methodname == 'RandomLearner':
        method = RandomLearner(externalActions, filename)

    #####################################################################################################

    elif methodname == "PPO2":
        from Catastrophic_Forgetting_NNs.A2C_Learner2 import PPO_Learner
        settings=get_A2C_configs(inputs,externalActions,filename,episodic)
        method = PPO_Learner( **settings)

    elif methodname == "MultiActorPPO2":
        from Catastrophic_Forgetting_NNs.PPO_MultiActor import PPO_MultiActor
        settings=get_A2C_configs(inputs,externalActions,filename,episodic)
        method = PPO_MultiActor(settings)


    elif methodname == "TaskDrift_PPO2":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDrift_PPO2 import TaskDriftPPO



        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            settings = get_A2C_configs(inputs, externalActions, filename, episodic)

            pols[pol] = TaskDriftPPO(settings)
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)


    elif methodname == "1to1_PPO2":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDrift_PPO2 import TaskDriftPPO



        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            settings = get_A2C_configs(inputs, externalActions, filename, episodic)

            pols[pol] = TaskDriftPPO(settings)
        homeostatic_params['one_to_one']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)


    elif methodname == "Unadaptive_PPO2":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDrift_PPO2 import TaskDriftPPO



        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            settings = get_A2C_configs(inputs, externalActions, filename, episodic)

            pols[pol] = TaskDriftPPO(settings)
        homeostatic_params['unadaptive']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)






    elif methodname == "DRQN":

        from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
        settings=get_DRQN_configs(inputs,externalActions,filename,episodic)
        method = DRQN_Learner( **settings)
    elif methodname == "SelectiveDRQN":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
        settings=get_DRQN_configs(inputs,externalActions,filename,episodic)
        method = DRQN_Learner( **settings)
        method.agent.init_selective_memory(FIFO=0)
    elif methodname == "SelectiveFifoDRQN":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
        settings=get_DRQN_configs(inputs,externalActions,filename,episodic)
        method = DRQN_Learner( **settings)
        method.agent.init_selective_memory(FIFO=50000)
    elif methodname == "EWC":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import EWC_Learner
        settings = get_DRQN_configs(inputs, externalActions, filename, episodic)
        settings["multigoal"] = True  # "We also allowed the DQN agents to maintain separate short-term memory buffers for each inferred task."
        settings["buffer_size"] = 400000 // 18  # distribute equally among tasks
        method = EWC_Learner(5*10**6, settings)
    elif methodname == "EWC_half":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import EWC_Learner
        settings = get_DRQN_configs(inputs, externalActions, filename, episodic)
        settings["multigoal"] = True  # "We also allowed the DQN agents to maintain separate short-term memory buffers for each inferred task."
        settings["buffer_size"] = 400000 // 18  # distribute equally among tasks
        method = EWC_Learner(2.5*10**6, settings)
    elif methodname == "EWC_fifth":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import EWC_Learner
        settings = get_DRQN_configs(inputs, externalActions, filename, episodic)
        settings["multigoal"] = True  # "We also allowed the DQN agents to maintain separate short-term memory buffers for each inferred task."
        settings["buffer_size"] = 400000 // 18  # distribute equally among tasks
        method = EWC_Learner(1*10**6 // 5, settings)
    elif methodname == "TaskDriftInit_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftMultiTaskDRQN
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftMultiTaskDRQN(DRQN_params)
        homeostatic_params['initialise_unseen'] = True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
    elif methodname == "1to1_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN
        args.policies=18
        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params)
        homeostatic_params['one_to_one']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
    elif methodname == "Unadaptive_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN
        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params)
        homeostatic_params['unadaptive']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
    elif methodname == "TaskDrift_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN

        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params)

        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)


    elif methodname == "TaskDriftInit_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN

        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params)
        homeostatic_params['initialise_unseen']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
    elif methodname == "TaskDriftInitAndOffline_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN

        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params)
        homeostatic_params['initialise_unseen']=True
        homeostatic_params['offline_updates'] = True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)


    #############################################################################

    elif methodname == 'SSA_WM':
        get_SSA_configs(SSA_WM_Params,filename,num_PLAs,episodic)
        method = SSA_with_WM(**SSA_WM_Params)
    elif methodname == 'SSA_gradientQsequence':
        from IS.SSA_gradientQ import SSA_gradientQ
        settings=get_SMP_DRQN_configs(SMP_DRQN_actions=internalActionsGradientQsequence,SSA_WM_Params=SSA_WM_Params,
                                      num_PLAs=num_PLAs,filename=filename,num_inputs=inputs,externalActions=externalActions,
                                      episodic=episodic)
        method = SSA_gradientQ(**settings)
    elif methodname == 'TraditionalSSA_gradientQsequence':
        from IS.SSA_gradientQ import SSA_gradientQ
        settings=get_SMP_DRQN_configs(SMP_DRQN_actions=internalActionsGradientQsequenceTrainReplay,SSA_WM_Params=SSA_WM_Params,
                                      num_PLAs=num_PLAs,filename=filename,num_inputs=inputs,externalActions=externalActions,
                                      episodic=episodic)
        SSA_WM_Params['conservative']=False
    elif methodname=="TaskBasedSMP_DRQN_nolearning":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN,LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)

        method = TaskBasedSMP_DRQN(LearningType.none,SSA_WM_Params,configs)
    elif methodname == "TaskBasedSMP_DRQN":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        method = TaskBasedSMP_DRQN(LearningType.taskbased, SSA_WM_Params, configs, reward_type="reward")
    elif methodname == "TaskBasedSMP_DRQN_lifetime":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)

        method = TaskBasedSMP_DRQN(LearningType.lifetime, SSA_WM_Params, configs, reward_type="reward")
    elif methodname == "TaskBasedSMP_DRQN_lifetimeTaskspecific":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams={'weighting_type':'fixed','absolute':True,'SSA_with_WMparams':SSA_WM_Params}

        method = TaskBasedSMP_DRQN(LearningType.lifetime_taskspecific, taskspecificSSAparams, configs, reward_type="reward")
    elif methodname == "TaskBasedSMP_DRQN_lifetimeTaskspecificRelative":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams = {'weighting_type': 'fixed', 'absolute': False, 'SSA_with_WMparams': SSA_WM_Params}
        method = TaskBasedSMP_DRQN(LearningType.lifetime_taskspecificrelative, taskspecificSSAparams, configs, reward_type="reward")

    elif methodname == "TaskBasedSMP_DRQN_inc":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs = get_DRQN_configs(inputs, externalActions, filename, episodic)

        get_TaskBasedSSA_configsINC(SSA_WM_Params, filename, num_PLAs)
        method = TaskBasedSMP_DRQN(LearningType.taskbased, SSA_WM_Params, configs, reward_type="reward")
    elif methodname == "TaskBasedSMP_DRQN_l_inc":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs = get_DRQN_configs(inputs, externalActions, filename, episodic)

        get_TaskBasedSSA_configsINC(SSA_WM_Params, filename, num_PLAs)

        method = TaskBasedSMP_DRQN(LearningType.lifetime, SSA_WM_Params, configs, reward_type="reward")

    elif methodname == "SingleSMP_DRQN_lifetime":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)

        method = TaskBasedSMP_DRQN(LearningType.single_lifetime, SSA_WM_Params, configs, reward_type="reward")
    elif methodname == "SingleSMP_DRQN_lifetimeTaskspecific":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams={'weighting_type':'fixed','absolute':True,'SSA_with_WMparams':SSA_WM_Params}

        method = TaskBasedSMP_DRQN(LearningType.single_lifetimetaskspecific, taskspecificSSAparams, configs, reward_type="reward")
    elif methodname == "SingleSMP_DRQN_lifetimeTaskspecificRelative":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams = {'weighting_type': 'fixed', 'absolute': False, 'SSA_with_WMparams': SSA_WM_Params}
        method = TaskBasedSMP_DRQN(LearningType.single_lifetimetaskspecificrelative, taskspecificSSAparams, configs, reward_type="reward")

    elif methodname == "SingleSMP_DRQN_inc":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs = get_DRQN_configs(inputs, externalActions, filename, episodic)

        get_TaskBasedSSA_configsINC(SSA_WM_Params, filename, num_PLAs)

        method = TaskBasedSMP_DRQN(LearningType.single_lifetime, SSA_WM_Params, configs, reward_type="reward")
    elif methodname == "SingleSMP_DRQN_Taskspecific_inc":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs = get_DRQN_configs(inputs, externalActions, filename, episodic)

        get_TaskBasedSSA_configsINC(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams = {'weighting_type': 'fixed', 'absolute': True, 'SSA_with_WMparams': SSA_WM_Params}

        method = TaskBasedSMP_DRQN(LearningType.single_lifetimetaskspecific, taskspecificSSAparams, configs,
                                   reward_type="reward")
    elif methodname == "SingleSMP_DRQN_TaskspecificRelative_inc":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs = get_DRQN_configs(inputs, externalActions, filename, episodic)

        get_TaskBasedSSA_configsINC(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams = {'weighting_type': 'fixed', 'absolute': False, 'SSA_with_WMparams': SSA_WM_Params}
        method = TaskBasedSMP_DRQN(LearningType.single_lifetimetaskspecificrelative, taskspecificSSAparams, configs,
                                   reward_type="reward")



    elif methodname == "TaskBasedSMP_DRQN_lossbased":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        method = TaskBasedSMP_DRQN(LearningType.taskbased, SSA_WM_Params, configs, reward_type="loss")
    elif methodname == "TaskBasedSMP_DRQN_lifetime_lossbased":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)

        method = TaskBasedSMP_DRQN(LearningType.lifetime, SSA_WM_Params, configs, reward_type="loss")
    elif methodname == "TaskBasedSMP_DRQN_lifetimeTaskspecific_lossbased":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams={'weighting_type':'fixed','absolute':True,'SSA_with_WMparams':SSA_WM_Params}

        method = TaskBasedSMP_DRQN(LearningType.lifetime_taskspecific, taskspecificSSAparams, configs, reward_type="loss")
    elif methodname == "TaskBasedSMP_DRQN_lifetimeTaskspecificRelative_lossbased":
        from Lifelong.TaskNGoalBasedExploration import TaskBasedSMP_DRQN, LearningType
        configs=get_DRQN_configs(inputs,externalActions,filename,episodic)

        get_TaskBasedSSA_configs(SSA_WM_Params, filename, num_PLAs)
        taskspecificSSAparams = {'weighting_type': 'fixed', 'absolute': False, 'SSA_with_WMparams': SSA_WM_Params}
        method = TaskBasedSMP_DRQN(LearningType.lifetime_taskspecificrelative, taskspecificSSAparams, configs, reward_type="loss")

    else:
        raise Exception("methodname %s not found" % (methodname))


    return method

class POcman(NavigationEnvironment):
    add_task_features = False
    reward_clearlevel = +100.
    reward_default = 0.

    reward_eatpower = +50.
    reward_eatghost = +100.
    reward_multipl = +1.  # multiply the eatghost reward when multiple caught during the same power-up
    reward_die = -50.
    reward_eatfood = +10.
    reward_hitwall = -10.
    reward_poison = -10.
    old_object=None
    old_F = None
    oldx = None
    oldy = None
    def __init__(self, agent, visual, params,actor_index=0):
        self.terminal = False



        #self.passage_y = -1
        self.smell_range = 1
        self.hear_range = 2
        self.food_prob = 0.5
        self.chase_prob = 0.50
        self.defensive_slip = 0.50
        self.power_numsteps = 15

        self.elementary_task_time=1000 if 'elementary_task_time' not in params else params['elementary_task_time']
        self.reward_die = -50. if 'reward_die' not in params else params['reward_die']
        self.reward_eatfood = +10. if 'reward_food' not in params else params['reward_eatfood']
        self.reward_hitwall = -10. if 'reward_hitwall' not in params else params['reward_hitwall']
        self.reward_poison = -10. if 'reward_poison' not in params else params['reward_poison']
        # 4: food available, 4 ghost visible 4 wall configuration 3: 2,3, or 4 manhattandist(food) 1: powerpill (13)
        # old version: 10
        self.obs_length = params["observation_length"]
        self.add_task_features = params["include_task_features"]
        self.inform_task_time = params['inform_task_time']
        self.num_observations = self.obs_length ** 2

        self.discount = 0.95
        self.num_food=0
        self.num_poison=0
        self.num_ghosts=0
        if actor_index>0:
            self.num_actors = 1# avoid infinite recursion
            self._is_actor=True
            prefix = params["filename"] + "worker"
            i = actor_index
            filename = prefix + str(i-1) + "R.txt"
            method = get_method(**agent)
            agent = PacmanAgent(method, params)
            # agent.learner.mazefile = open(prefix + "maze" + str(i) + ".txt", "w")
            agent.learner.Rfile = open(filename,"a")
        else:
            self._is_actor=False
            self.num_actors = params["num_actors"] if 'num_actors' in params else 1
            if self.num_actors>1:
                self.params = params
                self.agent_dict = agent
                method = get_method(**agent)
                agent = PacmanAgent(method, params)





        NavigationEnvironment.__init__(self, agent, visual, params)
        self.maze = np.zeros((self.sizeX, self.sizeY))
    #@overrides
    def is_actor(self):
        return self._is_actor
    def run_workers(self):

        for actor in self.actors:
            print("set task")
            ray.get(actor.set_task.remote(deepcopy(self.currentTask),self.running_time,self.t))
        stop=False
        while not stop:
            #get global weights
            w = self.agent.learner.get_weights()
            #print(w)
            #let actors work
            update_states=[]
            for actor in self.actors:
                #actor.hello.remote()
                # set weights of the actors
                ray.get(actor.set_weights.remote(w))
                #print(w)
                #run the actor
                stop=ray.get(actor.run_individual_worker.remote())
                update_states.append(ray.get(actor.get_update_states.remote()))

            #ready_ids, remaining_ids = ray.wait(update_states, num_returns=POcman.num_actors)
            # update the shared learner
            for terminal, update_state in update_states:
                print("length of update_state=", len(update_state[0]))
                self.agent.learner.process_update_states(update_state)
                #print("states outside=\n",self.agent.learner.update_states())
                if terminal:
                    self.agent.learner.update_model(True)
                else:
                    self.agent.learner.update_model(False)
        self.t = ray.get(self.actors[0].get_time.remote())
    def get_map(self, x, y):
        return self.map_flag_to_mapel(x,y)
    #@overrides
    def set_tasks(self,tasks,stat_freq):
        BaseEnvironment.set_tasks(self,tasks,stat_freq)
        self.initStats()
    def initStats(self):
        self.stats=[{} for i in range(self.slices)]


    def time_up(self):
        """
        check whether the current elementary task is supposed to end
        :return:
        """
        return self.t % self.elementary_task_time == 0
    @classmethod
    def get_reward_range(cls):
        """
        get the reward range independent of any settings (works only assuming no eat_ghost multiplier
        :return:
        """
        # worst case: die + hit_wall
        min_reward = cls.reward_die + cls.reward_hitwall + cls.reward_default
        # best case, eatfood, eatghost and clearlevel
        max_reward = cls.reward_eatghost + cls.reward_eatfood + cls.reward_clearlevel + cls.reward_default
        return min_reward,max_reward
    @classmethod
    def get_pacman_maze(cls):
        maze= [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0],
            [0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 0],
            [0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0],
            [0, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 1, 1, 1, 1],
            [0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0],
            [0, 7, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 7, 0],
            [0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0],
            [0, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        return np.transpose(maze)
    @classmethod
    def get_micropacman_maze(cls):
        maze=    [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 7, 3, 3, 3, 3, 3, 7, 0],
                [0, 3, 3, 0, 3, 0, 3, 3, 0],
                [0, 3, 0, 3, 3, 3, 0, 3, 0],
                [3, 3, 3, 3, 0, 3, 3, 3, 3],
                [0, 3, 0, 3, 3, 3, 0, 3, 0],
                [0, 3, 3, 0, 3, 0, 3, 3, 0],
                [0, 7, 3, 3, 3, 3, 3, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]

            ]
        return np.transpose(maze)
    @classmethod
    def get_minipacman_maze(cls):
        maze= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 0],
                [0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0],
                [0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0],
                [3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3],
                [0, 0, 0, 3, 0, 1, 1, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 0, 1, 1, 3, 3, 0, 0, 0],
                [0, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 0],
                [0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0],
                [0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0],
                [0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        return np.transpose(maze)
    def standard_pacman(self,num_ghosts=4):
        self.sizeX = 19
        self.sizeY = 21
        self.maze = self.get_pacman_maze()
        self.pocman_home = (8, self.sizeY - 1 - 6)
        self.num_ghosts = num_ghosts
        if num_ghosts > 0:
            self.ghost_range = 6

            self.ghost_home = (8, self.sizeY - 1 - 10)


    def micro_pacman(self,num_ghosts=1):
        self.sizeX=9
        self.sizeY=9
        self.maze = self.get_micropacman_maze()

        self.pocman_home = (4, self.sizeY - 2)
        self.num_ghosts = num_ghosts
        if self.num_ghosts > 0:

            self.ghost_range = 3
            self.ghost_home = (4, self.sizeY - 1 - 5)

    def mini_pacman(self,num_ghosts=3):
        self.sizeX=12
        self.sizeY=12
        self.maze = self.get_minipacman_maze()
        self.pocman_home = (5, self.sizeY - 1 - 3)
        self.num_ghosts = num_ghosts
        if num_ghosts>0:

            self.ghost_range = 4

            self.ghost_home = (5, self.sizeY - 1 - 6)

        #self.passage_y = 5
    def standardmaze_map(self,num_ghosts=0):
        self.sizeX = 11
        self.sizeY = 8

        self.maze=self.get_standard_maze()

        self.pocman_home = (1, 3)
        self.num_ghosts = num_ghosts
        if num_ghosts > 0:

            self.ghost_range = 3
            self.ghost_home = (9, 1)

          # we count downwards and from 0, so coords a little different from paper, but figure the same
    def cheesemaze_map(self,num_ghosts=0):
        self.sizeX = 7
        self.sizeY = 5

        self.maze=self.get_cheese_maze()

        self.pocman_home = (1, 2)
        self.num_ghosts = num_ghosts
        if num_ghosts > 0:

            self.ghost_range = 3
            self.ghost_home = (3, 3)

    @classmethod
    def fill_borders(cls,sizeX,sizeY):
        # Fill borders
        maze = np.array([[POcmanFlags.E_FREE] * sizeY for _ in range(sizeX)])
        for x in [0, sizeX - 1]:
            for y in range(sizeY):
                maze[x,y] = POcmanFlags.E_OBSTACLE
        for y in [0, sizeY - 1]:
            for x in range(sizeX):
                maze[x,y] = POcmanFlags.E_OBSTACLE
        return maze

    @classmethod
    def fill_other_obstacles(self,occupied,maze):
        for (x,y) in occupied:
            maze[x,y]=POcmanFlags.E_OBSTACLE
        return maze

    @classmethod
    def get_standard_maze(cls):
        sizeX = 11
        sizeY = 8
        maze=cls.fill_borders(sizeX,sizeY)
        occupiedCoords = [(3, 2), (3, 3), (3, 4), (6, 5), (8, 1), (8, 2), (
            8, 3)]
        return cls.fill_other_obstacles(occupiedCoords,maze)

    @classmethod
    def get_cheese_maze(cls):
        sizeX=7
        sizeY=5
        maze=cls.fill_borders(sizeX,sizeY)
        occupiedCoords = [(2, 2), (2, 3), (4, 2), (4, 3)]
        return cls.fill_other_obstacles(occupiedCoords, maze)
    # visualisation functions
    @classmethod
    def get_maze(cls,topology):
        if topology==PacmanTopology.standard:
            return cls.get_standard_maze()
        elif topology==PacmanTopology.pacman_micro:
            return cls.get_micropacman_maze()
        elif topology==PacmanTopology.pacman_mini:
            return cls.get_minipacman_maze()
        elif topology==PacmanTopology.pacman_standard:
            return cls.get_pacman_maze()
        elif topology==PacmanTopology.cheese:
            return cls.get_cheese_maze()
        else:
            raise Exception("not supported")
    def convert_to_map(self):
        self.map = [[EmptyObject()] * self.sizeY for _ in range(self.sizeX)]
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                self.map[x][y]= self.map_flag_to_mapel(x, y)
    def convertObservationBinary(self,agent):
        if FULLOBS:
            xbits=get_required_bits(self.sizeX)
            ybits=get_required_bits(self.sizeY)
            x=encode(agent.x,xbits)
            y=encode(agent.y,ybits)
            agent.learner.observation = x + y



        else:
            for i in range(len(agent.learner.observation)):
                agent.learner.observation[i]= True if agent.learner.observation[i]==1 else False
        if DEBUG_MODE:
            print("x,y = %d,%d"%(self.agent.x,self.agent.y))
            print(agent.learner.observation)

    def map_flag_to_mapel(self, x, y):
        # only the static elements
        flag = self.maze[x, y]
        if DEBUG_MODE:
            print(POcmanFlags.E_FOOD)
        if flag == int(POcmanFlags.E_FREE):
            return EmptyObject()
        elif flag == int(POcmanFlags.E_OBSTACLE):
            return Obstacle(None)
        elif flag == int(POcmanFlags.E_FOOD):
            return PacmanFood(None)
        elif flag == int(POcmanFlags.E_POISON):
            return PacmanPoison(None)
        elif flag == int(POcmanFlags.E_POWER):
            return Power(None)
        else:
            raise Exception("found element %d" % (flag))

    def add_other_agents(self):
        filename = "pacman_ghost.jpg" if self.power_steps > 0 else "pacman_normalghost.png"
        for ghost in self.ghost_pos:
            x, y = ghost
            self.vis.display.addPicture(x, y, shrink_factor=(1.0, 1.0),
                                        filename=filename)

    # end visualisation
    #@overrides
    def generateMap(self, params=None):
        self.new_level()

        #self.agent.learner.new_task(self.currentTask.task_feature)

    #@overrides
    def printStatistics(self):
        pass
    def reset(self):
        if DEBUG_MODE:
            print("reset")
        # gets done every tick:
        if not self.terminal:
            return

        self.new_level()

        # # if self.total > 100:
        # #     if self.successes / float(self.total) > .80:
        #         #self.agent.learner.Rfile.write("SUCCESS after " + str(self.t) + "\n")
        #         #self.agent.learner.Rfile.flush()
        #         #self.currentTask.solved = True
        self.endEpisode()
        self.terminal = False

    def endEpisode(self):
        if self.agent.learner.episodic:
            if self.t > 0:
                #self.currentCount = self.maxCount + 1
                self.agent.learner.setTerminalObservation(self.agent, self)  # set the final observation
                #print("added terminal obs")
        self.agent.learner.reset()
        self.currentTask.start_time=self.t






    # def Copy(
    # STATE & state)
    # {
    #
    # POCMAN_STATE & pocstate = safe_cast <
    # POCMAN_STATE & > (state)
    # POCMAN_STATE * newstate = MemoryPool.Allocate()
    # *newstate = pocstate
    # return newstate
    # }
    def check_passablepos(self, pos):
        return self.maze[int(pos[0]),int(pos[1])] != POcmanFlags.E_OBSTACLE

    def validate(self, state):
        assert self.check_passablepos((self.agent.x, self.agent.y))
        for g in range(self.num_ghosts):
            assert self.check_passablepos(self.ghost_pos)

    # def create_start_state(self):
    #     self.ghost_pos=np.zeros(self.num_ghosts)
    #     self.food=np.zeros(self.sizeX,self.sizeY,dtype=bool)
    #     self.new_level()



    # void
    # FreeState(STATE * state)
    #
    # {
    # POCMAN_STATE * pocstate = safe_cast < POCMAN_STATE * > (state)
    # MemoryPool.Free(pocstate)
    # }

    #
    def check_obstacle(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_OBSTACLE

    def check_powerpill(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_POWER

    def check_food(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_FOOD or self.maze[x, y] == POcmanFlags.E_POWER
    def check_poison(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_POISON
    def check_food_specific(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_FOOD

    def check_free(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_FREE
    @classmethod
    def check_free_maze(cls,x, y,maze):
        return maze[x, y] == POcmanFlags.E_FREE
    def check_foodwithindist(self, dist):
        minx = self.agent.x - dist
        miny = self.agent.y - dist
        maxx = self.agent.x + dist
        maxy = self.agent.y + dist
        for x in range(minx, maxx+1):
            for y in range(miny, maxy+1):
                xx = check_toroid_X(x, self)
                yy = check_toroid_Y(y, self)
                if self.check_food(xx, yy) and manhattan_dist((self.agent.x, self.agent.y), (xx, yy)) == dist:
                    if DEBUG_MODE:
                        print("see food within dist %d"%(dist))
                    return True
        if DEBUG_MODE:
            print("no food within dist %d" % (dist))
        return False
    def get_direction_based_on_manhattandist(self,dist):
        """
        get the directions leading to manhattan dist
        :param dist:
        :return:
        """
        # if 0, return (0,0)
        # if even, return (+-dist/2,+-dist/2), else return [(+-(dist//2+1),+-dist//2),(+-dist//2,+-(dist//2+1))]

        return {0:[(0,0)],
                1: [(0,1),(1,0),(-1,0),(0,-1)],
                2: [(1,1),(1,-1),(-1,1),(-1,-1),
                    (2,0),(0,2),(-2,0),(0,-2),
                    ],
                3: [(1,2),(1,-2),(-1,2),(-1,-2),
                    (2, 1), (2, -1), (-2, 1), (-2, -1),
                    (3,0),(0,3),(-3,0),(0,-3)],
                4: [(2,2),(2,-2),(-2,2),(-2,-2),
                    (3, 1), (3, -1), (-3, 1), (-3, -1),
                    (1, 3), (1, -3), (-1, 3), (-1, -3),
                    (4, 0), (0, 4), (-4, 0), (0, -4)
                    ]}[dist]
    def signed_bool(self,num):
        return 1. if num else -1.
    def check_objectwithindist(self, dists):
        """

        :param dists: sorted (ascending) list of distances (e.g., [2,3,4])
        :return:
        """
        l=[self.num_ghosts,self.num_food,self.num_poison]
        assert sum(l)==1, "%d ghosts, %d foods, %d poisons"%(self.num_ghosts,self.num_food,self.num_poison) # only one can be true in the meltingpot scenario
        object_location=self.get_object()
        distance=manhattan_dist((self.agent.x,self.agent.y),object_location)
        #print([self.signed_bool(distance <= dist) for dist in dists])
        return [1 if distance <= dist else -1 for dist in dists]
        # observations=np.zeros(len(dists)) - 1
        # min_index=0
        # for dist in range(maxdist+1):
        #     directions=self.get_direction_based_on_manhattandist(dist)
        #     mindist=dists[min_index]
        #     while dist > mindist:
        #         # need to change the min_index
        #         min_index+=1
        #         mindist=dists[min_index]
        #
        #
        #     for (x,y) in directions:
        #         if self.see_object((x, y)):
        #             observations[min_index:]=1
        #             print("dist=" + str(dist))
        #             print("mindist=" + str(mindist))
        #             print((x,y))
        #             print(observations)
        #             return observations

        # indexes_left=range(len(dists))
        # minx = - max(dists)
        # miny = - max(dists)
        # maxx = + max(dists)
        # maxy = + max(dists)
        # for x in range(minx, maxx+1):
        #     for y in range(miny, maxy+1):
        #         manhattandist=x+y
        #         if self.see_object((x,y)):
        #             if DEBUG_MODE:
        #                 print("see food within dist %d"%(manhattandist))
        #             for i in indexes_left:
        #                 if manhattandist <= dists[i]:
        #                     observations[i:]=1
        #                     indexes_left=indexes_left[0:i]
        #                     break
        #
        #             if not indexes_left:
        #                 if DEBUG_MODE:
        #                     print("observations="+str(observations))
        #                 return observations
        #
        #
        print("dist=" + str(dist+1))
        print("mindist=" + str(mindist))
        print("observations="+str(observations))
        return observations
    def setStandardObservation(self):
        observation = -1 + np.zeros((self.obs_length,))
        # wall configuration
        for d in range(4):
            if self.see_obstacle(directions[d]):
                observation[d] = 1.

        # ghost visible
        for d in range(4):
            if self.see_ghost(directions[d]) >= 0:
                observation[4 + d] = 1.

        # food visible
        for d in range(4):
            if self.see_food(directions[d]):
                observation[8 + d] = 1.
        # food smellable
        observation[12] = 1.0 if self.check_foodwithindist(2) else -1.0
        observation[13] = 1.0 if self.check_foodwithindist(3) else -1.0
        observation[14] = 1.0 if self.check_foodwithindist(4) else -1.0
        # power pill influence
        observation[15] = 1.0 if self.power_steps > 0 else -1.0
        add = 0
        if self.inform_task_time:
            add = 1
            observation[16] = min((self.t - self.currentTask.start_time)/1000,1.)  #0.001, .002, ... 1. , 1. , ...
        if self.add_task_features:
            observation[16+add:] = np.array(self.currentTask.task_feature)/3.

        if DEBUG_MODE:
            print(observation)
        self.agent.learner.observation = observation

    def setMeltingPotObservation(self):
        observation = -1 + np.zeros((self.obs_length,))
        # wall configuration
        for d in range(4):
            if self.see_obstacle(directions[d]):
                observation[d] = 1.

        # object visible
        for d in range(4):
            if self.see_object(directions[d]) >= 0:
                observation[4 + d] = 1.
        # object smellable
        observation[8:11] = self.check_objectwithindist([2,3,4])

        # power pill influence
        #observation[11] = 1.0 if self.power_steps > 0 else -1.0
        add=0
        if self.inform_task_time:
            add=1
            observation[11] = min((self.t - self.currentTask.start_time)/1000.,1.)  #0.001, .002, ... 1. , 1. , ...
        if self.add_task_features:
            observation[11+add:] = np.array(self.currentTask.task_feature)/3.

        if DEBUG_MODE:
            print("OBS="+str(observation))
        self.agent.learner.observation = observation
    def getCurrentSlice(self):
        return self.t//self.statfreq

    def mazestats_update(self,slice,F,key):
        if self.agent.learner.testing:
            return
        if key not in self.stats[slice]:
            self.stats[slice][key] = POcmanMeltingPotStatistics(self.pocman_home, F, self)
        self.stats[slice][key].update((self.agent.x, self.agent.y), self.get_object(), self.agent.learner.chosenAction)
    #@overrides
    def updateStat(self):
        slice=self.getCurrentSlice()
        if self.timePolicy.stepCondition(self):
            if hasattr(self.agent.learner,'pols'):
                for pol in range(len(self.agent.learner.pols)):
                    F=tuple(self.currentTask.task_feature)
                    key = "task%s" % (str(F)) + "pol%d"%(pol)
                    self.mazestats_update(slice, F,key)

            else:
                F = tuple(self.currentTask.task_feature)
                key = "task%s"%(str(F))
                self.mazestats_update(slice,F,key)
        self.agent.learner.stats.update(self.agent.learner)
        # oldCoord = (self.oldx,self.oldy)
        # newCoord = (self.agent.x, self.agent.y)
        # new_object=self.get_object()
        # oldCoord = (self.old_F,oldCoord,self.old_object)
        # newCoord = (F,newCoord,new_object)
        # self.agent.learner.track_q(oldCoord,newCoord,self.agent.learner.intervals)
        # self.old_object=new_object
        # self.old_F = F
        # self.oldx = self.agent.x
        # self.oldy = self.agent.y
    #@overrides
    def printStatistics(self):
        POcmanMeltingPotStatistics.getAllStats(self.filename,self.stats)
        dump_incremental(self.filename+"_POcmanStats",self.stats)
        self.agent.learner.printStatistics()
    def setObservation(self, agent):

        if self.currentTask.task_type==PacmanTaskType.MELTINGPOT:
            self.setMeltingPotObservation()
        else:
            self.setStandardObservation()


    def see_poison(self, direction):
        x,y = direction
        if DEBUG_MODE:
            if self.check_poison(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self)):
                print("see food in direction %d,%d" % (x, y))
            else:
                print("no food in direction %d,%d" % (x, y))
        return self.check_poison(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self))


    def see_food(self, direction):
        x,y = direction
        if DEBUG_MODE:
            if self.check_food(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self)):
                print("see food in direction %d,%d" % (x, y))
            else:
                print("no food in direction %d,%d" % (x, y))
        return self.check_food(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self))
    def get_object(self):

        if self.num_ghosts > 0 :
            return self.ghost_pos[0]
        else:
            return self.object_location
    def see_object(self,direction):
        """
        see object (exclusively for the meltingpot scenario)
        :param direction:
        :return:
        """

        if self.num_ghosts > 0:
            return self.see_ghost(direction)
        elif self.num_food > 0:
            return self.see_food(direction)
        elif self.num_poison > 0:
            return self.see_poison(direction)
        else:
            if self.num_ghosts==0 and self.num_food==0 and self.num_poison==0:
                raise Exception("no object !?")
            else:
                if DEBUG_MODE:
                    print(self.num_poison)
                    print(self.num_food)
                    print(self.num_ghosts)
                raise Exception('cannot have both objects in meltingpot task')


    def see_obstacle(self, direction):
        x,y = direction
        if DEBUG_MODE:
            if self.maze[check_toroid_X(self.agent.x + x, self), check_toroid_Y(self.agent.y + y,
                                                                                self)] == POcmanFlags.E_OBSTACLE:
                print("see obstacle in direction %d,%d" % (x, y))
            else:
                print("no obstacle in direction %d,%d" % (x, y))
        return self.maze[check_toroid_X(self.agent.x + x, self), check_toroid_Y(self.agent.y + y,
                                                                                self)] == POcmanFlags.E_OBSTACLE

    def see_ghost(self, direction):
        x = check_toroid_X(self.agent.x + direction[0], self)
        y = check_toroid_Y(self.agent.y + direction[1], self)
        eyepos = (x, y)
        if DEBUG_MODE:
            print(self.ghost_pos)
            print(self.num_ghosts)
        for g in range(self.num_ghosts):
            if (self.ghost_pos[g] == eyepos):
                #print("ghost pos "+str(self.ghost_pos[g]))
                if DEBUG_MODE:
                    print("see ghost in direction %d,%d" % (x, y))

                return 1.
        if DEBUG_MODE:
            print("no ghost in direction %d,%d" % (x, y))
        return -1.

    # strange, not in liine with paper and does not seem to be used
    # def LocalMove(self,history,stepObs, status) :
    #
    #
    #     numGhosts = np.random.randint(1, 3) # numpy also exclusive high
    #     # // Change
    #     # 1 or 2
    #     # ghosts
    #     # at
    #     # a
    #     # time
    #     for i in range(self.num_ghosts):
    #         g = np.random.randint(self.num_ghosts)
    #         self.ghost_pos[g] = np.random.random()*self.sizeX
    #         if ( not self.check_passablepos(self.ghost_pos[g]) or  self.ghost_pos[g] == (self.agent.x,self.agent.y)):
    #             return False
    #
    #
    #
    #     smellPos=[]
    #     for x in range(self.smell_range):
    #         for y in range(self.smell_range):
    #
    #             pos = (self.agent.x,self.agent.y) + smellPos
    #             if (smellPos != (0, 0) and self.maze[pos] == E_SEED):
    #                 self.Food[Maze.Index(pos)] = Bernoulli(self.food_prob)
    #
    #
    #         # // Just
    #         # check
    #         # the
    #         # last
    #         # time - step, don
    #         # 't check for full consistency
    #         if (history.Size() == 0)
    #             return True
    #
    #         observation = self.setObservation(pocstate)
    #          return history.Back().Observation == observation

    def move_objects_meltingpot(self,g):
        if self.pacman_dynamic==PacmanDynamic.pacman:
            # move ghost
            self.move_ghosts(g)
            return
        elif self.pacman_dynamic==PacmanDynamic.random:
            if self.t % 20 == 0:
                self.move_ghost_random(g)
            return
        elif self.pacman_dynamic==PacmanDynamic.static:
            return
        else:
            raise Exception("dynamic %s does not exist !"%(str(self.pacman_dynamic)))

    def move_ghosts(self, g):

        if manhattan_dist((self.agent.x, self.agent.y), self.ghost_pos[g]) < self.ghost_range:
            if (self.power_steps > 0):
                self.move_ghost_defensive(g)
            else:
                self.move_ghost_aggressive(g)
        else:

            self.move_ghost_random(g)

        return True

    def check_passablepos_ghost(self,pos,g):
        return self.check_passablepos(pos) and not pos in self.ghost_pos[0:g-1]
    def Bernoulli(self,prob):
        return self.rng.rand() < prob
    def move_ghost_aggressive(self, g):
        if (not self.Bernoulli(self.chase_prob)):
            self.move_ghost_random(g)
            return

        bestDist = self.sizeX + self.sizeY
        bestDir = 0  # stay
        bestPos = self.ghost_pos[g]
        x, y = self.ghost_pos[g]
        if DEBUG_MODE:
            print("ghost %d : move aggressive" % (g))
            print("current pos ghost %d : %d,%d" % (g, x, y))
        gdir = VonNeumannNeighbourhoodPlus[self.ghost_direction[g]]
        for dir in range(1, 5):
            direction = VonNeumannNeighbourhoodPlus[dir]
            dist = directional_dist(
                (self.agent.x, self.agent.y), self.ghost_pos[g], direction)
            vx, vy = direction
            newpos = (check_toroid_X(x + vx, self), check_toroid_Y(y + vy, self))
            if (dist <= bestDist and self.check_passablepos(newpos)
                and not opposite_direction(direction, gdir)):
                bestDist = dist
                bestPos = newpos
                bestDir = dir

        self.ghost_pos[g] = bestPos
        self.ghost_direction[g] = bestDir
        if DEBUG_MODE:
            print("new pos ghost %d : %s" % (g, str(self.ghost_pos[g])))

    def move_ghost_defensive(self, g):
        if (self.Bernoulli(self.defensive_slip) and self.ghost_direction[g] >= 0):
            self.ghost_direction[g] = 0
            return

        bestDist = 0
        bestDir = 0
        x, y = self.ghost_pos[g]
        bestPos = self.ghost_pos[g]
        if DEBUG_MODE:
            print("ghost %d : move defensive" % (g))
            print("current pos ghost %d : %d,%d" % (g, x, y))
        gdir = VonNeumannNeighbourhoodPlus[self.ghost_direction[g]]

        for dir in range(1, 5):
            dist = directional_dist((self.agent.x, self.agent.y), self.ghost_pos[g], VonNeumannNeighbourhoodPlus[dir])
            direction = VonNeumannNeighbourhoodPlus[dir]
            vx, vy = VonNeumannNeighbourhoodPlus[dir]
            newpos = (check_toroid_X(x + vx, self), check_toroid_Y(y + vy, self))
            if (dist >= bestDist and self.check_passablepos(newpos)
                and not opposite_direction(direction, gdir)):
                bestDist = dist
                bestPos = newpos
                bestDir = dir

        self.ghost_pos[g] = bestPos
        self.ghost_direction[g] = bestDir
        if DEBUG_MODE:
            print("new pos ghost %d : %s" % (g, str(self.ghost_pos[g])))

    def move_ghost_random(self, g):

        # // Never
        # switch
        # to
        # opposite
        # direction
        # // Currently
        # assumes
        # there
        # are
        # no
        # dead - ends.


        gdir = VonNeumannNeighbourhoodPlus[self.ghost_direction[g]]
        x, y = self.ghost_pos[g]
        if DEBUG_MODE:
            print("ghost %d : move random" % (g))
            print("current pos ghost %d : %d,%d" % (g, x, y))
        choices=set(range(1,5))
        while True:
            dir = self.rng.choice(list(choices),1)[0]
            choices = choices - set([dir])
            vx, vy = VonNeumannNeighbourhoodPlus[dir]

            newpos = (check_toroid_X(x + vx, self), check_toroid_Y(y + vy, self))

            if (not opposite_direction(VonNeumannNeighbourhoodPlus[dir], gdir) and self.check_passablepos(newpos)):
                break
            if not choices:
                self.ghost_direction[g] = 0
                return
        self.ghost_pos[g] = newpos
        if DEBUG_MODE:
            print("new pos ghost %d : %s" % (g, str(self.ghost_pos[g])))
        self.ghost_direction[g] = dir



    # def set_food_locations(self):
    #     # food is randomly located, other mapelements the same over episodes
    #     for x in range(self.sizeX):
    #         for y in range(self.sizeY):
    #             if self.maze[x,y]==POcmanFlags.E_FOOD:
    #                 self.maze[x,y]==POcmanFlags.E_FREE
    #             if self.maze[x,y]==POcmanFlags.E_FREE:
    #                  if Bernoulli(self.food_prob):
    #                      self.maze[x, y] = POcmanFlags.E_FOOD
    def init_pacman(self):
        if self.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS:
            # initialise around a powerpill
            coords = (self.maze == POcmanFlags.E_POWER).nonzero()
            self.agent.x = self.rng.choice(coords[0])
            self.agent.y = self.rng.choice(coords[1])
            pill_coord = (self.agent.x, self.agent.y)
            while True:
                self.agent.x = min(self.sizeX - 1, max(self.agent.x + randint(-1, 1), 0))
                self.agent.y = min(self.sizeY - 1, max(self.agent.y + randint(-1, 1), 0))
                if self.check_passablepos((self.agent.x, self.agent.y)) and (self.agent.x, self.agent.y) != pill_coord:
                    print("agent pos=%d,%d" % (self.agent.x, self.agent.y))
                    print("pill coord=%d,%d" % (pill_coord))
                    return pill_coord
        else:
            (self.agent.x, self.agent.y) = self.pocman_home

    def within_manhattandist(self, coord, coord2, min, max):
        dist = manhattan_dist(coord, coord2)
        return dist >= min and dist <= max

    def random_passable_pos(self):
        while True:
            x = self.rng.randint(0, self.sizeX - 1)
            y = self.rng.randint(0, self.sizeY - 1)
            if self.check_passablepos((x, y)):
                return (x, y)

    def init_ghosts(self, powerpill_location):
        # recall that num_ghosts=0 if task=EAT_FOOD
        if self.currentTask.task_type==PacmanTaskType.EAT_FOOD:
            self.num_ghosts=0
        self.ghost_pos = []
        self.ghost_direction = []
        for g in range(self.num_ghosts):
            if self.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS:
                xx, yy = powerpill_location
                while True:
                    x = min(self.sizeX - 1, max(xx + self.rng.randint(-4, 4), 0))
                    y = min(self.sizeY - 1, max(yy + self.rng.randint(-4, 4), 0))
                    if self.check_passablepos((x, y)) and self.within_manhattandist((x, y), (xx, yy),
                                                                                    self.ghost_range / 2,
                                                                                    self.ghost_range):
                        break

            else:
                x, y = self.ghost_home
                x += g % 2
                y += g / 2
                if not self.check_passablepos((x,y)):
                    raise Exception()
            self.ghost_pos.append((x, y))
            self.ghost_direction.append(-1)
    def set_food(self,num):
        """
        set the number of food:
        if dynamic, then there are only ghosts
        however, if
        :return:
        """
        self.num_ghosts=0
        self.num_poison=0
        self.num_food=num


    def set_poison(self,num):
        self.num_food=0
        self.num_ghosts=0
        self.num_poison=num

    def set_num_ghosts(self,num):
        self.num_poison=0
        self.num_food=0
        self.num_ghosts=num

    def melting_pot_foodlocations(self,topology_type):
        if topology_type==PacmanTopology.cheese:
            return [(3,3),(5,3)]
        elif topology_type==PacmanTopology.pacman_micro:
            return [(1,1),(1,7),(7,1),(7,7)]
        elif topology_type==PacmanTopology.standard:
            return [(6,1),(9,1),(9,6)]
        else:
            raise Exception("topology not used")
    def random_foodlocation(self,topology_type):
        locations=self.melting_pot_foodlocations(topology_type)
        l=self.rng.randint(len(locations))
        return locations[l]
    def init_food(self, pillcoord):
        if self.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS:
            # only one powerpill allowed
            self.num_food = 1
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food(x, y):  # because standardmap has some food in it already
                        self.maze[x, y] = POcmanFlags.E_FREE
                    if pillcoord == (x, y):
                        self.maze[x, y] = POcmanFlags.E_POWER
        elif self.currentTask.task_type == PacmanTaskType.RUN_GHOSTS:
            # no food allowed
            self.num_food = 0
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food(x, y):  # because standardmap has some food in it already
                        self.maze[x, y] = POcmanFlags.E_FREE
        elif self.currentTask.task_type == PacmanTaskType.MELTINGPOT:

            self.initial_object_location=None
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food(x, y):  # because standardmap has some food in it already
                        self.maze[x, y] = POcmanFlags.E_FREE

            for i in range(self.num_food):
                x, y = self.random_foodlocation(self.currentTask.topology_type)
                self.maze[x, y] = POcmanFlags.E_FOOD
                self.object_location=(x,y)
            for i in range(self.num_poison):
                x, y = self.random_foodlocation(self.currentTask.topology_type)
                self.maze[x, y] = POcmanFlags.E_POISON
                self.object_location=(x,y)

        else:
            # FULL or EAT_FOOD: 4 power pills and additional randomly located foods
            self.num_food = 4  # power pills
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food_specific(x, y):  # specific: do not remove powerpills
                        self.maze[x, y] = POcmanFlags.E_FREE
                    if self.currentTask.task_type in [PacmanTaskType.EAT_FOOD,
                                                      PacmanTaskType.FULL]:  # grow food randomly
                        if self.check_free(x, y) and self.Bernoulli(self.food_prob):
                            self.maze[x,y] = POcmanFlags.E_FOOD
                            self.num_food += 1

    def choose_topology(self,topology_type,num_ghosts):
        if topology_type==PacmanTopology.cheese:
            self.cheesemaze_map(num_ghosts)
        elif topology_type==PacmanTopology.standard:
            self.standardmaze_map(num_ghosts)
        elif topology_type==PacmanTopology.pacman_micro:
            self.micro_pacman(num_ghosts)
        elif topology_type==PacmanTopology.pacman_mini:
            self.mini_pacman(num_ghosts)
        elif topology_type==PacmanTopology.pacman_standard:
            self.standard_pacman(num_ghosts)
        else:
            raise Exception("not supported topology")



    def new_level(self):
        self.power_steps = 0
        if self.currentTask.task_type == PacmanTaskType.MELTINGPOT:
            self.new_task_meltingpot()
            self.choose_topology(self.currentTask.topology_type,self.num_ghosts)
        else:
            self.choose_topology(self.currentTask.topology_type)
        pillcoord = self.init_pacman()
        self.init_ghosts(pillcoord)

        self.init_food(pillcoord)


        self.num_ghosts_caught = 0
    def move(self,g):
        if self.currentTask.task_type==PacmanTaskType.MELTINGPOT:
            self.move_objects_meltingpot(g)
        else:
            self.move_ghosts(g)

    def new_task_meltingpot(self):
        """
        call this when a new meltingpot task has been initialised
        :return:
        """
        reward,dynamic,_top=self.currentTask.task_feature
        if DEBUG_MODE:
            print('-------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------')
            print("new task meltingpot:")
            print("topology %s,  dynamic %s, reward %.2f"%(str(self.currentTask.topology_type),str(dynamic),reward))

            print('-------------------------------------------------------------------------')
        self.pacman_dynamic=dynamic
        num_objects=  1
        if PacmanDynamic.is_dynamic(dynamic):
            self.set_num_ghosts(num_objects)
            if reward > 0:
                self.power_steps=float('inf') # chase ghosts until task end
                self.reward_eatghost=reward
            else:
                self.power_steps = 0
                self.reward_die = reward  # chase/escape ghosts


        else:
            if reward > 0:
                self.reward_eatfood=reward
                self.power_steps = 0
                self.set_food(num_objects)
            else:
                self.reward_eatpoison=reward
                self.power_steps = 0
                self.set_poison(num_objects)


defaultmapobjfuns = {}


def reward_fun_pocman(agent, environment):
    if DEBUG_MODE:
        print("rewarding agent")
        print("current location: %d, %d" % (agent.x, agent.y))

    if (environment.power_steps > 0):
        environment.power_steps -= 1
    else:
        # reset the num ghosts hit
        environment.num_ghosts_caught = 0
    if DEBUG_MODE:
        print("power steps=%s" % (environment.power_steps))

    reward = environment.reward_default
    #check hit wall
    if environment.no_move:
        reward += environment.reward_hitwall
        if DEBUG_MODE:
            print("hit wall, %d" % (environment.reward_hitwall))

    # check hit ghost or no (powersteps-> reward, no powersteps--> die)
    hitGhost = -1
    for g in range(environment.num_ghosts):

        if (environment.ghost_pos[g] == (environment.agent.x, environment.agent.y)):
            hitGhost = g
            break
        environment.move(g)
        if (environment.ghost_pos[g] == (environment.agent.x, environment.agent.y)):
            hitGhost = g
            break
    if (hitGhost >= 0):
        if (environment.power_steps > 0):
            if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
                environment.num_ghosts_caught += 1
                reward += environment.reward_eatghost * environment.num_ghosts_caught
                environment.ghost_pos[hitGhost] = environment.ghost_home
                environment.ghost_direction[hitGhost] = 0
            else:
                reward += environment.reward_eatghost

            if DEBUG_MODE:
                print("eat ghost %d, %d" % (g, environment.reward_eatghost * environment.num_ghosts_caught))

        else:
            reward += environment.reward_die
            if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
                environment.terminal = True
                if DEBUG_MODE:
                    print("hit ghost %d -->  DEATH ! %d" % (g, environment.reward_die))
            else:
                if DEBUG_MODE:
                    print("hit ghost %d , %d" % (g, environment.reward_die))

    # observation = environment.setObservation(pocstate)


    x, y = (environment.agent.x, environment.agent.y)
    # check food or powerpill
    if (environment.check_food(x, y)):
        if environment.check_powerpill(x, y):
            environment.power_steps = environment.power_numsteps
            reward += environment.reward_eatpower
            if DEBUG_MODE:
                print("eat power pill, powersteps=%d, %d" % (environment.power_steps, environment.reward_eatpower))
        else:
            if DEBUG_MODE:
                print("eat food, %d" % (environment.reward_eatfood))
            reward += environment.reward_eatfood
        if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
            environment.maze[x, y] = POcmanFlags.E_FREE
            environment.num_food -= 1
    # check poison
    if (environment.check_poison(x, y)):
        reward += environment.reward_eatpoison
        if DEBUG_MODE:
            print("eat poison, %d" % (environment.reward_eatpoison))
        if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
            environment.maze[x, y] = POcmanFlags.E_FREE
            environment.num_food -= 1
    # check food gone or no
    if (environment.num_food == 0):
        if environment.currentTask.task_type in [PacmanTaskType.EAT_FOOD, PacmanTaskType.FULL] or\
                (environment.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS and environment.power_steps == 0):
            reward += environment.reward_clearlevel
            environment.terminal = True
            if DEBUG_MODE:
                print("clear level (all food gone), %d" % (environment.reward_clearlevel))
    # check time up or no
    if environment.time_up():
        # if DEBUG_MODE:
        #print("final time of the elementary task at time %d" % (environment.t))
        environment.terminal=True
    if environment.currentTask.task_type == PacmanTaskType.MELTINGPOT:
        if environment.currentTask.task_feature[0]==1.:
            assert reward==0. or reward==1.
        else:
            assert reward==0 or reward==-1
    return reward

#@ray.remote
class POcmanActor(POcman):
    def __init__(self, agent_params, visual, params, index):

        print("start init")
        POcman.__init__(self, agent_params,visual, params,actor_index=index)
        self.use_stats = False
        self.agent.learner.did_terminal = False
        print("Finished init")


    def hello(self):
        print("hello")
    def set_task(self,currentTask,running_time,t):
        self.currentTask = currentTask
        self.currentTask.initialized = False
        self.currentTask.initialize(self)
        print(self.currentTask)
        self.running_time=running_time
        self.agent.learner.total_t = t
        self.agent.total_t = t
        self.agent.learner.t = 0
        self.agent.t = 0
        self.t = t
        #self.initialize_task(self.currentTask)
    def run_individual_worker(self):

        self.agent.learner.update_time = False
        print("I'm process", getpid(), "starting to run task")
        print(self.currentTask)
        print("initialized task",self.currentTask.initialized)
        self.currentTask.run(self)
        #print("I'm process", getpid(),  " time to update")
        interrupt = self.interrupt
        self.interrupt = False # reset because the task is finished and this means we should pop the next task when resuming
        return self.task_stop() or interrupt
    def task_stop(self):
        print("environment t: "+str(self.t))
        print("environment end_time: " + str(self.currentTask.end_time))
        stop =  self.currentTask.task_stop(self)  # is this is a stop or just time to update?
        if stop:
            print("task stop")
        else:
            print("no task stop yet")
        return stop
    def get_update_states(self):
        if self.agent.learner.did_terminal:
            self.agent.learner.did_terminal=False
            update_s = deepcopy(self.agent.learner.agent.states)
            update_a = deepcopy(self.agent.learner.agent.actions)
            update_r = deepcopy(self.agent.learner.agent.rewards)
            update_states =  True, (update_s,update_a,update_r)
            self.agent.learner.agent.states = []
            self.agent.learner.agent.actions = []
            self.agent.learner.agent.rewards = []
            self.agent.learner.t = 0
            #print("reset t=0")
        else:
            # remove the last appended s,a,r since this was added after setObs
            update_s = deepcopy(self.agent.learner.agent.states[:-1])
            update_a = deepcopy(self.agent.learner.agent.actions[:-1])
            update_r = deepcopy(self.agent.learner.agent.rewards[:-1])
            update_states =  False, (update_s,update_a,update_r)
            #keep only the last for the next update
            self.agent.learner.agent.states = [self.agent.learner.agent.states[-1]]
            self.agent.learner.agent.actions = [self.agent.learner.agent.actions[-1]]
            self.agent.learner.agent.rewards = [self.agent.learner.agent.rewards[-1]]
            #print("continue elementary task")
        return update_states
    # def update(self,terminal):
    #     self.agent.learner.update(terminal)
    def set_weights(self,w):
        self.agent.learner.set_weights(w)
    def initialize_task(self):
        print("in init task")
        print("initialising task")
        self.generateMap()
        self.agent.learner.new_elementary_task()
        self.terminal = False

    def save_agent(self,filename,save_learner):
        if save_learner:
            self.agent.learner.save(filename)
        else:
            self.agent.learner = None

    def load_agent(self, filename):
        self.agent.learner.load(filename)

    def get_time(self):
        return self.t

    def get_statistics(self,filename):
        self.printStatistics()
        self.agent.learner.printPolicy()
        self.agent.learner.save_stats(filename)
    #@overrides
    def printStatistics(self):
        self.agent.learner.printStatistics()

#sizeX and sizeY just are the maximum (for visualisation)
pacmanparams={'sizeX': 19 ,'sizeY':21, 'tasks': [],'observation_size': 1, 'observation_length':16,'sampling_rate': 10000, 'dynamic':False,
              'real_time':False ,'network_types':['external_actions'],'use_stats': True,
               'eval':False,'agent_filename': "pacman_pic.jpg",'include_task_features': False, 'inform_task_time':False,
              'record_intervals': [],'reward_hitwall':0.,'elementary_task_time':1000}




