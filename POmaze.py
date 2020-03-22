
# -life long reinforcement learning environment
# -different tasks are presented to the learner, but he does not know the set of tasks nor which task when
# -Examples: task1: grow and gather food; task2: gather samples; task3: attack simple agents;
# -some tasks may be similar but with slightly different rules; some rule changes may be predictable others not
# -To ensure we are not cheating: different reward functions are specified randomly and distributed randomly across time,
#  with e.g. some penalising and others rewarding the very same behaviours
# -modularity in a neural network should hold advantages
# -sensors: localised pixelmap
# -actuators: N,W,S,E,grab_object,drop_object,eat_object,gather_sample,attack
# -parameters: observability proportion, map size, agents_density
#


import os
import sys

# for entry in sys.path:
#     if entry.startswith("/home/local/software/"):
#         sys.path.remove(entry)


import numpy
print ("numpy version" +str(numpy.__version__))



#sys.path.insert(0,"/usr/lib/python2.7/site-packages/")

#sys.path.append("/home/db2c15/.cache/pip/wheels/d5/5b/93/433299b86e3e9b25f0f600e4e4ebf18e38eb7534ea518eba13")
#sys.path.append('${HOME}')
#
# sys.path.append('${PhDPath}')
# sys.path.append('${PhDPath}Environments/')
# #sys.path.append('${PhDPath}/Theano')
#
#
#
# sys.path.append('${HOME}/peas-master/')

# sys.setrecursionlimit(200000)
from Methods.RandomLearner import RandomLearner
#from Methods.Qlearning import QLambdaLearner

from IS.SSA import SSA_with_WM
from IS.SSA_gradientQ import SSA_gradientQ
#from IS.Predictive_SSA_NEAT import Predictive_SSA_NEAT
# from MultiTaskSSA.MultiTask_LSTM_SSA import Predictive_LSTM_SSA
# from MultiTaskSSA.MultiTaskSSA import PredictiveSSA


from MazeUtils import *


from ExperimentUtils import *

from Environment import *

from Parsers.Parse_Arguments import *
from Parsers.Parse_Configurations import *




from Configs.InstructionSets import *
from Mazes import POmaze, ResetType, reward_fun_POmaze
# from deer.q_networks.q_net_keras import MyQNetwork

from Agents.Agents import NavigationAgent

import itertools

from random import randint


loss_intervals=[(30.0*10**6,32.000*10**6), (79.0*10**6,80*10**6),(80*10**6,80.001*10**6)] # minimum must be greater than zero

NO_SAVING=False

TEST_ITS = 1000000


MAZE_IDs=DIFFICULT_MAZE_IDs

DEBUG_MODE=False

STOPTIME = 5000 #just a default


STOPTIMES = {'Easy':5*10**6,'Medium': 30*10**6 , 'Difficult':80*10**6}

REAL_STOPTIMES = {'Easy': 5*3600,'Difficult':50*3600}

SAMPLING_RATE=10000
REAL_SAMPLING_RATE=30

PROGRAM_CELLS = {'Easy': 50, 'Medium': 75, 'Difficult': 100}

WORKING_CELLS = {'Easy': 80, 'Medium': 100, 'Difficult': 120}



parser.add_argument("-empweight",dest="empirical_weight",type=float)

parser.add_argument("-w",dest="working_memory",type=int)
parser.add_argument("-p",dest="program_cells",type=int)

parser.add_argument("-n",dest="ff",type=ParseBoolean)
parser.add_argument("-a",dest="actions")
parser.add_argument("-q",dest="probabilistic",type=ParseBoolean)
parser.add_argument("-o",dest="freeze",type=ParseBoolean)
parser.add_argument("-x",dest="switch",type=ParseBoolean)
parser.add_argument("-y",dest="inform_mazenum",type=ParseBoolean)
parser.add_argument("-k",dest="k",type=ParseBoolean) # the number of nets in the kNets impl
parser.add_argument("-l",dest="l",type=int) # the reset periodicity in longterm maze
parser.add_argument("-i",dest="i",type=int) # additional inputs
parser.add_argument("-ms",dest="ms",type=float) # additional inputs

args = parser.parse_args()


def getStates():  # return all possible states
    states = []
    if FULLOBS:
            for x in range(13):
                for y in range(13):
                    states.append((x,y))
                return states

    l = list(itertools.product([-1, 1], repeat=4))
    for state in l:
        states.append(tuple(state))
    print('states=' + str(states))
    return states




def reward_fun3(agent,environment):
    # use for real-time
    if eat(agent, environment):
        if environment.remove_food:
            environment.terminal = True
        return 1.
    elif pseudo_eat(agent,environment):
        if environment.remove_food:
            environment.terminal = True
        return .10
    else:
        return -1/(2.0*environment.optimum)
# def reward_fun2(agent,environment):
#     if open_door(agent,environment):
#         environment.reset()
#         return 1.
#     else:
#         return 0.



defaultmapobjfuns={}

reward_fun=reward_fun_POmaze


default_task=NavigationTask(reward_fun,defaultmapobjfuns,STOPTIME)



agent_params={}


defaultparams={'sizeX':MAPSIZEX,'sizeY':MAPSIZEY,'complexity':.10, 'density':.10,
               'tasks': [default_task],'observation_size': 1, 'observation_length':4,'sampling_rate': SAMPLING_RATE, 'dynamic':False,
               'reset_type':ResetType.random, 'real_time':False ,'network_types':['external_actions'],'use_stats': True,
               'eval':False}



inputs=4 if not FULLOBS else 2
externalActions=[ExternalAction(north,0),ExternalAction(south,0),ExternalAction(west,0),ExternalAction(east,0)]
networktypeMap = {'jump':{'jump':1},'external_actions': externalActions}
SSA_WM_Params['num_inputs']=inputs
SSA_WM_Params['actions']=externalActions

#environmentfile='/home/david/PycharmProjects/PhD/POmazeData/SSA_NEAT_RANDOM_new/POmazeCorrection_SSA_NEAT_WMwm50p50random8_environment'


def generateNewTask(time,generate_new=False,maze_ids=FOUR_EASY_MAZE_IDs):

    maze_id = random.choice(range(len(maze_ids)))  # the easy maze indices
    chosen = maze_ids[maze_id]
    mazenum = MAZES[chosen]
    difficulty=DIFFICULTY[mazenum]
    file =  maze_dir+difficulty+'/maze' + str(mazenum)
    #reward_fun, funs, end_time, files = None, environment = None, generate_new = False, maze_id = None, use_stats = False
    return NavigationTask(reward_fun1, defaultmapobjfuns, end_time=time, files=file, generate_new=generate_new,maze_id=maze_id)
def initializeSwitchingTasks(seed,stoptime=None,generate_new=False,start_time=0):

    stop = stoptime if stoptime is not None else STOPTIMES['Medium'] # give more time to learn because of switching
    print("from %d to %d"%(start_time,stop))
    random.seed(seed)
    params=[]
    if generate_new:
        params.append(generateNewTask(stop,generate_new=True))
    else:
        t = SWITCHING_FREQ + start_time
        i=0
        while t <= stop:
            #reward_fun, funs, end_time, files = None, environment = Nonetest_run
            params.append(generateNewTask(t))
            #print("end=%d"%(params[-1].end_time))
            t+=SWITCHING_FREQ
            i+=1
    #print('tasks='+str(params))
    defaultparams['tasks']=params
def get_A2C_configs(inputs,externalActions, filename, episodic):
    return {'num_neurons': 80, 'task_features': [], 'use_task_bias': False,
            'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 15,
            'actions': deepcopy(externalActions), 'episodic': episodic,'file':filename
            }
def get_filename(filename_args,methodname,wm,p,actionstring,probstring,ffstring,freeze_str,run):
    return filename_args + '_' + methodname + wm + p + actionstring + probstring + ffstring + freeze_str + str(run)
def get_exploration_schedule(run):
    recordings = read_incremental("POmazeFinal__SSA_gradientQsequencewm120p100"+str(run)+
                                  "actions_reset(30000000,32000000)_recordings")
    epsilons=recordings['epsilon']
    return {coord:np.mean(eps) for coord,eps in epsilons.items()}
def main(arg_list):
    #e = pickle.load(open("/home/david/PycharmProjects/PhD/Environments/POmazeFinal_LSTM_SSA_with_WM_deltasuncertainty38actions_reset_environment"))
    # stats=pickle.load(open("/home/david/PycharmProjects/PhD/Environments/POmazeFinalTest_SSA_NEAT_WMwm120p100ff20both_noreset_stats_object"))
    # for i in range(30):
    #     start_coords=pickle.load(open(maze_dir+"/"+DIFFICULTY[MAZES[i]]+"/maze"+str(MAZES[i])+"startCoords","rb"))
    #
    #     feasible = pickle.load(open(maze_dir + "/" + DIFFICULTY[MAZES[i]] + "/maze" + str(MAZES[i]) + "_feasible", "rb"))
    #     print("%d / %d "%(len(start_coords),len(feasible)))
    #     startCoord = pickle.load(
    #         open(maze_dir + "/" + DIFFICULTY[MAZES[i]] + "/maze" + str(MAZES[i]) + "start", "rb"))
    #     foodCoord = pickle.load(
    #         open(maze_dir + "/" + DIFFICULTY[MAZES[i]] + "/maze" + str(MAZES[i]) + "end", "rb"))


    # print (sys.version)

    #print(e.agent.learner.printStack())
    # args.filename = "POmazeExploration_"
    # args.config_file="actions_reset"
    # args.method="DRQN"
    # args.environment_file=False
    ms=args.ms if args.ms is not None else 10
    freeze=False
    freeze_str=''
    if args.freeze!=None:
        freeze=args.freeze
    if freeze:
        internalActionsSSA_WM['freeze']=2
    # else:
    #     freeze_str='no_freeze'

    walltime=60*3600# 60 hours by default
    if args.walltime:
        ftr = [3600, 60, 1]
        walltime = sum([a * b for a, b in zip(ftr, map(int, args.walltime.split(':')))])

    print(walltime)

    probabilistic=False
    probstring = ''
    if args.probabilistic:
        probabilistic=True
        probstring='prob'
    ffstring=''
    ff=True
    if args.ff==True:
        ff=True
        ffstring='ff'

    actionstring=''
    if args.actions=='random':
        SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATrandom
        actionstring='random'

    methodname = args.method if args.method is not None else "DRQN"


    run=args.run if args.run is not None else 50
    switching=False
    if args.switch is not None:
        switching = args.switch
    inform_mazenum=False
    if args.inform_mazenum is not None:
        inform_mazenum=args.inform_mazenum
    difficulty = getDifficultySwitch() if switching else getDifficulty(run)
    statfreq = 1*10 ** 6 if difficulty == "Easy" else 16 * 10 ** 6
    wm = ''

    stoptime=STOPTIMES[difficulty] if args.STOPTIME is None else args.STOPTIME
    defaultparams['stoptime']=stoptime


    if methodname.startswith("SSA"):
        if args.working_memory and methodname in ['SSA_WM','SSA_NEAT_WM']:
            SSA_WM_Params['wm_cells'] = args.working_memory
        else:
            SSA_WM_Params['wm_cells'] = WORKING_CELLS[difficulty]
        wm = 'wm' + str(SSA_WM_Params['wm_cells'])
    p = ''
    if  methodname.startswith("SSA"):
        if args.program_cells:
            SSA_WM_Params['num_program'] = args.program_cells

        else:
            SSA_WM_Params['num_program'] = PROGRAM_CELLS[difficulty]
        p = 'p' + str(SSA_WM_Params['num_program'])
    print(SSA_WM_Params)

    filename=''

    if args.filename:
        filename=get_filename(args.filename,methodname,wm,p,actionstring,probstring,ffstring,freeze_str,run)

    if args.config_file is not None:

        configfile=str(os.environ['HOME']) + '/PycharmProjects/PhD/Configs/IS-NEAT-Singlemaze_'+args.config_file+".ini"
        parse_config_file(filename,configfile,defaultparams)
        filename+=args.config_file


    setSSA_WM_Params( SSA_WM_Params)

    setIS_NEAT_params(defaultparams, SSANeat_Params,networktypeMap)
    print(defaultparams)
    print(SSA_WM_Params)
    print(SSANeat_Params)
    # if SSA_WM_Params['eval']:
    #     internalActionsSSA_WM.update(searchPset)
    # elif SSA_WM_Params['predictiveSelfMod']:
    #     internalActionsSSA_WM.update(predictiveSelfMod)
    # else:
    #     internalActionsSSA_WM.update(incPset)
    internalActionsSSA_WM.update(incPset)
    num_PLAs=0
    for key in internalActionsSSA_WM:
        if key in ['searchP','incP','decP','inc_means','dec_means','sample']:
            num_PLAs+=1

    visual=False
    args.record_video=False
    if args.VISUAL:
        visual=args.VISUAL
        print(visual)

    defaultparams['record_intervals']=None
    if args.record_video:
        visual = True
        defaultparams['record_intervals'] = [[1+80 * 10 ** 6, 80.005 * 10 ** 6]]
        print('record intervals set')
        #defaultparams['record_intervals'] = get_record_intervals(STOPTIMES[difficulty])
    #defaultparams['record_intervals']=[[.4*10**6,0.405*10**6]]


    reward_func=reward_fun3 if defaultparams['real_time'] else reward_fun

    if args.l is not None:
        defaultparams['reset_period']=args.l
        filename+="l"+str(args.l)


    print(defaultparams)
    print(internalActionsSSA_WM)


    environmentfile=None
    explor_schedule = False
    if args.environment_file:
        environmentfile=filename

    else:
        if filename.startswith("POmazeExploration"):
            environmentfile=get_filename("Exploration/POmaze30Mil_",methodname,wm,p,actionstring,probstring,ffstring,freeze_str,run)
            environmentfile+=args.config_file
            explor_schedule=get_exploration_schedule(run)
            print(environmentfile)
        if filename.startswith("POmazeTest"):
            environmentfile=get_filename("POmazeFinal_",methodname,wm,p,actionstring,probstring,ffstring,freeze_str,run)
            environmentfile+=args.config_file


    if args.run_type is not None and args.run_type.startswith("stats"):
        print("reading environmentfile:"+str(environmentfile))
        e=read_incremental("/media/david/BackupDrive/POmazeFinalData/"+environmentfile+"_environment")
        print("file read")
        # except:
        #     e = getEnvironmentSFTP(environmentfile)
        if args.run_type=="stats":
            getStatistics(e,filename)
            return
        elif args.run_type == "statsP":
            from StatsAndVisualisation.Statistics import PolType
            e.printStatistics(PolType.P_MATRIX)
            return
        elif args.run_type == "statsN":
            from StatsAndVisualisation.Statistics import PolType
            e.printStatistics(PolType.NETWORK)
            return
        elif args.run_type == "statsEpsilon":
            for interval in loss_intervals:
                recordings_file='/home/david/PycharmProjects/PhD/Environments/'+environmentfile
                for stat in e.statistics.values():
                    stat.make_epsilon_map(e,
                                     maze_dir='/home/david/PycharmProjects/PhD/Environments/Mazes/',
                                     recordings_file=recordings_file,time_interval=interval)
            print("")

            return
        else:
            from StatsAndVisualisation.Statistics import PolType
            e.printStatistics(PolType.NETWORK)





    #filename="POmazeFinal__DRQN80actions_reset"
    #environmentfile=filename

    if environmentfile is not None:
         print("reading enviroment file")

         # Getting back the objects:
         e=read_incremental(environmentfile+'_environment')
         e.agent.learner.load(environmentfile)
         e.agent.learner.continue_experiment(intervals=loss_intervals)
         if difficulty=="Difficult":
            stoptime=81*10**6
         else:
            stoptime=5*10**6
         if filename.startswith("POmaze30Mil"):
             stoptime=32.001*10**6
         if explor_schedule:
             e.agent.learner.exploration_schedule=explor_schedule
             print("exploration schedule set:")
             print(explor_schedule)
         if filename.startswith("POmazeExploration"):
             stoptime=40*10**6
         if filename.startswith("POmazeTest"):
             stoptime=82.1*10**6


         e.stoptime=stoptime

            #print(e.agent.learner.Stack)
            #print(e.agent.learner.Pol.summary())



         print("starting at "+str(e.t))

         #e.agent.learner.displayNetPolicy()
         e.start = time.time()
         # if args.run_type == "test":
         #     print("preparing test run")
         #     if switching:
         #         initializeSwitchingTasks(seed=run, stoptime=e.t+TEST_ITS, generate_new=GENERATE_NEW,start_time=e.t)
         #     else:
         #         initializeDefaultNavigationTask(filename,default_task,defaultparams,run,SAMPLING_RATE if not defaultparams['real_time'] else REAL_SAMPLING_RATE,
         #                                         stoptime,reward_func)
         #     e.set_tasks(defaultparams['tasks'],statfreq)
         #
         # else:
         if not (switching and GENERATE_NEW) or e.interrupt: #else the new task is already in place
             if stoptime: #assuming one task
                 print("a")
                 e.currentTask.end_time=stoptime
                 print(e.currentTask.end_time)
             else:
                 print("b")
                 e.currentTask.end_time = STOPTIMES[difficulty]
                 e.currentTask.initialized=True

             e.set_tasks([e.currentTask] + e.tasks,statfreq)

         if args.VISUAL:
             # use these lines to convert old-style stack
             #from Stack import Stack
             #e.agent.learner.Stack = Stack.listToStack(e.agent.learner.Stack)
             for action in e.agent.learner.actions:
                 print(action)
                 print(action.n_args)
             e.rng = np.random.RandomState(run)
             e.visual = True
             print(defaultparams['record_intervals'])
             e.initVisualisation( defaultparams['record_intervals'], filename+"_video")
             if defaultparams['record_intervals']:
                 e.vis.on=False
             print("video initialised")



         print(e.start)

    else:
            #config = NEATGenotype(parseGenome(n_input,n_output,NEATGenotype))
            # create environment



            n_input = inputs
            if switching:
                inform_mazenum=True
                initializeSwitchingTasks(seed=run, stoptime=stoptime,generate_new=GENERATE_NEW)
            else:
                initializeDefaultNavigationTask(filename,default_task, defaultparams, run,
                                                SAMPLING_RATE if not defaultparams['real_time'] else REAL_SAMPLING_RATE,
                                                stoptime, reward_func)



            if methodname=='SSA_WM':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=20
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                method = SSA_with_WM(**SSA_WM_Params)

            elif methodname=='SSA_gradientQ':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQ)
                method = SSA_gradientQ(len(externalActions),trace_length,input_addresses,ConversionType.double_index,SSA_WM_Params)
            elif methodname=='SSA_gradientQ2':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(0,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQ)
                method = SSA_gradientQ(len(externalActions),trace_length,input_addresses,ConversionType.double_index,SSA_WM_Params)
            elif methodname=='SSA_gradientQsequence':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence)
                method = SSA_gradientQ(len(externalActions),trace_length,input_addresses,ConversionType.double_index,
                                       SSA_WM_Params,intervals=loss_intervals)
                method.Qlearner.batch_size=32
                print("batch size:"+str(method.Qlearner.batch_size))
            elif methodname=='SSA_gradientQsequence_greedy':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequencenoeps)
                method = SSA_gradientQ(len(externalActions),trace_length,input_addresses,ConversionType.double_index,SSA_WM_Params)
            elif methodname=='SSA_gradientQsequence_nomodification':
                enhance_PLA=0
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence)
                del SSA_WM_Params['internal_actionsSSA']['incP']
                del SSA_WM_Params['internal_actionsSSA']['decP']
                method = SSA_gradientQ(len(externalActions),trace_length,input_addresses,ConversionType.double_index,SSA_WM_Params)
            elif methodname=='SSA_gradientQsequence_fixedexperience':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence_fixedexperience)
                method = SSA_gradientQ(len(externalActions),trace_length,input_addresses,ConversionType.double_index,SSA_WM_Params)
            elif methodname=='SSA_gradientQsequence_internalgreedy':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA=0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval']=1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA=0
                else:
                    enhance_PLA=18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequencenoeps)
                method = SSA_gradientQ(len(externalActions),trace_length,
                                       input_addresses,ConversionType.double_index,SSA_WM_Params,Q_internal=True)
            elif methodname == 'SSA_gradientQsequence_internal':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA = 0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval'] = 1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA = 0
                else:
                    enhance_PLA = 18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA / num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = " + str(SSA_WM_Params['enhance_PLA']))
                input_addresses = range(4, 8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence)
                method = SSA_gradientQ(len(externalActions), trace_length, input_addresses,
                                       ConversionType.double_index,SSA_WM_Params,Q_internal=True)
            elif methodname == 'SSA_gradientQsequence_direct':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA = 0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval'] = 1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA = 0
                else:
                    enhance_PLA = 18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA / num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = " + str(SSA_WM_Params['enhance_PLA']))
                input_addresses = range(4, 8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence)
                method = SSA_gradientQ(len(externalActions), trace_length, input_addresses,
                                       ConversionType.direct,SSA_WM_Params)
            elif methodname == 'SSA_gradientQsequence_notrainreplay':
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA = 0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval'] = 1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA = 0
                else:
                    enhance_PLA = 18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence_notrainreplay)
                method = SSA_gradientQ(len(externalActions), trace_length, input_addresses,
                                       ConversionType.direct,SSA_WM_Params,fixed_training=True,intervals=loss_intervals)
                method.Qlearner.batch_size=32

            elif methodname=='RandomLearner':
                method=RandomLearner(externalActions,filename)

            elif methodname=="DRQN":

                from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
                task_features=[]
                #batch_size=32
                n_input=inputs

                trace_length=40 if difficulty == "Difficult" else 25
                use_task_bias=False
                use_task_gain=False
                epsilon_change=True
                method = DRQN_Learner(task_features,use_task_bias,use_task_gain,n_input,trace_length,externalActions,
                                      filename,episodic=False,loss=None,num_neurons=50,epsilon_change=epsilon_change,target_model=True)
                method.agent.batch_size=32
                print("batch size:"+str(method.agent.batch_size))
            elif methodname == "A2C2":
                from Catastrophic_Forgetting_NNs.A2C_Learner2 import A2C_Learner
                settings = get_A2C_configs(inputs, externalActions, filename, True)
                method = A2C_Learner(**settings)
            else:
                if methodname!='':
                    raise Exception("methodname %s not found"%(methodname))
                if filename.startswith("POmazeFinalPrepEval"):
                    enhance_PLA = 0
                    del SSA_WM_Params['internal_actionsSSA']['endSelfMod']
                    SSA_WM_Params['internal_actionsSSA']['prepEval'] = 1
                elif filename.startswith("POmazeFinalNoDupl"):
                    enhance_PLA = 0
                else:
                    enhance_PLA = 18
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not defaultparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                input_addresses=range(4,8)
                from IS.SSA_gradientQ import ConversionType
                trace_length = 40 if difficulty == "Difficult" else 25
                SSA_WM_Params['internal_actionsSSA'].update(internalActionsGradientQsequence_notrainreplay)
                method = SSA_gradientQ(len(externalActions), trace_length, input_addresses,
                                       ConversionType.direct,SSA_WM_Params,fixed_training=True,intervals=loss_intervals)
                method.Qlearner.batch_size=32
            e = POmaze(NavigationAgent(method,defaultparams),visual,switching,defaultparams)
            e.set_tasks(defaultparams['tasks'],statfreq)

    if args.run_type == "create_mazes":
        e.createMazes()

    e.run(walltime)
    if args.run_type=="test":
	    return

    continue_experiment(e.interrupt,arg_list)
    save_stats=not e.interrupt
    print("save stats %s"%(save_stats))
    #e, filename, arg_list, no_saving, args, save_stats = True, save_learner = True
    finalise_experiment(e, filename, arg_list, NO_SAVING, args, save_stats=save_stats)





if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'
    #
    # with PyCallGraph(output=graphviz):
    main(sys.argv[1:])

