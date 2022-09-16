
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

import numpy
import os
import sys

print ("numpy version" +str(numpy.__version__))

maze_dir = str(os.environ['HOME']) + '/PycharmProjects/PhD/Environments/Mazes/'
sys.path.insert(0, str(os.environ['HOME']) + '/PycharmProjects/PhD/LSTM')
sys.path.insert(0, str(os.environ['HOME']) + '/PycharmProjects/PhD/Configs')
sys.path.insert(0, str(os.environ['HOME']) + '/PycharmProjects/PhD/Parsers')
sys.path.insert(0,'/home/db2c15/.local/lib/python2.7/site-packages')

#sys.path.append("/home/db2c15/.cache/pip/wheels/d5/5b/93/433299b86e3e9b25f0f600e4e4ebf18e38eb7534ea518eba13")
# sys.path.append('${HOME}')
#
# sys.path.append('${PhDPath}')
# sys.path.append('${PhDPath}Environments/')
# #sys.path.append('${PhDPath}/Theano')
#
#
#
# sys.path.append('${HOME}/peas-master/')
print(sys.path)
sys.setrecursionlimit(20000)
from RandomLearner import RandomLearner
#from Methods.Qlearning import QLambdaLearner

from IS.SSA_Neat import *

from MazeUtils import *

from dill import load,dump
from Environment import *
from InstructionSets import *
from Parse_Arguments import *
from Parse_Configurations import *




from Defaults import *
from Mazes import POmaze
# from deer.q_networks.q_net_keras import MyQNetwork

from peas.networks.rnn import *
from RL_LSTM import RL_LSTM
from RL_LSTM_nonepisodic import RL_LSTM_nonepisodic
from Agents.Agents import NavigationAgent

import itertools

from random import randint

#from deer.q_networks.q_net_keras import MyQNetwork


from Child.Child import *


TEST_ITS = 15000


MAZE_IDs=DIFFICULT_MAZE_IDs

DEBUG_MODE=False

STOPTIME = 50000 #just a default


STOPTIMES = {'Easy':5*10**6,'Medium': 30*10**6 , 'Difficult': 80*10**6}

REAL_STOPTIMES = {'Easy': 5*3600,'Difficult':50*3600}

SAMPLING_RATE=10000
REAL_SAMPLING_RATE=30

PROGRAM_CELLS = {'Easy': 50, 'Medium': 75, 'Difficult': 100}

WORKING_CELLS = {'Easy': 80, 'Medium': 100, 'Difficult': 120}


parser.add_argument("-w",dest="working_memory",type=int)
parser.add_argument("-p",dest="program_cells",type=int)

parser.add_argument("-n",dest="ff",type=ParseBoolean)
parser.add_argument("-a",dest="actions")
parser.add_argument("-q",dest="probabilistic",type=ParseBoolean)
parser.add_argument("-o",dest="freeze",type=ParseBoolean)
parser.add_argument("-x",dest="switch",type=ParseBoolean)
parser.add_argument("-y",dest="inform_mazenum",type=ParseBoolean)
parser.add_argument("-k",dest="k",type=ParseBoolean)



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
    return states
    print('states=' + str(states))

def reward_fun1(agent,environment):
    if eat(agent,environment):
        environment.terminal=True
        return 1.
    else:
        return 0

def reward_fun3(agent,environment):
    # use for real-time
    if eat(agent, environment):
        return 1.
    else:
        return -1/(2.0*environment.optimum)
# def reward_fun2(agent,environment):
#     if open_door(agent,environment):
#         environment.reset()
#         return 1.
#     else:
#         return 0.



defaultmapobjfuns={}
reward_fun=reward_fun1
sizeX=MAPSIZEX

default_task=NavigationTask(reward_fun,defaultmapobjfuns,STOPTIME)

defaultparams={'sizeX':sizeX,'sizeY':MAPSIZEY,'complexity':.10, 'density':.10,
               'tasks': [default_task],'observation_size': 1, 'sampling_rate': SAMPLING_RATE, 'dynamic':False,
               'random_reset':False, 'real_time':False}
externalActions=[ExternalAction(north,0),ExternalAction(south,0),ExternalAction(west,0),ExternalAction(east,0)]


networktypeMap = {'jump':{'jump':1},'external_actions': externalActions}


inputs=4
SSA_WM_Params['num_inputs']=inputs
SSA_WM_Params['actions']=externalActions

#environmentfile='/home/david/PycharmProjects/PhD/POmazeData/SSA_NEAT_RANDOM_new/POmazeCorrection_SSA_NEAT_WMwm50p50random8_environment'

def getDifficulty(run):
    maze_num = MAZES[run % 30]
    print("maze " + str(maze_num))
    return DIFFICULTY[maze_num]
def getDifficultySwitch():
    return 'Easy'

def initializeDefaultTasks(run,stoptime=None,fileprefix=None,real_time=False):
    maze_num = MAZES[run % 30]  # 1 + (run-1)%15
    difficulty = DIFFICULTY[maze_num]
    fileprefix =  maze_dir+difficulty + '/maze' + str(maze_num)
    if stoptime:
        default_task.end_time =stoptime
    else:
        default_task.end_time = STOPTIMES[difficulty] if not real_time else REAL_STOPTIMES[difficulty]

    defaultparams['sampling_rate']=SAMPLING_RATE if not real_time else REAL_SAMPLING_RATE
    default_task.files=fileprefix
    default_task.maze_id=run%30
    if defaultparams['real_time']:
        default_task.reward_fun=reward_fun3

    defaultparams['tasks']=[default_task]

def generateNewTask(time,generate_new=False,maze_ids=FOUR_EASY_MAZE_IDs):

    maze_id = random.choice(range(len(maze_ids)))  # the easy maze indices
    chosen = maze_ids[maze_id]
    mazenum = MAZES[chosen]
    difficulty=DIFFICULTY[mazenum]
    file =  maze_dir+difficulty+'/maze' + str(mazenum)
    #reward_fun, funs, end_time, files = None, environment = None, generate_new = False, maze_id = None, use_stats = False
    return NavigationTask(reward_fun1, defaultmapobjfuns, end_time=time, files=file, generate_new=generate_new,maze_id=maze_id,use_stats=True)
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
            #reward_fun, funs, end_time, files = None, environment = None
            params.append(generateNewTask(t))
            #print("end=%d"%(params[-1].end_time))
            t+=SWITCHING_FREQ
            i+=1
    #print('tasks='+str(params))
    defaultparams['tasks']=params



def submit_job(arg_list):
    arg_string =''
    found=False
    for i in range(len(arg_list)):
        if i > 0 and arg_list[i-1] == '-e':
            arg_string+='True'
            found=True
        else:
            arg_string+=arg_list[i]
        arg_string+=' '
    if not found:
        arg_string+='-e True'
    print(arg_string)
    os.system("export HOME=/home/db2c15")
    os.system("sh ${HOME}/POmazescript.sh "+arg_string)

def main(arg_list):
    print (sys.version)
    freeze=False
    freeze_str=''
    if args.freeze!=None:
        freeze=args.freeze
    if freeze:
        internalActionsSSA_WM['freeze']=2
    # else:
    #     freeze_str='no_freeze'

    walltime=60*3600 # 60 hours by default
    if args.walltime:
        ftr = [3600, 60, 1]
        walltime = sum([a * b for a, b in zip(ftr, map(int, args.walltime.split(':')))])

    print(walltime)
    #parse_config_file("IS-NEAT-Singlemaze.ini")
    probabilistic=False
    probstring = ''
    if args.probabilistic:
        probabilistic=True
        probstring='prob'
    ffstring=''
    ff=False
    if args.ff==True:
        ff=True
        ffstring='ff'

    actionstring=''
    if args.actions=='random':
        SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATrandom
        actionstring='random'
    methodname = args.method
    filename = 'new'

    run=args.run if args.run is not None else 666
    switching=False
    if args.switch is not None:
        switching = args.switch
    inform_mazenum=False
    if args.inform_mazenum is not None:
        inform_mazenum=args.inform_mazenum
    difficulty = getDifficultySwitch() if switching else getDifficulty(run)
    wm = ''


    if methodname in ['SSA_WM', 'SSA_NEAT_WM','SSA_WM_FixedNN']:
        if args.working_memory and methodname in ['SSA_WM','SSA_NEAT_WM']:
            SSA_WM_Params['wm_cells'] = args.working_memory
        else:
            SSA_WM_Params['wm_cells'] = WORKING_CELLS[difficulty]
        wm = 'wm' + str(SSA_WM_Params['wm_cells'])
    p = ''
    if  methodname in ['SSA_WM','SSA_NEAT_Environments/WM','SSA_WM_FixedNN']:
        if args.program_cells:
            SSA_WM_Params['num_program'] = args.program_cells

        else:
            SSA_WM_Params['num_program'] = PROGRAM_CELLS[difficulty]
        p = 'p' + str(SSA_WM_Params['num_program'])
    print(SSA_WM_Params)
    filename=''
    if args.filename:
        filename = args.filename+'_'+methodname+wm+p+actionstring+probstring+ffstring+freeze_str+str(run)

    if args.config_file is not None:
        configfile=str(os.environ['HOME']) + '/PycharmProjects/PhD/Configs/'+args.config_file
        filename=parse_config_file(filename,configfile,defaultparams)

        setSSA_WM_Params(defaultparams, SSA_WM_Params)

        setIS_NEAT_params(defaultparams, SSANeat_Params,networktypeMap)
    print(defaultparams)
    print(SSA_WM_Params)
    print(SSANeat_Params)
    if SSA_WM_Params['eval']:
        internalActionsSSA_WM.update(searchPset)
    else:
        internalActionsSSA_WM.update(incPset)
    num_PLAs=0
    for key in internalActionsSSA_WM:
        if key in ['searchP','incP','decP']:
            num_PLAs+=1

    visual=False
    if args.VISUAL:
        visual=args.VISUAL
        print(visual)


    #method = RandomLearner(externalActions)
    print(defaultparams)
    print(internalActionsSSA_WM)

    #method = SSA_with_WM(**SSA_WM_Params)
    environmentfile = None
    if args.environment_file:
        environmentfile=filename+'_environment'

    if environmentfile:
         print("reading enviroment file")
         # Getting back the objects:
         e = load(open(environmentfile))
         #e.agent.learner.displayNetPolicy()
         e.start = time.time()
         if args.test_run:
             print("preparing test run")
             if switching:
                 initializeSwitchingTasks(seed=run, stoptime=e.t+TEST_ITS, generate_new=GENERATE_NEW,start_time=e.t)
             else:
                 initializeDefaultTasks(run=run, stoptime=e.t+TEST_ITS,real_time=e.real_time)
             e.tasks = defaultparams['tasks']
             if args.VISUAL:

                 e.visual=True
                 e.initVisualisation()
         else:
             if not (switching and GENERATE_NEW) or e.interrupted: #else the new task is already in place
                 if args.STOPTIME: #assuming one task
                     e.currentTask.end_time=args.STOPTIME
                 else:
                     e.currentTask.end_time = STOPTIMES[difficulty]
                     e.currentTask.initialized=True
                     e.tasks = [e.currentTask] + e.tasks


         print(e.start)

    else:
    #config = NEATGenotype(parseGenome(n_input,n_output,NEATGenotype))
    # create environment
        defaultparams['seed'] = run
        n_input = inputs
        if switching:
            inform_mazenum=True
            initializeSwitchingTasks(seed=run, stoptime=args.STOPTIME,generate_new=GENERATE_NEW)
        else:
            initializeDefaultTasks(run=run, stoptime=args.STOPTIME,real_time=defaultparams['real_time'])
        # if FULLOBS:
        #     SSA_WM_Params['num_inputs'] = 3 if switching and inform_mazenum else 2
        # else:
        #     SSA_WM_Params['num_inputs'] = 5 if switching and inform_mazenum else 4
        # # create agent
        x = randint(0, MAPSIZEX - 1)
        y = randint(0, MAPSIZEY - 1)
        if methodname=='SSA_WM':
            SSA_WM_Params['filename'] = filename
            SSA_WM_Params['enhance_PLA'] = 20/num_PLAs - 1 if not defaultparams['real_time'] else 0
            method = SSA_with_WM(**SSA_WM_Params)
        elif methodname=='RandomLearner':
            method=RandomLearner(externalActions,filename)


        elif methodname=='SSA_NEAT_WM':
            n_input = SSA_WM_Params['num_inputs'] + 4 +SSA_WM_Params['additional_inputs']
            SSA_WM_Params['enhance_PLA'] = 16/num_PLAs - 1 if not defaultparams['real_time'] else 0

            SSA_WM_Params['filename'] = filename
            config = {'inputs': n_input, 'types': ['sigmoid'],
                      'input_type': 'ident',
                      'probabilistic': probabilistic, 'feedforward': ff,
                      'topology': None}
            configs=getIS_NEAT_configs(config,SSANeat_Params)
            #SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATfixed

            # self, SSA_args, SSA_NEAT_args, config, network_type
            method = SSA_NEAT_WM(SSA_WM_args=SSA_WM_Params, SSA_NEAT_args=SSANeat_Params, configs=configs)
        elif methodname=='SSA_NEAT_WM_Multi':
            SSANeat_Params['internal_actionsNEAT'].update(internalActionsMulti)
            n_input = SSA_WM_Params['num_inputs'] + 4 +SSA_WM_Params['additional_inputs']
            SSANeat_Params['use_setnet'] = True
            SSA_WM_Params['enhance_PLA'] = 16/num_PLAs - 1 if not defaultparams['real_time'] else 0

            SSA_WM_Params['filename'] = filename

            config = {'inputs': n_input, 'instruction_set':SSANeat_Params['instruction_sets'], 'types': ['sigmoid'],
                      'input_type': 'ident',
                      'probabilistic': probabilistic, 'feedforward': ff,
                      'topology': None}
            configs = getIS_NEAT_configs(config,SSANeat_Params)
            # self, SSA_args, SSA_NEAT_args, config, network_type
            method = SSA_NEAT_WM(SSA_WM_args=SSA_WM_Params, SSA_NEAT_args=SSANeat_Params, configs=configs)
        elif methodname=='SSA_NEAT_WM_kNets':
            if not inform_mazenum:
                SSANeat_Params['internal_actionsNEAT'].update(internalActions_kNets)
            SSANeat_Params['use_setnet']=True
            n_input = SSA_WM_Params['num_inputs'] + 4 +SSA_WM_Params['additional_inputs']
            SSA_WM_Params['enhance_PLA'] = 16/num_PLAs - 1 if not defaultparams['real_time'] else 0

            SSA_WM_Params['filename'] = filename
            k = args.k if args.k else NUM_MAZES
            config = {'inputs': n_input, 'instruction_set':SSANeat_Params['instruction_sets'], 'types': ['sigmoid'],
                      'input_type': 'ident',
                      'probabilistic': probabilistic, 'feedforward': ff,
                      'topology': None, 'k': k}
            configs = getIS_NEAT_configs(config,SSANeat_Params)
            # self, SSA_args, SSA_NEAT_args, config, network_type
            method = SSA_NEAT_WM(SSA_WM_args=SSA_WM_Params, SSA_NEAT_args=SSANeat_Params, configs=configs)
        elif methodname=='SSA_WM_FixedNN':
            SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATfixed
            n_input = SSA_WM_Params['num_inputs'] + 4 +SSA_WM_Params['additional_inputs']
            SSA_WM_Params['enhance_PLA'] = 18/num_PLAs - 1 if not defaultparams['real_time'] else 0

            SSA_WM_Params['filename'] = filename

            config = {'inputs': n_input, 'instruction_set':SSANeat_Params['instruction_sets'], 'types': ['sigmoid'],
                      'input_type': 'ident',
                      'probabilistic': probabilistic,'feedforward': ff,
                      'topology': [10,10]}
            configs = getIS_NEAT_configs(config,SSANeat_Params)
            # self, SSA_args, SSA_NEAT_args, config, network_type
            method = SSA_NEAT_WM(SSA_WM_args=SSA_WM_Params, SSA_NEAT_args=SSANeat_Params, configs=configs)

        # elif methodname == 'LSTM_Q_prior':
        #
        #     LSTMnet = NN
        #     seq_length=1
        #     inputdims = [(seq_length,4)]
        #     qnetwork = MyQNetwork(inputdims, len(externalActions),
        #                           Parameters.RHO,
        #                           Parameters.RMS_EPSILON,
        #                           Parameters.MOMENTUM,
        #                           Parameters.CLIP_DELTA,
        #                           Parameters.FREEZE_INTERVAL,
        #                           Parameters.BATCH_SIZE,
        #                           Parameters.UPDATE_RULE,
        #                           Parameters.RANDOM_STATE,
        #                           Parameters.DOUBLE_Q,
        #                           LSTMnet)
        #
        #     method = DeerLearner([inputdims], qnetwork, externalActions, filename, batch_type='prioritised',exp_priority=1.)
        #     method.states = getStates()
        # elif methodname == 'LSTM_Q_prior_rew':
        #
        #     LSTMnet = NN
        #     seq_length=100
        #     inputdims = [(seq_length,4+1)]
        #     if FULLOBS:
        #         inputdims = [(seq_length, 3)]
        #     qnetwork = MyQNetwork(inputdims, len(externalActions),
        #                           Parameters.RHO,
        #                           Parameters.RMS_EPSILON,
        #                           Parameters.MOMENTUM,
        #                           Parameters.CLIP_DELTA,
        #                           Parameters.FREEZE_INTERVAL,
        #                           Parameters.BATCH_SIZE,
        #                           Parameters.UPDATE_RULE,
        #                           Parameters.RANDOM_STATE,
        #                           Parameters.DOUBLE_Q,
        #                           LSTMnet)
        #
        #     method = DeerLearner(inputdims, qnetwork, externalActions, filename, batch_type='prioritised',exp_priority=1.,reward_as_input=True)
        #     method.states = getStates()
        # elif methodname == 'QLambda':
        #     states = getStates()
        #     method = QLambdaLearner(n_input=n_input, actions=externalActions,file=filename,states=states, alpha=0.02, horizon=101,gamma=0.95, qlambda=0.90,epsilon=.05)
        # elif methodname == 'LSTMQ':
        #     states = getStates()
        #     method = QLambdaLearner(n_input=n_input, actions=externalActions, file=filename, states=states, alpha=0.0002, horizon=2, gamma=0.95,
        #                             qlambda=0,network=True,epsilon=.10,stateful=True)
        elif methodname=="RL_LSTM":
            method=RL_LSTM(n_input,externalActions,filename)
        elif methodname=="RL_LSTM_nonepisodic":
            method = RL_LSTM_nonepisodic(n_input, externalActions, filename)
        elif methodname == 'CHILD':
            rew=True
            method = CHILD(actions=externalActions, n_inputs=n_input, reward_as_input=rew,file = filename)
        else:
            method = RL_LSTM(n_input, externalActions, filename)
            # method = RL_LSTM(n_input,externalActions,filename)

            # #method = RandomLearner(externalActions, filename)
            # # states = getStates()
            # from Methods.LSTM_RL import RL_LSTM
            # method = RL_LSTM(n_input,externalActions,"",alpha=.2)
            # method = QLambdaLearner(n_input=n_input, actions=externalActions, file=filename, states=states, alpha=0.0002,
            #                 horizon=101, gamma=0.95,
            #                 qlambda=0, network=True, epsilon=.05, stateful=False)
            # states = getStates()
            # method = QLambdaLearner(n_input=n_input, actions=externalActions, file=filename, states=states, alpha=0.080,
            #                 horizon=101, gamma=0.95,
            #                 qlambda=0, network=True, epsilon=.05, stateful=False)
        defaultparams["filename"]=filename

        e = POmaze(NavigationAgent(method,x,y,defaultparams),visual,switching,defaultparams)
        if inform_mazenum:
            e.inform_mazenum=True
            if methodname=='SSA_NEAT_WM_kNets':
                e.inform_NP=True
    print("starting from "+ str(e.t))
    #print("will run until " + str(stoptime))

    print("real time " + str(e.start))


    # run environment
    e.run(walltime)
    if args.test_run:
        return
    if not e.real_time and e.t < e.currentTask.end_time:
        submit_job(arg_list)


    # Saving the objects:
    begintime=time.time()
    dump(e, open(filename+'_environment', "w"))
    time_passed=time.time() - begintime
    print("save time=%.3f" % (time_passed))



if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'
    #
    # with PyCallGraph(output=graphviz):
        main(sys.argv[1:])