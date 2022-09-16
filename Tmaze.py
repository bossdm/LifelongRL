

import numpy
import os
import sys

print ("numpy version" +str(numpy.__version__))



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
from Methods.RandomLearner import RandomLearner
#from Methods.Qlearning import QLambdaLearner


from MazeUtils import *

from dill import load,dump
from Environment import *
from mapobjects import *
from Configs.InstructionSets import *
from Parsers.Parse_Arguments import *
from Parsers.Parse_Configurations import *

from IS.SSA import SSA_with_WM
from Actions.SpecialActions import ExternalAction
from MazeUtils import *


from Configs.Defaults import *
from Mazes import TMaze
# from deer.q_networks.q_net_keras import MyQNetwork


from Agents.Agents import NavigationAgent

import itertools

from random import randint

#from deer.q_networks.q_net_keras import MyQNetwork


TEST_ITS = 15000


MAZE_IDs=DIFFICULT_MAZE_IDs

DEBUG_MODE=True

STOPTIME = 50000 #just a default


STOPTIMES = {'Easy':5*10**6,'Medium': 30*10**6 , 'Difficult': 80*10**6}

REAL_STOPTIMES = {'Easy': 5*3600,'Difficult':50*3600}

SAMPLING_RATE=10000
REAL_SAMPLING_RATE=30

PROGRAM_CELLS = {'Easy': 50, 'Medium': 75, 'Difficult': 100}

WORKING_CELLS = {'Easy': 80, 'Medium': 100, 'Difficult': 120}





parser.add_argument("-i",dest="internal",type=ParseBoolean) #only in NEAT: NN outputs internal actions as well

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


corridor_length=3
sizeX=corridor_length+1+2


inputs=3




def reward_fun1(agent,environment):
    if eat(agent,environment):
        environment.terminal=True
        return 1.
    else:
        return 0

def reward_fun2(agent,environment):
    print(environment.currentCount)
    if not isinstance(agent.learner.chosenAction,ExternalAction):
        return 0.
    environment.actionCount+=1
    if environment.actionCount==environment.maxNrTakenActions:
        environment.actionCount=0
        environment.terminal=True

    # use for T-maze
    if eat(agent,environment):
        environment.success_rate=( (environment.NSUCCESS-1)*environment.success_rate + 1 )/float(environment.NSUCCESS)
        environment.terminal=True
        if DEBUG_MODE:
            print("reached goal")
        environment.actionCount=0
        return 4.
    elif pseudo_eat(agent,environment):
        # pseudofood eat, failure
        if DEBUG_MODE:
            print("max count--> failure")
        environment.success_rate = ((environment.NSUCCESS - 1) * environment.success_rate + 0) / float(
            environment.NSUCCESS)
        environment.terminal = True
        return -1.0
    else:
        if agent.learner.chosenAction.function == west:
            return 0.
        elif agent.learner.chosenAction.function == east:
            return 0.
        else:
            if DEBUG_MODE:
                print("no move")
            return -.10

        return None



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

defaultmapobjfuns={}
reward_fun=reward_fun2

default_task=NavigationTask(reward_fun,defaultmapobjfuns,STOPTIME)


externalActions=[ExternalAction(north,0),ExternalAction(south,0),ExternalAction(west,0),ExternalAction(east,0)]


networktypeMap = {'jump':{'jump':1},'external_actions': externalActions}


inputs=3

defaultparams={'sizeX':sizeX,'sizeY':MAPSIZEY,'observation_size':1,'observation_length':inputs,
               'tasks': [default_task],'sampling_rate': SAMPLING_RATE, 'dynamic':False,'use_stats':False,
                'eval': False, 'real_time':False, 'reset_type': ResetType.fixed}
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
    fileprefix =  os.environ["HOME"] + "/LifelongRL/Mazes/"+difficulty + '/maze' + str(maze_num)

    if stoptime:
        defaultparams['stoptime'] = stoptime
        default_task.end_time =stoptime
    else:

        default_task.end_time = STOPTIMES[difficulty] if not real_time else REAL_STOPTIMES[difficulty]
        defaultparams['stoptime'] = STOPTIMES[difficulty]

    defaultparams['sampling_rate']=SAMPLING_RATE if not real_time else REAL_SAMPLING_RATE
    default_task.files=fileprefix
    default_task.maze_id=run%30
    if defaultparams['real_time']:
        default_task.reward_fun=reward_fun2

    defaultparams['tasks']=[default_task]
    defaultparams['record_intervals']=None


def generateNewTask(time,generate_new=False,maze_ids=FOUR_EASY_MAZE_IDs):

    maze_id = random.choice(range(len(maze_ids)))  # the easy maze indices
    chosen = maze_ids[maze_id]
    mazenum = MAZES[chosen]
    difficulty=DIFFICULTY[mazenum]
    file =  maze_dir+difficulty+'/maze' + str(mazenum)
    #reward_fun, funs, end_time, files = None, environment = None, generate_new = False, maze_id = None, use_stats = False
    return NavigationTask(reward_fun2, defaultmapobjfuns, end_time=time, files=file, generate_new=generate_new,maze_id=maze_id,use_stats=True)
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
        elif methodname=="DRQN":
            from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
            task_features = []
            # batch_size=32
            n_input = inputs
            trace_length = 30
            use_task_bias = False
            use_task_gain = False
            epsilon_change = True
            method = DRQN_Learner(task_features, use_task_bias, use_task_gain, n_input, trace_length, externalActions,
                                  filename, episodic=True, loss=None, target_model=True,num_neurons=50, epsilon_change=epsilon_change)
        else:
            from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
            task_features = []
            # batch_size=32
            n_input = inputs
            trace_length = 15
            use_task_bias = False
            use_task_gain = False
            epsilon_change = True
            method = DRQN_Learner(task_features, use_task_bias, use_task_gain, n_input, trace_length, externalActions,
                                  filename, episodic=True, loss=None, target_model=True,num_neurons=50, epsilon_change=epsilon_change)

        defaultparams["filename"]=filename

        e = TMaze(NavigationAgent(method,defaultparams),visual,switching,defaultparams)
        e.set_tasks(defaultparams['tasks'],statfreq=1*10**6)
        if inform_mazenum:
            e.inform_mazenum=True
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