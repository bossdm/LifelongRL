from Methods.RandomLearner import RandomLearner
#from Methods.Qlearning import QLambdaLearner

from IS.SSA import SSA_with_WM
from POcmanUtils import *
from MazeUtils import *

from ExperimentUtils import *
#import pickle
from Environment import *
#from InstructionSets import *
from Parsers.Parse_Arguments import parser, ParseBoolean
from Parsers.Parse_Configurations import *


from Agents.Agents import PacmanAgent

from Configs.InstructionSets import *

# from deer.q_networks.q_net_keras import MyQNetwork

from Agents.Agents import NavigationAgent

import itertools

from random import randint




NO_SAVING=False

TEST_ITS = 1000000


MAZE_IDs=DIFFICULT_MAZE_IDs

DEBUG_MODE=False

STOPTIME=5*10**6

REAL_STOPTIMES = 100*3600

SAMPLING_RATE=10000
REAL_SAMPLING_RATE=30

PROGRAM_CELLS = 150
TRANSFER_DIMS = 10
WORKING_CELLS = 150


SSA_WM_Params['additional_inputs']=5


parser.add_argument("-w",dest="working_memory",type=int)
parser.add_argument("-p",dest="program_cells",type=int)

parser.add_argument("-n",dest="ff",type=ParseBoolean)
parser.add_argument("-a",dest="actions")
parser.add_argument("-q",dest="probabilistic",type=ParseBoolean)

parser.add_argument("-i",dest="i",type=int) # additional inputs

args = parser.parse_args()





defaultmapobjfuns={}



pocman_task=NavigationTask(reward_fun_pocman,defaultmapobjfuns,STOPTIME)



pacmanparams={'sizeX': 19 ,'sizeY':21,'tasks': [pocman_task],'observation_size': 1,'observation_length':16, 'sampling_rate': SAMPLING_RATE, 'dynamic':False,
              'real_time':False ,'network_types':['external_actions'],'use_stats': True,
               'eval':False,'agent_filename': "pacman_pic.jpg",'inform_task_time':True,'include_task_features': True}







externalActions=[ExternalAction(north,0),ExternalAction(south,0),ExternalAction(west,0),ExternalAction(east,0)]
networktypeMap = {'jump':{'jump':1},'external_actions': externalActions}

SSA_WM_Params['actions']=externalActions

#environmentfile='/home/david/PycharmProjects/PhD/POmazeData/SSA_NEAT_RANDOM_new/POmazeCorrection_SSA_NEAT_WMwm50p50random8_environment'
def calc_velocity_random():
    with open("randomR.txt","rb")  as f:
        R = [int(line) for line in f]


    eatfood_velocity=(R[25]-R[0])/float(250000)
    runghost_velocity=(R[50]-R[25])/float(250000)
    chase_ghost_velocity=(R[75]-R[50])/float(250000)
    full_velocity=(R[125]-R[75])/float(500000)
    return [eatfood_velocity,runghost_velocity,chase_ghost_velocity,full_velocity]
def catastrophicforgetting_tasks(T):
    tasks = [[0. / 2], [1. / 2], [2. / 2]]
    topology=PacmanTopology.pacman_micro
    chase_ghost_task=MultiTask(topology_type=topology,task_feature=tasks[0],task_type=PacmanTaskType.CHASE_GHOSTS,
                               reward_fun=reward_fun_pocman,funs=None,start_time=0,end_time=2*T/5)
    eatfood_task=MultiTask(topology_type=topology,task_feature=tasks[1],task_type=PacmanTaskType.EAT_FOOD,
                           reward_fun=reward_fun_pocman,funs=None,start_time=2*T/5,end_time=4*T/5)
    test_chase_ghost=MultiTask(topology_type=topology,task_feature=tasks[2],task_type=PacmanTaskType.CHASE_GHOSTS,
                               reward_fun=reward_fun_pocman,funs=None,start_time=4*T/5,end_time=T)
    pacmanparams['tasks']=[chase_ghost_task,eatfood_task,test_chase_ghost]

    weights=[2*T/5, 2*T/5, T / 5]
    return tasks, weights

def curriculum_tasks(T):
    feature_labels=["task_type","ghost_range", "chase_prob", "power_num_steps"]
    tasks = [[0. / 3,],
             [1. / 3,],
             [2. / 3,],
             [3. / 3,]
             ]

    topology=PacmanTopology.pacman_micro
    # eatfood_task1 = MultiTask(topology_type=topology,task_feature=tasks[0],task_type=PacmanTaskType.EAT_FOOD,
    #                           reward_fun=reward_fun_pocman, funs=None,start_time=0,end_time=T/20)
    # run_ghost_task1=MultiTask(topology_type=topology,task_feature=tasks[1],task_type=PacmanTaskType.RUN_GHOSTS,
    #                           reward_fun=reward_fun_pocman,funs=None,start_time=T/20, end_time=T/10)
    # chase_ghost_task1=MultiTask(topology_type=topology,task_feature=tasks[2],task_type=PacmanTaskType.CHASE_GHOSTS,
    #                             reward_fun=reward_fun_pocman,funs=None,start_time=T/10,end_time=3*T/20)
    full_task1=MultiTask(topology_type=topology,task_feature=tasks[3],task_type=PacmanTaskType.FULL,
                         reward_fun=reward_fun_pocman,funs=None,start_time=0,end_time=5*T/20)

    eatfood_task2 = MultiTask(topology_type=topology,task_feature=tasks[0],task_type=PacmanTaskType.EAT_FOOD,
                              reward_fun=reward_fun_pocman, funs=None,start_time=5*T/20,end_time=6*T/20)
    run_ghost_task2=MultiTask(topology_type=topology,task_feature=tasks[1],task_type=PacmanTaskType.RUN_GHOSTS,
                              reward_fun=reward_fun_pocman,funs=None,start_time=6*T/20,end_time=7*T/20)
    chase_ghost_task2=MultiTask(topology_type=topology,task_feature=tasks[2],task_type=PacmanTaskType.CHASE_GHOSTS,
                                reward_fun=reward_fun_pocman,funs=None,start_time=7*T/20,end_time=8*T/20)
    full_task2=MultiTask(topology_type=topology,task_feature=tasks[3],task_type=PacmanTaskType.FULL,
                         reward_fun=reward_fun_pocman,funs=None,start_time=7*T/20,end_time=T/2)

    test_task=MultiTask(topology_type=topology,task_feature=tasks[3],task_type=PacmanTaskType.FULL,
                        reward_fun=reward_fun_pocman,funs=None,start_time=T/2,end_time=T)
    pacmanparams['tasks']=[full_task1,
                           eatfood_task2,run_ghost_task2,chase_ghost_task2,full_task2,
                           test_task]

    weights=[T/10,    T/10,       T/10,     3*T/5]
    velocities=calc_velocity_random()
    return tasks,weights,velocities




def main(arg_list):
    # e = pickle.load(open("/home/david/PycharmProjects/PhD/Environments/POmazeFinalTest_SSA_WMwm120p100ff20actions_reset_stats_object"))
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

    V1,V2,V3,V4=calc_velocity_random()

    # else:
    #     freeze_str='no_freeze'

    walltime=60*3600 # 60 hours by default
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
    ff=False
    if args.ff==True:
        ff=True
        ffstring='ff'

    actionstring=''
    if args.actions=='random':
        SSANeat_Params['internal_actionsNEAT'] = internalActionsNEATrandom
        actionstring='random'

    methodname = args.method if args.method is not None else ""


    run=args.run if args.run is not None else 0
    pacmanparams['seed']=run

    wm = ''

    stoptime=STOPTIME if args.STOPTIME is None else args.STOPTIME

    #catastrophicforgetting_tasks(stoptime)
    features,weights,velocities=curriculum_tasks(stoptime)
    task_inputs =  0 if not pacmanparams['include_task_features'] else len(features[0])
    time_inputs = 0 if not pacmanparams['inform_task_time'] else 1
    pacmanparams['observation_length'] += task_inputs + time_inputs
    inputs=pacmanparams['observation_length']
    SSA_WM_Params['num_inputs'] = pacmanparams['observation_length']
    SSA_WM_Params['wm_cells']=WORKING_CELLS
    SSA_WM_Params['num_program']=PROGRAM_CELLS
    # if methodname.startswith("SSA")  or methodname.startswith("PredictiveSSA"):
    #     if args.working_memory and methodname in ['SSA_WM','SSA_NEAT_WM']:
    #         SSA_WM_Params['wm_cells'] = args.working_memory
    #     else:
    #         SSA_WM_Params['wm_cells'] = WORKING_CELLS
    #     wm = 'wm' + str(SSA_WM_Params['wm_cells'])
    # p = ''
    # if  methodname.startswith("SSA") or methodname.startswith("PredictiveSSA"):
    #     if args.program_cells:
    #         SSA_WM_Params['num_program'] = args.program_cells
    #
    #     else:
    #         SSA_WM_Params['num_program'] = PROGRAM_CELLS
    #     p = 'p' + str(SSA_WM_Params['num_program'])
    print(SSA_WM_Params)
    filename='POcman'
    if args.filename:
        filename = args.filename+'_'+methodname+actionstring+probstring+ffstring+str(run)

    if args.config_file is not None:

        configfile=str(os.environ['HOME']) + '/PycharmProjects/PhD/Configs/IS-NEAT-Singlemaze_'+args.config_file+".ini"
        parse_config_file(filename,configfile,pacmanparams)
        filename+=args.config_file


    setSSA_WM_Params( SSA_WM_Params)

    setIS_NEAT_params(pacmanparams, SSANeat_Params,networktypeMap)
    print(pacmanparams)
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
    if args.VISUAL:
        visual=args.VISUAL
        print(visual)

    pacmanparams['record_intervals'] = None
    if args.record_video:
        visual = True
        pacmanparams['record_intervals'] = get_record_intervals(STOPTIME)
    #pacmanparams['record_intervals']=[[.4*10**6,0.405*10**6]]

    #method = RandomLearner(externalActions)
    print(pacmanparams)
    print(internalActionsSSA_WM)

    #method = SSA_with_WM(**SSA_WM_Params)
    environmentfile=None
    if args.environment_file:
        environmentfile=filename+'_environment'
    # environmentfile = "/media/david/Acer/POmazeFinalData/POmazeFinal_RL_LSTM_nonepisodicff87actions_noreset_environment"
    #args.run_type = "stats"
    if args.run_type == "stats":
        #environmentfile="POmazeFinal_SSA_NEAT_WMwm120p100ff55actions_noreset_environment"
        #
        #e=getEnvironmentSFTP(environmentfile)
        #e.reset_type=ResetType.fixed
        #
        # try:
        print("reading environmentfile:"+str(environmentfile))
        with open(environmentfile,"rb") as f:
            e=pickle.load(f)
        print("file read")
        # except:
        #     e = getEnvironmentSFTP(environmentfile)
        getStatistics(e,filename)
        return




    if environmentfile is not None:
         print("reading enviroment file")

         # Getting back the objects:
         with open(environmentfile,"rb") as f:
            e = pickle.load(f)
            if hasattr(e.agent.learner,"action_model"):
                e.agent.learner.action_model.load(filename+'_actionmodel.h5')
            if hasattr(e.agent.learner,"evalPol"):
                e.agent.learner.evalPol.load(filename+'_evalPol.h5')

         print("starting at "+str(e.t))

         #e.agent.learner.displayNetPolicy()
         e.start = time.time()
         if args.run_type == "test":
             print("preparing test run")
             if switching:
                 initializeSwitchingTasks(seed=run, stoptime=e.t+TEST_ITS, generate_new=GENERATE_NEW,start_time=e.t)
             else:
                 initializeDefaultNavigationTask(filename,default_task,pacmanparams,run,SAMPLING_RATE if not pacmanparams['real_time'] else REAL_SAMPLING_RATE,
                                                 stoptime,reward_func)
             e.tasks = pacmanparams['tasks']
             if args.VISUAL:

                 e.visual=True
                 e.initVisualisation( pacmanparams['record_intervals'], filename+"_video")
         else:
             if not (switching and GENERATE_NEW) or e.interrupted: #else the new task is already in place
                 if stoptime: #assuming one task
                     print("a")
                     e.currentTask.end_time=stoptime
                     print(e.currentTask.end_time)
                 else:
                     print("b")
                     e.currentTask.end_time = STOPTIME
                     e.currentTask.initialized=True
                     e.tasks = [e.currentTask] + e.tasks
             if args.VISUAL:
                 e.visual = True
                 e.initVisualisation( pacmanparams['record_intervals'], filename+"_video")


         print(e.start)

    else:
            #config = NEATGenotype(parseGenome(n_input,n_output,NEATGenotype))
            # create environment



            n_input = inputs


            # if FULLOBS:
            #     SSA_WM_Params['num_inputs'] = 3 if switching and inform_mazenum else 2
            # else:
            #     SSA_WM_Params['num_inputs'] = 5 if switching and inform_mazenum else 4
            # # create agent
            x = randint(0, MAPSIZEX - 1)
            y = randint(0, MAPSIZEY - 1)
            if methodname=='SSA_WM':
                enhance_PLA=20
                SSA_WM_Params['filename'] = filename
                SSA_WM_Params['enhance_PLA'] = enhance_PLA/num_PLAs if not pacmanparams['real_time'] else 0
                print("enhance PLA = "+str(SSA_WM_Params['enhance_PLA']))
                method = SSA_with_WM(**SSA_WM_Params)
            elif methodname=='RandomLearner':
                method=RandomLearner(externalActions,filename)

            elif methodname == "DRQN":

                from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
                task_features = [1]
                batch_size = 32
                n_input = 4
                trace_length = 1
                use_task_bias = True
                use_task_gain = True
                method = DRQN_Learner(task_features, use_task_bias, use_task_gain, batch_size, n_input, trace_length,
                                      externalActions, filename, episodic=False, loss=None)
            else:

                from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
                task_features = [1]
                batch_size = 32
                n_input = 4
                trace_length = 50
                use_task_bias = True
                use_task_gain = True
                # (self, task_features, use_task_bias, use_task_gain, batch_size,n_inputs, trace_length, actions, file, episodic,
                #  loss = None)
                method = DRQN_Learner(task_features, use_task_bias, use_task_gain, n_input, trace_length,
                                      externalActions, filename, episodic=True, loss=None)
            #
            e = POcman(PacmanAgent(method,pacmanparams),visual,pacmanparams)

    e.run(walltime)

    contin = continue_experiment(e.interrupt, arg_list)
    save_stats = not contin
    finalise_experiment(e, filename, arg_list, NO_SAVING, args, save_stats=save_stats)




if __name__ == '__main__':
    main(sys.argv[1:])
