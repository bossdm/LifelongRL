"""
Experiment in which agent has to run on many environment
"""



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


import random

import numpy

print ("numpy version" +str(numpy.__version__))


from itertools import product


from Parsers.Parse_Arguments import parser, ParseBoolean
from Parsers.Parse_Configurations import *


from ExperimentUtils import *
from POcmanUtils import *



loss_intervals=[(30.0*10**6,31.000*10**6), (89.0*10**6,90.0*10**6)] # minimum must be greater than ze

NO_SAVING=False

TEST_ITS = 1000000



DEBUG_MODE=False

REAL_STOPTIMES = 60*60*100 # 100 hours

TOTAL_TASK_TIME = 20*10**6
BLOCK_LENGTH = 200000 # 200 repetitions of the same task (1000 per elementary task)


meltingpot_rewards=[-1.,1.]

def ParseTask (task):

    if task in ["chase_ghosts","CHASE_GHOSTS"]:
        return PacmanTaskType.CHASE_GHOSTS
    elif task in ["run_ghosts","run_GHOSTS"]:
        return PacmanTaskType.RUN_GHOSTS
    elif task in ["eat_food","EAT_FOOD"]:
        return PacmanTaskType.EAT_FOOD
    elif task in ["meltingpot","MELTINGPOT"]:
        return PacmanTaskType.MELTINGPOT
    elif task in ["full","FULL"]:
        return PacmanTaskType.FULL
    else:
        raise Exception("task type not recognised")



parser.add_argument("-w",dest="working_memory",type=int)
parser.add_argument("-p",dest="program_cells",type=int)

parser.add_argument("-P",dest="policies",type=int)
parser.add_argument("-x",dest="experiment_type",type=str) # lifelong, initial, interference or test
parser.add_argument("-F",dest="task",type=ParseTask)  # eat_food,run_ghosts,chase_ghosts, or full (in case NOT lifelong)
parser.add_argument("-C",dest="conservative",type=ParseBoolean)
parser.add_argument("-l",dest="load_file",type=int) # load file
parser.add_argument("-i",dest="i",type=int) # additional inputs
parser.add_argument("-a",dest="add_task",type=ParseBoolean)

args = parser.parse_args()


defaultmapobjfuns={}

agent_params={}



def generate_feature_weights(features,max_ind):
    num_features=len(features)
    min_prob = 0.5/float(num_features-1)
    max_prob = 0.5
    max_feature=features[max_ind]
    weights={}
    for feature in features:
        if feature==max_feature:
            weights[feature] = max_prob
        else:
            weights[feature]=min_prob
    return weights
def uniform_feature_weights(features):
    N_F=len(features)
    weights={F:1./N_F for F in features}
    return weights

def initialise_single_task(task_type,current_feature,runtime,environment):
    #task = get_task_type_transferexp(task_type)
    task=MultiTask(topology_type=PacmanTopology.pacman_micro, task_feature=current_feature,
                           task_type=task_type,
                           reward_fun=get_reward_fun(current_feature), start_time=environment.t, end_time=environment.t + runtime, funs={})
    return task
def init_agent(learner,params):
    return PacmanAgent(learner,params)


def initialise_meltingpot_environments(poc_agent,visual,cheesemaze_params,POmaze_params,POcman_params):
    cheesemaze=CheeseMaze(poc_agent,visual,False,cheesemaze_params)
    pomaze=StandardMaze(poc_agent,visual,False,POmaze_params)
    pocman=POcman(poc_agent,visual,POcman_params)
    return cheesemaze,pomaze,pocman
def initialise_meltingpot(poc_agent,visual,POcman_params):
    pocman=POcman(poc_agent,visual,POcman_params)
    return pocman

def get_random_feature(num_topologies,dynamic_options,minV,maxV):
    """

    :param num_topologies:
    :param dynamic_options: list of dynamic options; currently choice between true and false (1 or 0);
    :param minV:
    :param maxV:
    :return:
    """
    V_r = random.uniform(minV, maxV)
    V_r = round(V_r) if abs(V_r) > 0.50 else numpy.sign(V_r)
    topology = float(random.randint(0, num_topologies - 1))
    dynamic = random.choice(dynamic_options)
    return np.array((V_r,dynamic,topology))
def get_featureset(r_range,dynamic_options,topologies):
    """
    return a finite feature set
    :param cardinality: number of elements in the set
    :param V_range: range of V_r
    :param num_topologies: number of topologies
    :return: finite feature set (list)
    """
    lists=[r_range,dynamic_options,topologies]
    l=list(product(*lists))
    return l
def similarity(F1,F2,Fsize):
    """
    return the similarity of two feature vectors
    :return:
    """

    d=manhattan_dist_highdim(F1,F2,Fsize)
    return 1 - d

def feature_probability(previous_feature,feature_set):
    """
    return the probability of a feature vector in the feature set, based on similarity to feature vector in previous task
    :return:
    """
    min = feature_set.min(axis=0)
    max = feature_set.max(axis=0)
    Fsize=max-min
    similarities=np.array([similarity(feature_set[i], previous_feature,Fsize) for i in range(len(feature_set))])
    C=sum(similarities)
    return similarities/C
def sample_task_similaritybased(previous_feature,feature_set):
    """
    sample task according to their similarity
    :param previous_task:
    :param featureset:
    :return:
    """
    probabilities=feature_probability(previous_feature,feature_set)
    i = np.random.choice(len(feature_set),1,p=probabilities)
    return feature_set[i][0]
def sample_task_probabilitybased(feature_weights):
    """
    sample task according to their similarity
    :param previous_task:
    :param featureset:
    :return:
    """
    feature_set=list(feature_weights.keys())
    probabilities=list(feature_weights.values())
    index = np.random.choice(len(feature_set),1,p=probabilities)
    return list(feature_set[index[0]])
def sample_task_index(feature_weights):
    """
    sample task according to their similarity
    :param previous_task:
    :param featureset:
    :return:
    """
    feature_set=list(feature_weights.keys())
    probabilities=list(feature_weights.values())
    index = np.random.choice(len(feature_set),1,p=probabilities)
    return index[0]
# def get_task_type(feature):
#     """
#     return the task type for the given feature (topology==3 is the main indicator)
#     :param feature:
#     :return:
#     """
#     topology=feature[-1]
#     if topology==2:
#         return PacmanTaskType.MELTINGPOT
#     else:
#         return None
def get_task_type(feature):
    """
    return the task type for the given feature (topology==3 is the main indicator)
    :param feature:
    :return:
    """
    return PacmanTaskType.MELTINGPOT

# def get_reward_fun(feature):
#     """
#     return the reward function
#     :param feature:
#     :return:
#     """
#     topology=feature[-1]
#     if topology==0:
#         #cheesemaze
#         return reward_fun_meltingpot
#     elif topology==1:
#         #POmaze
#         return reward_fun_meltingpot
#     elif topology==2:
#         #POcman
#         return reward_fun_pocman
#
#     else:
#         raise Exception("not supported topology")
def get_reward_fun(feature):
    return reward_fun_pocman
def initialise_tasks(time_per_task,T,feature_set):
    #(V ^ r, dynamic, topology)
    current_feature=random.choice(feature_set) # random initial feature
    old_t = 0
    t=time_per_task
    tasks=[]
    num_tasks=T//time_per_task
    for i in range(num_tasks):
        # (task_feature, task_type, reward_fun, funs, start_time, end_time, topology_type=None,
        #                  files=None, environment=None, generate_new=False, maze_id=None):


        tasks.append(MultiTask(topology_type=current_feature[-1],task_feature=current_feature,
                            task_type=get_task_type(current_feature),
                               reward_fun=get_reward_fun(current_feature),start_time=old_t,end_time=t,funs={}))
        current_feature=sample_task_similaritybased(current_feature,feature_set)
        t+=time_per_task
        old_t += time_per_task
    return tasks
def create_single_task(current_feature,old_t,t):
    return MultiTask(topology_type=current_feature[-1], task_feature=current_feature,
              task_type=get_task_type(current_feature),
              reward_fun=get_reward_fun(current_feature), start_time=old_t, end_time=t, funs={})
def initialise_tasks_featureweights(time_per_task,T,task_weights):


    #(V ^ r, dynamic, topology)
    current_feature=list(random.choice(list(task_weights.keys())))# random initial feature
    old_t = 0
    t=time_per_task
    tasks=[]
    num_tasks=T//time_per_task
    for i in range(num_tasks):
        # (task_feature, task_type, reward_fun, funs, start_time, end_time, topology_type=None,
        #                  files=None, environment=None, generate_new=False, maze_id=None):


        tasks.append(create_single_task(current_feature,old_t,t))
        current_feature=sample_task_probabilitybased(task_weights)
        t+=time_per_task
        old_t += time_per_task
    return tasks
def initialise_tasks_featureweightsIndexes(time_per_task,T,task_weights):


    #(V ^ r, dynamic, topology)
    unique_tasks=len(task_weights)
    current_featureIndex=random.randint(0,unique_tasks-1)# random initial feature

    features=task_weights.keys()
    current_feature = features[current_featureIndex]
    old_t = 0
    t=time_per_task
    tasks=[[] for i in range(unique_tasks)]
    num_tasks=T//time_per_task
    for i in range(num_tasks):
        # (task_feature, task_type, reward_fun, funs, start_time, end_time, topology_type=None,
        #                  files=None, environment=None, generate_new=False, maze_id=None):


        tasks[0].append(MultiTask(topology_type=current_feature[-1],task_feature=current_feature,
                            task_type=get_task_type(current_feature),
                               reward_fun=get_reward_fun(current_feature),start_time=old_t,end_time=t,funs={}))
        current_featureIndex=sample_task_index(task_weights)
        current_feature = features[current_featureIndex]

        for run in range(1,unique_tasks):
            i=(current_featureIndex+run)%unique_tasks
            current_feature=features[i]
            tasks[run].append(MultiTask(topology_type=current_feature[-1], task_feature=current_feature,
                                      task_type=get_task_type(current_feature),
                                      reward_fun=get_reward_fun(current_feature), start_time=old_t, end_time=t,
                                      funs={}))
        t+=time_per_task
        old_t += time_per_task
    return tasks


def run_singleenv(tasks,POcman,walltime,arg_list,filename):
    """


    :param tasks:
    :param features:
    :param cheese:
    :param PO:
    :param POcman:
    :param walltime:
    :param arg_list:
    :param filename:
    :return:
    """
    POcman.set_tasks(tasks,stat_freq=18*10**6)
    POcman.run(walltime)
    if POcman.agent.learner.testing:
        return
    interrupted=POcman.t < POcman.stoptime

    save_stats = not interrupted
    finalise_experiment(POcman, filename, arg_list, NO_SAVING, args, save_stats=save_stats,
                        save_learner=True)
    continue_experiment(interrupted, arg_list, job_script='bash lifelong_experiments.sh ')  # if tasks remain, continue


def save_tasks(tasks,feature_weights,tag,task_filename):
    with open(tag+task_filename+'_tasks', 'wb') as file:
        pickle.dump(tasks,file, protocol=pickle.HIGHEST_PROTOCOL)
    with open(tag+task_filename+'_featureweights','wb') as file2:
        pickle.dump(feature_weights,file2,protocol=pickle.HIGHEST_PROTOCOL)

def create_tasks_lifelong(task_filename,tasktime,T,run,num_tasks,extended=False,tag=''):
    if extended:
        topologies, dynamic_opts, reward_range = convert_run_to_setting_x(run, num_tasks)
    else:
        topologies, dynamic_opts, reward_range = convert_run_to_setting(run, num_tasks)
    feature_set = get_featureset(reward_range, dynamic_opts, topologies)
    feature_weights = uniform_feature_weights(feature_set)
    tasks = initialise_tasks_featureweights(tasktime, T, feature_weights)
    save_tasks(tasks,feature_weights,tag,task_filename)


def create_tasks_lifelong_uniform(task_filename,tasktime,T,num_tasks,extended=False,tag=''):
    run=0
    if extended:
        topologies, dynamic_opts, reward_range = convert_run_to_setting_x(run, num_tasks)
    else:
        topologies, dynamic_opts, reward_range = convert_run_to_setting(run, num_tasks)
    feature_set = get_featureset(reward_range, dynamic_opts, topologies)
    feature_weights = uniform_feature_weights(feature_set)
    taskslist = initialise_tasks_featureweightsIndexes(tasktime, T, feature_weights)
    i=0
    for tasks in taskslist:
        save_tasks(tasks, feature_weights, tag, add_run_tag(task_filename,i))
        i+=1


def initialise_and_run_lifelong(filename,arg_list,walltime,agent,
                                visual,POcman_params,tasks,feature_weights):

    #cheese,PO,POcman=initialise_meltingpot_environments(agent,visual,cheesemaze_params,POmaze_params,POcman_params)

    POcman = initialise_meltingpot(agent, visual, POcman_params)

    if len(tasks) > 1:
        if hasattr(POcman.agent.learner,"set_tasks"):
            POcman.agent.learner.set_tasks(feature_weights)
    if POcman_params["num_actors"] > 1:
        return POcman,tasks,walltime,arg_list,filename
    run_singleenv(tasks,POcman,walltime,arg_list,filename)

def run_single_task(agent,visual,POcman_params,task_type,feature,runtime,walltime,arg_list,save_file,load_file=None):
    if load_file is not None:
        with open(load_file, "rb") as f:
            e = pickle.load(f)
            e.agent.learner.load(load_file)

    else:
        from POcmanUtils import POcman
        e = POcman(agent, visual, POcman_params)

    current_feature=feature
    tasks=[initialise_single_task(task_type,current_feature,runtime,environment=e)]
    run_singleenv(tasks,e,walltime,arg_list,save_file)


def set_actions(meltingpot,add_F=False):
    if meltingpot  :
        feature_size = 3
        observation_size = 11
        inputs = feature_size + observation_size if add_F else observation_size
        externalActions = [sa.ExternalAction(north, 0), sa.ExternalAction(south, 0), sa.ExternalAction(west, 0),
                           sa.ExternalAction(east, 0), sa.ExternalAction(stay, 0)]

    else:
        feature_size = 1
        observation_size = 16
        inputs = feature_size + observation_size if add_F else observation_size
        externalActions = [sa.ExternalAction(north, 0), sa.ExternalAction(south, 0), sa.ExternalAction(west, 0),
                           sa.ExternalAction(east, 0)]

    return inputs, externalActions

def get_reward_range(args,meltingpotreward_range):
    if args.task == "meltingpot":
        return meltingpotreward_range
    else:
        return POcman.get_reward_range()


# settings_ntask are lists of tuples of topology,dynamics,reward_range
default_top=[0]
default_dynam=[0]
default_reward=[1.]
all_topologies=range(3)
easy_dynamics=range(2)
all_dynamics=range(3)
all_rewards=[-1,1.]
settings_1task_experiments=[
    (0,0,-1.),
    (0,0,1.),
    (0,1,-1.),
    (0,1,1.),
    (0,2,-1.),
    (0,2,1.),

    (1,0,-1.),
    (1,0,1.),
    (1,1,-1.),
    (1,1,1.),
    (1,2,-1.),
    (1,2,1.),

    (2, 0, -1.),
    (2, 0, 1.),
    (2, 1, -1.),
    (2, 1, 1.),
    (2, 2, -1.),
    (2, 2, 1.),
]
settings_2task_experiments=[
    (default_top,all_dynamics,default_reward),
    ([0,1],default_dynam,default_reward),
    ([1,2],default_dynam,default_reward),
    ([0,2],default_dynam,default_reward),
    (default_top,default_dynam,all_rewards)
]
settings_4task_experiments=[
    # keep topology constant (9x)
    ([0],easy_dynamics,all_rewards),
    ([0], easy_dynamics, all_rewards),
    ([0], easy_dynamics, all_rewards),
    ([1], easy_dynamics,all_rewards),
    ([1], easy_dynamics, all_rewards),
    ([1], easy_dynamics,all_rewards),
    ([2], easy_dynamics, all_rewards),
    ([2], easy_dynamics, all_rewards),
    ([2], easy_dynamics, all_rewards),


]

extended_settings_4task_experiments = [
    # keep topology constant (9x)
    ([0], [0, 1], all_rewards),
    ([0], [0, 2], all_rewards),
    ([0], [1, 2], all_rewards),
    ([1], [0, 1], all_rewards),
    ([1], [0, 2], all_rewards),
    ([1], [1, 2], all_rewards),
    ([2], [0, 1], all_rewards),
    ([2], [0, 2], all_rewards),
    ([2], [1, 2], all_rewards),
    # keep dynamic constant (9x)
    ([0, 1], [0], all_rewards),
    ([0, 2], [0], all_rewards),
    ([1, 2], [0], all_rewards),
    ([0, 1], [1], all_rewards),
    ([0, 2], [1], all_rewards),
    ([1, 2], [1], all_rewards),
    ([0, 1], [2], all_rewards),
    ([0, 2], [2], all_rewards),
    ([1, 2], [2], all_rewards),

    # keep reward constant (18x)
    ([0, 1], [0, 1], [all_rewards[0]]),
    ([0, 2], [0, 1], [all_rewards[0]]),
    ([1, 2], [0, 1], [all_rewards[0]]),
    ([0, 1], [0, 2], [all_rewards[0]]),
    ([0, 2], [0, 2], [all_rewards[0]]),
    ([1, 2], [0, 2], [all_rewards[0]]),
    ([0, 1], [1, 2], [all_rewards[0]]),
    ([0, 2], [1, 2], [all_rewards[0]]),
    ([1, 2], [1, 2], [all_rewards[0]]),

    ([0, 1], [0, 1], [all_rewards[1]]),
    ([0, 2], [0, 1], [all_rewards[1]]),
    ([1, 2], [0, 1], [all_rewards[1]]),
    ([0, 1], [0, 2], [all_rewards[1]]),
    ([0, 2], [0, 2], [all_rewards[1]]),
    ([1, 2], [0, 2], [all_rewards[1]]),
    ([0, 1], [1, 2], [all_rewards[1]]),
    ([0, 2], [1, 2], [all_rewards[1]]),
    ([1, 2], [1, 2], [all_rewards[1]]),

]
settings_6task_experiments=[
    (all_topologies,default_dynam,all_rewards),
    (all_topologies,easy_dynamics,default_dynam)
]
settings_12task_experiments=[
    (all_topologies,easy_dynamics,all_rewards)
]
extended_settings_12task_experiments=[
    # reduce dynamics
    (all_topologies, [0, 1], all_rewards),
    (all_topologies,  [0, 2], all_rewards),
    (all_topologies, [1, 2], all_rewards),

    #reduce topologies
    ([0, 1], all_dynamics, all_rewards),
    ([0, 2], all_dynamics, all_rewards),
    ([1, 2], all_dynamics, all_rewards),

]

extended_settings_18task_experiments=[
    (all_topologies,all_dynamics,all_rewards)
]
extended_settings_9task_experiments=[
    (all_topologies,all_dynamics,[1.0])
]
def convert_run_to_setting(run,num_tasks):
    if num_tasks==1:
        setting=([0],[0],[1.0])
    elif num_tasks==2:
        m=len(settings_2task_experiments)
        setting= settings_2task_experiments[run%m]
    elif num_tasks==4:
        m = len(settings_4task_experiments)
        setting=settings_4task_experiments[run%m]
    elif num_tasks==6:
        m = len(settings_6task_experiments)
        setting=settings_6task_experiments[run%m]
    elif num_tasks==12:
        m = len(settings_12task_experiments)
        setting=settings_12task_experiments[run%m]
    else:
        raise Exception("number of tasks should be 2,4,6 or 12, was %d"%(num_tasks,))
        setting=None

    return setting
def convert_run_to_setting_x(run,num_tasks):
    if num_tasks==1:
        setting=([0],[0],[1.0])
    elif num_tasks==4:
        m = len(extended_settings_4task_experiments)
        setting=extended_settings_4task_experiments[run%m]
    elif num_tasks==9:
        m = len(extended_settings_9task_experiments)
        setting=extended_settings_9task_experiments[run%m]
    elif num_tasks==12:
        m = len(extended_settings_12task_experiments)
        setting=extended_settings_12task_experiments[run%m]
    elif num_tasks==18:
        m = len(extended_settings_18task_experiments)
        setting=extended_settings_18task_experiments[run%m]
    else:
        raise Exception("number of tasks should be 4, 12, or 18, was %d"%(num_tasks,))
        setting=None

    return setting
def add_run_tag(task_filename,run):
    return task_filename+'_r'+str(run)

def perform_single_task_lifelongx(run,filename,arg_list,walltime,stoptime,agent,visual):
    pacmanparams["stoptime"] = stoptime
    topology,dynamics,reward=settings_1task_experiments[run%18]
    task_feature=(reward,dynamics,topology)
    tasks=[MultiTask(topology_type=task_feature[-1],task_feature=task_feature,
                            task_type=get_task_type(task_feature),
                               reward_fun=get_reward_fun(task_feature),start_time=0,end_time=stoptime,funs={})]
    initialise_and_run_lifelong(filename,arg_list,walltime,agent,visual,pacmanparams,tasks,feature_weights=None)
    #if experiment_type

def perform_lifelong_setting(run,filename,task_filename,arg_list,walltime,num_tasks,stoptime,agent,visual):

    stoptime = stoptime*num_tasks
    task_time = BLOCK_LENGTH
    task_filename=add_run_tag(task_filename,run)
    with open("Tasks/"+task_filename+'_tasks',"rb") as file:
        tasks=pickle.load(file)
    with open("Tasks/"+task_filename+'_featureweights',"rb") as file2:
        feature_weights=pickle.load(file2)
    print("task_time=%d" % (task_time))
    pacmanparams["stoptime"]=stoptime
    return initialise_and_run_lifelong(filename, arg_list, walltime, agent, visual,
                                pacmanparams, tasks=tasks,feature_weights=feature_weights)



# def read_incremental(file_name):
#     import dill
#     with open(file_name,"rb") as f:
#             try:
#                 while True:
#                     article=dill.load(f)
#             except EOFError:
#                     pass
#     return article




def random_data():
    total_data_points=1*10**6
    data=-1 + 2*np.random.random_integers(0, 1, size=(total_data_points,11))  # 11-dim observation
    dump_incremental("data_diversity",data)



def get_filenames(methodname,filename_arg,experiment_type,num_pols,run,folder=None):
    homedir=str(os.environ['HOME'])
    if folder is None:
        if 'db2c15' in homedir:
            filename='/scratch/db2c15/'
        elif 'dmb1m19' in homedir:
            filename = '/scratch/dmb1m19/LifelongRL/'
        else:
            filename=str(os.environ['HOME']) + '/LifelongRL/'
    else:
        filename=folder+"/"
    task_filename=''
    add_task_str=''
    if args.add_task is None:
        args.add_task = False
    if args.add_task:
        add_task_str='addtask'
    if filename_arg is not None:
        filename+=filename_arg
    filename+=experiment_type+str(run)+'_'+methodname
    task_filename+=experiment_type
    filename+=str(num_pols)+"pols"+add_task_str
    return filename,task_filename


def get_num_tasks(experiment_type):
    if experiment_type == "lifelong_convergence_test":
        num_tasks = 1
    elif experiment_type == "lifelong2t":
        num_tasks = 2
    elif experiment_type == "lifelong4t":
        num_tasks = 4
    elif experiment_type == "lifelong6t":
        num_tasks = 6

    elif experiment_type == "lifelong12t":
        num_tasks = 12


    elif experiment_type == "lifelongx4t":
        num_tasks = 4

    elif experiment_type == "lifelongx12t":
        num_tasks = 12

    elif experiment_type == "lifelongx18t":
        num_tasks = 18
    elif experiment_type == "lifelongx9t":
        num_tasks = 9
    elif experiment_type == "lifelongx_uniform18t":
        num_tasks = 18
    else:
        num_tasks=1

    return num_tasks


def main(arg_list):
    # with open("MultiExp_lifelong6t0_2pols_stats_object","rb") as f:
    #     e=pickle.load(f)

    walltime= 60*3600 # 60 hours by default
    if args.walltime:
        ftr = [3600, 60, 1]
        walltime = sum([a * b for a, b in zip(ftr, map(int, args.walltime.split(':')))])

    print(walltime)



    ff=True


    conservative=args.conservative if args.conservative is not None else True
    methodname = args.method if args.method is not None else "MultiActorPPO2"

    if args.experiment_type is None:
        args.experiment_type = "lifelongx18t"


    run=args.run if args.run is not None else 0

    if args.policies is None:
        args.policies=18

    wm = ''
    stoptime=TOTAL_TASK_TIME if args.STOPTIME is None else args.STOPTIME


    print(SSA_WM_Params)
    filename,task_filename=get_filenames(methodname,args.filename,args.experiment_type,args.policies,run)

    if args.config_file is not None:

        configfile=str(os.environ['HOME']) + '/LifelongRL/Configs/IS-NEAT-Singlemaze_'+args.config_file+".ini"
        parse_config_file(filename,configfile,pacmanparams)
        filename+=args.config_file





    visual=False
    if args.VISUAL is not None:
        visual=args.VISUAL
        print(visual)
    pacmanparams['record_intervals']=None
    if args.record_video:
        visual = True
        pacmanparams['record_intervals'] = [[1+90 * 10 ** 6, 90.003 * 10 ** 6]]
        print('record intervals set')

    #pacmanparams['record_intervals']=[[.4*10**6,0.405*10**6]]

    #method = RandomLearner(externalActions)
    print(pacmanparams)
    print(internalActionsSSA_WM)

    #method = SSA_with_WM(**SSA_WM_Params)
    environmentfile=None
    if args.environment_file:
        environmentfile=filename+'_environment'

    if args.experiment_type == "stats":
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
    if args.experiment_type == "assess_testperformance":
        environmentfile=environmentfile.replace("assess_testperformance","lifelongx18t")
        filename=filename.replace("assess_testperformance","lifelongx18t")
        e=read_incremental(environmentfile)
        e.agent.learner.load(filename)
        tasks=e.agent.learner.occurence_weights
        e.agent.learner.testing = True

        for pol in e.agent.learner.pols:
            if "DRQN" in environmentfile:
                pol.agent.epsilon = 0.20  # recover start parameter after overwriting it wrongly
            pol.testing=True  # no learning can affect the performance


        added_time=50001
        e.use_stats=False
        e.stoptime = e.t + added_time*len(tasks)
        e.start=time.time()
        task_objects = [create_single_task(task,e.t+i*added_time,e.t+(i+1)*added_time) for i,task in enumerate(tasks)]
        print("stoptime"+str(e.stoptime))
        print("file read")
        test_performances={}
        for i,task in enumerate(tasks):
            startingR = e.agent.learner.R
            current_task=task_objects.pop(0)
            e.interrupt=False
            e.tasks=[current_task]
            e.run(walltime)
            print("after task run:"+str(e.t))
            test_performances[task]=(e.agent.learner.R-startingR)/float(added_time)



        dump_incremental(filename+'test_performance',test_performances)

        return

    if args.experiment_type == "print_taskpolmaps":
        environmentfile = environmentfile.replace("print_taskpolmaps", "lifelongx18t")
        filename = filename.replace("print_taskpolmaps", "lifelongx18t")
        e = read_incremental(environmentfile)
        e.agent.learner.load(filename)
        dump_incremental(filename+"_taskpolmap",e.agent.learner.taskpolmap)
        return
    # if args.experiment_type == "print_diversity":
    #     #environmentfile = environmentfile.replace("print_diversities", "lifelongx18t")
    #     filename = filename.replace("print_diversity", "lifelongx18t")
    #     parametric_diversities=[]
    #     performance_diversities=[]
    #     for f in stops.values()+[""]:
    #         e = read_incremental(filename+f+"_environment")
    #         e.agent.learner.load(filename+f)
    #         parametric_diversities.append(e.agent.learner.get_diversity())
    #         performance_diversities.append(e.agent.learner.get_performance_diversity())
    #     dump_incremental(filename+"_diversityRATIO",(parametric_diversities,performance_diversities))
    #     return

    if args.experiment_type == "print_diversity":
        #environmentfile = environmentfile.replace("print_diversities", "lifelongx18t")
        filename = filename.replace("print_diversity", "lifelongx18t")
        data=read_incremental("/scratch/db2c15/data_diversity")
        output_div=[]
        performance_diversities = []
        for f in stops.values() + [""]:
            e = read_incremental(filename + f + "_environment")
            e.agent.learner.load(filename + f)
            div=e.agent.learner.get_output_diversity(data,metric_type="totalvar")
            print("div = "+str(div))
            output_div.append(div)
            performance_diversities.append(None)
        dump_incremental(filename+"_outputdiversity_totalvar",(output_div,performance_diversities))
        return

    pacmanparams['num_actors'] = 5


    if environmentfile is not None:
         print("reading enviroment file")
         # Getting back the objects:
         # with open(environmentfile,"rb") as f:
         #    e = pickle.load(f)
         e=read_incremental(environmentfile)
         e.agent.learner.load(filename)

         #e.agent.learner.continue_experiment(intervals=loss_intervals)
         #print("loss intervals="+str(e.agent.learner.intervals))

         print("starting at "+str(e.t))


         #e.agent.learner.displayNetPolicy()
         e.start = time.time()
         #TODO remove this when main experiments finished
         #args.STOPTIME=5*10**6
         if args.STOPTIME is not None:
             e.stoptime = 90*10**6#args.STOPTIME * get_num_tasks(args.experiment_type)
         if args.record_video:
             e.stoptime=pacmanparams['record_intervals'][-1][-1]
             e.agent.learner.testing=True
             if args.VISUAL:
                 # use these lines to convert old-style stack
                 # from Stack import Stack
                 # e.agent.learner.Stack = Stack.listToStack(e.agent.learner.Stack)
                 for action in e.agent.learner.actions:
                     print(action)
                     print(action.n_args)
                 e.rng = np.random.RandomState(run)
                 e.visual = True
                 print(pacmanparams['record_intervals'])
                 e.initVisualisation(pacmanparams['sizeX'],pacmanparams['sizeY'],pacmanparams['record_intervals'], filename + "_video")
                 if pacmanparams['record_intervals']:
                     e.vis.on = False
                 print("video initialised")
         print("stopping at " + str(e.stoptime))

         if e.num_actors > 1:
             return e, e.tasks, walltime, arg_list, filename
         else:
             run_singleenv(e.tasks, e, walltime, arg_list, filename)


    else:

            # create environment


            if "lifelong" in args.experiment_type:
                meltingpot=True
            else:
                meltingpot=False

            inputs,externalActions = set_actions(meltingpot,add_F=args.add_task)
            pacmanparams['include_task_features']=args.add_task
            pacmanparams['observation_length']=inputs
            pacmanparams['filename'] = filename
            pacmanparams['seed'] = run


            n_input = inputs
            networktypeMap = {'jump': {'jump': 1}, 'external_actions': externalActions}

            SSA_WM_Params['num_inputs'] = inputs
            SSA_WM_Params['actions'] = externalActions
            setSSA_WM_Params(SSA_WM_Params)

            setIS_NEAT_params(pacmanparams, SSANeat_Params, networktypeMap)


            internalActionsSSA_WM.update(incPset)
            num_PLAs = 0
            for key in internalActionsSSA_WM:
                if key in ['searchP', 'incP', 'decP', 'inc_means', 'dec_means', 'sample']:
                    num_PLAs += 1


            SSA_WM_Params['conservative'] = conservative
            print("pacman params:" + str(pacmanparams))
            if "SSA" in methodname:
                print("ssa_wm_params:" + str(SSA_WM_Params))
                print("ssa-neat params:" + str(SSANeat_Params))
            if args.policies is not None:
                print("num policies : "+str(args.policies))



            episodic=True


            if pacmanparams["num_actors"] > 1:
                # just init the dictionary
                agent={"methodname":methodname,
                       "externalActions":externalActions,
                       "filename":filename,
                       "num_PLAs":num_PLAs,
                       "pacmanparams":pacmanparams,"inputs":inputs,"conservative":conservative,"episodic":episodic}
            else:
                method=get_method(methodname,externalActions,filename,num_PLAs,pacmanparams,inputs,conservative,episodic)
                print("episodic="+str(method.episodic))
                agent=PacmanAgent(method,pacmanparams)



            #filename, arg_list, walltime, tasktime, T, agent, visual, cheesemaze_params, POmaze_params, POcman_params, tasks = None





            savefile=filename+"_"+str(args.experiment_type)
            if args.load_file is None:
                loadfile=None
            else:
                loadfile=filename+args.load_file

            if args.task is None:
                args.task = PacmanTaskType.RUN_GHOSTS

            if args.experiment_type == "create_lifelong_tasks":
                for num_tasks in [2,4,6,12]:
                    task_filename = "lifelong%dt" % (num_tasks)
                    T = stoptime*num_tasks
                    for run in range(10):
                        new_task_filename=add_run_tag(task_filename,run)
                        task_time = BLOCK_LENGTH  # time for a task block
                        create_tasks_lifelong(new_task_filename,task_time,T,run,num_tasks)
            elif args.experiment_type == "create_lifelongx_tasks":
                for num_tasks in [4,9,12,18]:
                    task_filename = "lifelongx%dt" % (num_tasks)
                    T = stoptime*num_tasks
                    for run in range(36):
                        new_task_filename=add_run_tag(task_filename,run)
                        task_time = BLOCK_LENGTH  # time for a task block
                        create_tasks_lifelong(new_task_filename,task_time,T,run,num_tasks,extended=True)
            elif args.experiment_type == "create_lifelongx_tasks_uniform":
                  num_tasks=18
                  task_filename = "lifelongx_uniform%dt" % (num_tasks)
                  T = stoptime*num_tasks
                  task_time = BLOCK_LENGTH
                  create_tasks_lifelong_uniform(task_filename,task_time,T,num_tasks,extended=True)



            elif args.experiment_type == "create_lifelongx_test_tasks":
                num_tasks=4
                task_filename = "lifelongxtest"
                T = 200000*num_tasks
                for run in range(1):
                    new_task_filename=add_run_tag(task_filename,run)
                    task_time = 2000  # time for a task block
                    pacmanparams['elementary_task_time'] = 1000 # for simplicity one block==one elementary task
                    create_tasks_lifelong(new_task_filename,task_time,T,run,num_tasks,extended=True)
            elif args.experiment_type == "lifelongxtest":
                num_tasks=4
                pacmanparams['elementary_task_time']=1000
                stoptime=100000 if args.STOPTIME is None else args.STOPTIME
                return perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelong":
                topologies = all_topologies
                dynamic_opts = all_dynamics
                feature_set = get_featureset(all_rewards, dynamic_opts, topologies)

                # each run has a different probability initialisation
                feature_weights = generate_feature_weights(feature_set, max_ind=run)

                task_time = stoptime / N_BLOCK  # time for a task block

                initialise_and_run_lifelong(filename, arg_list, walltime, task_time, stoptime, agent, visual,
                                            pacmanparams, feature_weights=feature_weights)
            elif args.experiment_type == "lifelong_convergence_test":
                num_tasks=1
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)

            elif args.experiment_type == "lifelong2t":
                num_tasks=2
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelong4t":
                num_tasks=4
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelong6t":
                num_tasks=6
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelong12t":
                num_tasks=12
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)

            elif args.experiment_type == "lifelongx4t":
                num_tasks=4
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelongx12t":
                num_tasks=12
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelongx18t":
                num_tasks=18
                return perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)
            elif args.experiment_type == "lifelongx9t":
                num_tasks=9
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent, visual)

            elif args.experiment_type == "lifelongx18t_nonepisodic":
                num_tasks = 18
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent,
                                         visual)

            elif args.experiment_type == "lifelongxSingle":
                perform_single_task_lifelongx(run, filename, arg_list, walltime, stoptime, agent, visual)
            elif args.experiment_type == "lifelongx_uniform18t":
                num_tasks = 18
                perform_lifelong_setting(run, filename, task_filename, arg_list, walltime, num_tasks, stoptime, agent,
                                         visual)
            elif args.experiment_type == "simplified_lifelong":
                topologies=[0]
                dynamic_opts=[0.]
                feature_set = get_featureset(meltingpot_rewards, dynamic_opts, topologies)

                # each run has a different probability initialisation
                feature_weights=generate_feature_weights(feature_set,max_ind=run)

                task_time = stoptime / N_BLOCK
                print("task_time=%d"%(task_time))
                initialise_and_run_lifelong(filename,arg_list,walltime,task_time,stoptime,agent,visual,
                                            pacmanparams,feature_weights=feature_weights)
            elif args.experiment_type == "initial" or args.experiment_type is None:
                # (agent,visual,POcman_params,task,walltime,arg_list,save_file,load_file=None)
                print(str(args.task))
                feature=[0.]
                run_single_task(agent,visual,pacmanparams,args.task,feature,stoptime,walltime,arg_list,save_file=savefile,load_file=None)
            elif args.experiment_type == "interference":
                # (agent,visual,POcman_params,task,walltime,arg_list,save_file,load_file=None)
                savefile+=str(stoptime/200)
                feature = [1.]
                run_single_task(agent,visual,pacmanparams,args.task,feature,stoptime,walltime,arg_list,save_file=savefile,load_file=loadfile)
            elif args.experiment_type == "test":
                # (agent,visual,POcman_params,task,walltime,arg_list,save_file,load_file=None)
                feature = [0.]
                run_single_task(agent,visual,pacmanparams,args.task,feature,stoptime,walltime,arg_list,save_file=savefile,load_file=loadfile)
            else:
                raise Exception("no proper choice for experiment type; please use -x lifelong, -x initial, -x interference or -x stats")








if __name__ == '__main__':
    # from pycallgraph import PyCallGraph
    # from pycallgraph.output import GraphvizOutput
    # graphviz = GraphvizOutput()
    # graphviz.output_file = 'basic.png'
    #
    #stats=read_incremental("lifelongxtest0_MultiActorPPO218polsworker0_stats_object")
    # with PyCallGraph(output=graphviz)
    POcman,tasks,walltime,arg_list,filename = main(sys.argv[1:])
    #arg_list=["multi","-e","True"]
    import ray
    print(must_load(arg_list))
    ray.shutdown()
    ray.init()


    #     POcman =  read_incremental(filename + "_environment")
    #     POcman.actors = [POcmanActor.remote(POcman.agent_dict, POcman.visual, POcman.params,i+1) for i in range(POcman.num_actors)]
    #
    # else:
    # if must_load(arg_list):
    #     for i in range(POcman.num_actors):
    #         POcman.actors[i] = read_incremental(filename+"worker"+str(i)+"_environment")
    # else:
    POcman.actors = [POcmanActor.remote(POcman.agent_dict, POcman.visual, POcman.params,i+1) for i in range(POcman.num_actors)]

    # recreate the run_single_env
    POcman.set_tasks(tasks,stat_freq=18*10**6)
    POcman.start=time.time()
    POcman.run(walltime)
    interrupted=POcman.t < POcman.stoptime

    save_stats = not interrupted
    finalise_experiment_actors(POcman, filename, arg_list, NO_SAVING, args, save_stats=save_stats,
                        save_learner=True)
    #continue_experiment(interrupted, arg_list, job_script='bash lifelong_experiments.sh ')  # if tasks remain, continue

