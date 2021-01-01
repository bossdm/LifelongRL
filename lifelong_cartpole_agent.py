

import os
from gym import wrappers, logger

from gym.envs.classic_control.cartpole import *
from copy import deepcopy

from ExperimentUtils import dump_incremental, read_incremental

import time


TASK_BLOCK_SIZE =60000 # 300 episodes per task block
FRAMES_PER_TASK=1500000 # total of at least 7500 episodes per task, to be divided among at most 14 policies
# compare 5000 episodes for MultiExperiment with at most 9 policies
# ---> 25 blocks per task, and total 675 blocks (compare 450 in MultiExperiment)
NUM_BLOCKS=675



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
                   'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 1,
                   'actions': deepcopy(externalActions),'episodic': episodic,'recurrent': False,
                'target_model':True,'init_epsilon':0.1,'final_epsilon':.10,'epsilon_change':True,"learning_rate":lr}
def get_A2C_configs(inputs,externalActions, filename, episodic):
    paramsdict={}
    if os.environ["tuning_lr"]:
        paramsdict["learning_rate"]=float(os.environ["tuning_lr"])
    else:
        paramsdict["learning_rate"]=0.00025
    return {'num_neurons': 80, 'task_features': [], 'use_task_bias': False,
            'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 1, 'recurrent': False,
            'actions': deepcopy(externalActions), 'episodic': episodic,'file':filename, 'params':paramsdict,
            "large_scale": False, "terminal_known": True
            }


class LifelongCartpoleAgent(object):
    """The world's simplest agent!"""
    def __init__(self, args,filename,n_tasks):
        self.learner = select_learner(args,4,range(2),filename,n_tasks)
    def act(self,obs, done, total_t):
        self.learner.setTime(total_t)
        self.learner.atari_cycle(obs)
        return self.learner.chosenAction

    def reward(self,r):
        self.learner.setReward(r)
    def set_term(self,obs):
        self.learner.setAtariTerminalObservation(obs)

    def new_episode(self):
        self.learner.new_elementary_task()
    def end_episode(self):
        self.learner.reset()
def indices_convergence(run):
    np.random.seed(0)  # only first sequence is random
    environment_index = np.random.choice(27, size=27, replace=False)
    for j in range(27):
        environment_index[j] = (environment_index[j] + run) % 27  # increment tindex according to run
    assert len(np.unique(environment_index)) == 27 and len(environment_index) == 27
    return environment_index
def indices_lifelong(run):
    np.random.seed(0)  # only first sequence is random
    environment_index = [None for i in range(NUM_BLOCKS)]
    for j in range(NUM_BLOCKS):
        environment_index[j] = np.random.randint(0, 27)  # generate random index
        environment_index[j] = (environment_index[j] + run) % 27  # increment tindex according to run
    return environment_index

def get_games(args):
        environments=[]
        game = ["CartPole-v1"]

        masscarts=[0.5,1.0,2.0]
        masspoles=[0.05,0.1,0.2]
        lengths=[0.25,0.50,1.0]
        for mc in masscarts:
            for mp in masspoles:
                for l in lengths:
                    e = make_custom_environment(mc,mp,l)
                    environments.append(e)
                    print("game ", game[0])
                    print("obs ", environments[-1].observation_space)
                    print("act", environments[-1].action_space)
        if args.experiment_type == "lifelong_convergence":
            indices=indices_convergence(args.run)
            # now set the correct random state
            np.random.seed(args.run)  # only first sequence is random
            taskblockend=FRAMES_PER_TASK
            return environments, indices, taskblockend
        elif args.experiment_type == "lifelong":
            indices = indices_lifelong(args.run)
            # now set the correct random state
            np.random.seed(args.run)  # only first sequence is random
            taskblockend=TASK_BLOCK_SIZE
            return environments, indices, taskblockend
        else:
            environments = [environments[args.run]]
            taskblockend = FRAMES_PER_TASK
            return environments, [0], taskblockend

def make_custom_environment(masscart,masspole,length):
    e = CartPoleEnv()
    e.masscart = masscart
    e.masspole = masspole
    e.length = length  # actually half the pole's length
    e.settings = "masscart="+str(e.masscart)+" masspole="+str(e.masspole)+" length="+str(e.length)
    return e

def perform_episode(visual,env, agent, seed,total_t):
    #print("starting environment", env, "with mc=",env.masscart," mp=", env.masspole, "e.length=",env.length)
    env.seed(seed)
    print("seed ",seed)
    reward = 0
    done = False
    consumed_frames=0
    ob = env.reset()
    t=0
    agent.new_episode()
    while True:
        action = agent.act(ob, done, total_t)
        ob, r, done, _ = env.step(action)
        agent.reward(r)
        if done or t==200:  # should terminate at t==200 as this defines success
            agent.set_term(ob)
            break
        if visual:
            env.render()
        t+=1
        total_t+=1
        consumed_frames+=1

            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.
    agent.end_episode()
    # Close the env and write monitor result info to disk
    env.close()
    return consumed_frames


def select_learner(args,inputs,externalActions,filename,n_tasks,episodic=True):
    if args.method == "PPO":
        from Catastrophic_Forgetting_NNs.A2C_Learner2 import PPO_Learner
        settings = get_A2C_configs(inputs, externalActions, filename, episodic)
        method = PPO_Learner(**settings)
    elif args.method == "DRQN":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
        settings=get_DRQN_configs(inputs,externalActions,filename,episodic)
        method = DRQN_Learner( **settings)
    elif args.method == "TaskDrift_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN
        from Configs.InstructionSetsMultiExp import homeostatic_params
        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params, episodic_performance=True)

        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
        method.set_tasks({(i,): 1. / float(n_tasks) for i in range(n_tasks)})
    elif args.method == "TaskDrift_PPO":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDrift_PPO2 import TaskDriftPPO
        from Configs.InstructionSetsMultiExp import homeostatic_params
        # batch_size=32
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []
            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            PPO_params=get_A2C_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftPPO(PPO_params,episodic_performance=True)

        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
        method.set_tasks({(i,): 1. / float(n_tasks) for i in range(n_tasks)})
    elif args.method == "Unadaptive_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN
        from Configs.InstructionSetsMultiExp import homeostatic_params
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
        method.set_tasks({(i,): 1. / float(n_tasks) for i in range(n_tasks)})

    elif args.method == "Unadaptive_PPO":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDrift_PPO2 import TaskDriftPPO
        from Configs.InstructionSetsMultiExp import homeostatic_params
        num_pols = args.policies
        pols = [None] * num_pols
        for pol in range(num_pols):
            settings = get_A2C_configs(inputs, externalActions, filename, episodic)

            pols[pol] = TaskDriftPPO(settings)
        homeostatic_params['unadaptive']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
        method.set_tasks({(i,): 1. / float(n_tasks) for i in range(n_tasks)})
    elif args.method == "1to1_DRQN":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDriftDRQN import TaskDriftDRQN
        from Configs.InstructionSetsMultiExp import homeostatic_params
        #args.policies=27
        # batch_size=32
        num_pols = args.policies
        assert num_pols==27
        pols = [None] * num_pols
        for pol in range(num_pols):
            task_features = []

            # task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic, loss = None
            DRQN_params=get_DRQN_configs(inputs,externalActions,filename,episodic)
            pols[pol] = TaskDriftDRQN(DRQN_params)
        homeostatic_params['one_to_one']=True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
        method.set_tasks({(i,): 1. / float(n_tasks) for i in range(n_tasks)})
    elif args.method == "1to1_PPO":
        from Lifelong.HomeostaticPols import HomeostaticPol
        from Lifelong.TaskDrift_PPO2 import TaskDriftPPO
        from Configs.InstructionSetsMultiExp import homeostatic_params
        num_pols = args.policies
        assert num_pols == 27
        pols = [None] * num_pols
        for pol in range(num_pols):
            settings = get_A2C_configs(inputs, externalActions, filename, episodic)

            pols[pol] = TaskDriftPPO(settings)
        homeostatic_params['one_to_one'] = True
        method = HomeostaticPol(episodic=episodic, actions=externalActions, filename=filename, pols=pols, weights=None,
                                **homeostatic_params)
        method.set_tasks({(i,): 1. / float(n_tasks) for i in range(n_tasks)})
    #elif: 1to1 not needed since we just converge on the task here, can use single_task runs
    else:
        raise Exception("learner ",args.method," not supported")
    return method
if __name__ == '__main__':
    from Parsers.Parse_Arguments import *
    parser.add_argument("-P", dest="policies", type=int,default=1)
    parser.add_argument("-x", dest="experiment_type", type=str, default="single")  # single, lifelong_convergence , lifelong
    args = parser.parse_args()
    print("will start run ",args.run, " with experiment_type ",args.experiment_type, "and ",args.policies, " policies of ", args.method)
    # args.VISUAL=True
    # args.method="1to1_DRQN"
    # args.policies=27
    # args.run=5
    # args.experiment_type="lifelong_convergence"
    # args.filename="/home/david/LifelongRL"
    # args.environment_file=False
    filename=args.filename +args.experiment_type+str(args.run) + '_' + args.method + str(args.policies) + "pols" + os.environ["tuning_lr"]
    walltime = 60*3600 #60*3600  # 60 hours by default
    if args.walltime:
        ftr = [3600, 60, 1]
        walltime = sum([a * b for a, b in zip(ftr, map(int, args.walltime.split(':')))])



    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().

    envs, indices, taskblockend = get_games(args)
    if args.environment_file:
        # just to get stopping time
        interrupted=True
        agent = read_incremental(filename+"_agent") # load atari agent
        agent.learner.load(filename)   # load the learner
    else:
        interrupted=False
        inputs = envs[0].observation_space
        agent = LifelongCartpoleAgent(args,filename,len(envs))  # network has full action set, but only uses minimal for each task
        agent.total_t=0
        agent.num_episodes = 0
        agent.learner.printDevelopmentAtari(frames=0)
        agent.index = 0
    starttime = time.time()


    #print(agent.learner.__dict__)

    for i in range(agent.index,len(indices)):
        j=indices[i]
        env=envs[j]
        agent.index=i
        print("starting cartpole environment")
        print("block i =", i, "environment ", j , "\t settings: ", env.settings)
        if not interrupted:
            agent.taskblock_t=0
            agent.learner.new_task([i])
        # for item in env.__dict__.items():
        #     print(item)
        while agent.taskblock_t< taskblockend:
            print("starting new episode at taskblock_t: ", agent.taskblock_t)
            consumed_steps=perform_episode(args.VISUAL, env, agent, args.run*100000+agent.num_episodes, agent.total_t)
            agent.taskblock_t+=consumed_steps
            agent.total_t+=consumed_steps # need to add because primitive data types not passed by reference
            agent.num_episodes+=1
            agent.learner.printDevelopmentAtari(frames=agent.total_t)
            walltime_consumed = time.time() - starttime
            if walltime_consumed >= 0.9*walltime:
                break
        agent.learner.end_task()

    agent.learner.save(filename)
    dump_incremental(filename + "_agent", agent)
    print("stopping at time ", walltime_consumed)
    exit(0)