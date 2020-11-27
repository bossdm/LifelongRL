
from gym import wrappers, logger

from gym.envs.atari.atari_env import *
from gym.wrappers.atari_preprocessing import *
from copy import deepcopy

from ExperimentUtils import dump_incremental, read_incremental

import time

FRAMES_PER_TASK=10*10**6 # for tuning
#FRAMES_PER_TASK=50*10**6  # for full experiment
#FRAMES_PER_EPISODE=18000 # according to Arcade learning environment paper

def get_DQN_configs(inputs,externalActions,filename,episodic):
    d=get_DQN_agent_configs(inputs, externalActions, filename, episodic)
    d.update({'file': filename, 'loss': None})
    return d
def get_DQN_agent_configs(inputs,externalActions,filename,episodic):
    if os.environ["tuning_lr"]:
        lr=float(os.environ["tuning_lr"])
    else:
        lr=0.10
    return  {'num_neurons':None,'task_features': [], 'use_task_bias': False,
                   'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 3,
                   'actions': deepcopy(externalActions),'episodic': episodic,
                'target_model':True,'init_epsilon':1.0,'final_epsilon':.10,'epsilon_change':True,"learning_rate":lr}
def get_A2C_configs(inputs,externalActions, filename, episodic):
    paramsdict={}
    if os.environ["tuning_lr"]:
        paramsdict["learning_rate"]=float(os.environ["tuning_lr"])
    else:
        paramsdict["learning_rate"]=0.00025
    return {'num_neurons': None, 'task_features': [], 'use_task_bias': False,
            'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 3,
            'actions': deepcopy(externalActions), 'episodic': episodic,'file':filename, 'params':paramsdict,
            "large_scale": True, "terminal_known": True
            }


class LifelongAtariAgent(object):
    """The world's simplest agent!"""
    def __init__(self, args,filename,inputs, externalActions):
        self.learner = select_learner(args,inputs,externalActions,filename)
    def act(self,obs, reward, done, total_t, t):

        self.learner.setReward(reward)
        self.learner.atari_cycle(obs, reward, total_t, t)
        return self.learner.chosenAction
    def set_term(self,obs):
        self.learner.setAtariTerminalObservation(obs)

    def new_episode(self):
        self.learner.new_elementary_task()
    def end_episode(self):
        self.learner.reset()
def get_games(args,type):

    if type == "custom":
        environments=[]
        games = ["pong", "breakout", "bowling", "boxing", "battle_zone", # hitting target
             "ms_pacman", "alien", "up_n_down", "time_pilot", "frostbite"]  # navigation
        if args.experiment_type != "lifelong":
            games = [games[args.run]]  # a single game with exactly 18 discrete actions
        for game in games:
            e = make_custom_environment(game,frameskip=4)
            environments.append(e)
            print("game ", game)
            print("obs ", environments[-1].observation_space)
            print("act", environments[-1].action_space)
        return environments
    else: #using preprocessing wrapper
        environments = []
        games = ["PongNoFrameskip-v4", "BreakoutNoFrameskip-v4", "BowlingNoFrameskip-v4", "BoxingNoFrameskip-v4", "BattleZoneNoFrameskip-v4",  # hitting target
                 "MsPacmanNoFrameskip-v4", "AlienNoFrameskip-v4", "UpNDownNoFrameskip-v4", "TimePilotNoFrameskip-v4", "FrostbiteNoFrameskip-v4"]  # navigation
        if args.experiment_type != "lifelong":
            games = [games[args.run]]  # a single game with exactly 18 discrete actions
        for game in games:
            env = gym.make(game,full_action_space=True)
            e = AtariPreprocessing(env,grayscale_newaxis=True)
            # e._action_set = e.ale.getLegalActionSet()
            # e.action_space = spaces.Discrete(len(e._action_set))
            environments.append(e)
            print("game ", game)
            print("obs ", environments[-1].observation_space)
            print("act", environments[-1].action_space)
        return environments
def make_custom_environment(game,frameskip):
    e = AtariEnv(game=game, mode=None, difficulty=None, obs_type="image", frameskip=frameskip,
                 repeat_action_probability=0.25,
                 full_action_space=True
                 )
    return e

def perform_episode(visual,env, agent, seed,total_t):
    print("starting environment")
    env.seed(seed)
    reward = 0
    done = False
    consumed_frames=0
    ob = env.reset()
    t=0
    agent.new_episode()
    while True:
        action = agent.act(ob, reward, done, total_t, t)
        ob, reward, done, _ = env.step(action)
        if done:
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


def select_learner(args,inputs,externalActions,filename,episodic=True):
    if args.method == "PPO":
        from Catastrophic_Forgetting_NNs.A2C_Learner2 import PPO_Learner
        settings = get_A2C_configs(inputs, externalActions, filename, episodic)
        method = PPO_Learner(**settings)
    elif args.method == "DRQN":
        from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
        settings=get_DQN_configs(inputs,externalActions,filename,episodic)
        method = DRQN_Learner( **settings)
    else:
        raise Exception("learner ",args.method," not supported")
    return method
if __name__ == '__main__':
    from Parsers.Parse_Arguments import *
    parser.add_argument("-P", dest="policies", type=int,default=1)
    parser.add_argument("-x", dest="experiment_type", type=str, default="single")  # lifelong, initial, interference or test
    args = parser.parse_args()
    print("will start run ",args.run)
    # args.VISUAL=False
    # args.method="PPO"
    # args.policies=1
    # args.run=0
    # args.experiment_type="single"
    # args.filename="/home/david/LifelongRL"
    # args.environment_file=True
    filename=args.filename + "/"+args.experiment_type+str(args.run) + '_' + args.method + str(args.policies) + "pols" + os.environ["tuning_lr"]
    walltime = 10 #60*3600  # 60 hours by default
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

    envs = get_games(args, "preprocessing")
    if args.environment_file:
        # just to get stopping time
        interrupted=True
        agent = read_incremental(filename+"_agent") # load atari agent
        agent.learner.load(filename)   # load the learner
    else:
        interrupted=False
        inputs = envs[0].observation_space
        agent = LifelongAtariAgent(args,filename,inputs.shape,range(len(ACTION_MEANING)))  # network has full action set, but only uses minimal for each task
        agent.total_t=0
        agent.num_episodes = 0
        agent.learner.printDevelopmentAtari(frames=0)
        agent.index = 0
    starttime = time.time()

    #print(agent.learner.__dict__)

    for i in range(agent.index,len(envs)):
        env=envs[i]
        print("starting ",env.game)
        agent.taskblock_t=0 if not interrupted else agent.taskblock_t
        for item in env.__dict__.items():
            print(item)
        while agent.taskblock_t<FRAMES_PER_TASK:
            print("starting new episode at taskblock_t: ", agent.taskblock_t)
            consumed_steps=perform_episode(args.VISUAL, env, agent, args.run*100000+agent.num_episodes, agent.total_t)
            agent.taskblock_t+=consumed_steps*env.frame_skip
            agent.total_t+=consumed_steps # need to add because primitive data types not passed by reference
            agent.num_episodes+=1
            agent.learner.printDevelopmentAtari(frames=agent.total_t*env.frame_skip)
            walltime_consumed = time.time() - starttime
            if walltime_consumed >= 0.9*walltime:
                break

    agent.learner.save(filename)
    dump_incremental(filename + "_agent", agent)
    print("stopping at time ", walltime_consumed)
    exit(0)