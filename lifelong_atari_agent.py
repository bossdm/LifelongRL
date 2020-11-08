
from gym import wrappers, logger

from gym.envs.atari.atari_env import *

from copy import deepcopy


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
                   'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 5,
                   'actions': deepcopy(externalActions),'episodic': episodic,
                'target_model':True,'init_epsilon':1.0,'final_epsilon':.10,'epsilon_change':True,"learning_rate":lr}
def get_A2C_configs(inputs,externalActions, filename, episodic):
    paramsdict={}
    if os.environ["tuning_lr"]:
        paramsdict["learning_rate"]=float(os.environ["tuning_lr"])
    else:
        paramsdict["learning_rate"]=0.00025
    return {'num_neurons': None, 'task_features': [], 'use_task_bias': False,
            'use_task_gain': False, 'n_inputs': inputs, 'trace_length': 5,
            'actions': deepcopy(externalActions), 'episodic': episodic,'file':filename, 'params':paramsdict
            }


class LifelongAtariAgent(object):
    """The world's simplest agent!"""
    def __init__(self, args,filename,inputs, externalActions):
        self.learner = select_learner(args,inputs,externalActions,filename)
    def act(self,obs, reward, done, total_t, t):
        if done:
            self.learner.setAtariTerminalObservation(obs)
        else:
            self.learner.setReward(reward)
            self.learner.atari_cycle(obs, reward, total_t, t)
        return self.learner.chosenAction


def generate_environment_sequence(args):
    environments = []
    games = ["pong", "breakout", "bowling", "boxing", "battle_zone", # hitting target
             "ms_pacman", "alien", "up_n_down", "time_pilot", "frostbite"]  # navigation
    # all games have observation space Box(0, 255, (210, 160, 3), uint8)
    if args.experiment_type!="lifelong":
        games=[games[args.run]] # a single game with exactly 18 discrete actions
    frameskip = 3
    for game in games:
        environments.append(AtariEnv(game=game,mode=None,difficulty=None,obs_type="image",frameskip=frameskip,
                                     repeat_action_probability=0.0,
                                     full_action_space=True
                                     ))
        print("game ",game)
        print("obs ",environments[-1].observation_space)
        print("act", environments[-1].action_space)
    return environments
def perform_episode(visual,env, agent, seed,total_t):
    print("starting environment")
    env.seed(seed)
    for item in env.__dict__.items():
        print(item)
    episode_count = 100
    reward = 0
    done = False

    for t in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done, total_t, t)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            if visual:
                env.render()
            total_t+=1
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()

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
    args.VISUAL=False
    filename=args.filename + "/"+args.experiment_type+str(args.run) + '_' + args.method + str(args.policies) + "pols" + os.environ["tuning_lr"]
    walltime = 60 * 3600  # 60 hours by default
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
    outdir = '/tmp/random-agent-results'

    envs = generate_environment_sequence(args)
    inputs = envs[0].observation_space
    num_actions = 18
    agent = LifelongAtariAgent(args,filename,inputs.shape,range(len(ACTION_MEANING)))
    total_t=0
    for env in envs:
        #visual,env, agent, seed,total_t
        perform_episode(args.VISUAL, env, agent, args.run, total_t)