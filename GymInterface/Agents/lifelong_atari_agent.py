import argparse
import sys

import gym
from gym import wrappers, logger

from gym.envs.atari.atari_env import *



class LifelongAtariAgent(object):
    """The world's simplest agent!"""
    def __init__(self, learner):
        self.learner = learner
    def act(self,obs,reward,done):
        return 0


def generate_environment_sequence():
    environments = []
    games=["pong","breakout","carnival","bowling","boxing","battle_zone","chopper_command",# hitting target
            "bank_heist","ms_pacman","alien","up_n_down","wizard_of_wor","time_pilot","skiing","frostbite"] # navigation
    frameskip = 3
    for game in games:
        environments.append(AtariEnv(game=game,mode=None,difficulty=None,obs_type="image",frameskip=frameskip,
                                     repeat_action_probability=0.0,
                                     full_action_space=False
                                     ))
    return environments
def perform_episode(env, agent, seed):
    env.seed(seed)
    agent = LifelongAtariAgent(env.action_space)
    episode_count = 100
    reward = 0
    done = False

    for i in range(episode_count):
        ob = env.reset()
        while True:
            action = agent.act(ob, reward, done)
            ob, reward, done, _ = env.step(action)
            if done:
                break
            # Note there's no env.render() here. But the environment still can open window and
            # render if asked by env.monitor: it calls env.render('rgb_array') to record video.
            # Video is not recorded every episode, see capped_cubic_video_schedule for details.

    # Close the env and write monitor result info to disk
    env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('env_id', nargs='?', default='CartPole-v0', help='Select the environment to run')
    args = parser.parse_args()

    # You can set the level to logger.DEBUG or logger.WARN if you
    # want to change the amount of output.
    logger.set_level(logger.INFO)


    # You provide the directory to write to (can be an existing
    # directory, including one with existing data -- all monitor files
    # will be namespaced). You can also dump to a tempdir if you'd
    # like: tempfile.mkdtemp().
    outdir = '/tmp/random-agent-results'

    envs = generate_environment_sequence()
    agent = LifelongAtariAgent(None)
    for env in envs:
        perform_episode(env,agent,0)


