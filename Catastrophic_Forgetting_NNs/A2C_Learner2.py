import sys, os
sys.path.insert(0, str(os.environ['HOME']) + '/VizDoom-Keras-RL/')

import numpy as np
import random

from ExperimentUtils import read_incremental,dump_incremental
from overrides import overrides

from Methods.Learner import CompleteLearner

from Catastrophic_Forgetting_NNs.A2C_small2 import A2CAgent,PPO_Agent
from Catastrophic_Forgetting_NNs.CustomNetworks import CustomNetworks

from keras.models import clone_model

def clone_kerasmodel(model):
    copy_m = clone_model(model)
    copy_m.set_weights(model.get_weights())
    return model

DEBUG_MODE=False


class A2C_Learner(CompleteLearner):
    loss=None
    recorded_Qs={}
    recorded_loss={}
    recorded_targets={}
    intervals=[]
    exploration_schedule={}
    terminal_states_known=False
    def __init__(self,task_features,use_task_bias,use_task_gain,n_inputs,trace_length,actions,file,episodic,loss=None,
                 target_model=False,num_neurons=80,
                 agent=None,intervals=[]):
        CompleteLearner.__init__(self,actions,file,episodic)

        self.init_variables()
        self.action_size=len(actions)
        if agent is None:
            self.agent=A2C_Learner.init_agent(n_inputs,actions,trace_length,episodic,
                                           task_features, use_task_bias, use_task_gain, num_neurons)
        self.state_size = A2C_Learner.set_state_size(n_inputs, trace_length)
        self.continue_experiment(intervals)

        self.s_t=np.zeros(self.state_size)
        # self.model_objective = EWC_objective(lbda_task,learning_rate,batch_size,model,n_in, n_out,lbda,output_type=OutputType.linear,epochs=200,
        #          objective_type=ObjectiveType.EWC,task_weights=None)
        # self.target_model_objective = EWC_objective_linoutput(out,lbda)
        # agent.model.compile(loss=self.model_objective.objective)
        #
        # agent.target_model.compile()

        print(self.__dict__)
        print(self.agent.__dict__)



    @classmethod
    def init_agent(cls,n_inputs,actions,trace_length,episodic,task_features, use_task_bias, use_task_gain,
                   num_neurons):
        action_size = len(actions)
        state_size = A2C_Learner.set_state_size(n_inputs, trace_length)
        agent = A2CAgent(state_size, action_size, trace_length,
                                episodic=episodic)

        #agent.epsilon=.05

        input_shape = (None,) + state_size
        if isinstance(n_inputs,tuple): # use convolution
            # input_shape, action_size, value_size, learning_rate, num_neurons
            agent.model = CustomNetworks.small_scale_a2c_lstm( state_size, action_size,  agent.value_size,num_neurons)

        else:
            # input_shape, action_size, learning_rate, task_features, use_task_bias, use_task_gain
            agent.model = CustomNetworks.small_scale_a2c_lstm( state_size, action_size, agent.value_size,num_neurons)


        return agent
    @classmethod
    def set_state_size(cls,n_inputs,trace_length):
        if isinstance(n_inputs,tuple):
            # is a shape
            state_size=(trace_length,)+n_inputs

        else:
            state_size=(trace_length,n_inputs,)
        return state_size
    def init_variables(self):

        self.action_idx=0

        self.total_t = 0

    def null_state(self,n_inputs):
        return np.zeros(n_inputs)-1
    def null_experience(self):
        _, n_inputs= self.state_size
        return [self.null_state(n_inputs), 0., 0., self.null_state(n_inputs)]
    def get_policy_copy(self):
        return clone_kerasmodel(self.agent.model)

    def set_policy(self,pol):
        self.agent.model=pol
    @overrides
    def printPolicy(self):
        pass
    @overrides
    def reset(self):
        if len(self.agent.rewards)>=1:
            if self.testing:
                return
            loss=self.agent.train_model(terminal=self.terminal_states_known)
            if DEBUG_MODE:
                print("loss="+str(loss))
    @overrides
    def new_elementary_task(self):
        print("new elementary task")
        if self.episodic :
            print("episodic")
            self.agent.reset_states()
            self.t = 0

    # def track_q(self,old_location,location,intervals):
    #     """
    #
    #     :return:
    #     """
    #
    #
    #     for min, max in intervals:
    #
    #         if min <= self.total_t < max:
    #             self.record_qs(old_location,location)
    #
    #         if self.total_t == max:
    #             self.terminate_qs(min,max)
    # def record_qs(self,old_location,location):
    #     if old_location not in self.recorded_Qs:
    #         self.recorded_Qs[old_location] = []
    #         self.recorded_targets[old_location] = []
    #         self.recorded_loss[old_location] = []
    #     s, a, r, s_ = self.agent.memory.buffer[self.agent.memory.sp]
    #     output, target = self.agent.compute_output_and_target(self.old_input, self.new_input, 1, [[a]],
    #                                                           [[r]])
    #     self.recorded_Qs[old_location].append(output)
    #     self.recorded_targets[old_location].append(target)
    #     self.recorded_loss[old_location].append(np.mean((output - target) ** 2))
    # def terminate_qs(self,min,max):
    #     del self.intervals[0]
    #     self.save_recordings(min, max)
    #     self.recorded_Qs = {}
    #     self.recorded_loss = {}
    #     self.recorded_targets = {}
    # def save_recordings(self,min,max,folder=''):
    #     recorded_stuff = {'Qs': self.recorded_Qs, 'targets': self.recorded_targets,
    #                       'loss': self.recorded_loss}
    #
    #     dump_incremental(folder+self.file + '(%d,%d)_recordings' % (min, max), recorded_stuff)

    @overrides
    def new_task(self,feature):
        """
        when new feature arrives, need to switch to task-specific
        :param feature:
        :return:
        """
        pass
    @overrides
    def setObservation(self,agent,environment):
        self.agent.total_t = self.total_t
        environment.setObservation(agent)
        self.observation=agent.learner.observation # in case of task drif
        obs=np.expand_dims(self.observation,axis=0)
        self. s_t = np.append(self.s_t[ 1:, :],obs, axis=0)

        if len(self.agent.rewards)==self.agent.update_freq:
            if self.testing:
                return
            self.agent.states.append(self.s_t)  # add final state
            loss = self.agent.train_model(terminal=False)
            if DEBUG_MODE:
                print("loss=" + str(loss))

    @overrides
    def setTerminalObservation(self,agent,environment):
        #print("setting terminal observation")
        self.setObservation(agent,environment)
        if self.t >= self.agent.trace_length and len(self.agent.rewards) >= 1:
            #print("appending state")
            self.agent.states.append(self.s_t) # add final state


    def process_datapoint(self,state):
        state=np.expand_dims(state,0)
        prob = self.agent.ppo.get_probability(state).flatten()
        return prob
    @overrides
    def setAction(self):
        if self.t < self.agent.trace_length:
            self.action_idx =  random.randrange(self.action_size)
        else:
            # try:
            self.action_idx = self.agent.get_action(np.expand_dims(self.s_t,0))[0]
            # except Exception as e:
            #     print(e)
            #     print("state = " + str(self.s_t))
            #     raise e


        self.chosenAction=self.actions[self.action_idx]



    def setTime(self,t):
        increment= t - self.total_t
        self.t += increment
        self.total_t += increment


    @overrides
    def continue_experiment(self,intervals):
        self.intervals=intervals
        if self.intervals:
            self.loss_file=open(self.file+"_loss",mode="wb")

    @overrides
    def learn(self):
        # if DEBUG_MODE:
        #     print("epsilon=%.4f"%(self.agent.epsilon))
        # # Update epsilon
        # Update the cache

        pass

    @overrides
    def save(self,filename):
        self.agent.save(filename)
    @overrides
    def load(self,filename):
        self.agent.load(filename)
        print(self.__dict__)
        print(self.agent.__dict__)

    @overrides
    def performAction(self, agent, environment):
        self.chosenAction.perform([agent,environment])
        #self.t = environment.t
    @overrides
    def cycle(self,agent,environment):
        self.setObservation(agent, environment)

        self.setAction()
        agent.learner.chosenAction = self.chosenAction # cf. task drift (Homeostatic Pols)
        self.performAction(agent,environment)
        self.learn()


    def add_sample(self):
        if self.t >= self.agent.trace_length:
            self.agent.append_sample(self.s_t, self.action_idx, self.r)


    def setReward(self,reward):
        CompleteLearner.setReward(self,reward)
        self.add_sample()



class PPO_Learner(A2C_Learner):
    def __init__(self, task_features, use_task_bias, use_task_gain, n_inputs, trace_length, actions, file, episodic,
                 loss=None,
                 target_model=False, num_neurons=80,large_scale=False,terminal_known=False,
                 agent=None, intervals=[],params={}):
        self.terminal_states_known = terminal_known
        CompleteLearner.__init__(self, actions, file, episodic)

        self.init_variables()
        self.action_size = len(actions)
        if agent is None:
            self.agent = PPO_Learner.init_agent(n_inputs, actions, trace_length, episodic,
                                                task_features, use_task_bias, use_task_gain, num_neurons,params,large_scale)
        self.state_size = PPO_Learner.set_state_size(n_inputs, trace_length)
        self.continue_experiment(intervals)

        self.s_t = np.zeros(self.state_size)
        # self.model_objective = EWC_objective(lbda_task,learning_rate,batch_size,model,n_in, n_out,lbda,output_type=OutputType.linear,epochs=200,
        #          objective_type=ObjectiveType.EWC,task_weights=None)
        # self.target_model_objective = EWC_objective_linoutput(out,lbda)
        # agent.model.compile(loss=self.model_objective.objective)
        #
        # agent.target_model.compile()

        print(self.__dict__)
        print(self.agent.__dict__)

    @classmethod
    def init_agent(cls,n_inputs,actions,trace_length,episodic,task_features, use_task_bias, use_task_gain,
                   num_neurons,params,large_scale):
        action_size = len(actions)
        state_size = A2C_Learner.set_state_size(n_inputs, trace_length)
        agent = PPO_Agent(state_size, action_size, trace_length,
                                episodic=episodic,params=params,large_scale=large_scale)
        return agent

    def add_sample(self):
        if self.t >= self.agent.trace_length:
            self.agent.append_sample(self.s_t, self.action_idx, self.r)
    @overrides
    def atari_cycle(self, observation, reward, total_t, t):
        self.total_t = total_t
        self.agent.total_t = total_t
        self.t = t
        self.r = reward
        self.set_atari_observation(observation)
        self.setAction()
        # ??? agent.learner.chosenAction = self.chosenAction  # cf. task drift (Homeostatic Pols)
    def set_atari_observation(self,obs):
        self.observation = obs  # in case of task drif
        obs = np.expand_dims(self.observation, axis=0)
        self.s_t = np.append(self.s_t[1:, :], obs, axis=0)

        if len(self.agent.rewards) == self.agent.update_freq:
            if self.testing:
                return
            self.agent.states.append(self.s_t)  # add final state
            loss = self.agent.train_model(terminal=False)
            if DEBUG_MODE:
                print("loss=" + str(loss))
    @overrides
    def setAtariTerminalObservation(self,obs):
        self.observation = obs  # in case of task drif
        obs = np.expand_dims(self.observation, axis=0)
        self.s_t = np.append(self.s_t[1:, :], obs, axis=0)
        self.agent.states.append(self.s_t)  # add final state



if __name__ == "__main__":
    # Avoid Tensorflow eats up GPU memory
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    game = DoomGame()
    game.load_config("../../scenarios/defend_the_center.cfg")
    game.set_sound_enabled(True)
    game.set_screen_resolution(ScreenResolution.RES_640X480)
    game.set_window_visible(False)
    game.init()

    # Maximum number of episodes
    max_episodes = 1000000

    game.new_episode()
    game_state = game.get_state()
    misc = game_state.game_variables  # [KILLCOUNT, AMMO, HEALTH]
    prev_misc = misc

    action_size = game.get_available_buttons_size()

    img_rows, img_cols = 64, 64
    img_channels = 3  # Color Channels
    # Convert image into Black and white
    trace_length = 4  # RNN states

    state_size = (trace_length, img_rows, img_cols, img_channels)
    agent = A2CAgent(state_size, action_size, trace_length)
    agent.model = Networks.a2c_lstm(state_size, action_size, agent.value_size, agent.learning_rate)

    # Start training
    GAME = 0
    t = 0
    max_life = 0  # Maximum episode life (Proxy for agent performance)

    # Buffer to compute rolling statistics
    life_buffer, ammo_buffer, kills_buffer = [], [], []

    for i in range(max_episodes):

        game.new_episode()
        game_state = game.get_state()
        misc = game_state.game_variables
        prev_misc = misc

        x_t = game_state.screen_buffer  # 480 x 640
        x_t = preprocessImg(x_t, size=(img_rows, img_cols))
        s_t = np.stack(tuple([x_t] * trace_length), axis=0)  # It becomes 4x64x64x3
        s_t = np.expand_dims(s_t, axis=0)  # 1x4x68x64x3

        life = 0  # Episode life

        while not game.is_episode_finished():

            loss = 0  # Training Loss at each update
            r_t = 0  # Initialize reward at time t
            a_t = np.zeros([action_size])  # Initialize action at time t

            x_t = game_state.screen_buffer
            x_t = preprocessImg(x_t, size=(img_rows, img_cols))
            x_t = np.reshape(x_t, (1, 1, img_rows, img_cols, img_channels))
            s_t = np.append(s_t[:, 1:, :, :, :], x_t, axis=1)  # 1x4x68x64x3

            # Sample action from stochastic softmax policy
            action_idx, policy = agent.get_action(s_t)
            a_t[action_idx] = 1

            a_t = a_t.astype(int)
            game.set_action(a_t.tolist())
            skiprate = agent.frame_per_action  # Frame Skipping = 4
            game.advance_action(skiprate)

            r_t = game.get_last_reward()  # Each frame we get reward of 0.1, so 4 frames will be 0.4
            # Check if episode is terminated
            is_terminated = game.is_episode_finished()

            if (is_terminated):
                # Save max_life
                if (life > max_life):
                    max_life = life
                life_buffer.append(life)
                ammo_buffer.append(misc[1])
                kills_buffer.append(misc[0])
                print ("Episode Finish ", prev_misc, policy)
            else:
                life += 1
                game_state = game.get_state()  # Observe again after we take the action
                misc = game_state.game_variables

            # Reward Shaping
            r_t = agent.shape_reward(r_t, misc, prev_misc, t)

            # Save trajactory sample <s, a, r> to the memory
            agent.append_sample(s_t, action_idx, r_t)

            # Update the cache
            t += 1
            prev_misc = misc

            if (is_terminated and t > agent.observe):
                # Every episode, agent learns from sample returns
                loss = agent.train_model()

            # Save model every 10000 iterations
            if t % 10000 == 0:
                print("Save model")
                agent.model.save_weights("models/a2c_lstm.h5", overwrite=True)

            state = ""
            if t <= agent.observe:
                state = "Observe mode"
            else:
                state = "Train mode"

            if (is_terminated):

                # Print performance statistics at every episode end
                print("TIME", t, "/ GAME", GAME, "/ STATE", state, "/ ACTION", action_idx, "/ REWARD", r_t, "/ LIFE",
                      max_life, "/ LOSS", loss)

                # Save Agent's Performance Statistics
                if GAME % agent.stats_window_size == 0 and t > agent.observe:
                    print("Update Rolling Statistics")
                    agent.mavg_score.append(np.mean(np.array(life_buffer)))
                    agent.var_score.append(np.var(np.array(life_buffer)))
                    agent.mavg_ammo_left.append(np.mean(np.array(ammo_buffer)))
                    agent.mavg_kill_counts.append(np.mean(np.array(kills_buffer)))

                    # Reset rolling stats buffer
                    life_buffer, ammo_buffer, kills_buffer = [], [], []

                    # Write Rolling Statistics to file
                    with open("statistics/a2c_lstm_stats.txt", "w") as stats_file:
                        stats_file.write('Game: ' + str(GAME) + '\n')
                        stats_file.write('Max Score: ' + str(max_life) + '\n')
                        stats_file.write('mavg_score: ' + str(agent.mavg_score) + '\n')
                        stats_file.write('var_score: ' + str(agent.var_score) + '\n')
                        stats_file.write('mavg_ammo_left: ' + str(agent.mavg_ammo_left) + '\n')
                        stats_file.write('mavg_kill_counts: ' + str(agent.mavg_kill_counts) + '\n')

        # Episode Finish. Increment game count
        GAME += 1