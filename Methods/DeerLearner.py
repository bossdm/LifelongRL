import deer.experiment.base_controllers as bc
from deer.agent import *
from overrides import overrides

from Learner import CompleteLearner

"""This module contains classes used to define the standard behavior of the agent.
It relies on the controllers, the chosen training/test policy and the learning algorithm
to specify its behavior in the environment.

.. Authors: Vincent Francois-Lavet, David Taralla
"""

from theano import config
import os
import numpy as np
import sys
import joblib
from warnings import warn

from deer.experiment import base_controllers as controllers
from deer.helper import tree
from deer.policies import EpsilonGreedyPolicy


class Parameters:
    # ----------------------
    # Experiment Parameters
    # ----------------------
    # STEPS_PER_EPOCH = 1000
    # EPOCHS = 200
    # STEPS_PER_TEST = 1000
    # PERIOD_BTW_SUMMARY_PERFS = 10

    # ----------------------
    # Environment Parameters
    # ----------------------
    FRAME_SKIP = 1

    # ----------------------
    # DQN Agent parameters:
    # ----------------------
    UPDATE_RULE = 'rmsprop'
    LEARNING_RATE = 0.0002
    LEARNING_RATE_DECAY =1.0
    DISCOUNT = 0.95
    DISCOUNT_INC = .99
    DISCOUNT_MAX = 0.95
    RMS_DECAY = 0.9
    RMS_EPSILON = 0.0001
    MOMENTUM = 0.5
    CLIP_DELTA = 100000
    EPSILON_START = 0.10
    EPSILON_MIN = .05
    EPSILON_DECAY = 1000
    UPDATE_FREQUENCY = 1 #how many time steps before training occurs
    REPLAY_MEMORY_SIZE = 100
    BATCH_SIZE = 100
    FREEZE_INTERVAL = 1000
    DETERMINISTIC = True
    DOUBLE_Q = True
    RANDOM_STATE = np.random.RandomState()
    RHO=0.9
    BATCH_TYPE='random'


class DeerLearner(CompleteLearner):
    """The NeuralAgent class wraps a deep Q-network for training and testing in a given environment.

    Attach controllers to it in order to conduct an experiment (when to train the agent, when to test,...).

    Parameters
    -----------
    environment : object from class Environment
        The environment in which the agent interacts
    q_network : object from class QNetwork
        The q_network associated to the agent
    replay_memory_size : int
        Size of the replay memory. Default : 1000000
        Size of the replay memory. Default : 1000000
    replay_start_size : int
        Number of observations (=number of time steps taken) in the replay memory before starting learning. Default: minimum possible according to environment.inputDimensions().
    batch_size : int
        Number of tuples taken into account for each iteration of gradient descent. Default : 32
    random_state : numpy random number generator
        Default : random seed.
    exp_priority : float
        The exponent that determines how much prioritization is used, default is 0 (uniform priority).
        One may check out Schaul et al. (2016) - Prioritized Experience Replay.
    train_policy : object from class Policy
        Policy followed when in training mode (mode -1)
    test_policy : object from class Policy
        Policy followed when in other modes than training (validation and test modes)
    only_full_history : boolean
        Whether we wish to train the neural network only on full histories or we wish to fill with zeroes the observations before the beginning of the episode
    """

    def __init__(self, inputDims, q_network, actions, file='',replay_memory_size=Parameters.REPLAY_MEMORY_SIZE, replay_start_size=Parameters.BATCH_SIZE, batch_size=Parameters.BATCH_SIZE,
                 random_state=np.random.RandomState(), exp_priority=0, batch_type='sequential',train_policy=None, test_policy=None,
                 only_full_history=True, reward_as_input=False):

        CompleteLearner.__init__(self,actions,file)
        self.polfile = open(self.file + 'policy.txt', "w")
        # if replay_start_size == None:
        #     replay_start_size = max(inputDims[i][0] for i in range(len(inputDims)))
        # elif replay_start_size < max(inputDims[i][0] for i in range(len(inputDims))):
        #     raise AgentError("Replay_start_size should be greater than the biggest history of a state.")



        self._controllers = []

        # --- Bind controllers to the agent ---
        # For comments, please refer to run_toy_env.py
        self.attach(bc.VerboseController(
            evaluate_on='epoch',
            periodicity=1))

        self.attach(bc.TrainerController(
            evaluate_on='action',
            periodicity=Parameters.UPDATE_FREQUENCY,
            show_episode_avg_V_value=True,
            show_avg_Bellman_residual=True))

        self.attach(bc.LearningRateController(
            initial_learning_rate=Parameters.LEARNING_RATE,
            learning_rate_decay=Parameters.LEARNING_RATE_DECAY,
            periodicity=10000))

        self.attach(bc.DiscountFactorController(
            initial_discount_factor=Parameters.DISCOUNT,
            discount_factor_growth=Parameters.DISCOUNT_INC,
            discount_factor_max=Parameters.DISCOUNT_MAX,
            periodicity=10000))

        self.attach(bc.EpsilonController(
            initial_e=Parameters.EPSILON_START,
            e_decays=Parameters.EPSILON_DECAY,
            e_min=Parameters.EPSILON_MIN,
            evaluate_on='action',
            periodicity=1000,
            reset_every='none'))

        # self.attach(bc.InterleavedTestEpochController(
        #     id=0,
        #     epoch_length=Parameters.STEPS_PER_TEST,
        #     controllers_to_disable=[0, 1, 2, 3, 4],
        #     periodicity=2,
        #     show_score=True,
        #     summarize_every=-1))

        self.obs=[]
        self.reward_as_input=reward_as_input
        self._network = q_network
        self._replay_memory_size = replay_memory_size
        self._replay_start_size = replay_start_size # make sure you gather this many observations before learning
        self._batch_size = batch_size
        self._batch_type = batch_type
        self._random_state = random_state
        self._exp_priority = exp_priority
        self._only_full_history = only_full_history
        #inputDimensions, n_actions, observation_type = np.float32, random_state = None, max_size = 1000, batch_type = 'random', only_full_history = True
        self._dataset = DataSet(inputDimensions=inputDims,n_actions=len(actions), max_size=replay_memory_size, random_state=random_state,
                                batch_type=self._batch_type, only_full_history=self._only_full_history)
        self._tmp_dataset = None  # Will be created by startTesting() when necessary
        self._mode = -1
        self._mode_epochs_length = 0
        self._total_mode_reward = 0
        self._training_loss_averages = []
        self._Vs_on_last_episode = []
        #self._in_episode = False

        self._selected_action = -1
        self._state = []
        for i in range(len(inputDims)):
            self._state.append(np.zeros(inputDims[i], dtype=config.floatX))
        if (train_policy == None):
            self._train_policy = EpsilonGreedyPolicy(q_network, len(actions), random_state, 0.05)
        else:
            self._train_policy = train_policy
        if (test_policy == None):
            self._test_policy = EpsilonGreedyPolicy(q_network, len(actions), random_state, 0.)
        else:
            self._test_policy = test_policy

        self.initEpisode()

    def initEpisode(self):
        for c in self._controllers: c.onStart(self)
        self._in_episode = True
        # initState = environment.getInitState()
        # inputDims = environment.inputDimensions()
        # for i in range(len(inputDims)):
        #     if inputDims[i][0] > 1:
        #         self._state[i][1:] = initState[i][1:]
    def setControllersActive(self, toDisable, active):
        """ Activate controller
        """
        for i in toDisable:
            self._controllers[i].setActive(active)

    def setLearningRate(self, lr):
        """ Set the learning rate for the gradient descent
        """
        self._network.setLearningRate(lr)

    def learningRate(self):
        """ Get the learning rate
        """
        return self._network.learningRate()

    def setDiscountFactor(self, df):
        """ Set the discount factor
        """
        self._network.setDiscountFactor(df)

    def discountFactor(self):
        """ Get the discount factor
        """
        return self._network.discountFactor()

    def overrideNextAction(self, action):
        """ Possibility to override the chosen action. This possibility should be used on the signal OnActionChosen.
        """
        self._selected_action = action

    def avgBellmanResidual(self):
        """ Returns the average training loss on the epoch
        """
        if (len(self._training_loss_averages) == 0):
            return -1
        return np.average(self._training_loss_averages)

    def avgEpisodeVValue(self):
        """ Returns the average V value on the episode (on time steps where a non-random action has been taken)
        """
        if (len(self._Vs_on_last_episode) == 0):
            return -1
        if (np.trim_zeros(self._Vs_on_last_episode) != []):
            return np.average(np.trim_zeros(self._Vs_on_last_episode))
        else:
            return 0

    def totalRewardOverLastTest(self):
        """ Returns the average sum of rewards per episode and the number of episode
        """
        return self._total_mode_reward / self._totalModeNbrEpisode, self._totalModeNbrEpisode

    def bestAction(self):
        """ Returns the best Action
        """
        action = self._network.chooseBestAction(self._state)
        V = max(self._network.qValues(self._state))
        return action, V

    def attach(self, controller):
        if (isinstance(controller, controllers.Controller)):
            self._controllers.append(controller)
        else:
            raise TypeError("The object you try to attach is not a Controller.")

    def detach(self, controllerIdx):
        return self._controllers.pop(controllerIdx)

    def mode(self):
        return self._mode

    # def startMode(self, mode, epochLength):
    #     if self._in_episode:
    #         raise AgentError("Trying to start mode while current episode is not yet finished. This method can be "
    #                          "called only *between* episodes for testing and validation.")
    #     elif mode == -1:
    #         raise AgentError("Mode -1 is reserved and means 'training mode'; use resumeTrainingMode() instead.")
    #     else:
    #         self._mode = mode
    #         self._mode_epochs_length = epochLength
    #         self._total_mode_reward = 0.
    #         del self._tmp_dataset
    #         self._tmp_dataset = DataSet(self._inputDimensions,len(self.actions),self._random_state, max_size=self._replay_memory_size,
    #                                     only_full_history=self._only_full_history, batch_type=self.batch_type)

    def resumeTrainingMode(self):
        self._mode = -1

    def summarizeTestPerformance(self):  # not needed
        pass

    def train(self):
        # We make sure that the number of elements in the replay memory
        # is strictly superior to self._replay_start_size before taking
        # a random batch and perform training
        if self._dataset.n_elems <= self._replay_start_size:
            return

        try:
            # states, actions, rewards, next_states, terminals, rndValidIndices = self._dataset.randomBatch(
            #     self._batch_size, self._exp_priority)
            loss, loss_ind = self._network.train(self.states, self.actions, self.rewards, self.next_states, self.terminals)
            self._training_loss_averages.append(loss)
            # if (self._exp_priority):
            #     self._dataset.updatePriorities(pow(loss_ind, self._exp_priority) + 0.0001, rndValidIndices[1])

        except SliceError as e:
            warn("Training not done - " + str(e), AgentWarning)

    def dumpNetwork(self, fname, nEpoch=-1):
        """ Dump the network

        Parameters
        -----------
        fname : string
            Name of the file where the network will be dumped
        nEpoch : int
            Epoch number (Optional)
        """
        try:
            os.mkdir("nnets")
        except Exception:
            pass
        basename = "nnets/" + fname

        for f in os.listdir("nnets/"):
            if fname in f:
                os.remove("nnets/" + f)

        all_params = self._network.getAllParams()

        if (nEpoch >= 0):
            joblib.dump(all_params, basename + ".epoch={}".format(nEpoch))
        else:
            joblib.dump(all_params, basename, compress=True)

    def setNetwork(self, fname, nEpoch=-1):
        """ Set values into the network

        Parameters
        -----------
        fname : string
            Name of the file where the values are
        nEpoch : int
            Epoch number (Optional)
        """

        basename = "nnets/" + fname

        if (nEpoch >= 0):
            all_params = joblib.load(basename + ".epoch={}".format(nEpoch))
        else:
            all_params = joblib.load(basename)

        self._network.setAllParams(all_params)

    def run(self, n_epochs, epoch_length):
        for c in self._controllers: c.onStart(self)
        i = 0
        while i < n_epochs or self._mode_epochs_length > 0:
            self._training_loss_averages = []

            if self._mode != -1:
                self._totalModeNbrEpisode = 0
                while self._mode_epochs_length > 0:
                    self._totalModeNbrEpisode += 1
                    self._mode_epochs_length = self._runEpisode(self._mode_epochs_length)
            else:
                length = epoch_length
                while length > 0:
                    length = self._runEpisode(length)
                i += 1
            for c in self._controllers: c.onEpochEnd(self)

    # def end(self):
    #     self._environment.end()
    #     for c in self._controllers: c.onEnd(self)
    @overrides
    def setObservation(self,agent,environment):
        environment.setObservation(agent)
    @overrides
    def performAction(self,agent,environment):
        self.chosenAction.perform(agent,environment)
    @overrides
    def setAction(self):
        pass
    @overrides
    def cycle(self,agent,environment):
        environment.setObservation(agent)

        if self.reward_as_input:
            self.observation.append(self.r)

        if len(self.obs) > self._dataset._batch_dimensions[0]:
            self.obs.append(np.array(self.observation))
            del self.obs[0]
            for i in range(self._dataset._batch_dimensions[0]):
                self._state[i][0:-1] = self._state[i][1:]
                self._state[i][-1] = self.observation[i]
        else:
            self.obs.append(np.array(self.observation))

        V, action = self._step()
        self.chosenAction = self.actions[action]
        self.chosenAction.perform([agent,environment])
        self.setReward(environment.currentTask.reward_fun(agent, environment))

        self._Vs_on_last_episode.append(V)
        if self._mode != -1:
            self._total_mode_reward += self.r
            self._totalModeNbrEpisode += 1 #we will just define an episode as one time step

        is_terminal = False #no terminal states in lifelong learning

        self._addSample(self.observation, action, self.r, is_terminal)
        self.environment = environment

        for c in self._controllers: c.onActionTaken(self)
        for c in self._controllers: c.onEpochEnd(self)
        # if is_terminal:
        #     self.endEpisode(is_terminal,self.r) #though not needed for life-long learning ! ?

    @overrides
    def learn(self):
        pass


    @overrides
    def printPolicy(self):
        pass
        # self.polfile.write("Final Pol: \n")
        # self.polfile.write("\n")
        # for s in len(self.states):
        #     qvals=self._network.qValues(self, s)
        #     for a in len(self.actions):
        #         self.polfile.write('('+str(s)+',' + self.actions[a].function.__name__+') : \t')
        #         self.polfile.write('Q=%.2f'%( qvals[a],))
    @overrides
    def reset(self):
        pass
    def endEpisode(self,is_terminal,reward):
        self._in_episode = False
        for c in self._controllers: c.onEpisodeEnd(self, is_terminal, reward)

    def _step(self):
        """
        This method is called at each time step. If the agent is currently in testing mode, and if its *test* replay
        memory has enough samples, it will select the best action it can. If there are not enough samples, FIXME.
        In the case the agent is not in testing mode, if its replay memory has enough samples, it will select the best
        action it can with probability 1-CurrentEpsilon and a random action otherwise. If there are not enough samples,
        it will always select a random action.
        Parameters
        -----------
        state : ndarray
            An ndarray(size=number_of_inputs, dtype='object), where states[input] is a 1+D matrix of dimensions
               input.historySize x "shape of a given ponctual observation for this input".
        Returns
        -------
        action : int
            The id of the action selected by the agent.
        V : float
            Estimated value function of current state.
        """

        action, V = self._chooseAction()

        # print('state='+str(self.observation))
        # print('action='+str(self.actions[action].function.__name__))
        # print('value='+str(V))
        return V, action

    def _addSample(self, ponctualObs, action, reward, is_terminal):
        if self._mode != -1:
            self._tmp_dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1)
        else:
            self._dataset.addSample(ponctualObs, action, reward, is_terminal, priority=1)

    def _chooseAction(self):

        if self._mode != -1:
            # Act according to the test policy if not in training mode
            action, V = self._test_policy.action(self._state)
        else:
            if self._dataset.n_elems >= self._replay_start_size:
                # follow the train policy
                # follow the train policy
                action, V = self._train_policy.action(
                    self._state)  # is self._state the only way to store/pass the state?
            else:
                # Still gathering initial data: choose dummy action
                action, V = self._train_policy.randomAction()

        for c in self._controllers: c.onActionChosen(self, action)
        return action, V


class AgentError(RuntimeError):
    """Exception raised for errors when calling the various Agent methods at wrong times.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


class AgentWarning(RuntimeWarning):
    """Warning issued of the various Agent methods.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """


class DataSet(object):
    """A replay memory consisting of circular buffers for observations, actions, rewards and terminals."""

    def __init__(self, inputDimensions, n_actions, observation_type = np.float32,random_state=None, max_size=1000, batch_type='random', only_full_history=True):
        """Initializer.
        Parameters
        -----------
        inputDims : list of tuples
            For each subject i, inputDims[i] is a tuple where the first value is the memory size for this
            subject and the rest describes the shape of each single observation on this subject (number, vector or
            matrix). See base_classes.Environment.inputDimensions() documentation for more info about this format.
        random_state : Numpy random number generator
            If None, a new one is created with default numpy seed.
        max_size : The replay memory maximum size.
        """

        self._batch_dimensions = inputDimensions
        self._max_history_size = np.max([self._batch_dimensions[i][0] for i in range(len(self._batch_dimensions))])
        self._size = max_size
        self._batch_type = batch_type
        self._only_full_history = only_full_history
        if (isinstance(n_actions, int)):
            self._actions = CircularBuffer(max_size, dtype="int8")
        else:
            self._actions = CircularBuffer(max_size, dtype='object')
        self._rewards = CircularBuffer(max_size)
        self._terminals = CircularBuffer(max_size, dtype="bool")
        if (batch_type=='prioritised'):
            self._priority_tree = tree.SumTree(max_size)
            self._translation_array = np.zeros(max_size)

        self._observations = np.zeros(len(self._batch_dimensions), dtype='object')
        # Initialize the observations container if necessary
        for i in range(len(self._batch_dimensions)):
            self._observations[i] = CircularBuffer(max_size, elemShape=self._batch_dimensions[i][1:],
                                                   dtype=observation_type,)

        if (random_state == None):
            self._random_state = np.random.RandomState()
        else:
            self._random_state = random_state

        self.n_elems = 0

    def actions(self):
        """Get all actions currently in the replay memory, ordered by time where they were taken."""

        return self._actions.getSlice(0)

    def rewards(self):
        """Get all rewards currently in the replay memory, ordered by time where they were received."""

        return self._rewards.getSlice(0)

    def terminals(self):
        """Get all terminals currently in the replay memory, ordered by time where they were observed.

        terminals[i] is True if actions()[i] lead to a terminal state (i.e. corresponded to a terminal
        transition), and False otherwise.
        """

        return self._terminals.getSlice(0)

    def observations(self):
        """Get all observations currently in the replay memory, ordered by time where they were observed.

        observations[s][i] corresponds to the observation made on subject s before the agent took actions()[i].
        """

        ret = np.zeros_like(self._observations)
        for input in range(len(self._observations)):
            ret[input] = self._observations[input].getSlice(0)

        return ret

    def updatePriorities(self, priorities, rndValidIndices):
        """
        """
        for i in range(len(rndValidIndices)):
            self._priority_tree.update(rndValidIndices[i], priorities[i])

    def randomBatch(self, size, use_priority):
        """Return corresponding states, actions, rewards, terminal status, and next_states for size randomly
        chosen transitions. Note that if terminal[i] == True, then next_states[s][i] == np.zeros_like(states[s][i]) for
        each subject s.

        Parameters
        -----------
        size : int
            Number of transitions to return.
        Returns
        -------
        states : ndarray
            An ndarray(size=number_of_subjects, dtype='object), where states[s] is a 2+D matrix of dimensions
            size x s.memorySize x "shape of a given observation for this subject". States were taken randomly in
            the data with the only constraint that they are complete regarding the histories for each observed
            subject.
        actions : ndarray
            An ndarray(size=number_of_subjects, dtype='int32') where actions[i] is the action taken after
            having observed states[:][i].
        rewards : ndarray
            An ndarray(size=number_of_subjects, dtype='float32') where rewards[i] is the reward obtained for
            taking actions[i-1].
        next_states : ndarray
            Same structure than states, but next_states[s][i] is guaranteed to be the information
            concerning the state following the one described by states[s][i] for each subject s.
        terminals : ndarray
            An ndarray(size=number_of_subjects, dtype='bool') where terminals[i] is True if actions[i] lead
            to terminal states and False otherwise
        Throws
        -------
            SliceError
                If a batch of this size could not be built based on current data set (not enough data or all
                trajectories are too short).
        """

        if (self._max_history_size - 1 >= self.n_elems):
            raise SliceError(
                "Not enough elements in the dataset to create a "
                "complete state. {} elements in dataset; requires {}"
                    .format(self.n_elems, self._max_history_size))

        if self._batch_type=='prioritised':
            # FIXME : take into account the case where self._only_full_history is false
            rndValidIndices, rndValidIndices_tree = self._randomPrioritizedBatch(size)
            if (rndValidIndices.size == 0):
                raise SliceError("Could not find a state with full histories")
        elif self._batch_type=='random':
            rndValidIndices = np.zeros(size, dtype='int32')
            if (self._only_full_history):
                for i in range(size):  # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(self._max_history_size)
            else:
                for i in range(size):  # TODO: multithread this loop?
                    rndValidIndices[i] = self._randomValidStateIndex(minimum_without_terminal=1)
        # elif self._batch_type=='sequential':
        #     rndValidIndices=np.array(range(size),dtype='int32')
        else:
            raise NotImplementedError()

        actions = np.vstack(self._actions.getSliceBySeq(rndValidIndices))
        rewards = self._rewards.getSliceBySeq(rndValidIndices)
        terminals = self._terminals.getSliceBySeq(rndValidIndices)

        states = np.zeros(len(self._batch_dimensions), dtype='object')
        next_states = np.zeros_like(states)
        # We calculate the first terminal index backward in time and set it
        # at maximum to the value self._max_history_size
        first_terminals = []
        for rndValidIndex in rndValidIndices:
            first_terminal = 1
            while first_terminal < self._max_history_size:
                if (self._terminals[rndValidIndex - first_terminal] == True or first_terminal > rndValidIndex):
                    break
                first_terminal += 1
            first_terminals.append(first_terminal)

        for input in range(len(self._batch_dimensions)):
            states[input] = np.zeros((size,) + self._batch_dimensions[input], dtype=self._observations[input].dtype)
            next_states[input] = np.zeros_like(states[input])
            for i in range(size):
                slice = self._observations[input].getSlice(
                    rndValidIndices[i] + 1 - min(self._batch_dimensions[input][0], first_terminals[i]),
                    rndValidIndices[i] + 1)
                if (len(slice) == len(states[input][i])):
                    states[input][i] = slice
                else:
                    for j in range(len(slice)):
                        states[input][i][-j - 1] = slice[-j - 1]
                        # If transition leads to terminal, we don't care about next state
                if rndValidIndices[i] >= self.n_elems - 1 or terminals[i]:
                    next_states[input][i] = np.zeros_like(states[input][i])
                else:
                    slice = self._observations[input].getSlice(
                        rndValidIndices[i] + 2 - min(self._batch_dimensions[input][0], first_terminals[i] + 1),
                        rndValidIndices[i] + 2)
                    if (len(slice) == len(states[input][i])):
                        next_states[input][i] = slice
                    else:
                        for j in range(len(slice)):
                            next_states[input][i][-j - 1] = slice[-j - 1]
                            # next_states[input][i] = self._observations[input].getSlice(rndValidIndices[i]+2-min(self._batch_dimensions[input][0],first_terminal), rndValidIndices[i]+2)

        if (self._batch_type=='prioritised'):
            return states, actions, rewards, next_states, terminals, [rndValidIndices, rndValidIndices_tree]
        elif (self._batch_type=='random' or self._batch_type=='sequential'):
            return states, actions, rewards, next_states, terminals, rndValidIndices
        else:
            raise NotImplementedError()

    def _randomValidStateIndex(self, minimum_without_terminal):
        """ Returns the index corresponding to a timestep that is valid
        """
        index_lowerBound = minimum_without_terminal - 1
        # We try out an index in the acceptable range of the replay memory
        index = self._random_state.randint(index_lowerBound, self.n_elems - 1)

        # Check if slice is valid wrt terminals
        # The selected index may correspond to a terminal transition but not
        # the previous minimum_without_terminal-1 transition
        firstTry = index
        startWrapped = False
        while True:
            i = index - 1
            processed = 0
            for _ in range(minimum_without_terminal - 1):
                if (i < 0 or self._terminals[i]):
                    break;

                i -= 1
                processed += 1
            if (processed < minimum_without_terminal - 1):
                # if we stopped prematurely, shift slice to the left and try again
                index = i
                if (index < index_lowerBound):
                    startWrapped = True
                    index = self.n_elems - 1
                if (startWrapped and index <= firstTry):
                    raise SliceError("Could not find a state with full histories")
            else:
                # else index was ok according to terminals
                return index

    def _randomPrioritizedBatch(self, size):
        indices_tree = self._priority_tree.getBatch(
            size, self._random_state, self)
        indices_replay_mem = np.zeros(indices_tree.size, dtype='int32')
        for i in range(len(indices_tree)):
            indices_replay_mem[i] = int(self._translation_array[indices_tree[i]] \
                                        - self._actions.getLowerBound())

        return indices_replay_mem, indices_tree

    def addSample(self, obs, action, reward, is_terminal, priority):
        """Store a (observation[for all subjects], action, reward, is_terminal) in the dataset.
        Parameters
        -----------
        obs : ndarray
            An ndarray(dtype='object') where obs[s] corresponds to the observation made on subject s before the
            agent took action [action].
        action :  int
            The action taken after having observed [obs].
        reward : float
            The reward associated to taking this [action].
        is_terminal : bool
            Tells whether [action] lead to a terminal state (i.e. corresponded to a terminal transition).
        priority : float
            The priority to be associated with the sample
        """
        # Store observations
        for i in range(len(self._batch_dimensions)):
            self._observations[i].append(obs[i])

        # Update tree and translation table
        if (self._batch_type=='prioritised'):
            index = self._actions.getIndex()
            if (index >= self._size):
                ub = self._actions.getUpperBound()
                true_size = self._actions.getTrueSize()
                tree_ind = index % self._size
                if (ub == true_size):
                    size_extension = true_size - self._size
                    # New index
                    index = self._size - 1
                    tree_ind = -1
                    # Shift translation array
                    self._translation_array -= size_extension + 1
                tree_ind = np.where(self._translation_array == tree_ind)[0][0]
            else:
                tree_ind = index

            self._priority_tree.update(tree_ind)
            self._translation_array[tree_ind] = index

        # Store rest of sample
        self._actions.append(action)
        self._rewards.append(reward)
        self._terminals.append(is_terminal)

        if (self.n_elems < self._size):
            self.n_elems += 1


class CircularBuffer(object):
    def __init__(self, size, elemShape=(), extension=0.1, dtype="float32"):
        self._size = size
        self._data = np.zeros((int(size + extension * size),) + elemShape, dtype=dtype)
        self._trueSize = self._data.shape[0]
        self._lb = 0
        self._ub = size
        self._cur = 0
        self.dtype = dtype

    def append(self, obj):
        if self._cur >= self._size:
            self._lb += 1
            self._ub += 1

        if self._ub > self._trueSize:
            self._data[0:self._size - 1] = self._data[self._lb:]
            self._lb = 0
            self._ub = self._size
            self._cur = self._size - 1

        self._data[self._cur] = obj

        self._cur += 1

    def __getitem__(self, i):
        return self._data[self._lb + i]

    def getSliceBySeq(self, seq):
        return self._data[seq + self._lb]

    def getSlice(self, start, end=sys.maxsize):
        if end == sys.maxsize:
            return self._data[self._lb + start:self._cur]
        else:
            return self._data[self._lb + start:self._lb + end]

    def getLowerBound(self):
        return self._lb

    def getUpperBound(self):
        return self._ub

    def getIndex(self):
        return self._cur

    def getTrueSize(self):
        return self._trueSize


class SliceError(LookupError):
    """Exception raised for errors when getting slices from CircularBuffers.
    Attributes:
        expr -- input expression in which the error occurred
        msg  -- explanation of the error
    """

    def __init__(self, value):
        self.value = value

    def __str__(self):
        return repr(self.value)


if __name__ == "__main__":
    pass
