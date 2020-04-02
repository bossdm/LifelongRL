import numpy as np
from ExperimentUtils import dump_incremental
from enum import Enum
from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
from Catastrophic_Forgetting_NNs.drqn_small import DoubleDRQNAgent
from IS.SSA import SSA_with_WM, SSAimplementor
from IS.StackEntries import *
from IS.ArgumentConversions import conversion, narrowing_conversion
from overrides import overrides
from StatsAndVisualisation.metrics import getDifferential

from IS.IS_LearnerStatistics import IS_LearnerStatistics, IS_NEAT_LearnerStatistics
from Actions.SpecialActions import *
from Catastrophic_Forgetting_NNs.CustomNetworks import CustomNetworks

DEBUG_MODE = False


class ConversionType(Enum):
    direct = 0
    single_index = 1
    double_index = 2


class SSA_gradientQ_Stats(IS_NEAT_LearnerStatistics):


    def __init__(self, learner):
        IS_NEAT_LearnerStatistics.__init__(self, learner)
    # def init_dict(self,learner,init):
    #     return {task: [deepcopy(init) for i in range(len(learner.Qlearners))] for task in learner.tasks}
    # def init_dict_nops(self,learner,init):
    #     return {task: [[deepcopy(init)  for i in range(learner.n_ops)] for i in range(len(learner.Qlearners))] for task in learner.tasks}
    def set_tasks(self,learner):
        self.slicerewards=0
        self.slicerewardsOverTime=[]
        self.match_jumpexperience = init=0
        self.match_jumpexperienceOverTime = init=[]
        self.num_experience_changes=init=0
        self.num_experience_changesOverTime=init=[]
        self.num_network_trainings=init=0
        self.num_network_trainingsOverTime=init=[]
        self.batch_size=init=0
        self.batch_sizeOverTime=init=[]
        self.time_outs=init=0
        self.time_outsOverTime=init=[]
        self.match_experience=init=0
        self.match_experienceOverTime=init=[]

        self.experience_diversity=init=[]
        self.buffer_diversity=init=[]
        self.Q_calls=init=0
        self.epsilon=(0,0)
        self.epsilonOverTime=[]

        self.Q_callsOverTime=[]

        self.loops=init=0
        self.loopsOverTime=[]
        self.match_steps=init=0
        self.match_stepsOverTime=[]
        self.until_time_steps=init=0
        self.until_time_stepsOverTime=[]
        if learner.Q_internal:
            self.Qinternals=init=0
            self.QinternalsOverTime=[]

    def initialise_statistics(self, numnets=1):
        IS_LearnerStatistics.initialise_statistics(self)
        self.developstats['correctnessNP'] = self.correctnessNP
        self.finalstats['NPStackSize'] = 0
        self.developstats['NPmodifications'] = self.numNPModificationsOverTime
        self.developstats['validNPmodifications'] = []
        self.developstats['validnonredNPmodifications'] = []
        self.developstats['experience_diversity'] = []
        self.finalstats['numnets'] = numnets

        self.finalstats['experience_set'] = self.final_experience_set
        if len(self.final_experience_set) > 0:
            m = SSA_gradientQ.matrix_format_exp(self.final_experience_set)
            self.finalstats['unique_experiences'] = len(np.unique(m))
        eps = [[self.epsilonOverTime[i][j]
                for j in range(len(self.epsilonOverTime[i])) if not isinstance(self.epsilonOverTime[i][j], tuple)]
               for i in range(len(self.epsilonOverTime))]
        self.epsilonOverTime = eps
        self.developstats['epsilonOverTime'] = [[] for i in range(len(self.epsilonOverTime))]
        self.developstats['batch_sizeOverTime'] = []
        self.developstats['match_stepsOverTime'] = []
        self.developstats['until_time_stepsOverTime'] = []

        self.developstats['loopsOverTime'] = []

        self.developstats['Q_callsOverTime'] = []
        self.developstats['num_experience_changesOverTime'] = []
        self.developstats['num_network_trainingsOverTime'] = []

        self.developstats['time_outsOverTime'] = []

        self.developstats['match_experienceOverTime'] = []
        self.developstats['match_jumpexperienceOverTime'] = []

    def development_statistics_iteration(self, j, Vstep, opt_speed=1.):
        IS_LearnerStatistics.development_statistics_iteration(self, j, Vstep, opt_speed)
        if self.numValidNPModificationsOverTime:
            NPentries = getDifferential(self.numValidNPModificationsOverTime, j, Vstep)
            # nonredNPentries=getDifferential(self.numValidNonRedNPModificationsOverTime,j,Vstep)
        else:
            return

        averageExperienceDiversity = np.mean(self.experience_diversity[j:j + Vstep])
        self.finalstats['NPStackSize'] += NPentries
        self.finalstats['StackSize'] += NPentries
        self.developstats['validNPmodifications'].append(NPentries)
        # self.developstats['validnonredNPmodifications'].append(nonredNPentries)
        self.developstats['experience_diversity'].append(averageExperienceDiversity)

        self.developstats['batch_sizeOverTime'].append(np.mean(self.batch_sizeOverTime[j:j + Vstep]))
        self.developstats['match_stepsOverTime'].append(np.mean(self.match_stepsOverTime[j:j + Vstep]))
        self.developstats['until_time_stepsOverTime'].append(np.mean(self.until_time_stepsOverTime[j:j + Vstep]))
        self.developstats['loopsOverTime'].append(np.sum(self.loopsOverTime[j:j + Vstep]))

        self.developstats['Q_callsOverTime'].append(np.sum(self.Q_callsOverTime[j:j + Vstep]))
        self.developstats['num_experience_changesOverTime'].append(
            np.sum(self.num_experience_changesOverTime[j:j + Vstep]))
        self.developstats['num_network_trainingsOverTime'].append(
            np.sum(self.num_network_trainingsOverTime[j:j + Vstep]))

        self.developstats['time_outsOverTime'].append(np.sum(self.time_outsOverTime[j:j + Vstep]))

        self.developstats['match_experienceOverTime'].append(np.sum(self.match_experienceOverTime[j:j + Vstep]))
        try:
            self.developstats['match_jumpexperienceOverTime'].append(
                np.sum(self.match_jumpexperienceOverTime[j:j + Vstep]))
        except:
            print("no match_jumpexperience ?")
        for i in range(len(self.epsilonOverTime)):
            self.developstats['epsilonOverTime'][i].append(np.mean(self.epsilonOverTime[i][j:j + Vstep]))


class SSA_gradientQ(SSA_with_WM):
    recorded_Qs = {}
    recorded_loss = {}
    recorded_targets = {}
    recorded_usage = {}
    recorded_time = {}
    recorded_match = {}
    recorded_timesteps = {}
    recorded_matchsteps = {}
    recorded_epsilon = {}
    intervals = None
    max_batchsize = 64
    sampling_start = max_batchsize
    network_usage = True
    maxNetworks = 1
    Q_internal = False
    fixed_training = False
    trained = False
    match = 0
    time = 0
    experience_type2 = False


    use_reflexes = False
    no_external_actions = False
    num_slices = 1
    add_IP = False
    epsilon_scale = .005  # multiplied by n_ops --> .20 --> strategy ranges from greedy to .20 random selection
    stat_frequency=50000
    def __init__(self, n_outputs, trace_length, input_addresses, conversion_type, SSA_WM_Params, Q_internal=False,
                 fixed_training=False, intervals=True, DRQN_params=None, init_with_training_freq=False):
        self.intervals = intervals
        self.Q_internal = Q_internal
        self.n_outputs = n_outputs
        self.fixed_training = fixed_training
        self.old_input = None
        self.new_input = None
        if self.Q_internal:
            self.n_outputs += 1

        self.r_steps = trace_length
        self.episode_buf = []
        self.trace_length = trace_length
        self.min_untiltime = 1
        self.max_untiltime = max(12, self.trace_length / 2.)
        self.next_chosenAction = None
        self.usedNPinstruction = None
        self.conversion_type = conversion_type
        self.net_key = 0
        self.s_t = None
        self.s_t1 = None
        self.experience_set={}

        #self.use_reflexes = False if SSA_NEAT_Params is None else True

        external_actions = SSA_WM_Params['actions']
        if self.no_external_actions:
                SSA_WM_Params['actions']=[]
        SSA_with_WM.__init__(self, **SSA_WM_Params)

        self.num_external_actions, self.external_actions = SSAimplementor.count_external_actions(external_actions)
        self.input_addresses = [a + self.Min for a in input_addresses]
        if self.use_jump_experience:
            self.last_jump=0
        n_inputs = len(input_addresses) + 1 if self.use_jump_experience and self.add_IP else len(input_addresses)
        state_size = (trace_length, n_inputs)
        self.slice_size = self.m/self.num_slices
        self.current_IP_slice = self.get_current_IP_slice(self.IP)
        if DRQN_params is None:
            self.Qlearner = DoubleDRQNAgent(state_size, n_outputs, self.trace_length, batch_size=None,
                                            episodic=False, double=False)
            input_shape = (None, trace_length, len(input_addresses))
            # input_shape, action_size, learning_rate, task_features, use_task_bias, use_task_gain
            self.Qlearner.model = CustomNetworks.small_scale_drqn(input_shape, n_outputs, task_features=[],
                                                                  use_task_bias=False, use_task_gain=False,
                                                                  num_neurons=50)
            self.Qlearner.target_model = CustomNetworks.small_scale_drqn(input_shape, n_outputs, task_features=[],
                                                                         use_task_bias=False, use_task_gain=False,
                                                                         num_neurons=50)
        else:
            assert trace_length == DRQN_params['trace_length']
            DRQN_params["n_inputs"] = n_inputs
            self.init_DRQN(self.num_slices,DRQN_params)
        self.state_size = self.Qlearner.state_size
        self.loop = False
        self.loopTime = 0
        self.epsilon_scale = 2. * self.Qlearner.final_epsilon / self.n_ops
        if init_with_training_freq:
            replay_freq = 0.35  # training frequency of 4 for usual methods, +1 due to loop
            avg_steps = float(self.min_untiltime + self.max_untiltime) / 2.

            # .95 the desired probability of using DRQN for each external action taken;
            loop_freq = 0.35 # if avg steps==10 and
            train_replay_index = next((i for i, a in enumerate(self.actions) if str(a) == "train_replay"), None)
            loop_index = next((i for i, a in enumerate(self.actions) if str(a) == "doQuntil" or str(a) == "doQuntil2"),
                              None)
            # if self.use_reflexes:
            #     # bias towards changing weights, just a little bit
            #     weight_change_index=next((i for i, a in enumerate(self.actions) if str(a) == "weight_change"), None)
            #     weight_change_freq=0.04
            # else:
            #     weight_change_freq=0.0
            if train_replay_index is not None:
                for IP in range(self.ProgramStart, self.Max + 1):
                    if self.separate_arguments and (IP - self.ProgramStart) % (1 + self.max_arguments) != 0:
                        continue
                    for index in range(self.n_ops):
                        if index == train_replay_index:
                            self.Pol[IP].p[index] = replay_freq
                        elif index == loop_index:
                            self.Pol[IP].p[index] = loop_freq
                        # elif index==weight_change_index:
                        #     self.Pol[IP].p[index] = weight_change_freq
                        else:
                            self.Pol[IP].p[index] -= (replay_freq + loop_freq) / float(self.n_ops - 2)
                    ps = self.Pol[IP].p
                    C = sum(self.Pol[IP].p)
                    self.Pol[IP].p = [p / C for p in ps]

    def get_normalised_IPslice(self):
        return (self.get_current_IP_slice() - self.ProgramStart) / float(self.m)

    def get_inputs(self, net_key):
        inputs = [self.get_normalised_IPslice()]
        for k in self.input_addresses:  # first one is IP
            inputs.append(self.c[k] / float(self.MaxInt))
        return inputs

    def get_inputs_or_data(self, net_key):
        return self.get_inputs(net_key)

    def initStats(self):
        self.stats = SSA_gradientQ_Stats(self)

    def init_DRQN(self, num_networks, DRQN_params):
        self.Qlearners=[]
        for network in range(num_networks):
            self.Qlearners.append(DRQN_Learner.init_agent(**DRQN_params))

        self.setQlearner(0)
    def set_tasks(self,tasks):
        self.tasks=tasks
        self.stats.set_tasks(self)

    def setQlearner(self,index):
        self.current_Q_index = index
        self.Qlearner = self.Qlearners[index]
    @overrides
    def reset(self):
        """
        at end of elementary task
        :return:
        """
        if self.episodic:
            self.loop = False
            if self.task_time > 0:
                self.zero_padding()
                self.Qlearner.memory.add(self.episode_buf)

    @overrides
    def new_elementary_task(self):
        """
        at start of the elementary task
        :return:
        """
        SSA_with_WM.new_elementary_task(self)
        if self.episodic:
            self.episode_buf = []  # Reset Episode Buf
            self.last_jump=0
    def new_task(self,feature):
        self.current_feature = tuple(feature)

    def zero_padding(self):
        trace_length, _ = self.state_size
        while len(self.episode_buf) < trace_length:
            self.episode_buf.insert(0, self.null_experience())

    def null_state(self, n_inputs):
        return np.zeros(n_inputs) - 1

    def null_experience(self):
        _, n_inputs = self.state_size
        return [self.null_state(n_inputs), 0., 0., self.null_state(n_inputs)]

    @overrides
    def continue_experiment(self, intervals):
        self.intervals = intervals
        if intervals:
            self.Qlearner.loss_file = open(self.file + "_loss", mode="wb")

    @overrides  # (SSAimplementor)
    def restore(self, entry):  # evaluate if condition is met
            if isinstance(entry, StackEntry):  # note: TaskSpecificSSA will also hit this
                self.Pol[entry.address] = entry.oldP  # to restore old policy
            elif isinstance(entry, PolicyStackEntry):
                self.Pol = entry.pol
            elif isinstance(entry, StackEntryNP):
                    if self.experience_type2:
                        IP, index = entry.address
                        self.experience_set[IP][index] = entry.oldNP
                    else:
                        index = entry.address
                        self.experience_set[index] = entry.oldNP


            else:

                raise Exception(str(type(entry)))

    def parseInternalActions(self, list):
        actions = []
        self.update_model_in_instructions = False
        self.use_jump_experience = False
        for key, value in list.items():
            function = getattr(self, key)
            if key in ['incP', 'decP', 'searchP', 'sample', 'inc_mean', 'dec_mean']:
                for i in range(1 + self.enhance_PLA):
                    actions.append(PLA(function, value))

            elif key in ['train_replay', 'set_experience']:
                actions.append(NeatPLA(function, value))
            elif key == "set_experience2":
                self.experience_type2 = True
                actions.append(NeatPLA(function, value))
            else:
                actions.append(Action(function, value))
                if key == "updateModel":
                    self.update_model_in_instructions = True
                if "jump_experience" in key:
                    self.use_jump_experience = True
            if value > self.max_arguments:
                self.max_arguments = value
            print(actions[-1].function.__name__)

        return actions

    def convert_arg(self, a, m, M):
        if self.conversion_type == ConversionType.direct:
            return conversion(a, (0, self.n_ops - 1), (m, M))
        elif self.conversion_type == ConversionType.single_index:
            return narrowing_conversion(self.c[a], (-self.MaxInt, self.MaxInt), (m, M))
        elif self.conversion_type == ConversionType.double_index:
            ca = self.c[a]
            if self.relaxedreadcondition(ca):
                return narrowing_conversion(self.c[ca], (-self.MaxInt, self.MaxInt), (m, M))
            else:
                return None
        else:
            raise Exception("impossible conversiontype %s" % (str(self.conversion_type)))

    def set_input_state(self):
        """
        define the new observation
        :return:
        """
        input = [self.c[input] / float(self.MaxInt) for input in self.input_addresses]
        if self.use_jump_experience and self.add_IP:
            input = [self.get_normalised_IPslice()] + input
        self.s_t1 = np.array(input)

    def check_replay_ready(self):
        return self.fixed_training and self.t > self.Qlearner.replay_start_size and self.t % self.Qlearner.update_freq == 0
        # if DEBUG_MODE:
        #     print("train")

    @overrides
    def setObservation(self, agent, environment):
        SSA_with_WM.setObservation(self, agent, environment)
        self.observation = agent.learner.observation  # in case of task drif

        if self.t % self.Qlearner.update_target_freq == 0:
            self.Qlearner.update_target_model()
        if not self.check_replay_ready():
            self.trained = False
        else:
            if not self.trained:
                self.trained = True
                Q_max, loss = self.Qlearner._train_replay(self.Qlearner.batch_size)
                if DEBUG_MODE:
                    print("Qmax=" + str(Q_max))
                    print("loss=" + str(loss))

        if not environment.observation_set:
            self.set_input_state()
            self.current_experience()
            self.new_input = self.get_input()
    def setTime(self,t):
        inc=t-self.t
        self.Qlearner.total_t += inc
        SSA_with_WM.setTime(self,t)

    def setReward(self,reward):
        SSA_with_WM.setReward(self,reward)

    def setTerminalObservation(self, agent, environment):
        self.setObservation(agent, environment)

    @overrides
    def printDevelopment(self):
        SSA_with_WM.printDevelopment(self)
        if self.t % self.stat_frequency==0:
                    b = 0 if self.stats.num_network_trainings == 0 else self.stats.batch_size / float(self.stats.num_network_trainings)
                    self.stats.batch_sizeOverTime.append(b)
                    self.stats.batch_size = 0

                    m = 0 if self.stats.match_experience == 0 else self.stats.match_steps / float(self.stats.match_experience)
                    self.stats.match_stepsOverTime.append(m)
                    self.stats.match_steps = 0

                    u = 0 if self.stats.time_outs == 0 else self.stats.until_time_steps / float(self.stats.time_outs)
                    self.stats.until_time_stepsOverTime.append(u)
                    self.stats.until_time_steps = 0

                    self.stats.loopsOverTime.append(self.stats.loops)
                    self.stats.loops = 0

                    self.stats.Q_callsOverTime.append(self.stats.Q_calls)
                    self.stats.Q_calls = 0

                    self.stats.num_experience_changesOverTime.append(self.stats.num_experience_changes)
                    self.stats.num_experience_changes = 0

                    self.stats.num_network_trainingsOverTime.append(self.stats.num_network_trainings)
                    self.stats.num_network_trainings = 0

                    self.stats.time_outsOverTime.append(self.stats.time_outs)
                    self.stats.time_outs = 0

                    self.stats.match_experienceOverTime.append(self.stats.match_experience)
                    self.stats.match_experience = 0
                    self.stats.match_jumpexperienceOverTime.append(self.stats.match_jumpexperience)
                    self.stats.match_jumpexperience = 0


                    if self.Qlearner.total_t > self.Qlearner.replay_start_size:
                        IPslice=self.ProgramStart+i*self.slice_size
                        self.stats.experience_diversity.append(self.get_experience_diversity(self.experience_set))
                        for j in range(len(self.experience_set)):
                            e = 0 if self.stats.epsilon[j][0] == 0 else self.stats.epsilon[j][0] / float(self.stats.epsilon[j][1])
                            self.stats.epsilonOverTime[j].append(e)
                            self.stats.epsilon[j] = (0., 0)
                    if self.Q_internal:
                        self.stats.QinternalsOverTime.append(self.stats.Qinternals)
                        self.stats.Qinternals = 0

    def is_P_modification(self, entry):
        return isinstance(entry, StackEntry)

    def is_NP_modification(self, entry):
        return isinstance(entry, StackEntryNP)

    @overrides
    def printStatistics(self):
        # see number of modifications and their success over time
        self.Stackfile = open(self.file + 'Stack.txt', "w")
        ind = 0
        s = 0
        s2 = 0
        s3 = 0
        self.stats.numValidPModificationsOverTime = []
        self.stats.numValidNPModificationsOverTime = []
        self.stats.numValidNonRedNPModificationsOverTime = []
        temp_set = [None for i in range(self.n_ops)]
        result = []
        resultNP = []
        num_slices = max(1, len(self.stats.R_overTime))
        for tt in range(0, int(self.t), int(self.t) / num_slices):
            # get all entries before
            time = tt + int(self.t) / num_slices
            tempP = []
            tempNP = []
            for i in range(ind, len(self.Stack)):
                if self.Stack[i].t >= time:
                    ind = i
                    break
                if self.is_P_modification(self.Stack[i]):
                    tempP.append(self.Stack[i])
                elif self.is_NP_modification(self.Stack[i]):
                    tempNP.append(self.Stack[i])

                    g = self.Stack[i].oldNP

                    index = self.Stack[i].address
                    if temp_set[index] is None or not self.compare_experience(temp_set[index], g):
                        temp_set[index] = g
                        s3 += 1

                # 1. count the number of valid modifications as time progresses
            result.append(len(tempP))
            resultNP.append(len(tempNP))
            s += len(tempP)
            s2 += len(tempNP)
            self.stats.numValidPModificationsOverTime.append(s)
            self.stats.numValidNPModificationsOverTime.append(s2)
            self.stats.numValidNonRedNPModificationsOverTime.append(s3)
        self.Stackfile.write("\n \n num Valid P-modifications over time: \n")
        for r in result:
            self.Stackfile.write(str(r) + "\t")
        self.Stackfile.write("\n \n num Valid NP-modifications over time: \n")
        for rr in resultNP:
            self.Stackfile.write(str(rr) + "\t")

        self.Stackfile.write("\n")
        self.Stackfile.flush()
        self.stats.final_experience_set = self.experience_set
        # sp_max=min(self.t-1,self.Qlearner.memory.buffer_size)
        # self.stats.buffer_diversity = self.get_experience_diversity(self.Qlearner.memory.buffer[0:sp_max])

    def get_input(self):
        """
        form the input for the qlearner, based on the last elements in the experience trace
        :return:
        """
        if self.Q_ready():
            buffer = self.Qlearner.memory.get_top_trace(self.Qlearner.trace_length)
            state_series = np.array([trace[-1] for trace in buffer])
            state_series = np.expand_dims(state_series, axis=0)
            return state_series
        else:
            return None

    def track_q(self, old_location, location, intervals):
        """

        :return:
        """

        if not isinstance(self.chosenAction, ExternalAction):
            if str(self.chosenAction) == "doQuntil":
                if old_location not in self.recorded_usage:
                    self.recorded_usage[old_location] = 0
                self.recorded_usage[old_location] += 1
            return
        for min, max in intervals:

            if min <= self.t < max:
                self.record_qs(old_location, location)
            if self.t == max:
                self.terminate_qs(min, max)
                return (min, max)

    def record_qs(self, old_location, location):

        if old_location not in self.recorded_Qs:
            self.recorded_Qs[old_location] = []
            self.recorded_targets[old_location] = []
            self.recorded_loss[old_location] = []
        s, a, r, s_ = self.Qlearner.memory.buffer[self.Qlearner.memory.sp]
        output, target = self.Qlearner.compute_output_and_target(self.old_input, self.new_input, 1, [[a]],
                                                                 [[r]])
        self.recorded_Qs[old_location].append(output)
        self.recorded_targets[old_location].append(target)
        self.recorded_loss[old_location].append(np.mean((output - target) ** 2))
        if self.loop:
            if old_location not in self.recorded_epsilon:
                self.recorded_epsilon[old_location] = []
            self.recorded_epsilon[old_location].append(self.epsilon)

        if self.match:
            if old_location not in self.recorded_match:
                self.recorded_match[old_location] = 0
                self.recorded_matchsteps[old_location] = []
            self.recorded_match[old_location] += 1
            self.recorded_matchsteps[old_location].append(self.match)
            self.match = 0
        if self.time:
            if old_location not in self.recorded_time:
                self.recorded_time[old_location] = 0
                self.recorded_timesteps[old_location] = []

            self.recorded_time[old_location] += 1
            self.recorded_timesteps[old_location].append(self.time)
            self.time = 0

    def terminate_qs(self, min, max):
        del self.intervals[0]
        self.save_recordings(min, max)
        self.recorded_Qs = {}
        self.recorded_loss = {}
        self.recorded_targets = {}
        self.recorded_usage = {}
        self.recorded_time = {}
        self.recorded_match = {}
        self.recorded_timesteps = {}
        self.recorded_matchsteps = {}
        self.recorded_epsilon = {}

    def save_recordings(self, min, max, folder=''):
        recorded_stuff = {'Qs': self.recorded_Qs, 'targets': self.recorded_targets, 'loss': self.recorded_loss,
                          'usage': self.recorded_usage, 'time': self.recorded_time, 'match': self.recorded_match,
                          'time_steps': self.recorded_timesteps, 'matchsteps': self.recorded_matchsteps,
                          'epsilon': self.recorded_epsilon}

        dump_incremental(folder + self.file + '(%d,%d)_recordings' % (min, max), recorded_stuff)

    def get_part(self):
        if self.slice_size is None:
            return self.m / float(self.n_ops)
        else:
            return self.slice_size

    def convert_index(self, a):
        """
        convert a number in [0,n_ops-1] to an IP in [0,a,2a,...,(num_slices-1)*a], where n_ops > num_slices
        :return:
        """
        slice_size = self.get_part()
        num_slices = self.m / float(slice_size)
        ops_parts = self.n_ops / float(num_slices)
        slice_number = int(a / ops_parts)
        return slice_number,self.ProgramStart + slice_number * slice_size

    def get_IP_slice(self, index):
        return self.convert_index(index)

    def get_current_IP_slice(self,IP):
        slice_size = self.get_part()
        index = int((IP - self.ProgramStart) / slice_size)
        new_ip = int(index * slice_size + self.ProgramStart)
        return new_ip
    def replay_ready(self):
        return self.Qlearner.total_t > self.Qlearner.replay_start_size
    def Q_ready(self):
        return self.Qlearner.total_t >  1000
    def jump_experience(self):
        """
        jump based on the current matching experience in the experience set
        the instruction pointers are divided in n_ops partitions depending on the matched experience
        this instruction puts the pointer at the start of such a partition
        :return:
        """
        if not self.Q_ready() or self.random_steps > 0 or self.get_last_experiences()<self.Qlearner.trace_length+1:
            return
        match = next(
            (i for i, e in enumerate(self.experience_set) if self.compare_experience(e, self.get_top_experience())),
            None)
        if match is None:
            return
        index,new_ip = self.get_IP_slice(match)

        self.setIP(new_ip)

        # add the data to the episode buffer

        self.add_last_experiences()



        self.stats.match_jumpexperience[self.current_feature][self.current_Q_index] += 1
        self.setQlearner(index)
    @overrides
    def setIP(self,ip):
        self.current_IP_slice=self.get_current_IP_slice(ip)
        if self.separate_arguments:
            ip=ip - ((ip-self.current_IP_slice)%self.max_arguments)
        SSA_with_WM.setIP(self,ip)


    def jump_experience2(self):
        """
        jump based on the current matching experience in the experience set of the current IP slice
        the instruction pointers are divided in n_ops partitions depending on the matched experience
        this instruction puts the pointer at the start of such a partition
        :return:
        """
        if not self.Q_ready() or self.random_steps > 0 or self.get_last_experiences()<self.Qlearner.trace_length+1:
            return
        IPslice = self.get_current_IP_slice(self.IP)
        if IPslice not in self.experience_set:
            return
        match = next((i for i, e in enumerate(self.experience_set[IPslice]) if
                      self.compare_experience(e, self.get_top_experience())), None)
        if match is None:
            return
        index,new_ip = self.get_IP_slice(match)
        self.current_IP_slice = new_ip
        self.setIP(new_ip)

        # add the data to the episode buffer
        self.add_last_experiences()
        self.setQlearner(index)
        self.stats.match_jumpexperience[self.current_feature][self.current_Q_index] += 1


    def jump_within(self,a1):
        """
        jump based on the current matching experience to determine next step inside the current slice
        :return:
        """
        if not self.Q_ready() or self.random_steps > 0 or self.get_last_experiences()<self.Qlearner.trace_length+1:
            return
        # add circular inside the current tape
        max=self.get_next_slice()
        min=self.current_IP_slice
        new_ip=self.IP+a1
        diff=self.get_next_slice()-new_ip
        if diff <= 0:
            new_ip=min - diff
        self.setIP(new_ip)



    def train_replay(self, a1):
        """
        do gradient descent for a1 experiences from the replay memory
        :param self:
        :param a1:
        :param a2:
        :return:
        """

        if not self.replay_ready():  return
        batch_size = self.convert_arg(a1, 1, self.max_batchsize)
        if batch_size is None: return
        self.stats.batch_size[self.current_feature][self.current_Q_index] += batch_size
        self.stats.num_network_trainings[self.current_feature][self.current_Q_index] += 1
        self.Q_max, self.loss = self.Qlearner._train_replay(batch_size)

        if DEBUG_MODE:
            print("Q_max=" + str(self.Q_max))
            print("loss=" + str(self.loss))

    def getQoutput(self, epsilon=0.0):

        if self.new_input is None:
            return

        Q = self.Qlearner.model.predict(self.new_input)[0]

        r = np.random.random()
        if r < epsilon:
            self.action_idx = np.random.randint(0, len(Q))
            self.Q = None
        else:
            self.Q = Q
            self.action_idx = np.argmax(self.Q)

        if self.Q_internal and self.action_idx == self.n_outputs - 1:
            self.stats.internalQs[self.current_feature][self.current_Q_index] += 1
            return  # chosen action remains unset
        if self.no_external_actions:
            self.next_chosenAction = self.external_actions[self.action_idx]
            if DEBUG_MODE:
                print("Qoutput:" + str(self.next_chosenAction))
            self.nextInstruction = [-(self.action_idx + 1)]
        else:
            self.next_chosenAction = self.actions[self.action_idx]
            if DEBUG_MODE:
                print("Qoutput:" + str(self.next_chosenAction))

            self.nextInstruction = [self.action_idx]

        self.usedNPinstruction = self.currentInstruction
        self.stats.Q_calls[self.current_feature][self.current_Q_index] += 1
    def get_next_slice(self):
        return min(self.current_IP_slice+self.slice_size,self.Max+1)
    @overrides
    def generateParamsEtc(self):
        nextIP=self.get_next_slice()
        if self.IP+self.get_IP_addition()-1 >=nextIP:
            self.jumpHome()
            return False

        self.generateParameters()
        self.afterParameters()
        if DEBUG_MODE:
            print("currentInstruction =" + str(self.currentInstruction))
        return True
    def afterParameters(self):
        # print(self.currentInstruction)
        # print(self.IP)
        # After successful setting, check whether IP > m - 1 again

        if (self.IP == self.get_next_slice()):
            #print("jumpHome")
            self.jumpHome()  # if so, reset IP for the following loop
        else:
            if(self.IP > self.get_next_slice()):
                raise Exception()

    @overrides
    def jumpHome(self):
        self.setIP(self.current_IP_slice)
    def create_experience_stack_entry(self, first, index):
        if self.experience_type2:
            IP, index = index
            return StackEntryNP(t=self.t, R=self.R, first=first, oldNP=deepcopy(self.experience_set[IP][index]),
                                address=(IP, index))
        else:
            return StackEntryNP(t=self.t, R=self.R, first=first, oldNP=deepcopy(self.experience_set[index]),
                                address=index)

    def preparePerceptionChange(self, index, proposedPol):
        if not self.time_passed_modification() or self.compare_experience(proposedPol, self.experience_set[index]):
            return False
        self.polChanged = True
        first = self.getPointerToFirstModification(
            len(self.Stack))  # pointer to the index of the first modification of this SMS on the stack
        newEntry = self.create_experience_stack_entry(first, index)
        self.Stack.push(newEntry)
        if DEBUG_MODE:
            print("New Stack Entry: " + str(newEntry))
            self.writeStack()
        return True

    def preparePerceptionChange2(self, IP, index, proposedPol):
        if not self.replay_ready():
            return False
        if not self.time_passed_modification() or self.compare_experience(proposedPol, self.experience_set[IP][index]):  # no check for redundancy here because order matters #or self.compare_experience(proposedPol,self.experience_set[IP][index]):
            return False
        self.polChanged = True
        first = self.getPointerToFirstModification(
            len(self.Stack))  # pointer to the index of the first modification of this SMS on the stack
        newEntry = self.create_experience_stack_entry(first, (IP, index))
        self.Stack.push(newEntry)
        if DEBUG_MODE:
            print("New Stack Entry: " + str(newEntry))
            self.writeStack()
        return True

    def set_experience(self, a1):
        """
        add the current experience to the perception module, usable for doQuntil
        :return:
        """
        if not self.replay_ready() or self.disablePLA or self.random_steps > 0:
            return
        proposedPol = self.get_top_experience()

        if not self.preparePerceptionChange(a1, proposedPol):
            return
        self.stats.num_experience_changes[self.current_feature][self.current_Q_index] += 1
        self.experience_set[a1] = proposedPol

    def set_experience2(self, a1):
        """
        add the current experience to the perception module, usable for doQuntil

        WITH redundancy check
        :return:
        """
        if not self.replay_ready() or self.disablePLA or self.random_steps > 0:
            return
        proposedPol = self.get_top_experience()
        IP = self.current_IP_slice
        if not self.preparePerceptionChange2(IP, a1, proposedPol):
            return
        self.stats.num_experience_changes[self.current_feature][self.current_Q_index] += 1
        self.experience_set[IP][a1] = proposedPol

    def doQuntilOld(self, a1, a2, a3=0.):
        """
        # used in the first experiments
        :param a1: experience index
        :param a2: maximal time to stop if experience is not encountered
        :param a3: the epsilon used for exploration
        :return:
        """
        if self.t <= self.Qlearner.replay_start_size or self.loop:
            return
        until_experience = list(self.experience_set[a1])

        if self.compare_experience(until_experience, self.get_top_experience()):
            return
        if DEBUG_MODE:
            print("looping")
        self.until_experience = until_experience
        self.loop = True
        self.stats.loops[self.current_feature][self.current_Q_index] += 1
        self.until_time = narrowing_conversion(a2, (0, self.n_ops - 1), (self.min_untiltime, self.max_untiltime))
        self.epsilon = a3 * self.epsilon_scale

        a, b = self.stats.epsilon[self.current_feature][self.current_Q_index][a1]
        a += self.epsilon  # inc the sum of epsilons
        b += 1  # inc the count of epsilons (each for the given goal state)
        self.stats.epsilon[self.current_feature][self.current_Q_index][a1] = (a, b)

    def doQuntil(self, a1, a2, a3=0.):
        """

        :param a1: experience index
        :param a2: maximal time to stop if experience is not encountered
        :param a3: the epsilon used for exploration
        :return:
        """
        if not self.replay_ready():
            if self.random_steps > 0 or self.no_external_actions:
                self.random_action()
            return
        if self.loop:
            if DEBUG_MODE:
                print("looping already")
            return
        until_experience = list(self.experience_set[a1])

        if self.compare_experience(until_experience, self.get_top_experience()):
            return
        if DEBUG_MODE:
            print("looping")
        self.until_experience = until_experience

        self.loop = True
        self.stats.loops[self.current_feature][self.current_Q_index] += 1
        self.until_time = narrowing_conversion(a2, (0, self.n_ops - 1), (self.min_untiltime, self.max_untiltime))
        self.epsilon = a3 * self.epsilon_scale

        a, b = self.stats.epsilon[self.current_feature][self.current_Q_index][a1]
        a += self.epsilon  # inc the sum of epsilons
        b += 1  # inc the count of epsilons (each for the given goal state)
        self.stats.epsilon[self.current_feature][self.current_Q_index][a1] = (a, b)

    def doQuntil2(self, a1, a2, a3=0.):
        """

        :param a1: experience index
        :param a2: maximal time to stop if experience is not encountered
        :param a3: the epsilon used for exploration
        :return:
        """
        if not self.replay_ready() or self.random_steps > 0:
            if self.no_external_actions:
                self.random_action()
            return
        if self.loop:
            if DEBUG_MODE:
                print("looping already")
            return
        IP = self.current_IP_slice
        if IP not in self.experience_set:
            return
        until_experience = list(self.experience_set[IP][a1])
        if not self.use_jump_experience:
            self.next_IP_slice=IP
        if DEBUG_MODE:
            print("looping")

        self.until_experience = until_experience
        self.loop = True
        self.stats.loops[self.current_feature][self.current_Q_index] += 1
        self.until_time = narrowing_conversion(a2, (0, self.n_ops - 1), (self.min_untiltime, self.max_untiltime))
        self.epsilon = a3 * self.epsilon_scale

        a, b = self.stats.epsilon[self.current_feature][self.current_Q_index][a1]
        a += self.epsilon  # inc the sum of epsilons
        b += 1  # inc the count of epsilons (each for the given goal state)
        self.stats.epsilon[self.current_feature][self.current_Q_index][a1] = (a, b)

    def compare_experience(self, exp, exp2):
        s, a, r, s_ = exp
        ss, aa, rr, ss_ = exp2

        return np.allclose(s,ss, atol=.01) and a == aa and np.allclose(r,rr, atol=10**(-4)) and np.allclose(s_,ss_, atol=.01)

    @classmethod
    def matrix_format_exp(cls, experience_set):
        matrix = []
        for i in range(len(experience_set)):
            s, a, r, s_ = experience_set[i]
            ss = np.append(s, s_)
            added = [a, r]
            exp = np.append(ss, added)
            matrix.append(exp)
        return matrix

    # def get_experience_diversity(self, experience_set, obs_range=2., r_range=1.):
    #     if self.experience_type2:
    #         return self._get_experience_diversity2(experience_set, obs_range, r_range)
    #     else:
    #         return self._get_experience_diversity(experience_set, obs_range, r_range)

    def get_experience_diversity(self, experience_set, obs_range=2., r_range=1.):
        matrix = []
        for i in range(len(experience_set)):
            s, a, r, s_ = experience_set[i]
            ss = np.append(s, s_)
            added = [a, r]
            exp = np.append(ss, added)
            matrix.append(exp)
        stds = np.array(matrix).std(axis=0)
        stds[0:-2] = np.divide(stds[0:-2], obs_range)
        stds[-2] = np.divide(stds[-2], self.n_ops - 1)
        stds[-1] = np.divide(stds[-1], r_range)
        return np.mean(stds)

    # def _get_experience_diversity2(self, experience_set, obs_range=2., r_range=1.):
    #     matrix = []
    #     for IP in experience_set:
    #         for i in range(len(experience_set[IP])):
    #             try:
    #                 s, a, r, s_ = experience_set[IP][i]
    #             except:
    #                 print("error")
    #                 print(experience_set[IP][i])
    #                 continue
    #             ss = np.append(s, s_)
    #             added = [a, r]
    #             exp = np.append(ss, added)
    #             matrix.append(exp)
    #     stds = np.array(matrix).std(axis=0)
    #     stds[0:-2] = np.divide(stds[0:-2], obs_range)
    #     stds[-2] = np.divide(stds[-2], self.n_ops - 1)
    #     stds[-1] = np.divide(stds[-1], r_range)
    #     return np.mean(stds)

    def looping(self):

        if self.loop:  # if loop is set true before
            self.usedNPinstruction = None
            self.next_chosenAction = None
            self.nextInstruction = None
            self.Q = None
            experience = self.get_top_experience()
            self.match = self.compare_experience(experience, self.until_experience)
            self.time = self.loopTime >= self.until_time

            if self.match or self.time:

                if DEBUG_MODE:
                    print("stop looping")
                    print("exp:" + str(self.match))
                    print("time:" + str(self.time))
                if self.match:
                    self.stats.match_experience[self.current_feature][self.current_Q_index] += 1
                    self.stats.match_steps[self.current_feature][self.current_Q_index] += self.loopTime
                else:
                    self.stats.time_outs[self.current_feature][self.current_Q_index] += 1
                    self.stats.until_time_steps[self.current_feature][self.current_Q_index] += self.loopTime
                self.loop = False
                if self.match:
                    self.match = self.loopTime
                else:
                    self.time = self.loopTime
                self.loopTime = 0
                # if not self.use_jump_experience:
                #     self.setIP(self.next_IP_slice)
                #     self.current_IP_slice=self.next_IP_slice
                #     self.next_IP_slice = None

                return
            else:
                self.getQoutput(self.epsilon)
                self.loopTime += 1
        # else:
        #     self.match=False
        #     self.time=False

    def get_top_experience(self):
        if self.episodic:
            return self.episode_buf[-1]
        else:
            top = self.Qlearner.memory.sp
            return self.Qlearner.memory.buffer[top]

    def add_single_experience(self, experience):
        if self.episodic:
            if self.task_time > 0:
                # save the sample <s, a, r, s'> to episode buffer
                self.episode_buf.append(experience)
                if DEBUG_MODE:
                    print("this experience:" + str(self.episode_buf[-1]))
        else:
            self.Qlearner.memory.add(experience)

    def get_last_experiences(self):
        if self.episodic:
            return len(self.episode_buf) - self.last_jump
        else:
            return self.Qlearner.sp+1 - self.last_jump
    def add_last_experiences(self):
        if self.episodic:
                # save the sample <s, a, r, s'> to episode buffer
                experiences=self.episode_buf[self.last_jump:]

                self.Qlearner.memory.add(experiences)
                if DEBUG_MODE:
                    print("this experience:" + str(self.episode_buf[-1]))
                self.last_jump = len(self.episode_buf) - self.trace_length
        else:
            if DEBUG_MODE:
                print("do nothing, already added to memory")
            self.last_jump = self.Qlearner.memory.sp - self.trace_length


    def add_experience(self):
        experience = [self.s_t, self.action_idx, self.r, self.s_t1]
        self.add_single_experience(experience)

    def init_experience_set(self):
        if self.experience_type2:
            num_slices=self.m/self.slice_size
            for i in range(num_slices):
                IP=self.ProgramStart+self.slice_size*i
                self.experience_set[IP] = [np.squeeze([self.Qlearner.memory.sample(1, 1)[0, 0] ]) for i in range(self.n_ops)]

        else:

            samples = [self.Qlearner.memory.sample(1, 1)[0, 0] for i in range(self.n_ops)]
            self.experience_set = np.squeeze(samples)

    def current_experience(self):
        """
        add the current experience to the memory
        :return:
        """
        if self.t > 0:

            if isinstance(self.chosenAction, ExternalAction):
                self.add_experience()
                if self.Qlearner.total_t == self.Qlearner.replay_start_size:  # initialise before allowing experience_set modification
                    self.init_experience_set()
                    #self.stats.experience_diversity[self.current_feature][self.current_Q_index].append(self.get_experience_diversity(self.experience_set[self.current_IP_slice]))
                    self.stats.epsilon[self.current_feature][self.current_Q_index] = [(0, 0) for i in range(self.n_ops)]
                    self.stats.epsilonOverTime[self.current_feature][self.current_Q_index] = [[] for i in range(self.n_ops)]

    @overrides
    def setAction(self):
        self.current_IP_slice = self.get_current_IP_slice(self.IP)
        self.looping()

        if DEBUG_MODE: print("IP=" + str(self.IP - self.ProgramStart))
        self.action_is_set = False

        # step 1: generate instructions anc check for IP and to-be-modified parameters
        # step 1: generate instructions anc check for IP and to-be-modified parameters
        if self.next_chosenAction is not None:  # do nothing, action already chosen
            self.chosenAction = self.next_chosenAction
            self.currentInstruction = self.nextInstruction
            self.next_chosenAction = None
        else:

            self.generateInstruction()
            if isinstance(self.chosenAction, ExternalAction):
                self.action_idx = self.currentInstruction[0]

            self.afterInstruction()  # do not increment IP when looping, too unpredictable
            if not self.generateParamsEtc():  # also no parameters needed with the given instruction set chosen by qlearning
                return

        # print("in setAction:" + str(self.chosenAction.function.__name__))
        if DEBUG_MODE:
            print("t=%d" % (self.t))
            print("chosenAction =" + str(self.chosenAction.function.__name__))
        # if all successful, return with action_is_set
        self.action_is_set = True
        self.s_t = self.s_t1
        if self.new_input is not None:
            self.old_input = self.new_input

    def load(self, name):
        # self.model.load_weights(name)
        for i in range(len(self.Qlearners)):
            self.Qlearners[i].load(name)

    # save the model which is under training
    def save(self, name):
        # self.model.save_weights(name)
        for i in range(len(self.Qlearners)):
            self.Qlearners[i].save(name)

    @overrides
    def performAction(self, agent, environment):

        argument_list = self.get_argument_list(agent, environment)

        self.chosenAction.perform(argument_list)  # record result for statistics

    
    def get_argument_list(self, agent, environment, additional_args=0):
        if DEBUG_MODE:
            print ("performing Action" + str(self.chosenAction.function.__name__))
            print("arguments " + str(self.chosenAction.n_args))
            print("additional args" + str(additional_args))
        argument_list = self.currentInstruction[1:]
        # print(type(self.chosenAction))
        if (isinstance(self.chosenAction, ExternalAction)):
            argument_list.extend([agent, environment])
        
        return argument_list
