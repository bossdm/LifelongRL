import numpy as np
from Lifelong.MultiBlockStackBaseOLD import MultiBlockStackBase
from Methods.Learner import CompleteLearner
from StatsAndVisualisation.LearnerStatistics import LearnerStatistics
from keras.models import Sequential, load_model
from keras.layers import Dense
import keras.backend as K
from overrides import overrides

from copy import deepcopy
from StatsAndVisualisation.metrics import getDifferential
DEBUG_MODE=False

#from MultiBlockStackBase import MultiBlockStackBase


class HomeostaticStats(LearnerStatistics):
    """ keep track of crucial homeostasis & task-drift statistics

    -number of deaths: keep track of how many policies have historically died
    -number of mindist_changes: how often does the mindist-policy change ?
        --> expected: high at first, e.g. once every 10 elementary tasks
    -number of most general policy changes
    -total distance travelled per task




    """
    mindist_time=None
    def __init__(self,learner,tasks):
        LearnerStatistics.__init__(self, learner)
        if learner.do_taskdrift():
            self.task_travel_distances = {task: [] for task in tasks}
            self.num_task_moves = {task:0 for task in tasks}
            self.num_deaths=[[] for pol in range(len(learner.pols))]
            self.pol_usage={task:[[] for pol in range(len(learner.pols))] for task in tasks}# how often is each pol selected
            self.task_coords={task: [learner.task_coords[task]] for task in tasks}
            self.snapshot_taskcoords={task: [] for task in tasks}
            self.mg = [[] for pol in range(len(learner.pols))]
            self.mindist_time = {task: [[] for pol in range(len(learner.pols))] for task in tasks}
            self.diversity = []
            self.provision = []

        else:
            self.diversity = []
            self.task_coords=False


        #self.rewards={task: [ [] for pol in learner.pols] for task in tasks} # cumulative reward (sum of accuracy in case of supervised learners)
        self.mindist_performances={task:[[] for pol in range(len(learner.pols))] for task in tasks}

        self.choices=0



    @overrides
    def update(self,learner):
        """
        update all the statistics based on the current tick
        (for now the included ones are best updated after calling other functions)
        :param learner:
        :return:
        """
        pass

    def add_task_coords(self,coords):
        if self.task_coords:
            for task,coord in coords.items():
                self.task_coords[task][-1]+=coord
            self.choices+=1
    def add_num_deaths(self,pol):
        self.num_deaths[pol][-1]+=1
    def add_pol_usage(self,task,pol,time):
        self.pol_usage[task][pol][-1]+=time
    def add_mg(self,pol,time):
        self.mg[pol][-1]+=time
    def add_mindist_time(self,task,pol,time):
        self.mindist_time[task][pol][-1] += time
    def add_travel_distance(self,task,distance):
        self.task_travel_distances[task][-1]+=distance
    def add_task_move(self,task):
        self.num_task_moves[task]+=1

    def new_task_initialise(self,task,learner):


        self.task_travel_distances[task] = []
        self.num_task_moves[task] = 0
        self.pol_usage[task]=[[] for pol in range(len(learner.pols))]  # how often is each pol selected
        self.task_coords[task] = [learner.task_coords[task]]
        self.snapshot_taskcoords[task]=[]
        self.mindist_time[task] = [[] for pol in range(len(learner.pols))]
        #self.rewards[task] = [[] for pol in learner.pols]
        # self.mindist_performances[task] = []
        self.new_sample_task(task)
    def new_sample_task(self,task):
        self.task_coords[task].append(np.zeros(len(self.num_deaths)))
        self.task_travel_distances[task].append(0)
        self.num_task_moves[task] = 0
        for pol in range(len(self.num_deaths)):
            self.mindist_time[task][pol].append(0)
            self.pol_usage[task][pol].append(0)

        #self.rewards[task].append(np.zeros(len(self.num_deaths)))
        # self.rewards[task].append([])  # cumulative reward (sum of accuracy in case of supervised learners)
        # self.mindist_performances[task].append(0)
    def new_sample(self):
        """ new samples for the stats that need to be incrementally updated over the next statistics interval"""
        for task in self.task_travel_distances:
            self.new_sample_task(task)
        for pol in range(len(self.num_deaths)):
            self.mg[pol].append(0)
            self.num_deaths[pol].append(0)


    def develop(self):
        """
        keep track of development stats (once every SAMPLE_FREQ steps)
        :return:
        """
        # divide travel distance by time spent in tasks:
        if self.task_coords:
            for task in self.task_travel_distances:
                if self.num_task_moves[task] > 0:
                    self.task_travel_distances[task][-1]/=self.num_task_moves[task]
                if self.task_coords and self.choices > 0:
                    self.task_coords[task][-1]/=self.choices # get the coordinate averaged over time steps
            self.choices=0
        self.new_sample()

    def do_taskdrift(self):
        return hasattr(self,'task_travel_distances')

    @overrides
    def initialise_statistics(self):
        LearnerStatistics.initialise_statistics(self)
        if not self.do_taskdrift(): return
        for task in self.task_travel_distances:
            # task-specific
            self.developstats['task_travel_distances'+str(task)] = []
            self.developstats['task_mindist_time'+str(task)] = []
            self.developstats['task_coords'+str(task)] = []
            self.developstats['task_snapshotcoords'+str(task)] =[]
            for pol in range(len(self.num_deaths)):
                # task-pol-specific
                self.developstats['taskpol_velocities'+str(task)+"p"+str(pol)] = []
                self.developstats['taskpol_avgvelocities'+str(task)+"p"+str(pol)]= []
                self.developstats['taskpol_rewards'+str(task)+"p"+str(pol)]= []
                self.developstats['taskpol_usage'+str(task)+"p"+str(pol)]= []
        for pol in range(len(self.num_deaths)):
            # pol specific
            self.developstats['pol_num_deaths'+"p"+str(pol)] = []
            self.developstats['pol_mg'+"p"+str(pol)] = []
            self.developstats['pol_task_cluster'+"p"+str(pol)] = []

        # just one number over time
        self.developstats['diversity'] = []
    def get_sum_of_slice(self,list,begin,Vstep):
        end=begin+Vstep
        return np.sum(list[begin:end])
    def get_avg_of_slice(self,list,begin,Vstep):
        end=begin+Vstep
        return np.mean(list[begin:end])
    @overrides
    def development_statistics_iteration(self,j,Vstep,opt_speed=1.):
        LearnerStatistics.development_statistics_iteration(self,j,Vstep,opt_speed)


    def get_taskpol_velocity(self,addedt,addedR):
        if addedt > 0:
            velocity = addedR / float(addedt)
        else:
            if addedt < 0:
                raise Exception("time cannot be negative")
            else:
                velocity = None
        return velocity
    def task_pol_development_iteration(self,j,Vstep):
        for task in self.task_travel_distances:
            self.developstats['task_travel_distances'+str(task)].append(self.get_sum_of_slice(self.task_travel_distances[task],j,Vstep))
            if self.mindist_time is not None:
                self.developstats['task_mindist_time'+str(task)].append(self.get_sum_of_slice(self.mindist_time[task],j, Vstep))
            self.developstats['task_coords'+str(task)].append(self.get_avg_of_slice(self.task_coords[task],j,Vstep))
            self.developstats['task_snapshotcoords'+str(task)].append(self.task_coords[task][j])

            sum=0
            for pol in range(len(self.num_deaths)):
                # totalRs=self.rewards[task][pol]
                totalts=self.pol_usage[task][pol]
                # addedR=getDifferential(totalRs, j, Vstep)
                addedt=self.get_sum_of_slice(totalts,j,Vstep)
                # velocity=self.get_taskpol_velocity(addedt,addedR)
                # avg_velocity=self.get_taskpol_velocity(totalts[j+Vstep-1],totalRs[j+Vstep-1])
                #self.developstats['taskpol_rewards'+str(task)+"p"+str(pol)].append(addedR)
                # self.developstats['taskpol_velocities'+str(task)+"p"+str(pol)].append(velocity)
                self.developstats['taskpol_avgvelocities'+str(task)+"p"+str(pol)].append(self.mindist_performances[task][pol][j+Vstep])#end of the interval == lifetime avg so far
                self.developstats['taskpol_usage'+str(task)+"p"+str(pol)].append(addedt)
                sum+=addedt



        for pol in range(len(self.num_deaths)):
            self.developstats['pol_num_deaths'+"p"+str(pol)].append(self.get_sum_of_slice(self.num_deaths[pol],j,Vstep))
            self.developstats['pol_mg'+"p"+str(pol)].append(self.get_sum_of_slice(self.mg[pol],j,Vstep))
        self.developstats['diversity'].append(self.get_avg_of_slice(self.diversity,j,Vstep))



    @classmethod
    def avg_final_N_pol_uses(cls,developstats,num_pols,task,N):
        """
        average the final N nonzero pol usages
        :param task:
        :param N:
        :return:
        """

        count=0
        index=-1
        length=len(developstats['taskpol_usage' + str(task) + "p" + str(0)])
        final_uses=[]
        while count<N and index >= -length:
            stats=np.zeros(num_pols)
            for pol in range(num_pols):

                stats[pol]=float(developstats['taskpol_usage' + str(task) + "p" + str(pol)][index])

            if sum(stats)>0:
                count+=1
                final_uses.insert(0,stats/sum(stats))


            index-=1
            print(index)


        return np.mean(final_uses,axis=0)

    @classmethod
    def avg_final_N_pol_Development(cls,developstats,num_pols,task,N,additional_key=""):
        """
        average the final N nonzero pol usages
        and get an additional statistic
        :param task:
        :param N:
        :return:
        """
        stats=[[] for pol in range(num_pols)]
        additional_stat=[[] for pol in range(num_pols)]
        count=0
        index=-1
        length=len(developstats['taskpol_usage' + str(task) + "p" + str(0)])
        while count<N and index >= -length:
            stat=np.zeros(num_pols)
            for pol in range(num_pols):
                stat[pol] = float(developstats['taskpol_usage' + str(task) + "p" + str(pol)][index])
            s=sum(stat)
            if s>0:
                count+=1
                for pol in range(num_pols):
                    stats[pol].insert(0,stat[pol]/s)
                    additional_stat[pol].insert(0,developstats[additional_key+str(task)+'p'+str(pol)][index])


            index-=1
            print(index)

        stats=[np.array(stat) for stat in stats]
        additional_stat=[np.array(stat) for stat in additional_stat]
        return stats,additional_stat

    @overrides
    def development_statistics(self, total_samples, Vstep, opt_speed=1.):
        LearnerStatistics.development_statistics(self,total_samples,Vstep,opt_speed)

        if not self.do_taskdrift():
            print("1to1/unadaptive policy")
            return # must be one-to-one
        self.task_pol_development_statistics(Vstep/2)
        D=len(self.num_deaths)
        pol_coords=HomeostaticPol.get_pol_coords(D)
        # add the final clustering

        for task in self.task_travel_distances:
            final_coords = self.developstats['task_snapshotcoords' + str(task)][-1]
            mindist=float("inf")
            mindist_pol=None
            for pol in range(D):
                distance=HomeostaticPol.class_minkowski_distance(final_coords,pol_coords[pol],D)
                if distance < mindist:
                    mindist=distance
                    mindist_pol=pol
            self.developstats['pol_task_cluster' + "p" + str(mindist_pol)].append(task)



    def task_pol_development_statistics(self,Vstep):
        num_samples=len(self.num_deaths[0])
        for j in range(0,num_samples-Vstep,Vstep):
            self.task_pol_development_iteration(j,Vstep)


class ProbabilityRule(object):

    distance=0
    epsilon=1

class HomeostaticPol(CompleteLearner):
    """


    features:
    -specialisation is encouraged by
        *using a default policy for a task
        *always maintaining a policy when it consistently outperforms other policies on one or more tasks
    -generalisation is encouraged by:
        *regularly testing other policies on a task
        *removing policies which never improve performance on other tasks


    """
    loss_intervals=[]
    alpha_off=0.05
    min_time_evaluation=0
    policy_space_picture_freq=20000 # equal to blocksize
    epsilon=.02 # minimal fraction of the maxdist, use for clipping proportional policy choice
    intervals = [(50000, 60000)]
    unadaptive=False
    probability_rule=ProbabilityRule.distance
    testing=False
    # (at most 50 times more likely than the theoretical maximal distance policy, about 25 times more likely than an average performing policy )
    def __init__(self,episodic,actions,filename,pols,weights,decision_frequency=100,stepsize=0.10,init_provision=1.0,
                 consumption=.10,reward_scale=.50,initialise_unseen=False,one_to_one=False,unadaptive=False,
                 probability_rule=ProbabilityRule.epsilon):
        CompleteLearner.__init__(self,actions,filename)
        self.occurence_weights=weights
        self.pols=pols
        self.num_policies=len(pols)
        self.init_provision=init_provision*self.num_policies
        self.provision=[self.init_provision for _ in range(self.num_policies)]
        self.testing_ticks=0
        self.consumption=consumption
        self.reward_scale=reward_scale*consumption*self.num_policies
        self.stepsize=stepsize
        self.decision_frequency=decision_frequency
        self.current_feature=None
        self.one_to_one = one_to_one
        self.unadaptive=unadaptive
        self.probability_rule=probability_rule
        if self.do_taskdrift():
            self.construct_policy_space()
        self.current_pol=0 # convention
        self.policy_chosen=False
        self.episodic=episodic
        self.initialise_unseen=initialise_unseen






        if self.initialise_unseen:
            self.N_trainings={}
        #self.offline_updates=offline_updates
        if self.episodic:
            self.task_t=0
            self.task_R=0.0
        if weights is not None:
            self.initialise_task_coords()
    @overrides
    def continue_experiment(self,intervals):
        self.intervals=intervals
    # def track_q(self, old_location, location,intervals):
    #     print("track q"+str(intervals))
    #     for min,max in intervals:
    #         if min <= self.t < max :
    #             self.pols[self.current_pol].record_qs(old_location,location)
    #         if self.t == max:
    #             for i in range(self.num_policies):
    #                 self.terminate_qs(min,max)
    @overrides
    def printStatistics(self):
        for key in self.stats.__dict__:
            print("%s :   \n %s"%(key,self.stats.__dict__[key]))

    def get_minkowski_distance(self,coord1,coord2):
        return np.sum(np.power(np.abs(coord1-coord2),self.D))**(1./self.D)
    @classmethod
    def class_minkowski_distance(cls,coord1,coord2,D):
        return np.sum(np.power(np.abs(coord1-coord2),D))**(1./D)
    def get_inverse_minkowski_distance(self,coord1,coord2):
        clipped_dist=np.clip(self.get_minkowski_distance(coord1, coord2),self.max_dist*self.epsilon,self.max_dist)
        return 1/float(clipped_dist)

    def reset_init_network_weights(self):

        session = K.get_session()
        for layer in self.init_network.layers:
            if hasattr(layer, 'kernel_initializer'):
                layer.kernel.initializer.run(session=session)
    def get_init_network_output(self,F):
        """
        train the init_network on the current data and then get its output on the new task, obtaining its coordinate in
        policy space
        :return:
        """
        self.reset_init_network_weights()
        Fs=self.task_coords.keys()
        n=1000
        N = n * len(Fs) # repeat each unique task 1000 times
        feature_dims=len(Fs[0])
        x=np.zeros((N,feature_dims))
        y = np.zeros((N, self.num_policies))
        w = np.zeros(N)
        mu=0.
        sigmaF=0.05
        sigmaP=0.05
        i=0
        for F,P in self.task_coords.items():
            x[i*n:(i+1)*n]=np.array(F)+np.random.normal(mu, sigmaF, (n,feature_dims))
            y[i * n:(i + 1) * n] = np.array(P) + np.random.normal(mu, sigmaP, (n, self.num_policies))
            w[i * n:(i + 1) * n] = float(self.N_trainings[F])/sum(self.N_trainings.values())
            i+=1
        indexes = np.random.permutation(N)
        x=x[indexes]
        y=y[indexes]
        w=w[indexes]
        if DEBUG_MODE:
            verbose=1
        else:
            verbose=0
        self.init_network.fit(x,y,epochs=20,sample_weight=w,validation_split=0.2,verbose=verbose)
        return self.init_network.predict(np.array([F]))[0]
    def do_taskdrift(self):
        return self.num_policies > 1 and not self.one_to_one and not self.unadaptive
    def initialise_new_task(self,F):
        """
        if enough tasks already initialised around the center, a new task will be initialised according to
            the functional representation
        :return:
        """
        if not self.do_taskdrift(): return
        if F in self.task_coords: return
        if not self.initialise_unseen: return

        if len(self.task_coords) >= 2:
            self.initialise_transfer(F)
        else:
            self.initialise_around_center(F)
        self.stats.new_task_initialise(F,self)
        self.N_trainings[F] = 0

        if DEBUG_MODE:
            self.make_policy_space_picture(debug=True)

    def initialise_transfer(self,F):
        """

        :return:
        """
        self.task_coords[F] = self.get_init_network_output(F)
        self.mindist_policies[F] = self.get_mindistpolicy_for(F)



    # def construct_policy_space(self):
    #     """
    #     construct a n points in a n-1-dimensional space with the policies on equal distances from each other
    #     :return:
    #     """
    #     assert self.num_policies>1, "need more than 1 policy, otherwise just use normal SSA learner"
    #     self.D=self.num_policies-1
    #     self.policy_space=[None]*self.num_policies
    #     for policy in range(self.num_policies):
    #         if policy < self.num_policies - 1:
    #             # put a 1 on the first dimension
    #             dimensions=np.zeros(self.D)
    #             dimensions[policy]=1.0
    #             self.policy_space[policy]=dimensions
    #         else:
    #             # put all 1s
    #             self.policy_space[policy]=np.ones(self.D)
    #
    #     # do a check-up
    #     for policy in range(self.num_policies):
    #         for policy2 in range(self.num_policies):
    #             if policy==policy2:
    #                 continue
    #             #assert self.get_minkowski_distance(self.policy_space[policy],self.policy_space[policy2])==2, "something wrong with policy space init"



    def construct_policy_space(self):
        """
        construct a n points in a n-1-dimensional space with the policies on equal distances from each other
        :return:
        """
        #assert self.num_policies>1, "need more than 1 policy, otherwise just use normal SSA learner"

        self.D=self.num_policies
        self.pol_coords=HomeostaticPol.get_pol_coords(self.D)


        if DEBUG_MODE:
            # do a check-up
            previous_dist=None
            for policy in range(self.num_policies):
                for policy2 in range(self.num_policies):
                    if policy==policy2:
                        continue
                    dist=self.get_minkowski_distance(self.pol_coords[policy],self.pol_coords[policy2])
                    assert previous_dist is None or dist==previous_dist
                    previous_dist=dist
        #set the maximal distance (equals the distance to travel from one policy to another)
        self.max_dist=self.get_minkowski_distance(self.pol_coords[0],self.pol_coords[1])
    @classmethod
    def get_pol_coords(cls,D):
        pol_coords = [None for i in range(D)]
        for policy in range(D):
            # put a 1 on the first dimension
            dimensions=np.zeros(D)
            dimensions[policy]=1.0
            pol_coords[policy]=dimensions
        return pol_coords
    @classmethod
    def get_center(cls,D):
        return np.zeros(D)+0.5




    def coord_noise(self,level):
        return np.random.randn(self.D)*level
    def initialise_task_coords(self):
        """
        -initialise all tasks at the center of the allowed space (equidistant to all policies
        -initialise the best policies
        :return:
        """
        if not self.do_taskdrift():
            return
        self.task_coords = {}
        self.mindist_policies = {}

        if self.initialise_unseen:

            self.init_network=Sequential()
            Fs=self.occurence_weights.keys()
            inputs=len(Fs[0])
            hidden=5*inputs
            outputs=self.num_policies
            self.init_network.add(Dense(hidden,input_shape=(inputs,), activation='relu'))
            self.init_network.add(Dense(hidden, activation='relu'))
            self.init_network.add(Dense(outputs))
            self.init_network.compile(loss='mean_squared_error', optimizer='adam')
        else:
            for task in self.occurence_weights:
                self.initialise_around_center(task)

        self.most_general_policy()
        self.avg_velocities={F:[-float('inf') for pol in self.pols] for F in self.occurence_weights}
    def initialise_around_center(self,F):

        self.task_coords[F] = self.get_center(self.D) + self.coord_noise(level=self.stepsize / 4)
        self.mindist_policies[F]=self.get_mindistpolicy_for(F)

    def clip_coord(self,coord):
        return np.clip(coord,0.,1.)  # clip to the unit-hypercube of D-dimensions

    def set_tasks(self, weights=None):
        """
        before starting experiments
        :param weights:
        :param total_num_tasks:
        :return:
        """

        if weights is not None:
            if self.one_to_one:
                assert not self.unadaptive
                assert self.num_policies == len(weights)
                i=0
                self.taskpolmap={}
                for F in weights:
                    self.taskpolmap[F]=i
                    i+=1
            else:
                if self.unadaptive:
                    assert not self.one_to_one
                    assert self.num_policies < len(weights)

                    self.taskpolmap = {}
                    num_repetitions=int(np.ceil(len(weights)/float(self.num_policies)))
                    policy_order=list(range(self.num_policies))*num_repetitions
                    tasks=list(weights.keys())
                    np.random.shuffle(tasks)
                    for task in tasks:
                        self.taskpolmap[task] = policy_order.pop(0)
            self.occurence_weights = weights
            self.initialise_task_coords()

            tasks=self.task_coords.keys() if self.do_taskdrift() else self.occurence_weights.keys()
            self.stats = HomeostaticStats(self, tasks)
        if DEBUG_MODE:
            self.improvements=[{F:0 for F in self.occurence_weights} for _ in range(self.num_policies)]
            self.tests = [0 for _ in range(self.num_policies)]
            self.ignored_t = 0
            self.ignored_R = 0

        for i in range(self.num_policies):
            self.pols[i].set_tasks(self.occurence_weights)



    # def set_tasks(self,weights,num_tasks):
    #     if weights is not None:
    #         self.occurence_weights=weights
    #         features=np.array(list(self.occurence_weights.keys()))
    #         self.feature_range = features.max(axis=0) - features.min(axis=0)
    #         indexes=np.where(self.feature_range==0.0)
    #         self.feature_range[indexes]=1.0 # just in case the dimensions don't vary at all, don't divide 0/0
    #         #self.feature_sets = {key:frozenset([key]) for key in self.occurence_weights}
    def get_mindistpolicy_for(self,feature):
        """
        return the arg-min of the dist(P,F) (the best policy for a given feature)
        :param feature:
        :return:
        """
        if self.num_policies==1:
            return 0
        policy_dists=[self.get_minkowski_distance(self.pol_coords[pol],self.task_coords[feature]) for pol in range(self.num_policies)]
        return np.argmin(policy_dists)




    def new_task(self,feature):
        """
        start a new task. note that this is called at the start of the task block
        instead of the reset which is called
        :param feature:
        :return:
        """

        self.current_feature=tuple(feature)
        self.initialise_new_task(self.current_feature)


    def end_pol(self):


        self.pols[self.current_pol].end_pol()
        if self.do_taskdrift():
            self.stats.add_mg(self.mg_pol, +1)
            self.stats.add_mindist_time(self.current_feature, self.mindist_policies[self.current_feature], +1)
            self.stats.add_pol_usage(self.current_feature, self.current_pol, +1)
            if self.probability_rule==ProbabilityRule.distance:
                self.update_task_and_policy()
            else:
                self.update_velocity()

    def update_velocity(self):
        #print("pol ",self.current_pol)
        V_avg = self.pols[self.current_pol].get_avg_velocity(self.current_feature)
        self.avg_velocities[self.current_feature][self.current_pol] = V_avg

    def new_pol(self,new_feature,new_pol):

        self.current_feature=new_feature
        self.current_pol = new_pol
        self.policy_chosen = True
        self.pols[self.current_pol].new_task(self.current_feature)

        if DEBUG_MODE:
            print("new pol\npolicy %d set"%(self.current_pol))
    # def set_testing_pol(self):
    #     self.set_new_pol(self.current_feature,self.testing_pol)
    # def set_new_pol(self,feature,policy_index):
    #     self.best_policies[feature] = policy_index
    #     self.pols[self.current_pol] = self.pols[policy_index]


    def end_task(self):
        """
        set the final policy such that it can be recovered if the new 'initial policy' is not successful
        :return:
        """
        if DEBUG_MODE:
            print("ending task block")
        if not self.episodic and not self.policy_chosen:  # otherwise done on reset anyway
            self.end_pol()
        #otherwise, end_pol after each elementary task (cf. reset())



        #self.finalise_stack_variables(self.min_time_evaluation)


    # def test_policy(self,pol,testing_time):
    #     """
    #
    #     :param pol: processed argument from an SSA learner, indicates the policy index
    #     :param testing_time: processed argument from an SSA learner, indicates the time length of the test
    #     :return:
    #     """
    #     # evvaluate current pol and finalise the current_pol's stack
    #     self.finish_SMS()
    #     self.finalise_stack_variables()
    #
    #     #
    #     self.testing_ticks=testing_time
    #     self.testing_pol=pol
    #     self.disablePLA=True
    #     self.set_testing_pol()

    # def end_test(self):
    #
    #     self.pols[self.current_pol].disablePLA=False
    def update_task_in_policyspace(self,direction):
        """
        update the current task coordinate in the given direction;

        :param direction:
        :return:
        """

        self.task_coords[self.current_feature]=self.update_task(self.task_coords[self.current_feature],direction)
        distance=self.get_minkowski_distance(direction*self.stepsize,np.zeros(self.D))
        self.stats.add_travel_distance(self.current_feature,distance)
        self.stats.add_task_move(self.current_feature)
        self.most_general_policy() # simply used for updating statistics here
        if DEBUG_MODE:
            self.make_policy_space_picture(debug=True)
    def update_task(self,coord,direction):
        c=coord
        c+=direction*self.stepsize
        return self.clip_coord(c)
    @classmethod
    def policy_space_video(cls,filename,sample_range,task_coordsOverTime,D):
        assert D==2
        for t in range(sample_range):
            coords={task:task_coordsOverTime[task][t] for task in task_coordsOverTime}
            HomeostaticPol.policy_space_pic(filename,t,coords,D)

    @classmethod
    def policy_space_pic(cls,filename,time,task_coords,D,replaced_policy=None,annot=True):

        import matplotlib as mpl
        mpl.use('agg')
        from matplotlib import pyplot as PLT
        if D != 2: return
        pol_coords=HomeostaticPol.get_pol_coords(D)

        x=[c[0] for c in task_coords.values()]
        y=[c[1] for c in task_coords.values()]
        PLT.scatter(x,y,s=100)
        PLT.xticks([0.0,1.0], (0,"$\mathcal{P}^1$"),fontsize=20)
        PLT.yticks([0.0,1.0], (0,"$\mathcal{P}^2$"),fontsize=20)
        if replaced_policy is not None:
            coord=pol_coords[replaced_policy]+.10*(HomeostaticPol.get_center(D)-pol_coords[replaced_policy])
            PLT.annotate("Replaced $P_%d$"%(replaced_policy),coord,)
        if annot:
            for F,coord in task_coords.items():
                #if F==(1.,1,1):
                PLT.annotate(F, (coord[0]+0.05,coord[1]+0.01),fontsize=10)
        PLT.ylim(-0.08,1.08)
        PLT.xlim(-0.08,1.08)
        PLT.title(r'$t=%d$'%(time))
        PLT.savefig(filename+"policy_space%d.png" % (time))
        PLT.gcf().clear()
    def make_policy_space_picture(self,debug=False,replaced_policy=None):
        time = self.t if debug else self.t / self.policy_space_picture_freq
        filename=self.file
        HomeostaticPol.policy_space_pic(filename,self.task_coords,self.D,replaced_policy)
    # def is_mindist_policy(self):
    #     """
    #     check if the tested policy is better for the task
    #     :return:
    #     """
    #     return self.current_pol==self.get_mindistpolicy_for(self.current_feature)
    def get_mindist_velocity(self,feature):
        """
        get the mindist velocity for a given feature;
        to avoid "lucky hits" which can never be improved on, we take the most recent velocity of the min-distance policy
            rather than the highest ever velocity
        :param feature:
        :return:
        """
        p=self.mindist_policies[feature]
        return self.avg_velocities[feature][p]
    def get_velocity(self):
        """
        get the velocity of the current policy
        :return:
        """
        if not self.episodic:
            self.pols[self.current_pol].set_velocity(self.decision_frequency)
            return self.pols[self.current_pol].Stack.velocity
        else:

            if self.task_t == 0:
               V=None
            else:
                V = self.task_R / self.task_t
            self.task_R=0.0
            self.task_t=0.0
            return V

        #return self.pols[self.current_pol].get_velocity()
    def set_mindist_policy(self,old_pol,new_pol,V_avg):
        """

        :param old_pol: the old mindist
        :param new_pol: the new mindist
        :param V_avg:
        :return:
        """
        # self.stats.add_mindist_time(self.current_feature,old_pol,self.mindist_time_passed)
        self.avg_velocities[self.current_feature][new_pol] = V_avg
        self.mindist_policies[self.current_feature] = new_pol
    def get_homeostatic_reward(self,feature,reward_direction=1.0):
        return len(self.occurence_weights) * self.occurence_weights[feature] * self.reward_scale * reward_direction
    def is_improvement(self,V_new,V_best):
        return (V_new-V_best) > 0

    def update_task_and_policy(self):
        """
        evaluate the reward speed of the current testing policy, and use it to move the task in policy space
        :return:
        """
        if self.initialise_unseen:
            self.N_trainings[self.current_feature] += 1
        min_dist_policy=self.mindist_policies[self.current_feature]



        V_new = self.get_velocity()
        if V_new is None:
            if DEBUG_MODE:
                print("not ready for velocity calculation, pol=%d, time=%d" % (
                self.current_pol, self.pols[self.current_pol].t))
            return
        V_best= self.avg_velocities[self.current_feature][min_dist_policy]
        improvement=self.is_improvement(V_new,V_best)
        if DEBUG_MODE:
            self.tests[self.current_pol]+=1
            if improvement:
                self.improvements[self.current_pol][self.current_feature]+=1

        reward_direction=1.0 if improvement else 0.0
        reward = self.get_homeostatic_reward(self.current_feature,reward_direction)
        self.provision[self.current_pol] += reward
        survived=self.homeostasis(self.current_pol)
        if not survived:
            if DEBUG_MODE:
                print("policy %d died and is replaced"%(self.current_pol))
                self.make_policy_space_picture(debug=True,replaced_policy=self.current_pol)
            return
        # remove any scaling by simply ignoring the magnitude of the change, both for space update and provision update
        if improvement:
            direction=self.pol_coords[self.current_pol] - self.task_coords[self.current_feature]
            self.update_task_in_policyspace(direction)

        else:
            direction = self.pol_coords[min_dist_policy] - self.task_coords[self.current_feature]
            if self.current_pol != min_dist_policy:
                # evidence for min_dist_policy
                self.update_task_in_policyspace(direction)
            else:
                # means that mindist-policy is difficult to improve itself
                # move away from min_dist_policy
                self.update_task_in_policyspace(-direction)

        new_mindist = self.get_mindistpolicy_for(self.current_feature) # check which one is mindist policy now and set its velocity
        Vavg=self.pols[new_mindist].get_avg_velocity(self.current_feature)
        if self.set_mindist_condition(new_mindist):

            self.set_mindist_policy(min_dist_policy,new_mindist,Vavg)


        # note that the strength of the adjustment depends inversely on dist(task,pol) --> convergence



        if DEBUG_MODE:
            print("pol_coords: %s "%(str(self.pol_coords)))
            print("task_coords: %s" % (str(self.task_coords)))
        # self.end_pol(self.testing_pol)
        # self.new_pol()
    def set_mindist_condition(self,new_mindist):
        return self.current_pol != new_mindist

    def survive(self,pol):
        return self.provision[pol] > 0
    def check_provision(self,pol):
        """

        :return:
        """
        epsilon=self.epsilon
        target=sum(self.improvements[pol][F]*self.get_homeostatic_reward(F) for F in self.task_coords) # if predictive evaluation not used, w=1/|F|
        target-=self.tests[pol]*self.consumption
        target+=self.init_provision
        assert abs(self.provision[pol] - target) < epsilon, "provision=%.4f, target=%.4f"%(self.provision[pol],target)
    def check_convergence(self):
        """"
        check that given repeatedly being the best on a task, the best policy will be:
            -the mindist_policy
            -chosen according to the correct probability

            verified elsewhere
        """
        pass
    def homeostasis(self,pol):
        """

        :param pol: the index of the policy
        :return:
        """
        self.provision[pol]-=self.consumption

        if DEBUG_MODE:
            self.check_provision(pol)

        if not self.survive(pol):
            self.stats.add_num_deaths(pol)
            self.create_new_policy(pol)
            return False
        return True
    # def kill_policy(self,pol):
    #     """
    #     remove the current policy for that index
    #
    #     :param pol: the index of the policy to be removed
    #     :return: None
    #     """
    #     for F in self.occurence_weights:
    #         stack=self.Stack
    def weighted_dist(self,pol_coord):
        return np.sum(self.occurence_weights[F]*self.get_minkowski_distance(pol_coord,self.task_coords[F])  for F in self.task_coords)
    def most_general_policy(self):
        """
        get the most-general policy (the one that minimises summed distance to tasks)
        :return:
        """
        min=float('inf')
        min_pol=None
        for i in range(self.num_policies):
            weighted_dist=self.weighted_dist(self.pol_coords[i])
            if weighted_dist< min:
                min=weighted_dist
                min_pol = i

        self.mg_pol=min_pol
        return min_pol

    def create_new_policy(self,pol):
        """
        create a new policy on index pol, similar to the policy with least weighted average distance to tasks

        (essential part for predictive evaluation; search the space of general policies especially those that perform well at
         the most important tasks !!)
        :param pol:
        :return:
        """
        if DEBUG_MODE:
            #self.kill_policy(pol)
            print("creating new policy")
            self.tests[pol] =0
            self.improvements[pol]={F:0 for F in self.occurence_weights}
            self.ignored_t, self.ignored_R=self.pols[pol].add_to_ignored(self.ignored_t,self.ignored_R)

            # if CHECK_LOCATION:
            #     self.initial_locations=[deepcopy(self.task_coords[F]) for F in self.occurence_weights]

        min_pol=self.most_general_policy()
        self.pols[pol]=self.pols[min_pol].create_similar_policy()
        self.pols[pol].set_tasks(self.occurence_weights)
        self.provision[pol]=self.init_provision
        print("outside:"+str(self.pols[pol].actions))

    def get_policy_probs(self,task):

        dists=np.zeros(self.num_policies)
        for p in range(self.num_policies):
            dists[p]=self.get_inverse_minkowski_distance(self.pol_coords[p],self.task_coords[task])
        C=np.sum(dists)
        dists/=C

        return dists
    def get_dist_based_policy(self):
        probs = self.get_policy_probs(self.current_feature)
        new_pol = np.random.choice(self.num_policies, 1, p=probs)[0]
        if DEBUG_MODE:
            print("choose policy")
            print("probabilities=" + str(probs))
        return new_pol
    def get_random_policy(self):

        new_pol = np.random.choice(self.num_policies, 1)[0]
        return new_pol
    def argmax_randomtie(self,b):
        return np.random.choice(np.flatnonzero(b == b.max()))
    def choose_best_policy(self):
        if self.one_to_one or self.unadaptive:
            # one-to-one
            new_pol = self.taskpolmap[self.current_feature]
        else:
            new_pol = self.best_policy(self.current_feature)
            # in case of distance-based, can be that mindist temporarily changed to suboptimal
        self.new_pol(self.current_feature, new_pol)
    def best_policy(self,task):
        return self.argmax_randomtie(np.array(self.avg_velocities[task]))
    def choose_policy(self):
        """
        choose policy, proportional to distance
        :return:
        """
        if self.testing:
            self.choose_best_policy()
            return
        if self.num_policies > 1:

            if self.one_to_one or self.unadaptive:
                # one-to-one
                new_pol=self.taskpolmap[self.current_feature]

            else:
                # task drift
                self.stats.add_task_coords(self.task_coords)
                if self.probability_rule == ProbabilityRule.epsilon:
                    r=np.random.random()
                    epsilon=0.10
                    if r < epsilon:
                        # take random policy based on distance
                        new_pol=self.get_random_policy()
                    else:
                        # get mindist policy
                        new_pol=self.best_policy(self.current_feature)
                else:

                    new_pol=self.get_dist_based_policy()

            self.new_pol(self.current_feature, new_pol)
        else:
            new_pol=0
            self.new_pol(self.current_feature, new_pol)
            return


    def get_diversity(self):
        cls=type(self.pols[0])
        if hasattr(cls,"get_diversity"):
            return cls.get_diversity(self.pols)
        return None
    def get_output_diversity(self,data,metric_type):
        cls=type(self.pols[0])
        if hasattr(cls,"get_output_diversity"):
            return cls.get_output_diversity(self.pols,data,metric_type)
        return None

    def get_performance_diversity(self):
        stds=[]
        for task in self.occurence_weights:
            avg_velocities=[pol.get_avg_velocity(task)   for pol in self.pols]
            num_infs=sum(np.isinf(avg_velocities))
            if num_infs>0:
                continue  # no estimate available
            stds.append(np.std(avg_velocities))
        return np.mean(stds)
    @overrides
    def save(self,filename):
        for i in range(self.num_policies):
            self.pols[i].save(filename+'pol%d'%(i))
        # self.model.save_weights(name)
        if self.initialise_unseen:
            self.init_network.save(filename + "_initnetwork.h5")
            del self.init_network

    @overrides
    def load(self,filename):
        for i in range(self.num_policies):
            self.pols[i].load(filename+'pol%d'%(i))
        if self.initialise_unseen:
            self.init_network=load_model(filename+"_initnetwork.h5")
    @overrides
    def printDevelopment(self):
       CompleteLearner.printDevelopment(self)
       if self.t % self.policy_space_picture_freq == 0:


           if self.do_taskdrift():
               self.stats.develop()
               self.get_performance_diversity()
               self.stats.diversity.append(self.get_diversity())

               self.stats.provision.append([prov for prov in self.provision])
               for task in self.task_coords:
                        self.stats.snapshot_taskcoords[task].append(self.task_coords[task])
                        for pol in range(self.num_policies):
                            self.stats.mindist_performances[task][pol].append(self.avg_velocities[task][pol])

           else:
               if self.num_policies > 1:
                   self.stats.diversity.append(self.get_diversity())
           # self.make_policy_space_picture()
       # for pol in self.pols:
       #      pol.printDevelopment()
    @overrides
    def printDevelopmentAtari(self,frames):
        CompleteLearner.printDevelopmentAtari(self,frames)
        if self.t % self.policy_space_picture_freq == 0:

            if self.do_taskdrift():
                self.stats.develop()
                self.get_performance_diversity()
                self.stats.diversity.append(self.get_diversity())

                self.stats.provision.append([prov for prov in self.provision])
                for task in self.task_coords:
                    self.stats.snapshot_taskcoords[task].append(self.task_coords[task])
                    for pol in range(self.num_policies):
                        self.stats.mindist_performances[task][pol].append(self.avg_velocities[task][pol])

            else:
                if self.num_policies > 1:
                    self.stats.diversity.append(self.get_diversity())
            # self.make_policy_space_picture()
        # for pol in self.pols:
        #      pol.printDevelopment()
    @overrides
    def initStats(self):
        pass # initialise instead at set_tasks


    @overrides
    def reset(self):
        # at the end of episodes
        if self.episodic:
            if self.task_t > 0:
                self.pols[self.current_pol].reset()
                self.end_pol()

                if DEBUG_MODE:
                    print("end of elementary task --> reset")


    @overrides
    def new_elementary_task(self):
        # at start of new elementary task
        if self.episodic :
            if not self.policy_chosen:
                self.choose_policy()
                self.pols[self.current_pol].new_elementary_task()
    @overrides
    def setTerminalObservation(self,agent,environment):
        self.pols[self.current_pol].setTerminalObservation(agent,environment)

    @overrides
    def setAtariTerminalObservation(self,obs):
        self.pols[self.current_pol].setAtariTerminalObservation(obs)
        self.observation = self.pols[self.current_pol].observation

    @overrides
    def printPolicy(self):
        #self.make_policy_space_picture()
        pass
    @overrides
    def save_stats(self,filename):
        CompleteLearner.save_stats(self,filename)
        for pol in range(self.num_policies):
            self.pols[pol].save_stats(filename+'pol%d'%(pol))

    @overrides
    def cycle(self,agent,environment):

        time=self.pols[self.current_pol].t
        if not self.episodic and not self.policy_chosen and time % self.decision_frequency == 0 :
            self.end_pol()
            self.choose_policy()
        if DEBUG_MODE:
            print("cycle outer:"+str(self))
            print("pol t = %d" % (self.t))
            print("current pol=%d"%(self.current_pol))

        self.pols[self.current_pol].cycle(agent,environment)
        assert np.array_equal(self.observation,self.pols[self.current_pol].observation), "observation mismatch"
        assert np.array_equal(self.chosenAction,self.pols[self.current_pol].chosenAction), "action mismatch"
    @overrides
    def atari_cycle(self,observation):

        time=self.pols[self.current_pol].t
        if not self.episodic and not self.policy_chosen and time % self.decision_frequency == 0 :
            self.end_pol()
            self.choose_policy()
        if DEBUG_MODE:
            print("cycle outer:"+str(self))
            print("total t = %d" % (self.t))
            print("policy t = %d" % (time))
            print("policy total_t = %d" % (self.pols[self.current_pol].total_t))
            print("current pol=%d"%(self.current_pol))

        self.pols[self.current_pol].atari_cycle(observation)
        self.observation = self.pols[self.current_pol].observation
        self.chosenAction = self.pols[self.current_pol].chosenAction
        assert np.array_equal(self.observation,self.pols[self.current_pol].observation), "observation mismatch"
        assert np.array_equal(self.chosenAction,self.pols[self.current_pol].chosenAction), "action mismatch"

    # setReward and setTime are done outside the cycle, and require difference in top level vs bottom level
    @overrides
    def setReward(self,reward):
        CompleteLearner.setReward(self,reward)
        self.pols[self.current_pol].update_task_reward(reward)
        if self.episodic:
            self.task_R+=reward

        # if DEBUG_MODE:
        #     R=0
        #     for pol in self.pols:
        #         for feature in self.occurence_weights:
        #             R+=pol.get_task_R(feature)
        #     assert R==self.R
    @overrides
    def setTime(self,t):

        increment=t-self.t
        if self.episodic:
            self.task_t+=increment
        self.pols[self.current_pol].update_task_time(increment)
        CompleteLearner.setTime(self,t)

        if self.num_policies > 1:
            self.policy_chosen=False # allow choose_policy again when time has passed
        if DEBUG_MODE:
            self.check_policy_variables()
        # if DEBUG_MODE:
        #     t=0
        #     for pol in self.pols:
        #         for feature in self.occurence_weights:
        #             t+=pol.get_task_t(feature)
        #     assert t==self.t
        #     if hasattr(self.pols[self.current_pol],'check_policy_variables'):
        #         self.pols[self.current_pol].check_policy_variables()

    def check_policy_variables(self):
        if isinstance(self.pols[0],MultiBlockStackBase):
            features = list(self.occurence_weights.keys())
            t = 0
            R = 0
            for pol in self.pols:
                if pol.BlockStack.sp < 0:
                    continue
                # assert pol.t == pol.BlockStack.top().t, "at time %d pol t not equal to block stack top t" % (self.t)
                # assert pol.t == pol.Stack.t, "pol t not equal to stack t" % (self.t)
                # assert pol.R == pol.BlockStack.top().R, "at time %d pol R not equal to block stack top R" % (self.t)
                # assert pol.R == pol.Stack.R, "pol t not equal to stack R" % (self.t)
                tt, RR = pol.BlockStack.total_t_and_R_consumption(features)

                t += tt
                R += RR

            t += self.ignored_t
            R += self.ignored_R
            assert t == self.t, "at time %d != time sum %d" % (self.t, t)

            assert R == self.R, "reward %.2f != reward sum %.2f" % (self.R, R)
        else:
            t = 0
            R = 0
            for pol in self.pols:
                t,R=pol.add_to_ignored(t,R)


            t += self.ignored_t
            R += self.ignored_R
            assert t == self.t, "at time %d != time sum %d" % (self.t, t)

            assert R == self.R, "reward %.2f != reward sum %.2f" % (self.R, R)
    # def check_location_in_policyspace(self):
    #     """
    #     check that the task is located in correct place in policy space:
    #     -improvements*(C(mindistpolicy)- initial_F
    #     (uses assumption mindistpolicy does not change)
    #     :return:
    #     """
    #
    #     for F in self.occurence_weights:
    #         self.check_single_task_in_policyspace(F)
    # def check_single_task_in_policyspace(self,F):
    #     expected_location=self.initial_location
    #     for pol in self.pols:
    #         expected_location=self.update_task(expected_location,direction)
    #
    #     return self.task_coords[F]==expected_location
    # @overrides
    # def printDevelopment(self):
    #     self.printR()
    # @overrides
    # def printStatistics(self):
    #     pass
    # @overrides
    # def initStats(self):
    #     self.stats=LearnerStatistics(self)


#
# def test_TaskSpecificSSA():
#     from MazeUtils import north,south,west,east
#     method=HomeostaticPol.get_standard_instance([north,west,south,east],filename='')
#     p=0
#
if __name__ == "__main__":
    from MazeUtils import north, south, west, east
    from Actions.SpecialActions import ExternalAction
    method=HomeostaticPol.get_standard_instance([ExternalAction(north)],filename='')
    method.occurence_weights={(1.0):.50,(0.):.50}
    method.check_policy_variables()

    method.check_convergence()