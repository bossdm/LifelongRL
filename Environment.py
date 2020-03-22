
import time
import numpy as np
from abc import abstractmethod

from overrides import overrides

from Actions.SpecialActions import *
from IS.SSA import SSA
from StatsAndVisualisation.visual import IS_Visualisation, Visual
from mapobjects import EmptyObject, Obstacle

from ExperimentUtils import save_intermediate

stops={1*10**6:"1Mil",30*10**6:'30Mil',60*10**6:'60Mil'}

DEBUG_MODE=False
#MAX_RUN_TIME=20
class BaseEnvironment(object):

    def __init__(self, agent, params, visual=False):
        self.interrupt = False  # was task interrupted or not
        self.visual = visual
        self.filename=params["filename"]
        self.stoptime=params["stoptime"]
        print("stoptime="+str(self.stoptime))
        self.sampling_rate=params['sampling_rate'] if 'sampling_rate' in params else 10000
        self.real_start=time.time()
        self.start=self.real_start # this may be changed, e.g. second iridis run of single experiment --> compare to start not real_start
        self.real_time=params['real_time']
        self.timePolicy=RealTime if self.real_time else ExternalActionTime
        self.rng = np.random.RandomState(params['seed'])
        self.agent = agent
        self.observation_length = params['observation_length']
        self.agent.learner.observation = np.zeros(self.observation_length)
        self.current_sample=None if not self.real_time else 1 #use to count samples
        self.t = 0
        self.real_t = 0
    def set_tasks(self,tasks,statfreq):
        self.slices = 5

        self.tasks = tasks
        #end_time = self.tasks[-1].end_time
        self.statfreq = statfreq
    def convert_to_map(self):
        pass
    @abstractmethod
    def printStatistics(self):
        pass
    @abstractmethod
    def setObservation(self, agent):
        pass
    @abstractmethod
    def convertObservationBinary(self):
        pass


    def tick(self):
        self.currentTask.tick(self)



    def run(self,walltime):
        self.running_time = walltime*.82 # leave some time for finalisation
        self.agent.learner.print_initialisation()
        if self.t==0:
            self.agent.learner.printDevelopment()

        while (self.tasks and time.time() - self.start < self.running_time and self.t < self.stoptime):

            #if not self.interrupt:  # if environment was interrupted just keep going with the current task !
            if not self.interrupt:
                self.currentTask = self.tasks.pop(0)
                self.currentTask.initialize(self)  # in case you interrupted

            self.currentTask.run(self)
            print(time.time()-self.start)

        end = time.time()
        print("Simulation took %d seconds" % (end - self.start,))
        print("ended task "+str(self.currentTask))

    @abstractmethod
    def newTask(self):
        pass

    @abstractmethod
    def updateStat(self):
        pass



class NavigationEnvironment(BaseEnvironment):
    def __init__(self,agent,visual,params):

        self.sizeX = params['sizeX']
        self.sizeY = params['sizeY']
        self.observation_size=params['observation_size']
        self.dynamic = params['dynamic']
        self.use_stats = params['use_stats']

        record_intervals = params['record_intervals']

        video_file = params['filename'] if record_intervals else None
        BaseEnvironment.__init__(self, agent, params,visual)


        if self.visual:
            self.initVisualisation(params['sizeX'],params['sizeY'],record_intervals,video_file)
    def add_other_agents(self):
        pass
    def initVisualisation(self, sizeX,sizeY, record_intervals, video_file):
        buttons=True if not record_intervals else False
        canvas=True if not record_intervals else False
        record=True if record_intervals else False
        self.vis = Visual(sizeX, sizeY, (self.observation_size, self.observation_size), 500,500,buttons,record_intervals, video_file,tk_canvas=canvas)
        if not record and isinstance(self.agent.learner, SSA):
            num_to_display=10
            total_rows=18
            use_NP=False
            if hasattr(self.agent.learner, "NP"):
                use_NP=True
            self.vis.IS_vis = IS_Visualisation(use_NP,self.agent.learner,total_rows,num_to_display+1, 1600, 1000 ,
                                               self.vis.display.root)
    @abstractmethod
    def checkLegalMove(self,agent,coord):
        pass
    @abstractmethod
    def checkBounds(self,coord):
        pass
    @abstractmethod
    def generateMap(self,params):
        pass
    @overrides
    def printStatistics(self):
        pass

    def fill_borders(self):
        # Fill borders
        self.map = [[EmptyObject()] * self.sizeY for _ in range(self.sizeX)]
        for x in [0, self.sizeX - 1]:
            for y in range(self.sizeY):
                self.map[x][y] = Obstacle(None)
        for y in [0, self.sizeY - 1]:
            for x in range(self.sizeX):
                self.map[x][y] = Obstacle(None)
    def getMapString(self):
        mapstring=''
        for y in range(self.sizeY):
            for x in range(self.sizeX):
                mapstring+=self.map[x][y].__class__.__name__+'  '
            mapstring+='\n'
        return mapstring
    def getObservationString(self):
        return str(self.agent.learner.observation)
    def printMapState(self):
        print('MAP: \n'+self.getMapString())
        print('Observation: \n'+self.getObservationString())
        print('agent location: \n (' + str(self.agent.x)+','+str(self.agent.y)+')')

    @abstractmethod
    def setAgentMap(self):
        pass
    # @abstractmethod
    # def getInitState(self):
    #     pass
    @abstractmethod
    def inputDimensions(self):
        pass
    @abstractmethod
    def inputSize(self):
        pass
    def check_obstacle(self,x,y):
        return isinstance(self.map[x][y], Obstacle)
    def getStates(self): #return all possible states
        pass
    def tick_mapelements(self):
        if self.dynamic:
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    self.map[x][y].tick(self)

    def tick(self):
        self.tick_mapelements()
        BaseEnvironment.tick(self)

class TimeMeasurement(object): # policy for determining whether or not time has passed
    def __init__(self):
        pass
    @abstractmethod
    def incrementTime(self,environment):
        pass
    @abstractmethod
    def timeCriterion(self,environment):
        pass
    @abstractmethod
    def stepCondition(self,environment):
        pass

    @classmethod
    def sampleCondition(cls, environment):
        pass

class ExternalActionTime(TimeMeasurement):
    def __init__(self):
        TimeMeasurement.__init__(self)

    @classmethod
    def incrementTime(cls, environment):
        if cls.stepCondition(environment):
            environment.t += 1

    @classmethod
    def incrementTimeAct(cls, environment,action):
        if cls.stepConditionAct(action):
            environment.t += 1
        #print(" %d / %d"%(environment.t ,environment.currentTask.end_time))
    @classmethod
    def setTime(cls,environment):
        environment.agent.learner.setTime(environment.t)



    @classmethod
    def timeCriterion(cls,environment):
        return environment.t < environment.currentTask.end_time
    @classmethod
    def stepCondition(cls, environment):
        return isinstance(environment.agent.learner.chosenAction, ExternalAction)

    @classmethod
    def stepConditionAct(cls, chosen):
        return isinstance(chosen, ExternalAction)
    @classmethod
    def sampleCondition(cls,environment):
        return environment.t % environment.sampling_rate == 0

    @classmethod
    def getStatCounter(cls, environment):
        return environment.t // environment.statfreq


class CycleTime(TimeMeasurement):
    def __init__(self):
        TimeMeasurement.__init__(self)
    @overrides
    def incrementTime(self,environment): #assuming it is only called after agent tick !
        environment.agent.learner.t = environment.t

    @classmethod
    def timeCriterion(cls,environment):
        return environment.t % environment.sampling_rate == 0



class PseudoEnvironment(object):

    """
    environment for sole purpose of measuring time in simulated outcomes
    """
    def __init__(self,t,timePolicy):
        self.t=t
        self.timePolicy=timePolicy

    def tick(self,action):
        self.t0=time.time()
        self.timePolicy.incrementTimeAct(self,action)

class RealTime(TimeMeasurement):
    def __init__(self):
        TimeMeasurement.__init__(self)
    @classmethod
    def incrementTime(self,environment): #assuming it is only called after agent tick
        # the actual measurement is done outside in case you want to use other time but record real_time as well
        environment.real_t += time.time() - environment.t0
    @classmethod
    def incrementTimeAct(cls, environment, action):
        if cls.stepConditionAct(action):
            environment.t += time.time() - environment.t0
    @classmethod
    def setTime(cls, environment):
        environment.agent.learner.setTime(environment.real_t)

    @classmethod
    def timeCriterion(cls, environment):
        return environment.real_t <= environment.currentTask.end_time

    @classmethod
    def stepCondition(cls, environment):
        return True

    @classmethod
    def stepConditionAct(cls, chosen):
        return True
    @classmethod
    def sampleCondition(cls, environment):
        if int(environment.real_t)>0 and environment.real_t/environment.sampling_rate > environment.current_sample:
            environment.current_sample+=1
            return True
        return False

    @classmethod
    def getStatCounter(cls, environment):
        return int(environment.real_t) // environment.statfreq
class Task(object):
    def __init__(self, reward_fun, end_time,  environment=None,generate_new=False):
        self.reward_fun = reward_fun
        self.end_time = end_time
        print("end_time=%d"%(self.end_time))
        self.environment = environment
        self.solved=False
        self.generate_new=generate_new
        self.interrupt=False

    def initialize(self, environment):
        self.terminal = False

    def tick(self,environment):

        environment.observation_set=False

        while True:
            environment.t0=environment.real_t + environment.real_start
            environment.agent.tick(environment)
            if environment.visual:
                environment.vis.tick(environment,intermediate=True)
            if environment.use_stats:
                environment.updateStat()

            RealTime.incrementTime(environment)
            ExternalActionTime.incrementTime(environment)

            if environment.timePolicy.stepCondition(environment):
                environment.timePolicy.setTime(environment)
                if environment.visual:
                    environment.vis.tick(environment)
                break

            environment.observation_set=True
        # once time has passed get the reward
        environment.agent.learner.setReward(environment.currentTask.reward_fun(environment.agent, environment))
        if DEBUG_MODE:
            print("rewarding agent: r=%.2f, R=%.2f" % (environment.agent.learner.r, environment.agent.learner.R))

        if DEBUG_MODE:
            print("r =" + str( environment.agent.learner.r))
            print("R=" + str( environment.agent.learner.R))
            print("t="+str( environment.t))
            print("environment time : " + str(environment.agent.learner.t))
            print("end_time:" + str(self.end_time))
            print("\n \n ")









    def check_environment_interrupt(self,environment):
        if time.time() -environment.start >= environment.running_time:
            self.interrupt=True
            return
        if environment.t >= environment.stoptime:
            self.interrupt=True

    def iteration(self,environment):
        self.tick(environment)

    def run(self,environment):

        print("end= %d" %(self.end_time))
        self.interrupt=False
        stop=False
        while(not stop):

            self.check_environment_interrupt(environment)

            if self.interrupt:
                print("interrupted")
                environment.interrupt=True
                return
            if DEBUG_MODE:
                print("LoopT:"+str(environment.t))
                print("endtime"+str(self.end_time))
            environment.tick()
            term=environment.terminal
            environment.reset()
            if environment.timePolicy.sampleCondition(environment):
                environment.agent.learner.printDevelopment()
            if not environment.timePolicy.timeCriterion(environment) or self.solved:
                stop=True# prepare to stop don't do yet because need to check for intermediate saves
            else:
                if term:
                    environment.agent.learner.new_elementary_task()




            if environment.t in stops.keys():
                save_intermediate(environment, environment.filename + stops[environment.t], save_stats=False,
                                  save_learner=True)
                environment.agent.learner.load(environment.filename + stops[environment.t])
                print("done intermediate saving at time " + str(environment.t))


        environment.interrupt=False

        if self.generate_new and self.solved:
            environment.newTask()


class NonInterruptibleTask(Task):
    def __init__(self, reward_fun, end_time, environment=None, generate_new=False):
        Task.__init__(self, reward_fun, end_time, environment, generate_new)
    @overrides
    def check_environment_interrupt(self,environment):
        if not environment.terminal:
            return
        if time.time() -environment.start >= environment.running_time:
            self.interrupt=True
            return
        if environment.t >= environment.stoptime:
            self.interrupt=True

class FunctionTask(Task):
    def __init__(self, reward_fun, end_time,  environment=None):
        Task.__init__(self,reward_fun,end_time,environment)
    def tick(self,environment):
        Task.tick(self,environment)
        #print(environment.agent.answer)
        if environment.agent.learner.r == 1.0 or environment.t >= self.end_time:
            self.solved=True
            environment.reset()


class NavigationTask(Task):

    def __init__(self, reward_fun, funs, end_time, files=None,environment=None,generate_new=False,maze_id=None):
        Task.__init__(self, reward_fun, end_time, environment,generate_new=generate_new)
        self.solved = False
        self.funs=funs
        self.files = files
        self.initialized = False
        self.maze_id = maze_id

    def initialize(self,environment):
        if not self.initialized:
            environment.generateMap()
            environment.reset()
            if environment.dynamic:
                self.initializeMapObjFuns(environment)
            self.initialized=True
            environment.terminal = False
    # for dynamic environments
    def initializeMapObjFuns(self,environment):
        for x in range(environment.sizeX):
            for y in range(environment.sizeY):
                value=self.funs.get(type(environment.map[x][y]))
                if value is not None:
                    environment.map[(x,y)].function=value



class MultiTask(NavigationTask):

    def __init__(self, task_feature,task_type,reward_fun, funs, start_time,end_time, topology_type=None,files=None,environment=None,generate_new=False,maze_id=None):
        NavigationTask.__init__(self,reward_fun, funs, end_time, files,environment,generate_new,maze_id)
        self.task_type=task_type
        self.task_feature=task_feature
        self.topology_type=topology_type
        self.start_time=start_time
    def new_task(self,learner):
        learner.new_task(self.task_feature)
    def end_task(self,learner):
        learner.end_task()

    def initialize(self,environment):
        if not self.initialized:
            environment.generateMap()
            self.new_task(environment.agent.learner)
            environment.agent.learner.new_elementary_task()
            self.initialized = True
        environment.terminal = False
    def run(self,environment):
        Task.run(self,environment)
        self.end_task(environment.agent.learner)
    def __str__(self):
        return "Multi-task with task feature %s, task type %s, topology %s and start_time %d"%\
        (str(self.task_feature),self.task_type,self.topology_type,self.start_time)
    __repr__  = __str__

