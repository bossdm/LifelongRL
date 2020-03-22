
from MazeUtils import *

from Environment import *


from ExperimentUtils import dump_incremental
from StatsAndVisualisation.Statistics import POcmanMeltingPotStatistics
from random import randint



from overrides import overrides
from mapobjects import EmptyObject, PacmanFood, PacmanPoison,NormalGhost, ChasedGhost, Power, Obstacle
from MazeUtils import manhattan_dist, directions, directional_dist, opposite_direction, VonNeumannNeighbourhoodPlus, \
    check_toroid_X, check_toroid_Y

from mapobjects import *

DEBUG_MODE = False

from POcmanEnums import *

class POcman(NavigationEnvironment):
    add_task_features = False
    reward_clearlevel = +100.
    reward_default = 0.

    reward_eatpower = +50.
    reward_eatghost = +100.
    reward_multipl = +1.  # multiply the eatghost reward when multiple caught during the same power-up
    reward_die = -50.
    reward_eatfood = +10.
    reward_hitwall = -10.
    reward_poison = -10.
    old_object=None
    old_F = None
    oldx = None
    oldy = None
    def __init__(self, agent, visual, params):
        self.terminal = False
        NavigationEnvironment.__init__(self, agent, visual, params)


        self.maze = np.zeros((self.sizeX, self.sizeY))
        #self.passage_y = -1
        self.smell_range = 1
        self.hear_range = 2
        self.food_prob = 0.5
        self.chase_prob = 0.50
        self.defensive_slip = 0.50
        self.power_numsteps = 15

        self.elementary_task_time=1000 if 'elementary_task_time' not in params else params['elementary_task_time']
        self.reward_die = -50. if 'reward_die' not in params else params['reward_die']
        self.reward_eatfood = +10. if 'reward_food' not in params else params['reward_eatfood']
        self.reward_hitwall = -10. if 'reward_hitwall' not in params else params['reward_hitwall']
        self.reward_poison = -10. if 'reward_poison' not in params else params['reward_poison']
        # 4: food available, 4 ghost visible 4 wall configuration 3: 2,3, or 4 manhattandist(food) 1: powerpill (13)
        # old version: 10
        self.obs_length = params["observation_length"]
        self.add_task_features = params["include_task_features"]
        self.inform_task_time = params['inform_task_time']
        self.num_observations = self.obs_length ** 2

        self.discount = 0.95
        self.num_food=0
        self.num_poison=0
        self.num_ghosts=0

    def get_map(self, x, y):
        return self.map_flag_to_mapel(x,y)
    @overrides
    def set_tasks(self,tasks,stat_freq):
        BaseEnvironment.set_tasks(self,tasks,stat_freq)
        self.initStats()
    def initStats(self):
        self.stats=[{} for i in range(self.slices)]


    def time_up(self):
        """
        check whether the current elementary task is supposed to end
        :return:
        """
        return self.t % self.elementary_task_time == 0
    @classmethod
    def get_reward_range(cls):
        """
        get the reward range independent of any settings (works only assuming no eat_ghost multiplier
        :return:
        """
        # worst case: die + hit_wall
        min_reward = cls.reward_die + cls.reward_hitwall + cls.reward_default
        # best case, eatfood, eatghost and clearlevel
        max_reward = cls.reward_eatghost + cls.reward_eatfood + cls.reward_clearlevel + cls.reward_default
        return min_reward,max_reward
    @classmethod
    def get_pacman_maze(cls):
        maze= [
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0],
            [0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 7, 0],
            [0, 3, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 3, 0],
            [0, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 0, 0, 0, 0],
            [1, 1, 1, 1, 3, 0, 1, 0, 1, 1, 1, 0, 1, 0, 3, 1, 1, 1, 1],
            [0, 0, 0, 0, 3, 0, 1, 0, 0, 0, 0, 0, 1, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 1, 1, 1, 1, 1, 1, 1, 0, 3, 0, 0, 0, 0],
            [0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 3, 0, 0, 0, 3, 0, 3, 0, 0, 0, 3, 0, 0, 3, 0],
            [0, 7, 3, 0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0, 3, 7, 0],
            [0, 0, 3, 0, 3, 0, 3, 0, 0, 0, 0, 0, 3, 0, 3, 0, 3, 0, 0],
            [0, 3, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 0, 3, 3, 3, 3, 0],
            [0, 3, 0, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 3, 0],
            [0, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        ]
        return np.transpose(maze)
    @classmethod
    def get_micropacman_maze(cls):
        maze=    [
                [0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 7, 3, 3, 3, 3, 3, 7, 0],
                [0, 3, 3, 0, 3, 0, 3, 3, 0],
                [0, 3, 0, 3, 3, 3, 0, 3, 0],
                [3, 3, 3, 3, 0, 3, 3, 3, 3],
                [0, 3, 0, 3, 3, 3, 0, 3, 0],
                [0, 3, 3, 0, 3, 0, 3, 3, 0],
                [0, 7, 3, 3, 3, 3, 3, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0]

            ]
        return np.transpose(maze)
    @classmethod
    def get_minipacman_maze(cls):
        maze= [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                [0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 0],
                [0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0],
                [0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0],
                [3, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 3],
                [0, 0, 0, 3, 0, 1, 1, 3, 3, 0, 0, 0],
                [0, 0, 0, 3, 0, 1, 1, 3, 3, 0, 0, 0],
                [0, 3, 3, 3, 0, 0, 0, 0, 3, 3, 3, 0],
                [0, 3, 0, 3, 3, 3, 3, 3, 3, 0, 3, 0],
                [0, 3, 0, 0, 3, 0, 0, 3, 0, 0, 3, 0],
                [0, 7, 3, 3, 3, 3, 3, 3, 3, 3, 7, 0],
                [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
            ]
        return np.transpose(maze)
    def standard_pacman(self,num_ghosts=4):
        self.sizeX = 19
        self.sizeY = 21
        self.maze = self.get_pacman_maze()
        self.pocman_home = (8, self.sizeY - 1 - 6)
        self.num_ghosts = num_ghosts
        if num_ghosts > 0:
            self.ghost_range = 6

            self.ghost_home = (8, self.sizeY - 1 - 10)


    def micro_pacman(self,num_ghosts=1):
        self.sizeX=9
        self.sizeY=9
        self.maze = self.get_micropacman_maze()

        self.pocman_home = (4, self.sizeY - 2)
        self.num_ghosts = num_ghosts
        if self.num_ghosts > 0:

            self.ghost_range = 3
            self.ghost_home = (4, self.sizeY - 1 - 5)

    def mini_pacman(self,num_ghosts=3):
        self.sizeX=12
        self.sizeY=12
        self.maze = self.get_minipacman_maze()
        self.pocman_home = (5, self.sizeY - 1 - 3)
        self.num_ghosts = num_ghosts
        if num_ghosts>0:

            self.ghost_range = 4

            self.ghost_home = (5, self.sizeY - 1 - 6)

        #self.passage_y = 5
    def standardmaze_map(self,num_ghosts=0):
        self.sizeX = 11
        self.sizeY = 8

        self.maze=self.get_standard_maze()

        self.pocman_home = (1, 3)
        self.num_ghosts = num_ghosts
        if num_ghosts > 0:

            self.ghost_range = 3
            self.ghost_home = (9, 1)

          # we count downwards and from 0, so coords a little different from paper, but figure the same
    def cheesemaze_map(self,num_ghosts=0):
        self.sizeX = 7
        self.sizeY = 5

        self.maze=self.get_cheese_maze()

        self.pocman_home = (1, 2)
        self.num_ghosts = num_ghosts
        if num_ghosts > 0:

            self.ghost_range = 3
            self.ghost_home = (3, 3)

    @classmethod
    def fill_borders(cls,sizeX,sizeY):
        # Fill borders
        maze = np.array([[POcmanFlags.E_FREE] * sizeY for _ in range(sizeX)])
        for x in [0, sizeX - 1]:
            for y in range(sizeY):
                maze[x,y] = POcmanFlags.E_OBSTACLE
        for y in [0, sizeY - 1]:
            for x in range(sizeX):
                maze[x,y] = POcmanFlags.E_OBSTACLE
        return maze

    @classmethod
    def fill_other_obstacles(self,occupied,maze):
        for (x,y) in occupied:
            maze[x,y]=POcmanFlags.E_OBSTACLE
        return maze

    @classmethod
    def get_standard_maze(cls):
        sizeX = 11
        sizeY = 8
        maze=cls.fill_borders(sizeX,sizeY)
        occupiedCoords = [(3, 2), (3, 3), (3, 4), (6, 5), (8, 1), (8, 2), (
            8, 3)]
        return cls.fill_other_obstacles(occupiedCoords,maze)

    @classmethod
    def get_cheese_maze(cls):
        sizeX=7
        sizeY=5
        maze=cls.fill_borders(sizeX,sizeY)
        occupiedCoords = [(2, 2), (2, 3), (4, 2), (4, 3)]
        return cls.fill_other_obstacles(occupiedCoords, maze)
    # visualisation functions
    @classmethod
    def get_maze(cls,topology):
        if topology==PacmanTopology.standard:
            return cls.get_standard_maze()
        elif topology==PacmanTopology.pacman_micro:
            return cls.get_micropacman_maze()
        elif topology==PacmanTopology.pacman_mini:
            return cls.get_minipacman_maze()
        elif topology==PacmanTopology.pacman_standard:
            return cls.get_pacman_maze()
        elif topology==PacmanTopology.cheese:
            return cls.get_cheese_maze()
        else:
            raise Exception("not supported")
    def convert_to_map(self):
        self.map = [[EmptyObject()] * self.sizeY for _ in range(self.sizeX)]
        for x in range(self.sizeX):
            for y in range(self.sizeY):
                self.map[x][y]= self.map_flag_to_mapel(x, y)
    def convertObservationBinary(self,agent):
        if FULLOBS:
            xbits=get_required_bits(self.sizeX)
            ybits=get_required_bits(self.sizeY)
            x=encode(agent.x,xbits)
            y=encode(agent.y,ybits)
            agent.learner.observation = x + y



        else:
            for i in range(len(agent.learner.observation)):
                agent.learner.observation[i]= True if agent.learner.observation[i]==1 else False
        if DEBUG_MODE:
            print("x,y = %d,%d"%(self.agent.x,self.agent.y))
            print(agent.learner.observation)

    def map_flag_to_mapel(self, x, y):
        # only the static elements
        flag = self.maze[x, y]
        if DEBUG_MODE:
            print(POcmanFlags.E_FOOD)
        if flag == int(POcmanFlags.E_FREE):
            return EmptyObject()
        elif flag == int(POcmanFlags.E_OBSTACLE):
            return Obstacle(None)
        elif flag == int(POcmanFlags.E_FOOD):
            return PacmanFood(None)
        elif flag == int(POcmanFlags.E_POISON):
            return PacmanPoison(None)
        elif flag == int(POcmanFlags.E_POWER):
            return Power(None)
        else:
            raise Exception("found element %d" % (flag))

    def add_other_agents(self):
        filename = "pacman_ghost.jpg" if self.power_steps > 0 else "pacman_normalghost.png"
        for ghost in self.ghost_pos:
            x, y = ghost
            self.vis.display.addPicture(x, y, shrink_factor=(1.0, 1.0),
                                        filename=filename)

    # end visualisation
    @overrides
    def generateMap(self, params=None):
        self.new_level()

        #self.agent.learner.new_task(self.currentTask.task_feature)

    @overrides
    def printStatistics(self):
        pass
    def reset(self):
        if DEBUG_MODE:
            print("reset")
        # gets done every tick:
        if not self.terminal:
            return

        self.new_level()

        # # if self.total > 100:
        # #     if self.successes / float(self.total) > .80:
        #         #self.agent.learner.Rfile.write("SUCCESS after " + str(self.t) + "\n")
        #         #self.agent.learner.Rfile.flush()
        #         #self.currentTask.solved = True
        self.endEpisode()
        self.terminal = False

    def endEpisode(self):
        if self.agent.learner.episodic:
            if self.t > 0:
                #self.currentCount = self.maxCount + 1
                self.agent.learner.setTerminalObservation(self.agent, self)  # set the final observation
        self.agent.learner.reset()
        self.currentTask.start_time=self.t






    # def Copy(
    # STATE & state)
    # {
    #
    # POCMAN_STATE & pocstate = safe_cast <
    # POCMAN_STATE & > (state)
    # POCMAN_STATE * newstate = MemoryPool.Allocate()
    # *newstate = pocstate
    # return newstate
    # }
    def check_passablepos(self, pos):
        return self.maze[int(pos[0]),int(pos[1])] != POcmanFlags.E_OBSTACLE

    def validate(self, state):
        assert self.check_passablepos((self.agent.x, self.agent.y))
        for g in range(self.num_ghosts):
            assert self.check_passablepos(self.ghost_pos)

    # def create_start_state(self):
    #     self.ghost_pos=np.zeros(self.num_ghosts)
    #     self.food=np.zeros(self.sizeX,self.sizeY,dtype=bool)
    #     self.new_level()



    # void
    # FreeState(STATE * state)
    #
    # {
    # POCMAN_STATE * pocstate = safe_cast < POCMAN_STATE * > (state)
    # MemoryPool.Free(pocstate)
    # }

    #
    def check_obstacle(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_OBSTACLE

    def check_powerpill(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_POWER

    def check_food(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_FOOD or self.maze[x, y] == POcmanFlags.E_POWER
    def check_poison(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_POISON
    def check_food_specific(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_FOOD

    def check_free(self, x, y):
        return self.maze[x, y] == POcmanFlags.E_FREE
    @classmethod
    def check_free_maze(cls,x, y,maze):
        return maze[x, y] == POcmanFlags.E_FREE
    def check_foodwithindist(self, dist):
        minx = self.agent.x - dist
        miny = self.agent.y - dist
        maxx = self.agent.x + dist
        maxy = self.agent.y + dist
        for x in range(minx, maxx+1):
            for y in range(miny, maxy+1):
                xx = check_toroid_X(x, self)
                yy = check_toroid_Y(y, self)
                if self.check_food(xx, yy) and manhattan_dist((self.agent.x, self.agent.y), (xx, yy)) == dist:
                    if DEBUG_MODE:
                        print("see food within dist %d"%(dist))
                    return True
        if DEBUG_MODE:
            print("no food within dist %d" % (dist))
        return False
    def get_direction_based_on_manhattandist(self,dist):
        """
        get the directions leading to manhattan dist
        :param dist:
        :return:
        """
        # if 0, return (0,0)
        # if even, return (+-dist/2,+-dist/2), else return [(+-(dist//2+1),+-dist//2),(+-dist//2,+-(dist//2+1))]

        return {0:[(0,0)],
                1: [(0,1),(1,0),(-1,0),(0,-1)],
                2: [(1,1),(1,-1),(-1,1),(-1,-1),
                    (2,0),(0,2),(-2,0),(0,-2),
                    ],
                3: [(1,2),(1,-2),(-1,2),(-1,-2),
                    (2, 1), (2, -1), (-2, 1), (-2, -1),
                    (3,0),(0,3),(-3,0),(0,-3)],
                4: [(2,2),(2,-2),(-2,2),(-2,-2),
                    (3, 1), (3, -1), (-3, 1), (-3, -1),
                    (1, 3), (1, -3), (-1, 3), (-1, -3),
                    (4, 0), (0, 4), (-4, 0), (0, -4)
                    ]}[dist]
    def signed_bool(self,num):
        return 1. if num else -1.
    def check_objectwithindist(self, dists):
        """

        :param dists: sorted (ascending) list of distances (e.g., [2,3,4])
        :return:
        """
        l=[self.num_ghosts,self.num_food,self.num_poison]
        assert sum(l)==1, "%d ghosts, %d foods, %d poisons"%(self.num_ghosts,self.num_food,self.num_poison) # only one can be true in the meltingpot scenario
        object_location=self.get_object()
        distance=manhattan_dist((self.agent.x,self.agent.y),object_location)
        #print([self.signed_bool(distance <= dist) for dist in dists])
        return [1 if distance <= dist else -1 for dist in dists]
        # observations=np.zeros(len(dists)) - 1
        # min_index=0
        # for dist in range(maxdist+1):
        #     directions=self.get_direction_based_on_manhattandist(dist)
        #     mindist=dists[min_index]
        #     while dist > mindist:
        #         # need to change the min_index
        #         min_index+=1
        #         mindist=dists[min_index]
        #
        #
        #     for (x,y) in directions:
        #         if self.see_object((x, y)):
        #             observations[min_index:]=1
        #             print("dist=" + str(dist))
        #             print("mindist=" + str(mindist))
        #             print((x,y))
        #             print(observations)
        #             return observations

        # indexes_left=range(len(dists))
        # minx = - max(dists)
        # miny = - max(dists)
        # maxx = + max(dists)
        # maxy = + max(dists)
        # for x in range(minx, maxx+1):
        #     for y in range(miny, maxy+1):
        #         manhattandist=x+y
        #         if self.see_object((x,y)):
        #             if DEBUG_MODE:
        #                 print("see food within dist %d"%(manhattandist))
        #             for i in indexes_left:
        #                 if manhattandist <= dists[i]:
        #                     observations[i:]=1
        #                     indexes_left=indexes_left[0:i]
        #                     break
        #
        #             if not indexes_left:
        #                 if DEBUG_MODE:
        #                     print("observations="+str(observations))
        #                 return observations
        #
        #
        print("dist=" + str(dist+1))
        print("mindist=" + str(mindist))
        print("observations="+str(observations))
        return observations
    def setStandardObservation(self):
        observation = -1 + np.zeros((self.obs_length,))
        # wall configuration
        for d in range(4):
            if self.see_obstacle(directions[d]):
                observation[d] = 1.

        # ghost visible
        for d in range(4):
            if self.see_ghost(directions[d]) >= 0:
                observation[4 + d] = 1.

        # food visible
        for d in range(4):
            if self.see_food(directions[d]):
                observation[8 + d] = 1.
        # food smellable
        observation[12] = 1.0 if self.check_foodwithindist(2) else -1.0
        observation[13] = 1.0 if self.check_foodwithindist(3) else -1.0
        observation[14] = 1.0 if self.check_foodwithindist(4) else -1.0
        # power pill influence
        observation[15] = 1.0 if self.power_steps > 0 else -1.0
        add = 0
        if self.inform_task_time:
            add = 1
            observation[16] = min((self.t - self.currentTask.start_time)/1000,1.)  #0.001, .002, ... 1. , 1. , ...
        if self.add_task_features:
            observation[16+add:] = np.array(self.currentTask.task_feature)/3.

        if DEBUG_MODE:
            print(observation)
        self.agent.learner.observation = observation

    def setMeltingPotObservation(self):
        observation = -1 + np.zeros((self.obs_length,))
        # wall configuration
        for d in range(4):
            if self.see_obstacle(directions[d]):
                observation[d] = 1.

        # object visible
        for d in range(4):
            if self.see_object(directions[d]) >= 0:
                observation[4 + d] = 1.
        # object smellable
        observation[8:11] = self.check_objectwithindist([2,3,4])

        # power pill influence
        #observation[11] = 1.0 if self.power_steps > 0 else -1.0
        add=0
        if self.inform_task_time:
            add=1
            observation[11] = min((self.t - self.currentTask.start_time)/1000.,1.)  #0.001, .002, ... 1. , 1. , ...
        if self.add_task_features:
            observation[11+add:] = np.array(self.currentTask.task_feature)/3.

        if DEBUG_MODE:
            print("OBS="+str(observation))
        self.agent.learner.observation = observation
    def getCurrentSlice(self):
        return self.t//self.statfreq

    def mazestats_update(self,slice,F,key):
        if self.agent.learner.testing:
            return
        if key not in self.stats[slice]:
            self.stats[slice][key] = POcmanMeltingPotStatistics(self.pocman_home, F, self)
        self.stats[slice][key].update((self.agent.x, self.agent.y), self.get_object(), self.agent.learner.chosenAction)
    @overrides
    def updateStat(self):
        slice=self.getCurrentSlice()
        if self.timePolicy.stepCondition(self):
            if hasattr(self.agent.learner,'pols'):
                for pol in range(len(self.agent.learner.pols)):
                    F=tuple(self.currentTask.task_feature)
                    key = "task%s" % (str(F)) + "pol%d"%(pol)
                    self.mazestats_update(slice, F,key)

            else:
                F = tuple(self.currentTask.task_feature)
                key = "task%s"%(str(F))
                self.mazestats_update(slice,F,key)
        self.agent.learner.stats.update(self.agent.learner)
        # oldCoord = (self.oldx,self.oldy)
        # newCoord = (self.agent.x, self.agent.y)
        # new_object=self.get_object()
        # oldCoord = (self.old_F,oldCoord,self.old_object)
        # newCoord = (F,newCoord,new_object)
        # self.agent.learner.track_q(oldCoord,newCoord,self.agent.learner.intervals)
        # self.old_object=new_object
        # self.old_F = F
        # self.oldx = self.agent.x
        # self.oldy = self.agent.y
    @overrides
    def printStatistics(self):
        POcmanMeltingPotStatistics.getAllStats(self.filename,self.stats)
        dump_incremental(self.filename+"_POcmanStats",self.stats)
        self.agent.learner.printStatistics()
    def setObservation(self, agent):

        if self.currentTask.task_type==PacmanTaskType.MELTINGPOT:
            self.setMeltingPotObservation()
        else:
            self.setStandardObservation()


    def see_poison(self, direction):
        x,y = direction
        if DEBUG_MODE:
            if self.check_poison(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self)):
                print("see food in direction %d,%d" % (x, y))
            else:
                print("no food in direction %d,%d" % (x, y))
        return self.check_poison(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self))


    def see_food(self, direction):
        x,y = direction
        if DEBUG_MODE:
            if self.check_food(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self)):
                print("see food in direction %d,%d" % (x, y))
            else:
                print("no food in direction %d,%d" % (x, y))
        return self.check_food(check_toroid_X(x+self.agent.x,self), check_toroid_Y(y+self.agent.y,self))
    def get_object(self):

        if self.num_ghosts > 0 :
            return self.ghost_pos[0]
        else:
            return self.object_location
    def see_object(self,direction):
        """
        see object (exclusively for the meltingpot scenario)
        :param direction:
        :return:
        """

        if self.num_ghosts > 0:
            return self.see_ghost(direction)
        elif self.num_food > 0:
            return self.see_food(direction)
        elif self.num_poison > 0:
            return self.see_poison(direction)
        else:
            if self.num_ghosts==0 and self.num_food==0 and self.num_poison==0:
                raise Exception("no object !?")
            else:
                if DEBUG_MODE:
                    print(self.num_poison)
                    print(self.num_food)
                    print(self.num_ghosts)
                raise Exception('cannot have both objects in meltingpot task')


    def see_obstacle(self, direction):
        x,y = direction
        if DEBUG_MODE:
            if self.maze[check_toroid_X(self.agent.x + x, self), check_toroid_Y(self.agent.y + y,
                                                                                self)] == POcmanFlags.E_OBSTACLE:
                print("see obstacle in direction %d,%d" % (x, y))
            else:
                print("no obstacle in direction %d,%d" % (x, y))
        return self.maze[check_toroid_X(self.agent.x + x, self), check_toroid_Y(self.agent.y + y,
                                                                                self)] == POcmanFlags.E_OBSTACLE

    def see_ghost(self, direction):
        x = check_toroid_X(self.agent.x + direction[0], self)
        y = check_toroid_Y(self.agent.y + direction[1], self)
        eyepos = (x, y)
        if DEBUG_MODE:
            print(self.ghost_pos)
            print(self.num_ghosts)
        for g in range(self.num_ghosts):
            if (self.ghost_pos[g] == eyepos):
                #print("ghost pos "+str(self.ghost_pos[g]))
                if DEBUG_MODE:
                    print("see ghost in direction %d,%d" % (x, y))

                return 1.
        if DEBUG_MODE:
            print("no ghost in direction %d,%d" % (x, y))
        return -1.

    # strange, not in liine with paper and does not seem to be used
    # def LocalMove(self,history,stepObs, status) :
    #
    #
    #     numGhosts = np.random.randint(1, 3) # numpy also exclusive high
    #     # // Change
    #     # 1 or 2
    #     # ghosts
    #     # at
    #     # a
    #     # time
    #     for i in range(self.num_ghosts):
    #         g = np.random.randint(self.num_ghosts)
    #         self.ghost_pos[g] = np.random.random()*self.sizeX
    #         if ( not self.check_passablepos(self.ghost_pos[g]) or  self.ghost_pos[g] == (self.agent.x,self.agent.y)):
    #             return False
    #
    #
    #
    #     smellPos=[]
    #     for x in range(self.smell_range):
    #         for y in range(self.smell_range):
    #
    #             pos = (self.agent.x,self.agent.y) + smellPos
    #             if (smellPos != (0, 0) and self.maze[pos] == E_SEED):
    #                 self.Food[Maze.Index(pos)] = Bernoulli(self.food_prob)
    #
    #
    #         # // Just
    #         # check
    #         # the
    #         # last
    #         # time - step, don
    #         # 't check for full consistency
    #         if (history.Size() == 0)
    #             return True
    #
    #         observation = self.setObservation(pocstate)
    #          return history.Back().Observation == observation

    def move_objects_meltingpot(self,g):
        if self.pacman_dynamic==PacmanDynamic.pacman:
            # move ghost
            self.move_ghosts(g)
            return
        elif self.pacman_dynamic==PacmanDynamic.random:
            if self.t % 20 == 0:
                self.move_ghost_random(g)
            return
        elif self.pacman_dynamic==PacmanDynamic.static:
            return
        else:
            raise Exception("dynamic %s does not exist !"%(str(self.pacman_dynamic)))

    def move_ghosts(self, g):

        if manhattan_dist((self.agent.x, self.agent.y), self.ghost_pos[g]) < self.ghost_range:
            if (self.power_steps > 0):
                self.move_ghost_defensive(g)
            else:
                self.move_ghost_aggressive(g)
        else:

            self.move_ghost_random(g)

        return True

    def check_passablepos_ghost(self,pos,g):
        return self.check_passablepos(pos) and not pos in self.ghost_pos[0:g-1]
    def Bernoulli(self,prob):
        return self.rng.rand() < prob
    def move_ghost_aggressive(self, g):
        if (not self.Bernoulli(self.chase_prob)):
            self.move_ghost_random(g)
            return

        bestDist = self.sizeX + self.sizeY
        bestDir = 0  # stay
        bestPos = self.ghost_pos[g]
        x, y = self.ghost_pos[g]
        if DEBUG_MODE:
            print("ghost %d : move aggressive" % (g))
            print("current pos ghost %d : %d,%d" % (g, x, y))
        gdir = VonNeumannNeighbourhoodPlus[self.ghost_direction[g]]
        for dir in range(1, 5):
            direction = VonNeumannNeighbourhoodPlus[dir]
            dist = directional_dist(
                (self.agent.x, self.agent.y), self.ghost_pos[g], direction)
            vx, vy = direction
            newpos = (check_toroid_X(x + vx, self), check_toroid_Y(y + vy, self))
            if (dist <= bestDist and self.check_passablepos(newpos)
                and not opposite_direction(direction, gdir)):
                bestDist = dist
                bestPos = newpos
                bestDir = dir

        self.ghost_pos[g] = bestPos
        self.ghost_direction[g] = bestDir
        if DEBUG_MODE:
            print("new pos ghost %d : %s" % (g, str(self.ghost_pos[g])))

    def move_ghost_defensive(self, g):
        if (self.Bernoulli(self.defensive_slip) and self.ghost_direction[g] >= 0):
            self.ghost_direction[g] = 0
            return

        bestDist = 0
        bestDir = 0
        x, y = self.ghost_pos[g]
        bestPos = self.ghost_pos[g]
        if DEBUG_MODE:
            print("ghost %d : move defensive" % (g))
            print("current pos ghost %d : %d,%d" % (g, x, y))
        gdir = VonNeumannNeighbourhoodPlus[self.ghost_direction[g]]

        for dir in range(1, 5):
            dist = directional_dist((self.agent.x, self.agent.y), self.ghost_pos[g], VonNeumannNeighbourhoodPlus[dir])
            direction = VonNeumannNeighbourhoodPlus[dir]
            vx, vy = VonNeumannNeighbourhoodPlus[dir]
            newpos = (check_toroid_X(x + vx, self), check_toroid_Y(y + vy, self))
            if (dist >= bestDist and self.check_passablepos(newpos)
                and not opposite_direction(direction, gdir)):
                bestDist = dist
                bestPos = newpos
                bestDir = dir

        self.ghost_pos[g] = bestPos
        self.ghost_direction[g] = bestDir
        if DEBUG_MODE:
            print("new pos ghost %d : %s" % (g, str(self.ghost_pos[g])))

    def move_ghost_random(self, g):

        # // Never
        # switch
        # to
        # opposite
        # direction
        # // Currently
        # assumes
        # there
        # are
        # no
        # dead - ends.


        gdir = VonNeumannNeighbourhoodPlus[self.ghost_direction[g]]
        x, y = self.ghost_pos[g]
        if DEBUG_MODE:
            print("ghost %d : move random" % (g))
            print("current pos ghost %d : %d,%d" % (g, x, y))
        choices=set(range(1,5))
        while True:
            dir = self.rng.choice(list(choices),1)[0]
            choices = choices - set([dir])
            vx, vy = VonNeumannNeighbourhoodPlus[dir]

            newpos = (check_toroid_X(x + vx, self), check_toroid_Y(y + vy, self))

            if (not opposite_direction(VonNeumannNeighbourhoodPlus[dir], gdir) and self.check_passablepos(newpos)):
                break
            if not choices:
                self.ghost_direction[g] = 0
                return
        self.ghost_pos[g] = newpos
        if DEBUG_MODE:
            print("new pos ghost %d : %s" % (g, str(self.ghost_pos[g])))
        self.ghost_direction[g] = dir



    # def set_food_locations(self):
    #     # food is randomly located, other mapelements the same over episodes
    #     for x in range(self.sizeX):
    #         for y in range(self.sizeY):
    #             if self.maze[x,y]==POcmanFlags.E_FOOD:
    #                 self.maze[x,y]==POcmanFlags.E_FREE
    #             if self.maze[x,y]==POcmanFlags.E_FREE:
    #                  if Bernoulli(self.food_prob):
    #                      self.maze[x, y] = POcmanFlags.E_FOOD
    def init_pacman(self):
        if self.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS:
            # initialise around a powerpill
            coords = (self.maze == POcmanFlags.E_POWER).nonzero()
            self.agent.x = self.rng.choice(coords[0])
            self.agent.y = self.rng.choice(coords[1])
            pill_coord = (self.agent.x, self.agent.y)
            while True:
                self.agent.x = min(self.sizeX - 1, max(self.agent.x + randint(-1, 1), 0))
                self.agent.y = min(self.sizeY - 1, max(self.agent.y + randint(-1, 1), 0))
                if self.check_passablepos((self.agent.x, self.agent.y)) and (self.agent.x, self.agent.y) != pill_coord:
                    print("agent pos=%d,%d" % (self.agent.x, self.agent.y))
                    print("pill coord=%d,%d" % (pill_coord))
                    return pill_coord
        else:
            (self.agent.x, self.agent.y) = self.pocman_home

    def within_manhattandist(self, coord, coord2, min, max):
        dist = manhattan_dist(coord, coord2)
        return dist >= min and dist <= max

    def random_passable_pos(self):
        while True:
            x = self.rng.randint(0, self.sizeX - 1)
            y = self.rng.randint(0, self.sizeY - 1)
            if self.check_passablepos((x, y)):
                return (x, y)

    def init_ghosts(self, powerpill_location):
        # recall that num_ghosts=0 if task=EAT_FOOD
        if self.currentTask.task_type==PacmanTaskType.EAT_FOOD:
            self.num_ghosts=0
        self.ghost_pos = []
        self.ghost_direction = []
        for g in range(self.num_ghosts):
            if self.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS:
                xx, yy = powerpill_location
                while True:
                    x = min(self.sizeX - 1, max(xx + self.rng.randint(-4, 4), 0))
                    y = min(self.sizeY - 1, max(yy + self.rng.randint(-4, 4), 0))
                    if self.check_passablepos((x, y)) and self.within_manhattandist((x, y), (xx, yy),
                                                                                    self.ghost_range / 2,
                                                                                    self.ghost_range):
                        break

            else:
                x, y = self.ghost_home
                x += g % 2
                y += g / 2
                if not self.check_passablepos((x,y)):
                    raise Exception()
            self.ghost_pos.append((x, y))
            self.ghost_direction.append(-1)
    def set_food(self,num):
        """
        set the number of food:
        if dynamic, then there are only ghosts
        however, if
        :return:
        """
        self.num_ghosts=0
        self.num_poison=0
        self.num_food=num


    def set_poison(self,num):
        self.num_food=0
        self.num_ghosts=0
        self.num_poison=num

    def set_num_ghosts(self,num):
        self.num_poison=0
        self.num_food=0
        self.num_ghosts=num

    def melting_pot_foodlocations(self,topology_type):
        if topology_type==PacmanTopology.cheese:
            return [(3,3),(5,3)]
        elif topology_type==PacmanTopology.pacman_micro:
            return [(1,1),(1,7),(7,1),(7,7)]
        elif topology_type==PacmanTopology.standard:
            return [(6,1),(9,1),(9,6)]
        else:
            raise Exception("topology not used")
    def random_foodlocation(self,topology_type):
        locations=self.melting_pot_foodlocations(topology_type)
        l=self.rng.randint(len(locations))
        return locations[l]
    def init_food(self, pillcoord):
        if self.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS:
            # only one powerpill allowed
            self.num_food = 1
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food(x, y):  # because standardmap has some food in it already
                        self.maze[x, y] = POcmanFlags.E_FREE
                    if pillcoord == (x, y):
                        self.maze[x, y] = POcmanFlags.E_POWER
        elif self.currentTask.task_type == PacmanTaskType.RUN_GHOSTS:
            # no food allowed
            self.num_food = 0
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food(x, y):  # because standardmap has some food in it already
                        self.maze[x, y] = POcmanFlags.E_FREE
        elif self.currentTask.task_type == PacmanTaskType.MELTINGPOT:

            self.initial_object_location=None
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food(x, y):  # because standardmap has some food in it already
                        self.maze[x, y] = POcmanFlags.E_FREE

            for i in range(self.num_food):
                x, y = self.random_foodlocation(self.currentTask.topology_type)
                self.maze[x, y] = POcmanFlags.E_FOOD
                self.object_location=(x,y)
            for i in range(self.num_poison):
                x, y = self.random_foodlocation(self.currentTask.topology_type)
                self.maze[x, y] = POcmanFlags.E_POISON
                self.object_location=(x,y)

        else:
            # FULL or EAT_FOOD: 4 power pills and additional randomly located foods
            self.num_food = 4  # power pills
            for x in range(self.sizeX):
                for y in range(self.sizeY):
                    if self.check_food_specific(x, y):  # specific: do not remove powerpills
                        self.maze[x, y] = POcmanFlags.E_FREE
                    if self.currentTask.task_type in [PacmanTaskType.EAT_FOOD,
                                                      PacmanTaskType.FULL]:  # grow food randomly
                        if self.check_free(x, y) and self.Bernoulli(self.food_prob):
                            self.maze[x,y] = POcmanFlags.E_FOOD
                            self.num_food += 1

    def choose_topology(self,topology_type,num_ghosts):
        if topology_type==PacmanTopology.cheese:
            self.cheesemaze_map(num_ghosts)
        elif topology_type==PacmanTopology.standard:
            self.standardmaze_map(num_ghosts)
        elif topology_type==PacmanTopology.pacman_micro:
            self.micro_pacman(num_ghosts)
        elif topology_type==PacmanTopology.pacman_mini:
            self.mini_pacman(num_ghosts)
        elif topology_type==PacmanTopology.pacman_standard:
            self.standard_pacman(num_ghosts)
        else:
            raise Exception("not supported topology")



    def new_level(self):
        self.power_steps = 0
        if self.currentTask.task_type == PacmanTaskType.MELTINGPOT:
            self.new_task_meltingpot()
            self.choose_topology(self.currentTask.topology_type,self.num_ghosts)
        else:
            self.choose_topology(self.currentTask.topology_type)
        pillcoord = self.init_pacman()
        self.init_ghosts(pillcoord)

        self.init_food(pillcoord)


        self.num_ghosts_caught = 0
    def move(self,g):
        if self.currentTask.task_type==PacmanTaskType.MELTINGPOT:
            self.move_objects_meltingpot(g)
        else:
            self.move_ghosts(g)

    def new_task_meltingpot(self):
        """
        call this when a new meltingpot task has been initialised
        :return:
        """
        reward,dynamic,_top=self.currentTask.task_feature
        if DEBUG_MODE:
            print('-------------------------------------------------------------------------')
            print('-------------------------------------------------------------------------')
            print("new task meltingpot:")
            print("topology %s,  dynamic %s, reward %.2f"%(str(self.currentTask.topology_type),str(dynamic),reward))

            print('-------------------------------------------------------------------------')
        self.pacman_dynamic=dynamic
        num_objects=  1
        if PacmanDynamic.is_dynamic(dynamic):
            self.set_num_ghosts(num_objects)
            if reward > 0:
                self.power_steps=float('inf') # chase ghosts until task end
                self.reward_eatghost=reward
            else:
                self.power_steps = 0
                self.reward_die = reward  # chase/escape ghosts


        else:
            if reward > 0:
                self.reward_eatfood=reward
                self.power_steps = 0
                self.set_food(num_objects)
            else:
                self.reward_eatpoison=reward
                self.power_steps = 0
                self.set_poison(num_objects)


defaultmapobjfuns = {}


def reward_fun_pocman(agent, environment):
    if DEBUG_MODE:
        print("rewarding agent")
        print("current location: %d, %d" % (agent.x, agent.y))

    if (environment.power_steps > 0):
        environment.power_steps -= 1
    else:
        # reset the num ghosts hit
        environment.num_ghosts_caught = 0
    if DEBUG_MODE:
        print("power steps=%s" % (environment.power_steps))

    reward = environment.reward_default
    #check hit wall
    if environment.no_move:
        reward += environment.reward_hitwall
        if DEBUG_MODE:
            print("hit wall, %d" % (environment.reward_hitwall))

    # check hit ghost or no (powersteps-> reward, no powersteps--> die)
    hitGhost = -1
    for g in range(environment.num_ghosts):

        if (environment.ghost_pos[g] == (environment.agent.x, environment.agent.y)):
            hitGhost = g
            break
        environment.move(g)
        if (environment.ghost_pos[g] == (environment.agent.x, environment.agent.y)):
            hitGhost = g
            break
    if (hitGhost >= 0):
        if (environment.power_steps > 0):
            if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
                environment.num_ghosts_caught += 1
                reward += environment.reward_eatghost * environment.num_ghosts_caught
                environment.ghost_pos[hitGhost] = environment.ghost_home
                environment.ghost_direction[hitGhost] = 0
            else:
                reward += environment.reward_eatghost

            if DEBUG_MODE:
                print("eat ghost %d, %d" % (g, environment.reward_eatghost * environment.num_ghosts_caught))

        else:
            reward += environment.reward_die
            if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
                environment.terminal = True
                if DEBUG_MODE:
                    print("hit ghost %d -->  DEATH ! %d" % (g, environment.reward_die))
            else:
                if DEBUG_MODE:
                    print("hit ghost %d , %d" % (g, environment.reward_die))

    # observation = environment.setObservation(pocstate)


    x, y = (environment.agent.x, environment.agent.y)
    # check food or powerpill
    if (environment.check_food(x, y)):
        if environment.check_powerpill(x, y):
            environment.power_steps = environment.power_numsteps
            reward += environment.reward_eatpower
            if DEBUG_MODE:
                print("eat power pill, powersteps=%d, %d" % (environment.power_steps, environment.reward_eatpower))
        else:
            if DEBUG_MODE:
                print("eat food, %d" % (environment.reward_eatfood))
            reward += environment.reward_eatfood
        if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
            environment.maze[x, y] = POcmanFlags.E_FREE
            environment.num_food -= 1
    # check poison
    if (environment.check_poison(x, y)):
        reward += environment.reward_eatpoison
        if DEBUG_MODE:
            print("eat poison, %d" % (environment.reward_eatpoison))
        if environment.currentTask.task_type != PacmanTaskType.MELTINGPOT:
            environment.maze[x, y] = POcmanFlags.E_FREE
            environment.num_food -= 1
    # check food gone or no
    if (environment.num_food == 0):
        if environment.currentTask.task_type in [PacmanTaskType.EAT_FOOD, PacmanTaskType.FULL] or\
                (environment.currentTask.task_type == PacmanTaskType.CHASE_GHOSTS and environment.power_steps == 0):
            reward += environment.reward_clearlevel
            environment.terminal = True
            if DEBUG_MODE:
                print("clear level (all food gone), %d" % (environment.reward_clearlevel))
    # check time up or no
    if environment.time_up():
        if DEBUG_MODE:
            print("final time of the elementary task at time %d" % (environment.t))
        environment.terminal=True
    if environment.currentTask.task_type == PacmanTaskType.MELTINGPOT:
        if environment.currentTask.task_feature[0]==1.:
            assert reward==0. or reward==1.
        else:
            assert reward==0 or reward==-1
    return reward





#sizeX and sizeY just are the maximum (for visualisation)
pacmanparams={'sizeX': 19 ,'sizeY':21, 'tasks': [],'observation_size': 1, 'observation_length':16,'sampling_rate': 10000, 'dynamic':False,
              'real_time':False ,'network_types':['external_actions'],'use_stats': True,
               'eval':False,'agent_filename': "pacman_pic.jpg",'include_task_features': False, 'inform_task_time':False,
              'record_intervals': [],'reward_hitwall':0.,'elementary_task_time':1000}