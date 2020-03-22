
import sys,os

from tkinter import *
import dill



from Environment import *
from MazeUtils import *
from StatsAndVisualisation.Statistics import MazeStatistics, PolType
from Catastrophic_Forgetting_NNs.DRQN_Learner import DRQN_Learner
from IS.SSA_gradientQ import SSA_gradientQ

from mapobjects import *


class LongTermType(object):
    complex=1
    pseudo_goal=2
    memory_perturb=3

longterm_type=LongTermType.complex

class Maze(NavigationEnvironment):
    reward_food=1.0
    reward_default=0.0
    reset_type= ResetType.random
    dynamic_food=False
    task_feature_length=0
    def __init__(self,agent,visual,switching,params=None):
        self.statcounter = 0
        self.statistics = {}
        self.inform_mazenum = False
        self.inform_NP = False
        self.no_move = False
        self.terminal = True

        if 'reward_food' in params:
            self.reward_food=params['reward_food']
        if 'reward_default' in params:
            self.reward_default=params['reward_default']




        self.setResetType(params)


        self.switching=switching
        self.externalactions=["north","west","east","south"]

        # Only odd shapes
        NavigationEnvironment.__init__(self, agent, visual, params)



      #self.observability=params['observability']
    def get_map(self,x,y):
        return self.map[x][y]
    def setResetType(self, params):
        reset_type = params['reset_type']
        #print(reset_type)
        if reset_type == "random":
            self.reset_type = ResetType.random
        elif reset_type == "longterm":
            self.reset_type = ResetType.longterm
            self.reset_period = params['reset_period']
        else:
            self.reset_type = ResetType.fixed

    def new_task_meltingpot(self, dynamic, reward):
        """
               call this when a new meltingpot task has been initialised
               :return:
               """

        self.dynamic_food=dynamic
        self.reward_food = reward

    def currentStat(self):
        return None
    @overrides
    def updateStat(self):

        self.statcounter=self.timePolicy.getStatCounter(self)
        stat=self.currentStat()
        if not stat:
            return
        stat.add_slice(self.statcounter,self.agent.learner)

        oldCoord = (self.oldx, self.oldy)
        newCoord = (self.agent.x, self.agent.y)
        if self.reset_type==ResetType.longterm:
            oldCoord=(oldCoord,self.foodCoord)
            newCoord = (newCoord, self.foodCoord)

        stat.setNetOutputDistribution(self.statcounter,oldCoord,self.agent.learner) # for SMP-Constrthe new coord of old tick is old coord of new tick ,
        stat.setDistribution(self.statcounter,oldCoord,self.agent.learner) # given old coordinates what action was taken ?
        if hasattr(self.agent.learner,"IP"):
            stat.setIPdistribution(self.statcounter,newCoord,self.agent.learner.IP) # what IP for current location ?
        if self.agent.learner.network_usage:
            stat.setnetworkusagedistribution(self.statcounter,oldCoord,self.agent.learner) # given old coordinates which network was used ?
            if hasattr(self.agent.learner,"IP"):
                stat.set_output_distribution(self.statcounter,self.agent.learner) # how much IP which action chosen ?
        if isinstance(self.agent.learner,DRQN_Learner) or isinstance(self.agent.learner,SSA_gradientQ):
            self.agent.learner.track_q(oldCoord,newCoord,self.agent.learner.intervals)
        stat.countSteps(self.agent)
        self.agent.learner.stats.update(self.agent.learner)
    @overrides
    def printStatistics(self,poltype=PolType.GLOBAL):
        maze_dir = str(os.environ['HOME']) + '/PycharmProjects/PhD/Environments/Mazes/'
        self.statcounter = self.timePolicy.getStatCounter(self)
        print(self.statcounter)
        print("statfreq="+str(self.statfreq))


        for mazenum in self.statistics:
            maxcount=self.statcounter if self.t % self.statfreq==0 else self.statcounter+1
            for tt in range(maxcount):
                print("t=%d" % (tt))
                self.statistics[mazenum].getStats(self, tt, self.filename + "arrowMap%d" % (MAZES[mazenum]),maze_dir,pol_type=poltype)
        self.agent.learner.printStatistics()
    def setCorrectResponses(self):
        """ for each feasible coordinate, determine the optimal action(s) (1) and suboptimal actions (0)"""

        self.optimalactionmap={}
        count=0
        for (x,y) in self.feasible_coords:
            if (x,y)==self.foodCoord:
                continue
            begin=time.time()
            oldCoord=(x,y)
            optimals=[]
            minimum=float("inf")
            actionmap={north:"north",west:"west",east:"east",south:"south"}
            for action in [north,west,east,south]:
                self.agent.x,self.agent.y=oldCoord
                action(self.agent,self)
                distance,_ = self.getOptimalLength((self.agent.x,self.agent.y),self.foodCoord)
                if distance == minimum:
                    optimals.append(actionmap[action])
                if distance < minimum:
                    optimals=[actionmap[action]] #remove previous optimals, as they aren't optimal
                    minimum=distance
            count+=1
            t_passed = time.time() - begin
            print("%d/%d"%(count,len(self.feasible_coords)))
            print("%.2f seconds"%(t_passed))
            self.optimalactionmap[oldCoord]=optimals # for each coordinate a list of optimal responses

        dill.dump(self.optimalactionmap, open(self.currentTask.files + "_optimalResponses", "w"))

    def setCorrectResponsesMultigoal(self):
        """
        assuming multiple goals exist:
        for each feasible coordinate, determine the optimal action(s) (1) and suboptimal actions (0)"""

        self.optimalactionmap={}
        count=0
        for g in self.goal_coords:
            print(str(g))
            for (x,y) in self.feasible_coords:
                print(str((x,y)))
                if (x,y)==g:
                    continue
                begin=time.time()
                oldCoord=((x,y),g)
                optimals=[]
                minimum=float("inf")
                actionmap={north:"north",west:"west",east:"east",south:"south"}
                for action in [north,west,east,south]:
                    self.agent.x,self.agent.y=oldCoord[0]
                    print("ag="+str((self.agent.x,self.agent.y)))
                    print("g=" + str(g))
                    action(self.agent,self)
                    distance,_ = self.getOptimalLength((self.agent.x,self.agent.y),g)
                    print(distance)
                    if distance == minimum:
                        optimals.append(actionmap[action])
                    if distance < minimum:
                        optimals=[actionmap[action]] #remove previous optimals, as they aren't optimal
                        minimum=distance
                count+=1
                t_passed = time.time() - begin
                print("%d/%d"%(count,len(self.feasible_coords)))
                print("%.2f seconds"%(t_passed))
                self.optimalactionmap[oldCoord]=optimals # for each coordinate a list of optimal responses
        maze_dir = str(os.environ['HOME']) + '/PycharmProjects/PhD/Environments/Mazes/'
        dill.dump(self.optimalactionmap, open(maze_dir+self.currentTask.files + "_optimalResponsesMultigoal", "wb"))

    def setStartLocations(self):
        """ based on the feasible coords, remove all locations that do not satisfy the distance criterion"""

        feasible = self.feasible_coords

        min_radius = 0.8 * self.optimum
        max_radius = 1.2 * self.optimum
        for coord in self.feasible_coords:

            if coord == self.foodCoord:
                feasible = [value for value in feasible if value != coord]  # remove from feasible
                continue
            manh_distance = manhattan_dist(coord, self.foodCoord)
            # if manhattan distance greater than max_radius, you can easily reject
            if manh_distance > max_radius:
                feasible = [value for value in feasible if value != coord]  # remove from feasible
                continue
            # if select:
            #     # preselect points relatively close to the original startCoord
            #     manh_distance = manhattan_dist(coord, self.startCoord)
            distance,_ = self.getOptimalLength(coord, self.foodCoord)
            if distance > max_radius or distance < min_radius:
                feasible = [value for value in feasible if value != coord]  # remove from feasible
            else:
                pass

                    # now store back into member variable
        self.start_coords = feasible
        dill.dump(self.start_coords, open(self.currentTask.files + "startCoords", "w"))
        print("done saving startCoords")
    def setLongtermLocations(self):
        """ based on the feasible coords and a goal G1,

        create a single other goal G2 and start S1 such that

            dist(S,G1)~=dist(S,G2)=self.optimum
            dist(G1,G2) is maximal

        """

        feasible_starts = self.feasible_coords
        G2=None
        S=None
        max_dist=0
        min_radius = .90*self.optimum   # .50,1.20 for easy  0.90,1.10 times optimum for difficult
        max_radius = 1.10*self.optimum
        for s in self.feasible_coords:

            if s == self.foodCoord:
                feasible_starts = [value for value in feasible_starts if value != s]  # remove from feasible
                continue
            manh_distance = manhattan_dist(s, self.foodCoord)
            # if manhattan distance greater than max_radius, you can easily reject
            if manh_distance > max_radius:
                feasible_starts = [value for value in feasible_starts if value != s]  # remove from feasible
                continue
            # if select:
            #     # preselect points relatively close to the original startCoord
            #     manh_distance = manhattan_dist(coord, self.startCoord)
            distance,_ = self.getOptimalLength(s, self.foodCoord)
            if distance > max_radius or distance < min_radius:
                feasible_starts = [value for value in feasible_starts if value != s]  # remove from feasible
            else:
                pass
        if not feasible_starts:
            print("No feasible starts found  ! ! ! ! !")
            import sys
            sys.exit()
        for s in feasible_starts:
            for g2 in self.feasible_coords:
                #1. find a goal g2 that is at self.optimum distance to s
                if s == g2 or g2==self.foodCoord:
                    continue

                manh_distance = manhattan_dist(s, g2)
                # if manhattan distance greater than max_radius, you can easily reject
                if manh_distance > max_radius:
                    continue
                # if select:
                #     # preselect points relatively close to the original startCoord
                #     manh_distance = manhattan_dist(coord, self.startCoord)
                distance,path = self.getOptimalLength(s, g2)
                if distance > max_radius or distance < min_radius:
                    continue
                # goals should not be in each other path
                if self.foodCoord in path:
                    continue
                _, path2 = self.getOptimalLength(s, self.foodCoord)
                if g2 in path2:
                    continue
                #2. if distance ~= optimum, check whether dist(g1,g2) is maximal
                distance,_ = self.getOptimalLength(self.foodCoord, g2)
                if distance > max_dist:
                    max_dist=distance
                    G2=g2
                    S=s


        # now store back into member variable
        self.startCoord = S
        self.goal_coords = [self.foodCoord,G2]
        print("goals="+str(self.goal_coords))
        print("start="+str(self.startCoord))
        maze_dir = str(os.environ['HOME']) + '/PycharmProjects/PhD/Environments/Mazes/'
        dill.dump(self.goal_coords, open(maze_dir+self.currentTask.files + "goalCoordsLongterm", "w"))
        dill.dump(self.startCoord, open(maze_dir+self.currentTask.files + "startCoordLongterm", "w"))
        print("done saving goalCoords")

    def setCurrentGoal(self):
        self.goal_counter += 1

        # if (self.agent.x,self.agent.y) == self.goal_coords[id]:
        if self.goal_counter==self.reset_period:

            id = self.currentGoalIndex
            g = self.goal_coords[id] if longterm_type != LongTermType.complex else (self.agent.x,self.agent.y)
            self.map[g[0]][g[1]] = PseudoFood(None) if longterm_type!=LongTermType.memory_perturb else EmptyObject() # current achieved goal
            self.goal_counter=0
            self.currentGoalIndex=0 if id==1 else 1 #switch the goal index
            g = self.goal_coords[self.currentGoalIndex] # new goal is given

            self.map[g[0]][g[1]] = Food(None)
            self.foodCoord = g
            # id = self.currentGoalIndex
            # self.goal_counter=0
            # self.currentGoalIndex=0 if id==1 else 1 #switch the goal index
            # g = self.goal_coords[self.currentGoalIndex] # new goal is given
            # self.map[self.agent.x][self.agent.y] = PseudoFood(None) # current achieved goal
            # self.map[g[0]][g[1]] = Food(None)
            # self.foodCoord = g

    def setGoal(self,g1s,g2s): # g1: pseudogoals, g2 real goals
        for g1 in g1s:
            for g2 in g2s:
                self.map[g1[0]][g1[1]] = PseudoFood(None)
                self.map[g2[0]][g2[1]] = Food(None)


    def resetLocation(self):
        if self.reset_type==ResetType.random:
            idx=self.rng.randint(len(self.start_coords))
            self.currentStart=self.start_coords[idx]
        elif self.reset_type==ResetType.longterm:
            self.setCurrentGoal()
            self.currentStart=self.startCoord

        else:
            self.currentStart=self.startCoord
        self.agent.x, self.agent.y = self.currentStart
        self.oldx,self.oldy=(self.agent.x,self.agent.y)
    def endEpisode(self):
        if self.t > 0:
            if self.agent.learner.episodic and self.terminal:

                    self.agent.learner.setTerminalObservation(self.agent, self)  # set the final observation
            stat=self.currentStat()
            if stat:
                stat.setGoalAchieved(self.statcounter, self.currentStart, self.foodCoord)

        self.agent.learner.reset()
    def reset(self):

        if not self.terminal: return
        # is called when starting and when the reward is given and the coming state is terminal

        self.endEpisode()
        self.resetLocation()

        self.terminal=False
    def getOptimalLength(self, start, end,bfs=False):
        """ get the optimal length, for a given startcoord"""
        if start == end:
            return (0,[])
        if bfs:
            path = shortest_path_bfs(self.graph, start, end)
            return len(path)
        return self.shortest_path_astar(start,end)

    def setFeasibleCoords(self):
        self.feasible_coords = []
        self.feasible_coords.append(self.foodCoord)
        index = 0
        # recursively check all the neighbours until all neighbours have no more neighbours not in the list
        neighbours = [self.foodCoord]

        while neighbours:
            old_neighbours = neighbours
            neighbours = []
            for neighbour in old_neighbours:
                ns = self.getNeighboursPlus(neighbour)

                neighbours += ns
        self.graph = {}
        for coord in self.feasible_coords:
            self.graph[coord] = set(self.getNeighbours(coord))

        # dill.dump(self.feasible_coords,open(self.currentTask.files + "_feasible", "w"))
        # dill.dump(self.graph, open(self.currentTask.files + "_graph", "w"))

    def getNeighbours(self, tuple):
        """ return the neighbours of (x,y)      """
        x=tuple[0]
        y=tuple[1]
        neighbours = []
        for option in directions:
            new_x = x + option[0]
            new_y = y + option[1]
            new_x = check_toroid_Y(new_x, self)
            new_y = check_toroid_Y(new_y, self)
            if (check_obstacle((new_x, new_y), self)):
                continue
            neighbours.append((new_x, new_y))
        return neighbours
    def shortest_path_astar(self, start, goal):
        from pypaths import astar
        finder = astar.pathfinder(cost=manhattan_dist,neighbors=self.getNeighbours)
        result = finder(start,goal)
        return result
    def getNeighboursPlus(self, tuple):
        """ return the neighbours of (x,y)"""
        x = tuple[0]
        y = tuple[1]
        neighbours = []
        for option in directions:
            new_x = x + option[0]
            new_y = y + option[1]
            new_x = check_toroid_Y(new_x, self)
            new_y = check_toroid_Y(new_y, self)
            if (check_obstacle((new_x, new_y), self)):
                continue
            if (new_x, new_y) not in self.feasible_coords:
                neighbours.append((new_x, new_y))
                self.feasible_coords.append((new_x, new_y))
        return neighbours

    @overrides
    def inputSize(self):
        add = 1 if self.inform_mazenum and self.switching else 0
        if FULLOBS:
            return (2 + add,)
        return (4 + add,)  # just one dimension with length 4

    @overrides
    def newTask(self):
        # print("new task")
        self.tasks.append(generateNewTask(self.currentTask.end_time, generate_new=True))



    @overrides
    def checkLegalMove(self, agent, tuple):
        x=tuple[0]
        y=tuple[1]
        (xx, yy) = copy.copy((self.agent.x, self.agent.y))
        self.agent.x += x
        self.agent.y += y
        if (check_obstacle(self.agent, self)):
            self.agent.x, self.agent.y = (xx, yy)

    @overrides
    def checkBounds(self, tuple):
        x = tuple[0]
        y = tuple[1]
        return (x, y)  # always legal, since obstacle at the bounds (automatically taken care of)

    @overrides
    def setAgentMap(self):

        for y in [-1, 0, +1]:
            for x in [-1, 0, +1]:
                x_real = self.agent.x + x
                y_real = self.agent.y + y

                obs_x = x + 1
                obs_y = y + 1
                if (x, y) in directions:
                    # note: visualisation slightly different from observation: agent does not actually see the target
                    filename=self.map[x_real][y_real].filename


                else:
                    if x==0 and y==0:
                        filename=self.agent.filename
                    else:
                        filename ='invisible.png'
                self.vis.display.addObsPicture(obs_x, obs_y, filename)
    def setObs(self,agent):
        i=0
        for (x, y) in directions:
            xx = check_toroid_X(agent.x + x,self)
            yy = check_toroid_Y(agent.y + y,self)
            if (check_obstacle((xx, yy), self)):
                agent.learner.observation.append(1.)
            else:
                agent.learner.observation.append(-1.)
            i+=1


            # if (check_obstacle((xx, yy), self)):
            #     agent.learner.observation[i]=1.
            # else:
            #     agent.learner.observation[i]=-1.
            # i+=1

class StandardMaze(Maze):

    def __init__(self, agent, visual, switching, params=None):
        Maze.__init__(self, agent, visual, switching, params)

    @overrides
    def generateMap(self, filename=None):
        # print("time= %d"%(self.t))
        self.occupiedCoords = [(3, 2), (3, 3), (3, 4), (6, 5), (8, 1), (8, 2), (
        8, 3)]  # we count downwards and from 0, so coords a little different from paper, but figure the same
        self.foodCoord = (9, 1)  # goal location
        self.startCoord = (1, 3)
        self.sizeX = 11
        self.sizeY = 8
        self.map = []
        self.fill_borders()
        for (x, y) in self.occupiedCoords:
            self.map[x][y] = Obstacle(None)
        self.map[self.foodCoord[0]][self.foodCoord[1]] = Food(None)

            # if self.random_reset:
            #     self.setFeasibleCoords()
            #     self.setStartLocations()


class POmaze(Maze):

    def __init__(self, agent, visual, switching, params=None):
        self.setResetType(params)
        self.complexity = params['complexity']
        self.density = params['density']
        self.terminal=True

        if 'reward_pseudofood' in params:
            self.reward_default=params['reward_pseudofood']
        Maze.__init__(self, agent, visual, switching, params)

    @overrides
    def currentStat(self):
        return self.statistics[self.currentTask.maze_id]
    def custom_maze(self):
        # Adjust complexity and density relative to maze size
        complexity = int(self.complexity * (5 * (self.sizeX + self.sizeY))) # length of the island
        density = int(self.density * ((self.sizeX // 2) * (self.sizeY // 2))) # number of islands

        self.fill_borders()


        # Make aisles
        for i in range(density):
            x, y = self.rng.randint(1, self.sizeX-1), self.rng.randint(1, self.sizeY-1)
            self.map[x][y] = EmptyObject()
            for j in range(complexity):
                neighbours = [] #neighbours in each direction two steps
                if y > 1:             neighbours.append((x,y - 2))
                if y < self.sizeY - 2:  neighbours.append((x, y + 2))
                if x > 1:             neighbours.append((x - 2, y))
                if x < self.sizeX - 2:  neighbours.append((x + 2, y))
                if len(neighbours):
                    x_, y_ = neighbours[self.rng.randint(0, len(neighbours))] # select a random neighbour
                    print((x_,y_))
                    if self.map[x_][y_].walkable:
                        self.map[x_][y_] = Obstacle(None)
                        self.map[x_ + (x - x_) // 2][y_ + (y - y_) // 2] = Obstacle(None)
                        x, y = x_, y_
    def createMazes(self):
        j=0
        for i in range(31,45):

            while True:
                self.custom_maze()
                self.vis.display.canvas.delete(ALL)
                self.vis.display.setEnvironmentMap(self)
                self.vis.display.canvas.pack(side='left')

                accept=str(input("y/n"))
                # if accept == "y":
                #     dill.dump(self.map, open('maze' + str(i), "w"))
                #     break
            # while True:
            #     x=int(raw_input("start_x:"))
            #     y=int(raw_input("start_y:"))
            #     foodx = int(raw_input("end_x:"))
            #     foody = int(raw_input("end_y:"))
            #     self.map[foodx][foody] = Food(None)
            #     self.agent.x = x
            #     self.agent.y = y
            #     self.startCoord=(x,y)
            #     self.foodCoord=(foodx,foody)
            #     self.vis.display.canvas.delete(ALL)
            #     self.vis.display.setEnvironmentMap(self)
            #     self.vis.display.canvas.pack(side='left')
            #     accept=str(raw_input("y/n"))
                # if accept=="y":
                #
                #     dill.dump(self.startCoord, open('maze'+str(i)+'start', "w"))
                #     dill.dump(self.foodCoord, open('maze' + str(i) + 'end', "w"))
                #     break

    @overrides
    def generateMap(self, filename=None):
        # print("time= %d"%(self.t))

        if filename is None:
            maze_dir = str(os.environ['HOME']) + '/LifelongRL/Mazes/'
            filename = maze_dir  + self.currentTask.files
            # self.standard_maze()
            # print(self.currentTask.files)

        self.map = dill.load(open(filename, "rb"))



        self.foodCoord = dill.load(open(filename + 'end', "rb"))


        self.startCoord = dill.load(open(filename + 'start', "rb"))

        self.map[self.foodCoord[0]][self.foodCoord[1]]=EmptyObject()

        dill.dump(self.map,open(filename, "wb"))
        self.map = dill.load(open(filename, "rb"))

        with open(filename + "optimum") as f:
            self.optimum = int(f.readline())
            manh_dist = manhattan_dist(self.foodCoord, self.startCoord)
            # calculate the ratio of
            self.path_complexity = self.optimum / float(manh_dist)


        # import sys
        # sys.exit()
        self.feasible_coords = dill.load(open(filename + '_feasible', "rb"))

        # self.setFeasibleCoords()
        # self.setLongtermLocations()
        # import sys
        # sys.exit()
        # self.setStartLocations()
        # self.setCorrectResponses()
        # import sys
        # sys.exit()
        print(self.reset_type)
        if self.reset_type==ResetType.random:
            # self.setFeasibleCoords()
            # self.setStartLocations()
            self.start_coords = dill.load(open(filename + 'startCoords', 'rb'))
            # optima=[]
            # for s in self.start_coords:
            #     optima.append(self.getOptimalLength(s,self.foodCoord))
            # dill.dump(optima,open(filename+'optimaReset',"wb"))
            # import sys
            # sys.exit()

        elif self.reset_type==ResetType.longterm:
            print("longterm")
            self.startCoord = dill.load(open(filename + 'startCoordLongterm', "rb"))
            self.goal_coords = dill.load(open(filename + 'goalCoordsLongterm', 'rb'))
            # self.setCorrectResponsesMultigoal()
            # optima=[]
            # for g in self.goal_coords:
            #     optima.append(self.getOptimalLength(self.startCoord,g))
            # dill.dump(optima,open(filename+'optimaLongterm',"wb"))
            # import sys
            # sys.exit()
            self.currentGoalIndex=0
            self.goal_counter=-1 #reset will do +1 before the agent gets first goal achievement
            g = self.goal_coords[0]
            self.map[g[0]][g[1]] = Food(None)
            if longterm_type != LongTermType.memory_perturb:
                g2=self.goal_coords[1]
                self.map[g2[0]][g2[1]] = PseudoFood(None)


        else:
            pass


        # self.correctresponses=dill.load(open(filename+'_optimalResponses',"rb"))
        # self.start_coords = dill.load(open(filename + 'startCoords', 'rb'))
        self.map[self.foodCoord[0]][self.foodCoord[1]] = Food(None)

        if self.currentTask.maze_id not in self.statistics:
            start=self.start_coords if self.reset_type == ResetType.random else [self.startCoord]
            goal=self.goal_coords if self.reset_type == ResetType.longterm else [self.foodCoord]
            file = self.filename + "stats" + str(MAZES[self.currentTask.maze_id])
            self.statistics[self.currentTask.maze_id] = MazeStatistics(start,goal,self.feasible_coords,self.externalactions, self.agent.learner,
                                                                       self.sizeX, self.sizeY, mazefilename=self.currentTask.files,
                                                                   filename=file,slices=self.slices)
    @overrides
    def setObservation(self, agent):
                # print("%d / %d"%(self.t,self.currentTask.end_time))
                # print("current maze=%d"%(self.currentTask.maze_id))


        agent.learner.observation = []
        if FULLOBS:
            agent.learner.observation = [agent.x / float(self.sizeX), agent.y / float(self.sizeY)]
        else:
            self.setObs(agent)

        if self.inform_mazenum:
            agent.learner.observation.append(self.currentTask.maze_id / 10.0)
            if self.inform_NP:
                agent.learner.set_net(self.currentTask.maze_id, direct=True)
        if DEBUG_MODE: print("observation=" + str(agent.learner.observation))



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
        print("x,y = %d,%d"%(self.agent.x,self.agent.y))
        print(agent.learner.observation)



class TMaze(Maze):


    def __init__(self, agent, visual, switching, params=None):
        self.terminal = True
        Maze.__init__(self, agent, visual, switching, params)
        self.setResetType(params)
        self.NSUCCESS = 100
        self.success_rate=0.50
        self.currentCount=1 #keep separate count, more handy for dealing with it in the reward f
        self.maxCount=self.sizeX - 1
        self.maxNrTakenActions=300
        self.actionCount=0

    def createMaze(self):
        # Fill borders
        self.sizeY = 5
        self.map = [[EmptyObject()] * self.sizeY for _ in range(self.sizeX)]
        for x in [0, self.sizeX - 2, self.sizeX - 1]:
            for y in range(self.sizeY):
                self.map[x][y] = Obstacle(None)
        for y in [0, 1, self.sizeY - 2, self.sizeY - 1]:
            for x in range(self.sizeX):
                self.map[x][y] = Obstacle(None)
        ############         ##########
        ############         ######## #
        #         ##   --->  #        #
        ############         ######## #
        ############         ##########
        # remove right central obstacles
        self.map[self.sizeX - 2][1] = EmptyObject()
        self.map[self.sizeX - 2][2] = EmptyObject()
        self.map[self.sizeX - 2][3] = EmptyObject()

    def generateMap(self,files=None):
        self.createMaze()
        self.startCoord = (1, 2)
        self.random_reset = False
        self.food_coords = [(self.sizeX - 2, 1), (self.sizeX - 2, 3)]
        food=self.rng.choice(len(self.food_coords), 1)[0]
        self.foodCoord = self.food_coords[food]

    @overrides
    def setObservation(self, agent):

        # Tmaze obs
        if self.t % 10000 == 0:
            self.agent.learner.Rfile.write("SUCCES =" + str(self.success_rate) + "\n")
            self.agent.learner.Rfile.flush()
        self.setTmazeObs(agent)
    def setTmazeObs(self,agent):

        if self.currentCount==1:
            if self.rotation == 90:
                agent.learner.observation = [0, 1, 1]  # up
            else:
                agent.learner.observation = [1, 1, 0]  # down
        elif self.currentCount==self.maxCount:
            # junction
            agent.learner.observation = [0, 1, 0]
        elif self.currentCount>self.maxCount:
            #terminal state
            agent.learner.observation = [1,1,1]
        else:
            # corridor
            agent.learner.observation = [1,0,1]


        print("x,y=(%d,%d)"%(agent.x,agent.y))
        print("obs=" + str(agent.learner.observation))
        print()
    def resetTarget(self):
        self.map[self.foodCoord[0]][self.foodCoord[1]] = EmptyObject()
        # ind=0 #if you want easy problem
        ind = self.rng.randint(0, 2)  # if you want difficult
        self.rotation = 90 if ind == 0 else 270
        self.foodCoord = self.food_coords[ind]
        self.pseudoCoord = self.food_coords[not ind]
        self.map[self.foodCoord[0]][self.foodCoord[1]] = Food(None)
        self.map[self.pseudoCoord[0]][self.pseudoCoord[1]] = PseudoFood(None)
        self.map[self.startCoord[0]][self.startCoord[1]] = Arrow(None, self.rotation)
    def endEpisode(self):
        if self.t > 0:
            if self.agent.learner.episodic and self.terminal:

                    self.currentCount = self.maxCount + 1
                    self.agent.learner.setObservation(self.agent, self)  # set the final observation
        self.agent.learner.reset()
    def reset(self):
        #gets done every tick:
        if not self.terminal:
            return

        self.resetTarget()

        # # if self.total > 100:
        # #     if self.successes / float(self.total) > .80:
        #         #self.agent.learner.Rfile.write("SUCCESS after " + str(self.t) + "\n")
        #         #self.agent.learner.Rfile.flush()
        #         #self.currentTask.solved = True
        self.endEpisode()
        self.resetLocation()
        self.currentCount=1
        self.terminal=False


class FourRoomMaze(Maze):
    """ non-episodic version of four-room maze """
    terminal = True

    def __init__(self, agent, visual, switching, params=None):
        Maze.__init__(self, agent, visual, switching, params)
        self.NSUCCESS = 100
        self.success_rate = 0.50
        self.currentCount = 1  # keep separate count, more handy for dealing with it in the reward f
        self.maxCount = self.sizeX - 1

    def createMaze(self):
        # Fill borders
        self.sizeY = 5
        self.map = [[EmptyObject()] * self.sizeY for _ in range(self.sizeX)]
        for x in [0, self.sizeX - 2, self.sizeX - 1]:
            for y in range(self.sizeY):
                self.map[x][y] = Obstacle(None)
        for y in [0, 1, self.sizeY - 2, self.sizeY - 1]:
            for x in range(self.sizeX):
                self.map[x][y] = Obstacle(None)
        ############         ##########
        ############         ######## #
        #         ##   --->  #        #
        ############         ######## #
        ############         ##########
        # remove right central obstacles
        self.map[self.sizeX - 2][1] = EmptyObject()
        self.map[self.sizeX - 2][2] = EmptyObject()
        self.map[self.sizeX - 2][3] = EmptyObject()

    def generateMap(self, files=None):
        self.createMaze()
        self.startCoord = (1, 2)
        self.random_reset = False
        self.food_coords = [(self.sizeX - 2, 1), (self.sizeX - 2, 3)]
        self.foodCoord = self.rng.choice(self.food_coords,1)[0]

    @overrides
    def setObservation(self, agent):

        # Tmaze obs
        if self.t % 10000 == 0:
            self.agent.learner.Rfile.write("SUCCES =" + str(self.success_rate) + "\n")
            self.agent.learner.Rfile.flush()
        self.setTmazeObs(agent)

    def setTmazeObs(self, agent):

        if self.currentCount == 1:
            if self.rotation == 90:
                agent.learner.observation = [0, 1, 1]  # up
            else:
                agent.learner.observation = [1, 1, 0]  # down
        elif self.currentCount == self.maxCount:
            # junction
            agent.learner.observation = [0, 1, 0]
        elif self.currentCount > self.maxCount:
            # terminal state
            agent.learner.observation = [1, 1, 1]
        else:
            # corridor
            agent.learner.observation = [1, 0, 1]

        print("x,y=(%d,%d)" % (agent.x, agent.y))
        print("obs=" + str(agent.learner.observation))
        print()

    def resetTarget(self):
        self.map[self.foodCoord[0]][self.foodCoord[1]] = EmptyObject()
        # ind=0 #if you want easy problem
        ind = self.rng.randint(0, 2)  # if you want difficult
        self.rotation = 90 if ind == 0 else 270
        self.foodCoord = self.food_coords[ind]
        self.map[self.foodCoord[0]][self.foodCoord[1]] = Food(None)
        self.map[self.startCoord[0]][self.startCoord[1]] = Arrow(None, self.rotation)

    def endEpisode(self):
        if self.agent.learner.episodic and self.terminal:
            if self.t > 0:
                self.currentCount = self.maxCount + 1
                self.agent.learner.setObservation(self.agent, self)  # set the final observation
        self.agent.learner.reset()

    def reset(self):
        pass



class CheeseMaze(Maze):
    """ non-episodic version of four-room maze """
    terminal = True
    reward_collision=0.0
    def __init__(self, agent, visual, switching, params=None):
        Maze.__init__(self, agent, visual, switching, params)
        if 'reward_collision' in params:
            self.reward_collision=params['reward_collision']

    def createMaze(self):
        # Fill borders
        self.sizeX = 7
        self.sizeY = 5

        self.map = [[EmptyObject()] * self.sizeY for _ in range(self.sizeX)]
        self.fill_borders()
        # now we have:
        ####################
        #
        #
        #
        #
        ####################
        self.map[2][2]=Obstacle(None)
        self.map[2][3]=Obstacle(None)

        self.map[4][2]=Obstacle(None)
        self.map[4][3]=Obstacle(None)

        self.map[self.foodCoord[0]][self.foodCoord[1]]=Food(None)
    def generateMap(self, files=None):
        self.startCoord = (1, 2)

        self.foodCoord = (3, 3)
        self.createMaze()


    @overrides
    def setObservation(self, agent):
        self.setObs(agent)

def reward_fun_POmaze(agent,environment):
    if eat(agent,environment):
        environment.terminal=True
        return environment.reward_food
    elif pseudo_eat(agent,environment):
        environment.terminal=True
        return environment.reward_pseudofood
    else:
        return environment.reward_default
def reward_fun_meltingpot(agent,environment):
    if eat(agent,environment):
        # no terminal state; food is not removed
        return environment.reward_food
    else:
        return environment.reward_default
def reward_fun_cheese_maze(agent,environment):
    if environment.no_move:
        return environment.reward_collision
    elif eat(agent,environment):
        if environment.remove_food:
            environment.terminal=True
        return environment.reward_food
    else:
        return environment.reward_default

standardmaze_defaultparams={'tasks':[],'sampling_rate': 10000, 'dynamic':False,
                'record_intervals':[],'sizeX':11,'sizeY':8,'reset_type':ResetType.random, 'real_time':False ,
                            'network_types':['external_actions'],
                      'use_stats': True,'observation_size':1,'observation_length':4}
POmaze_defaultparams={'sizeX':MAPSIZEX,'sizeY':MAPSIZEY,'complexity':.10, 'density':.10,
               'tasks': [],'sampling_rate': 10000, 'dynamic':False,
                'record_intervals':[],
               'reset_type':ResetType.random, 'real_time':False ,'network_types':['external_actions'],
                      'use_stats': True,'observation_size': 1,'observation_length':4}

cheesemaze_defaultparams={
               'tasks': [], 'sampling_rate': 10000, 'dynamic':False,'observation_size':1,'record_intervals':[],
                'sizeX':7,'sizeY':5,
               'reset_type':ResetType.random, 'real_time':False ,'network_types':['external_actions'],'use_stats': True,
                'observation_size': 1,'observation_length':4}