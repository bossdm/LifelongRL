
from mapobjects import Obstacle, Food, PseudoFood

import Actions.SpecialActions as sa
import numpy as np
MAZES = [17,18,20,22,27,29,30,31,32,33,
         3,6,10,11,14,16,19,21,23,26,
         1,2,4,5,7,8,9,12,13,15]

DIFFICULTY = {17:'Easy',18:'Easy',20:'Easy',22:'Easy',27:'Easy',29:'Easy',30:'Easy',31:'Easy',32:'Easy',33:'Easy',
         3: 'Medium', 6:'Medium', 10:'Medium', 11:'Medium',14:'Medium',16:'Medium',19:'Medium',21:'Medium',23:'Medium',26:'Medium',
         1: 'Difficult',2:'Difficult',4:'Difficult',5:'Difficult',7:'Difficult',8:'Difficult',9:'Difficult',12:'Difficult',13:'Difficult',15:'Difficult'
         }


SWITCHING_FREQ = 2500 # for switching mazes: switch once every SWITCHING_FREQ time steps
SWITCHING=False
GENERATE_NEW=False

NUM_MAZES=4
MAPSIZEX=13#use uneven numbers > 4 !
MAPSIZEY=13

# custom pairs of difficulty and maze_ids here
FOUR_EASY_MAZE_IDs=range(4)
EASY_MAZE_IDs = range(10)
MEDIUM_MAZE_IDs= range(10,20)
FOUR_DIFFICULT_MAZE_IDs=range(20,24)
DIFFICULT_MAZE_IDs = range(20,30)


FULLOBS=False

# N, S, W, E
directions = [(0,-1), (0,+1), (-1,0),(+1,0)]





def opposite(dir1,dir2):
    if abs(np.sign(dir1) - np.sign(dir2))==2:
        return 2
    if dir1==0 and dir2==0:
        return 1
    return 0
def opposite_direction(dir1,dir2):
    x,y=dir1
    xx,yy=dir2
    oppositex=opposite(x,xx)
    oppositey=opposite(y,yy)
    return oppositex+oppositey >=3
class ResetType(object):
    random = 1
    longterm = 2
    fixed = 3


def manhattan_dist(coord,coord2):
    x,y=coord
    xx,yy=coord2
    return (abs(x - xx) + abs(y - yy))
def manhattan_dist_highdim(coord,coord2,norm=None):
    """

    :param coord: numpy array
    :param coord2: numpy array
    :param norm: numpy array to normalise for size of the dimension
    :return:
    """
    if norm is None:
        norm=np.ones(coord.shape)
    difference=(coord-coord2)/norm
    return sum(abs(difference))/len(coord)
def euclidian_dist(coord,coord2):
    x,y=coord
    xx,yy=coord2
    return ((x-xx)**2+(y-yy)**2)
def directional_dist(pos1,pos2,direction):
    x,y = pos1
    xx,yy = pos2
    if direction==(0,1):
        return yy - y
    elif direction==(+1,0):
        return xx - x
    elif direction==(0,-1):
        return y - yy
    elif direction==(-1,0):
        return y - yy
    else:
        raise Exception()
def dfs_paths(graph, start, goal):
    stack = [(start, [start])]
    while stack:
        (vertex, path) = stack.pop()
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                stack.append((next, path + [next]))
def bfs_paths(graph, start, goal):
    queue = [(start, [start])]
    while queue:
        (vertex, path) = queue.pop(0)
        for next in graph[vertex] - set(path):
            if next == goal:
                yield path + [next]
            else:
                queue.append((next, path + [next]))

def shortest_path_bfs(graph, start, goal):
    try:
        return next(bfs_paths(graph, start, goal))
    except StopIteration:
        return None

def check_toroid_X(x,environment):
    if (x >= environment.sizeX):
        return 0
    elif (x < 0):
      return environment.sizeX - 1
    else:
        return x

def check_toroid_Y(y,environment):
    if (y >= environment.sizeY):
       return 0
    elif (y < 0):
       return environment.sizeY - 1
    else:
        return y
def check_obstacle(pos,environment):
    #TODO: add conditions in which obstacles are permeable
    x,y=pos
    return  environment.check_obstacle(x,y)
def stay(agent,environment):
    return
def north(agent,environment):
    environment.oldx = agent.x
    environment.oldy = agent.y
    #environment.currentStat().setDistribution(north,(agent.x,agent.y))
    agent.y-=1
    agent.y=check_toroid_Y(agent.y,environment)
    if(check_obstacle((agent.x,agent.y),environment)):
        agent.y=environment.oldy
        environment.no_move=True


def south(agent,environment):
    environment.oldx=agent.x
    environment.oldy=agent.y
    agent.y+=1
    agent.y=check_toroid_Y(agent.y,environment)
    if (check_obstacle((agent.x,agent.y), environment)):
        agent.y = environment.oldy
        environment.no_move = True

def west(agent,environment):
    #environment.currentStat().setDistribution(west, (agent.x, agent.y))
    environment.oldx = agent.x
    environment.oldy = agent.y
    agent.x-=1
    agent.x=check_toroid_X(agent.x, environment)
    if (check_obstacle((agent.x,agent.y), environment)):
        agent.x = environment.oldx
        environment.no_move = True

def east(agent,environment):
    #environment.currentStat().setDistribution(east, (agent.x, agent.y))
    environment.oldx = agent.x
    environment.oldy = agent.y
    agent.x+=1
    agent.x=check_toroid_X(agent.x, environment)
    if (check_obstacle((agent.x,agent.y), environment)):
        agent.x = environment.oldx
        environment.no_move = True

#################################################
## movements as in the non-episodic four-room maze (NEFRM)
def north_NEFRM(agent,environment):
    environment.oldx = agent.x
    environment.oldy = agent.y
    #environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.y-=1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess)/2:

            agent.x-=1
        else:
            agent.x+=1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if(check_obstacle((agent.x,agent.y),environment)):
        agent.x,agent.y=(environment.old,environment.oldy)
        environment.no_move=True
def northwest_NEFRM(agent,environment):
    environment.oldy = agent.y
    environment.oldx = agent.x
    #environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.y-=1
        agent.x-=1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess)/2:

            agent.x-=1
            agent.y-=1
        else:
            agent.x-=1
            agent.y+=1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if(check_obstacle((agent.x,agent.y),environment)):
        agent.x,agent.y=(environment.oldx,environment.oldy)
        environment.no_move=True

def northeast_NEFRM(agent,environment):
    environment.oldy = agent.y
    environment.oldx = agent.x
    #environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.y-=1
        agent.x+=1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess)/2:

            agent.x-=1
            agent.y-=1
        else:
            agent.x+=1
            agent.y+=1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if(check_obstacle((agent.x,agent.y),environment)):
        agent.x,agent.y=(environment.oldx,environment.oldy)
        environment.no_move=True

def south_NEFRM(agent,environment):
    environment.oldy = agent.y
    environment.oldx = agent.x
    # environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.y += 1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess) / 2:

            agent.x -= 1
        else:
            agent.x += 1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if (check_obstacle((agent.x, agent.y), environment)):
        agent.x, agent.y = (environment.oldx, environment.oldy)
        environment.no_move = True


def southwest_NEFRM(agent,environment):
    environment.oldy = agent.y
    environment.oldx = agent.x
    #environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:
        agent.x -= 1
        agent.y+=1

    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess)/2:

            agent.x-=1
            agent.y-=1
        else:
            agent.x+=1
            agent.y+=1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if(check_obstacle((agent.x,agent.y),environment)):
        agent.x,agent.y=(environment.oldx,environment.oldy)
        environment.no_move=True

def southeast_NEFRM(agent,environment):
    environment.oldy = agent.y
    environment.oldx = agent.x
    #environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.y+=1
        agent.x+=1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess)/2:

            agent.x-=1
            agent.y+=1
        else:
            agent.x+=1
            agent.y-=1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if(check_obstacle((agent.x,agent.y),environment)):
        agent.x,agent.y=(environment.oldx,environment.oldy)
        environment.no_move=True


def west_NEFRM(agent,environment):
    #environment.currentStat().setDistribution(west, (agent.x, agent.y))

    environment.oldy = agent.y
    environment.oldx = agent.x
    # environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.x -= 1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess) / 2:

            agent.y -= 1
        else:
            agent.y += 1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if (check_obstacle((agent.x, agent.y), environment)):
        agent.x, agent.y = (environment.oldx, environment.oldy)
        environment.no_move = True

def east_NEFRM(agent,environment):
    #environment.currentStat().setDistribution(east, (agent.x, agent.y))

    environment.oldy = agent.y
    environment.oldx = agent.x
    # environment.currentStat().setDistribution(north,(agent.x,agent.y))
    r = environment.rng.rand()
    if r <= environment.pr_actionsuccess:

        agent.x -= 1
    else:

        if r <= environment.pr_actionsuccess + (1 - environment.pr_actionsuccess) / 2:

            agent.y -= 1
        else:
            agent.y += 1
    agent.x = check_toroid_X(agent.x, environment)
    agent.y = check_toroid_Y(agent.y, environment)
    if (check_obstacle((agent.x, agent.y), environment)):
        agent.x, agent.y = (environment.oldx, environment.oldy)
        environment.no_move = True



def eat(agent,environment):
    if environment.map[agent.x][agent.y].edible:
        if environment.switching and GENERATE_NEW: environment.currentTask.solved=True
        return True
    return False

def pseudo_eat(agent,environment):
    if (isinstance(environment.map[agent.x][agent.y],PseudoFood)):
        if environment.switching and GENERATE_NEW: environment.currentTask.solved=True
        return True
    return False

def getDifficulty(run):
    maze_num = MAZES[run % 30]
    print("maze " + str(maze_num))
    return DIFFICULTY[maze_num]
def getMazeId(run):
    return run % 30
def getDifficultySwitch():
    return 'Easy'
def getMazeFileName(run):
    maze_num = MAZES[run % 30]  # 1 + (run-1)%15
    difficulty = DIFFICULTY[maze_num]
    fileprefix = difficulty + '/maze' + str(maze_num)
    return fileprefix
def initializeDefaultNavigationTask(filename,default_task,defaultparams,run,sampling_rate,stoptime,reward_fun):


    default_task.end_time =stoptime

    defaultparams['sampling_rate']=sampling_rate
    default_task.files=getMazeFileName(run)
    default_task.maze_id=getMazeId(run)
    default_task.reward_fun = reward_fun

    defaultparams['tasks']=[default_task]
    defaultparams['seed'] = run
    defaultparams["filename"] = filename


#technically the first is the same as "directions" but just keep code consistent for now
VonNeumannNeighbourhood=[(0,1),(1,0),(0,-1),(-1,0)]
MooreNeighbourhood=[(-1,-1),(1,-1),(1,1),(-1,-1),(0,1),(1,0),(0,-1),(-1,0)]

VonNeumannNeighbourhoodPlus=[(0,0),(0,1),(1,0),(0,-1),(-1,0)]
MooreNeighbourhoodPlus=[(0,0),(-1,-1),(1,-1),(1,1),(-1,-1),(0,1),(1,0),(0,-1),(-1,0)]
navigation_set1 = [sa.ExternalAction(north,0),sa.ExternalAction(south,0),sa.ExternalAction(west,0),sa.ExternalAction(east,0)]
navigation_set2 = [sa.ExternalAction(north_NEFRM,0),sa.ExternalAction(northwest_NEFRM,0),sa.ExternalAction(west_NEFRM,0),
                   sa.ExternalAction(southwest_NEFRM,0),sa.ExternalAction(south_NEFRM,0),
                   sa.ExternalAction(southeast_NEFRM,0),sa.ExternalAction(east_NEFRM,0),
                   sa.ExternalAction(northeast_NEFRM,0)]