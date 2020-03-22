
from enum import Enum

class POcmanFlags(object):
    E_FREE = 3
    E_OBSTACLE = 0
    E_POWER = 7
    E_FOOD = 1
    E_POISON=9
    # E_OBSTACLE=3




class PacmanTaskType(Enum):
    EAT_FOOD = 0
    RUN_GHOSTS = 1
    CHASE_GHOSTS = 2
    FULL = 3
    MELTINGPOT = 4
    GENERALISATION_TEST=5




class PacmanTopology(object):
    cheese=0
    standard = 1
    pacman_micro = 2
    pacman_mini = 3
    pacman_standard = 4

    @classmethod
    def is_pacman(cls,topology):
        return topology > 1

class PacmanDynamic(object):
    static=0
    random = 1
    pacman = 2


    @classmethod
    def is_dynamic(cls,pacman_dynamic):
        return pacman_dynamic > 0