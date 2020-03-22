

from numpy import inf

class MapObject(object):
    edible=False
    def __init__(self,function,object_id):
        self.function=function
        self.object_id=object_id
        self.grabbable=False
        self.droppable=False
        self.ready_for_food=False
        self.walkable=True
        self.limited=True
        self.filename = None
    def tick(self,environment):
        if self.function is not None:
            self.function(self,environment)
    def get_filename(self):
        return self.filename

class SimpleAgent(MapObject):
    def __init__(self,function):
        MapObject.__init__(self,function,0)
        self.filename = "SimpleAgent.png"

class Obstacle(MapObject):
    walkable=True
    def __init__(self,function):
        MapObject.__init__(self,function,1)
        self.filename = "Obstacle.png"
        self.walkable=False
class Seed(MapObject): # seeds must be planted in fertile area
    def __init__(self,function,growth_time):
        MapObject.__init__(self,function,2)
        self.added_time=growth_time
        self.grabbable=True
        #self.planted=False #wait for dropping it in fertile area
        self.filename = "Seed.png"
    def plant(self,current_time):
        """
        after planting, the time after which it can turn into a food item is determined
        :param current_time:
        :return:
        """
        self.time = current_time + self.added_time


class FertileArea(MapObject):
    def __init__(self,function):
        MapObject.__init__(self,function,3)
        self.filename = "FertileArea.png"
class Food(MapObject):  # if agent can remember where seeds were planted, more efficient!
    edible=True
    def __init__(self,function):
        MapObject.__init__(self,function,4)
        self.grabbable = True
        self.filename = "Goal.jpeg" #"Food.png"
        self.fertilisation_ticks=0
        self.quantity=50




class Interestingness(MapObject): # agent can take record of interesting places
    def __init__(self,function):
        MapObject.__init__(self,function,5)
        self.filename = "Interestingness.png"
class Knife(MapObject): #can be used to attack simple agent
    def __init__(self,function):
        MapObject.__init__(self,function,6)
        self.grabbable = True
        self.filename = "Knife.png"
class EmptyObject(MapObject): # nothing
    def __init__(self):
        MapObject.__init__(self,None,7)
        self.droppable=True

class Door(MapObject): # nothing
    def __init__(self):
        MapObject.__init__(self,None,8)
        self.filename = "Door.png"

class Arrow(MapObject): # nothing
    def __init__(self,function,rotation):
        MapObject.__init__(self,None,9)
        self.filename = "Arrow.png"

class PseudoFood(MapObject):
    def __init__(self,function):
        MapObject.__init__(self,function,10)
        self.grabbable = True
        self.filename = "PseudoFood.jpeg"

class Text(MapObject):
    def __init__(self,text):
        MapObject.__init__(self,None,11)
        self.text=text

class Mine(MapObject):
    def __init__(self):
        MapObject.__init__(self, None, 12)
        self.filename = "Mine.png"
class Unexplored(MapObject):
    def __init__(self):
        MapObject.__init__(self, None, 13)
        self.filename = "Grey.png"


class Home(MapObject):
    """
    A home object contains
    -seeds
    -a storage place for all the food gathered

    """
    def __init__(self, num_seeds=inf):
        self.food_stored=0
        self.num_seeds=num_seeds
        MapObject.__init__(self,None,14)
        self.filename = "home.jpeg"

    def take_seed(self):
        self.num_seeds-=1
    def add_food(self,quantity=1):
        self.food_stored+=quantity



class Fertiliser(MapObject):


    def __init__(self,function):
        MapObject.__init__(self,function,15)
        self.filename="fertiliser.jpeg"
        self.limited=False # unlimited supply --> picking up does not result in removal

class NormalGhost(MapObject):

    def __init__(self,function):
        MapObject.__init__(self,function,16)
        self.filename="pacman_normalghost.png"
class ChasedGhost(MapObject):
    def __init__(self,function):
        MapObject.__init__(self, function, 16)
        self.filename = "pacman_ghost.jpg"


class Power(MapObject):
    def __init__(self,function):
        MapObject.__init__(self, function, 17)
        self.filename = "pacman_powerpill.jpeg"
class PacmanFood(MapObject):
    def __init__(self,function):
        MapObject.__init__(self, function, 18)
        self.filename = "pacman_food.jpeg"

class PacmanPoison(MapObject):
    def __init__(self,function):
        MapObject.__init__(self, function, 19)
        self.filename = "pacman_poison.jpeg"