import sys
# if sys.version_info[0] == 2:
#     import tkFont
# else:
#     from tkinter import font as tkFont

from mapobjects import EmptyObject
from abc import abstractmethod

class Agent(object):
    def __init__(self, learner):
        self.learner = learner
        self.energy_supply = 100

    def tick(self, environment):
        self.learner.cycle(environment.agent,environment)
        if self.learner.chosenAction.energy_cost is not None:
            self.energy_supply -= self.learner.chosenAction.energy_cost
    @abstractmethod
    def addAgentToCanvas(self, environment):
        pass

class NavigationAgent(Agent):
    def __init__(self, learner, params):
        Agent.__init__(self,learner)
        self.filename = "Agent.png" if "agent_filename" not in params else params["agent_filename"]
        self.x = 0
        self.y = 0 # wait for initialisation

    # def addAgentToCanvas(self,environment):
    #     # finally add agent and the object he is holding
    #     xx, yy = environment.vis.setCoords(self.x, self.y, (environment.vis.item_width, environment.vis.item_height), (0.7, 0.7))
    #     self.ag_img = environment.vis.getPicture(self.filename, environment.vis.canvas,
    #                                              (environment.vis.item_width, environment.vis.item_height), (0.4, 0.6))
    #     if environment.vis.checkPicture(self.ag_img, self):
    #         environment.vis.canvas.create_image(xx, yy, image=self.ag_img)
    #     xx, yy = environment.vis.setCoords(environment.agent.x, environment.agent.y,
    #                                        (environment.vis.item_width, environment.vis.item_height), (0.7, 0.4))
    #
    #     font = tkFont.Font(family="Arial")
    #     environment.vis.canvas.create_text((xx, yy), text=str(self.learner.chosenAction), font=font)
    def addAgentToCanvas(self,environment):
        # finally add agent and the object he is holding
        #works well for both coordinate systems
        offset=(.30 ,.40 )
        environment.vis.display.addPicture(self.x,self.y,offset=offset,shrink_factor=(0.4,0.6),filename=self.filename)
        #add a little above bottom right
        environment.vis.display.addText(self.x,self.y,offset=(offset[0],.10),txt=str(self.learner.chosenAction),font_t="Arial",font_size=12)

        environment.add_other_agents()
        # xx, yy = environment.vis.setCoords(self.x, self.y, (environment.vis.item_width, environment.vis.item_height), (0.7, 0.7))
        # self.ag_img = environment.vis.getPicture(self.filename, environment.vis.canvas,
        #                                          (environment.vis.item_width, environment.vis.item_height), (0.4, 0.6))
        # if environment.vis.checkPicture(self.ag_img, self):
        #     environment.vis.canvas.create_image(xx, yy, image=self.ag_img)
        # xx, yy = environment.vis.setCoords(environment.agent.x, environment.agent.y,
        #                                    (environment.vis.item_width, environment.vis.item_height), (0.7, 0.4))
        #
        # font = tkFont.Font(family="Arial")
        # environment.vis.canvas.create_text((xx, yy), text=str(self.learner.chosenAction), font=font)

    def tick(self, environment):
        environment.no_move = False
        #print("%d,%d"%(self.x,self.y))
        Agent.tick(self,environment)
        #print("%s : --> %d,%d" % (self.learner.chosenAction.function.__name__,self.x, self.y))
class ClassificationAgent(Agent):
    def __init__(self, learner):
        Agent.__init__(self,learner)
        self.chosenClass=None

class FF_Agent(NavigationAgent):
    def __init__(self,learner,x,y,params):
        NavigationAgent.__init__(self,learner,params)
        self.holdingObject=EmptyObject() # no object holding initially
        self.observation_size=params['observation_size']

    def addAgentToCanvas(self,environment):
        # finally add agent and the object he is holding
        # finally add agent and the object he is holding
        offset = (.30, .40)
        xx, yy = environment.vis.display.setCoords(self.x, self.y, (environment.vis.display.item_width, environment.vis.display.item_height),
                                           (0.7, 0.7))
        self.ag_img = environment.vis.display.getPicture(self.filename, environment.vis.display.canvas, (environment.vis.display.item_width,environment.vis.display.item_height),(0.4, 0.6))
        if environment.vis.display.checkPicture(self.ag_img,self):
            environment.vis.display.canvas.create_image(xx, yy,image=self.ag_img )
        xx, yy = environment.vis.display.setCoords(environment.agent.x, environment.agent.y,(environment.vis.display.item_width,environment.vis.display.item_height), (0.4, 0.4))
        if self.holdingObject is not None:
            self.obj_img=environment.vis.display.getPicture(self.holdingObject.filename, environment.vis.display.canvas, (environment.vis.display.item_width,environment.vis.display.item_height),(0.25, 0.25))
            if environment.vis.display.checkPicture(self.obj_img,self.holdingObject):
               environment.vis.display.canvas.create_image(xx, yy, image=self.obj_img)


        environment.vis.display.addText(self.x, self.y, offset=(offset[0], .10), txt=str(self.learner.chosenAction),
                                        font_t="Arial",font_size=14)
        # offset = (.30, .40)
        # self.ag_img=environment.vis.display.addPicture(self.x, self.y, offset=offset, shrink_factor=(0.4, 0.6),
        #                                    filename=self.filename)
        # # add a little above bottom right
        # self.text=environment.vis.display.addText(self.x, self.y, offset=(offset[0], .10), txt=str(self.learner.chosenAction),
        #                                 font_t="Arial")
        # offset = (.25, .25)
        # self.obj_img=environment.vis.display.addPicture(self.x, self.y, offset=offset, shrink_factor=(0.4, 0.4),
        #                                    filename=self.holdingObject.filename)

class POmaze_Agent(NavigationAgent):
    def __init__(self,learner,params):
        NavigationAgent.__init__(self,learner,params)


class PacmanAgent(NavigationAgent):
    """

    """

    def __init__(self, learner, params):
        NavigationAgent.__init__(self, learner, params)
        self.filename="pacman_pic.jpg"

    def addAgentToCanvas(self, environment):
        offset = (.50, .50)
        environment.vis.display.addPicture(self.x, self.y, shrink_factor=(1.0,1.0),
                                           filename=self.filename)
        environment.vis.display.addText(self.x, self.y-0.80, offset=(offset[0],- .20), txt=str(self.learner.chosenAction),
                                        font_t="Arial",font_size=15)
        environment.add_other_agents()
        # # add a little above bottom right
        # environment.vis.display.addText(self.x, self.y, offset=(.50, .10), txt=str(self.learner.chosenAction),
        #                                     font_t="Arial",font_size=5)


    # def new_container_item(self, environment):
    #     self.num_container_items += 1
    #     self.container += environment.get_value(self.x, self.y)
    #     if DEBUG_MODE:
    #         print("added to container")
    #         print("num_container_items is now %d" % (self.num_container_items))
    #         print("container value %.4f" % (self.container))
    #
    # def reset_container(self):
    #     self.num_container_items = 0
    #     self.container = 0


