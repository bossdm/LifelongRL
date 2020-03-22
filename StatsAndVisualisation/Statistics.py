

from PIL import ImageFont

import dill



from Environment import *
from POcmanEnums import POcmanFlags,PacmanDynamic
from MazeUtils import *


from random import sample

import os

class PolType(Enum):
    GLOBAL=0,
    P_MATRIX=1,
    NETWORK=2

class IS_Statistics(object):
    # statistics not specific to maze coordinates, but specific to IS
   def __init__(self,learner,actionslist,slices):
        self.p_output_distribution = []
        self.net_output_distribution =  []
        self.ignore_next=False
        for t in range(slices):
            self.p_output_distribution.append({})
            self.net_output_distribution.append({})
            for ip in range(learner.ProgramStart, learner.Max + 1):
                self.p_output_distribution[t][ip] = {}
                if learner.network_usage:
                    self.net_output_distribution[t][ip] = {}
                for action in actionslist:
                    self.p_output_distribution[t][ip][action] = 0
                    if learner.network_usage:
                        self.net_output_distribution[t][ip][action] = 0

   def set_output_distribution(self,t,learner):
       if learner.usedNPinstruction is not None:
           if isinstance(learner.next_chosenAction, ExternalAction): # TODO: allow other networks with other functions?
               self.net_output_distribution[t][learner.IP][learner.next_chosenAction.function.__name__]+=1
       else:
           if isinstance(learner.chosenAction,ExternalAction):
                if self.ignore_next:
                    self.ignore_next=False
                    return
                self.p_output_distribution[t][learner.IP][learner.chosenAction.function.__name__]+=1

   def print_dictionaries(self,file,t):
       if self.net_output_distribution:
           netoutputdict = "{" + "\n".join("{}: {}".format(k, v) for k, v in self.net_output_distribution[t].items()) + "}"
           file.write(
               "\n\n --------------- \n \n net-output-distribution over (x,y,t) and actions : \n" + str(netoutputdict))

       outputdict = "{" + "\n".join("{}: {}".format(k, v) for k, v in self.p_output_distribution[t].items()) + "}"
       file.write(
           "\n\n --------------- \n \n direct Pol-output-distribution over (x,y,t) and actions: \n" + str(outputdict))
class MazeStatistics(object):

    def __init__(self,start_coords,goal_coords,feasible_coords,external_actions, learner,Xsize,Ysize,mazefilename,filename="stats.txt",slices=1):
        self.distribution = []
        self.best = []
        self.worst = []
        self.variation = []
        self.mean = []
        self.sd = []
        self.goals_achieved = []

        if learner.network_usage:
            self.max_nets = learner.maxNetworks
            self.net_output_distribution = []
            self.network_usage_distribution = []
        if hasattr(learner, "IP"):
            self.IPdistribution = []
        self.Xsize=Xsize
        self.Ysize=Ysize
        self.filename=filename
        self.file = open(self.filename, "w")
        self.externalactions={}
        self.actionslist=external_actions
        self.slices=slices
        self.mazefilename=mazefilename
        self.IS_stats=None
        self.ignore_next=False

        if isinstance(learner,SSA):
            self.IS_stats=IS_Statistics(learner,external_actions,slices)


        # keep track of the path
        self.path = []







        for i in range(len(self.actionslist)):
            self.externalactions[self.actionslist[i]]=i
        for t in range(self.slices):
            self.distribution.append({})
            self.best.append({})
            self.worst.append({})
            self.mean.append({})
            self.variation.append({})
            self.sd.append({})
            self.goals_achieved.append({})

            if learner.network_usage:
                self.net_output_distribution.append([])
                self.network_usage_distribution.append([])
                for netind in range(self.max_nets):
                    self.net_output_distribution[t].append({})
                    self.network_usage_distribution[t].append({})
            if hasattr(learner, "IP"):
                self.IPdistribution.append({})

            add_g=True if len(goal_coords)>1 else False
            for g in goal_coords:
                for coord in feasible_coords:
                        if add_g:
                            coord=(coord,g)
                        self.distribution[t][coord]=[0 for _ in range(len(self.actionslist))]

                        if hasattr(learner,"IP"):
                            self.IPdistribution[t][coord]={ip:0 for ip in range(learner.ProgramStart,learner.Max+1)}
                        if learner.network_usage:
                            for netind in range(self.max_nets):
                                self.network_usage_distribution[t][netind][coord] = 0
                                self.net_output_distribution[t][netind][coord] = [0 for _ in range(len(self.actionslist))]
            for coord in start_coords:
                for g_coord in goal_coords:
                    self.best[t][(coord,g_coord)]=(100000,None)
                    self.worst[t][(coord,g_coord)]=(1,None)
                    self.mean[t][(coord,g_coord)]=0 # initialisation for these should not matter
                    self.variation[t][(coord,g_coord)]=0
                    self.sd[t][(coord,g_coord)] = 0
                    self.goals_achieved[t][(coord,g_coord)]=0 # this one always 0

    def add_slice(self,t,learner):
        if t < len(self.distribution):
            return
        self.distribution.append({})
        self.best.append({})
        self.worst.append({})
        self.mean.append({})
        self.variation.append({})
        self.sd.append({})
        self.goals_achieved.append({})

        if learner.network_usage:
            self.net_output_distribution.append([])
            self.network_usage_distribution.append([])
            for netind in range(self.max_nets):
                self.net_output_distribution[t].append({})
                self.network_usage_distribution[t].append({})
        if hasattr(learner, "IP"):
            self.IPdistribution.append({})

        for coord in self.distribution[t-1]:
            self.distribution[t][coord] = [0 for _ in range(len(self.actionslist))]

            if hasattr(learner, "IP"):
                self.IPdistribution[t][coord] = {ip: 0 for ip in range(learner.ProgramStart, learner.Max + 1)}
            if learner.network_usage:
                for netind in range(self.max_nets):
                    self.network_usage_distribution[t][netind][coord] = 0
                    self.net_output_distribution[t][netind][coord] = [0 for _ in range(len(self.actionslist))]
        for key in self.best[t-1]:
                self.best[t][key] = (100000, None)
                self.worst[t][key] = (1, None)
                self.mean[t][key] = 0  # initialisation for these should not matter
                self.variation[t][key] = 0
                self.sd[t][key] = 0
                self.goals_achieved[t][key] = 0  # this one always 0

    def setNetOutputDistribution(self,t,coord,learner):
        if learner.network_usage and learner.usedNPinstruction is not None:
            if learner.chosenAction is not None:

                if isinstance(learner.chosenAction, ExternalAction):
                    if DEBUG_MODE:
                        print(learner.t)

                    # print("counted net_output_distr")
                    action = learner.chosenAction.function.__name__
                    index = self.externalactions[action]
                    self.net_output_distribution[t][learner.net_key][coord][index] += 1
                if isinstance(learner.next_chosenAction, ExternalAction):
                    if DEBUG_MODE:
                        print(learner.t)

                    # print("counted net_output_distr")
                    action = learner.next_chosenAction.function.__name__
                    index = self.externalactions[action]
                    self.net_output_distribution[t][learner.net_key][coord][index] += 1


    # add the currently chosen action to the count of chosen actions on the given (x,y,t)
    def setDistribution(self,t,coord,learner):
        # print("coord=" +str(coord))
        # print("net_key=%d" %(learner.net_key))
        #print("used NPinstr="+str(learner.usedNPinstruction))
        # print("nextChosen="+str(None if learner.next_chosenAction is None else learner.next_chosenAction.function.__name__))
        #print(action)



        if isinstance(learner.chosenAction, ExternalAction):


            action = learner.chosenAction.function.__name__
            index = self.externalactions[action]
            self.distribution[t][coord][index] += 1

    def setIPdistribution(self,t,coord,IP):
        self.IPdistribution[t][coord][IP]+=1

    def countSteps(self,agent):
        if isinstance(agent.learner.chosenAction,ExternalAction):
            self.path.append((agent.x,agent.y))
    def resetSteps(self):
        self.path=[]

    def setGoalAchieved(self,t,start,goal):
        steps=len(self.path)
        length = min(500, steps)  # display the first 500 steps
        if steps < self.best[t][(start,goal)][0]:
            self.best[t][(start,goal)]=(steps,copy(self.path[0:length]))
        if steps > self.worst[t][(start,goal)][0]:
            self.worst[t][(start,goal)]=(steps,copy(self.path[0:length]))
        self.goals_achieved[t][(start,goal)]+=1
        g=self.goals_achieved[t][(start,goal)]
        #calc average
        prop_prev=(g-1)/float(g) # how much proportion is the previous observations of this start_goal
        old_m=self.mean[t][(start, goal)]
        new_m=prop_prev*self.mean[t][(start,goal)] + (1-prop_prev)*steps
        #calc sd
        if g-1 ==0:
            self.variation[t][(start, goal)]=0
        else:
            self.variation[t][(start,goal)]+=((float(g-1)*steps - float(g-1)*old_m)**2)/(float(g-1)*float(g))
            self.sd[t][(start,goal)]=np.sqrt(self.variation[t][(start,goal)]/float(g))
        self.mean[t][(start, goal)]=new_m

        # then reset the steps
        self.resetSteps()

    def setnetworkusagedistribution(self,t,coord,learner):
        # print("coord=" +str(coord))
        # print("net_key=%d" %(learner.net_key))
        #print("used NPinstr="+str(learner.usedNPinstruction))
        if learner.usedNPinstruction is not None:
            # print("counted in networkusage")
            # when net_key is None, get_output will return and do nothing
            if hasattr(learner,"NP"):
                id = learner.NP.representation[learner.net_key].id
            else:
                id=0
            self.network_usage_distribution[t][id][coord]+=1

    def set_output_distribution(self,t,learner):
        self.IS_stats.set_output_distribution(t,learner)
    def printDistributions(self):
        self.file=open(self.filename, "w")
        # print the actions
        for a in self.actionslist:
            self.file.write(a)
            self.file.write("\t")
        self.file.write("\n ")
        # print the column labels
        for x in range(self.Xsize):
            self.file.write("%d \t" % (x))
        self.file.write("\n ")
        for y in range(self.Ysize):
            #print the row label
            self.file.write("%d \t"%(y))
            for x in range(self.Xsize):
                distr=self.distribution[(x,y)]
                s = float(sum(distr))
                if s == 0:
                    self.file.write("NA")
                #print the distribution
                else:
                    distr=[el/s for el in distr]
                    self.file.write("(")
                    for i in distr:
                        self.file.write("%.2f "%(i))
                    self.file.write(")")
                self.file.write("\t")
            self.file.write("\n")
            self.file.flush()
    def heatMap(self,filename="heatmap.png"):
        import matplotlib.pyplot as plt
        import numpy as np

        # North as Red dimension, East as red dimension
        # Make a 9x9 grid...
        image = np.zeros((self.Xsize, self.Ysize, 3))

        # Set every other cell to a random number (this would be your data)
        max=10
        min=0
        i=0
        for x in range(self.Xsize):
            for y in range(self.Ysize):
                distr=self.distribution[(x,y)]
                r=sum(distr)
                g=0.5
                b=0.5 #west
                image[x][y] = np.array([r,g,b])
                i+=1

        # Reshape things into a 9x9 grid.

        row_labels = range(self.Xsize)
        col_labels = range(self.Ysize)
        plt.gca().invert_yaxis()
        plt.imshow(image)
        plt.xticks(range(self.Xsize), col_labels)
        plt.yticks(range(self.Ysize), row_labels)
        plt.show()


        # Make the figure:

        fig, axs = plt.subplots(1, 1, figsize=(9, 9))
        fig.subplots_adjust(left=0.02, bottom=0.06, right=0.95, top=0.94, wspace=0.05)
        plt.savefig(filename)

    def ambiguities(self,envir,t):
        #return all the ambiguities for the map
        observations={}
        g=envir.reset_type==ResetType.longterm
        for coord in self.distribution[t]:
            if sum(self.distribution[t][coord])>0: # if possibly can contain an agent
                if g:
                    x=coord[0][0]
                    y=coord[0][1]
                else:
                    x=coord[0]
                    y=coord[1]
                envir.agent.x=x
                envir.agent.y=y
                envir.setObservation(envir.agent)
                observations[coord]=envir.agent.learner.observation
            else:
                continue
                #print("%d,%d,%d"%(x,y,t))
        ambiguities={}
        for coord in observations:
            #count the occurences of same observations in this timeslice
            for c,obs in observations.items():
                if obs == observations[coord]:
                    if coord in ambiguities:
                        ambiguities[coord]+=1
                    else:
                        ambiguities[coord]=1
        return ambiguities
    @classmethod
    def getVisitationRate(cls,statfreq,size,sumact):
        return sumact / (float(statfreq)/size)

    @classmethod
    def getRGB(cls,visitation_rate):

        red = min(255, int(visitation_rate * 255))
        # print(sumact)
        # print(red)
        rgb = (red, 122, 122)
        return rgb
    def set_goal_text(self,visual,x,y,dir):
        x_not, y_not = visual.display.setCoords(x, y, (
            visual.display.item_width, visual.display.item_height), (0.40, 0.50))
        visual.display.canvas.text((x_not, y_not), text="G", font= ImageFont.truetype(dir+"Times New Roman.ttf",size=24),
                                   fill="black")
    def scale_epsilon(self,epsilon):
        return min(1., epsilon / float(0.16))
    def make_epsilon_map(self,envir,maze_dir,recordings_file,time_interval,num_actions=4,relative=True,):
        t = min(1,envir.timePolicy.getStatCounter(envir))
        #create the map
        actions=envir.agent.learner.actions[0:num_actions]
        action_strings=[action.function.__name__ for action in envir.agent.learner.actions[0:num_actions]]
        direction_map={"west":(-1.0,0.0),"south":(0,1.0),"east":(1.0,0.0),"north":(0.0,-1.0)}
        dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
        font_correctness = ImageFont.truetype(dir+"Times New Roman.ttf",size=16)
        a,b=time_interval
        mazefile = maze_dir+self.mazefilename

        envir.generateMap(mazefile)

        if envir.reset_type == ResetType.longterm:
            goals=envir.goal_coords
            optimalresponses= dill.load(open(mazefile + "_optimalResponsesMultigoal"))
        else:
            goals=[envir.foodCoord]
            optimalresponses = dill.load(open(mazefile + "_optimalResponses", "rb"))

        coords = dill.load(open(mazefile+"_feasible","rb"))
        try:
            recordings=dill.load(open(recordings_file+ '(%d,%d)_recordings'%(a,b),"r"))
        except Exception:

            print("can't open "+recordings_file+ '(%d,%d)_recordings'%(a,b))
            return
        epsilon={}
        if "epsilon" in recordings:
            epsilon=recordings['epsilon']
        match={}
        if "match" in recordings:

            n_coords=len(coords)

            match=recordings['match']
            total_matches=sum(match.values())
        if "usage" in recordings:

            n_coords=len(coords)

            initialisations=recordings['usage']
            timeouts=recordings['time']
            total_initialisations=sum(initialisations.values())
            total_timeouts=sum(timeouts.values())
            if total_timeouts==0:
                print("no timeouts--must be faulty recording")
                return
        else:
            initialisations={}
            timeouts={}
        dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
        font_start = ImageFont.truetype(dir + "Times New Roman.ttf", size=26)

        font_correctness = ImageFont.truetype(dir + "Times New Roman.ttf", size=18)
        boldfont = ImageFont.truetype(dir + "timesbd.ttf", size=18)
        start_pos = (0.90, 0.20)
        Q=recordings['Qs']
        arrow_pos = (0.30, 0.30)
        text_increment=.25
        g_index=0
        for goal in goals:
            g_index+=1
            visual = self.init_visual(envir)
            match_visual = self.init_visual(envir)
            init_visual = self.init_visual(envir)
            time_visual = self.init_visual(envir)


            for coord in coords:
                    x,y =coord
                    if coord in epsilon:
                        e_scaled=min(1.,np.mean(epsilon[coord])/float(0.16))
                    else:
                        e_scaled=0.10/0.16
                    rgb=MazeStatistics.getRGB(e_scaled)
                    col = '#%02x%02x%02x' % rgb
                    x_arrow, y_arrow = visual.display.setCoords(x, y, (
                        visual.display.item_width, visual.display.item_height), arrow_pos)
                    if coord in Q or coord in epsilon:
                        visual.display.canvas.rectangle(
                            [x_arrow - .30 * visual.display.item_width, y_arrow - .30 * visual.display.item_height,
                             x_arrow + .70 * visual.display.item_width, y_arrow + .70 * visual.display.item_height],
                            fill=col, outline="black")
                    if coord in match:
                        match_scaled=match[coord]/float(total_matches)*n_coords

                        rgb=MazeStatistics.getRGB(match_scaled)
                        col = '#%02x%02x%02x' % rgb
                        x_arrow, y_arrow = visual.display.setCoords(x, y, (
                            match_visual.display.item_width, visual.display.item_height), arrow_pos)
                        match_visual.display.canvas.rectangle(
                            [x_arrow - .30 * visual.display.item_width, y_arrow - .30 * visual.display.item_height,
                             x_arrow + .70 * visual.display.item_width, y_arrow + .70 * visual.display.item_height],
                            fill=col, outline="black")
                    if coord in initialisations:
                        if relative:
                            init_scaled=initialisations[coord]/float(sum(self.distribution[t][coord]))
                        else:
                            init_scaled = initialisations[coord] /float(total_initialisations) * n_coords


                        rgb=MazeStatistics.getRGB(init_scaled)
                        col = '#%02x%02x%02x' % rgb
                        x_arrow, y_arrow = visual.display.setCoords(x, y, (
                            init_visual.display.item_width, visual.display.item_height), arrow_pos)
                        init_visual.display.canvas.rectangle(
                            [x_arrow - .30 * visual.display.item_width, y_arrow - .30 * visual.display.item_height,
                             x_arrow + .70 * visual.display.item_width, y_arrow + .70 * visual.display.item_height],
                            fill=col, outline="black")
                    if coord in timeouts:
                        time_scaled=timeouts[coord]/float(total_timeouts)*n_coords

                        rgb=MazeStatistics.getRGB(time_scaled)
                        col = '#%02x%02x%02x' % rgb
                        x_arrow, y_arrow = visual.display.setCoords(x, y, (
                            time_visual.display.item_width, visual.display.item_height), arrow_pos)
                        time_visual.display.canvas.rectangle(
                            [x_arrow - .30 * visual.display.item_width, y_arrow - .30 * visual.display.item_height,
                             x_arrow + .70 * visual.display.item_width, y_arrow + .70 * visual.display.item_height],
                            fill=col, outline="black")
                    if coord in Q:
                        Qvalues=Q[coord]
                        N=len(Qvalues)
                        QQ=np.array([Q[coord][i][0] for i in range(N)])
                        maxind=np.argmax(np.mean(QQ,axis=0))
                        for action in range(num_actions):
                            mQ = np.mean(QQ[:,action])
                            stdQ = np.std(QQ[:,action])
                            name=envir.agent.learner.actions[action].function.__name__
                            dx,dy=direction_map[name]
                            x_not=x+dx*text_increment
                            y_not=y+dy*text_increment
                            x_not, y_not = visual.display.setCoords(x_not, y_not, (
                                visual.display.item_width, visual.display.item_height), (0.40,0.50))
                            visual.display.canvas.text((x_not,y_not), text="%.4f"%(mQ),
                                                       font=boldfont if action == maxind else font_correctness,
                                                   fill="black")

                    if (envir.reset_type == ResetType.random and coord in envir.start_coords) or (
                            envir.reset_type == ResetType.fixed and coord == envir.startCoord) \
                            or (envir.reset_type == ResetType.longterm and coord[0] == envir.startCoord):
                        self.create_starting_pic(visual, x, y, start_pos, font_start)
                        self.create_starting_pic(match_visual, x, y, start_pos, font_start)
                        self.create_starting_pic(init_visual, x, y, start_pos, font_start)
                        self.create_starting_pic(time_visual, x, y, start_pos, font_start)
                    # if (x,y )==goal:
                    #     self.set_goal_text(visual,x,y,dir)
                    #     self.set_goal_text(match_visual, x, y,dir)
                    #     self.set_goal_text(init_visual, x, y,dir)
                    #     self.set_goal_text(time_visual, x, y,dir)



            visual.display.save(recordings_file+'epsilonmap_(%d,%d)'%(a,b), extension=".png")
            match_visual.display.save(recordings_file + 'matchmap_(%d,%d)' % (a, b), extension=".png")
            time_visual.display.save(recordings_file+'timemap_(%d,%d)'%(a,b), extension=".png")
            init_visual.display.save(recordings_file + 'relativeinitmap_(%d,%d)' % (a, b), extension=".png")
    def init_visual(self,envir):
        visual = Visual(envir.sizeX, envir.sizeY, (envir.observation_size, envir.observation_size), frameX=1600,
                              frameY=1600, buttons=False,
                              recording_intervals=None, videofile=None, tk_canvas=False)
        visual.display.createCanvas()
        visual.display.setEnvir(envir)
        return visual
    def create_starting_pic(self,visual,x,y,start_pos, font_start, annotations=False):
        x_start, y_start = visual.display.setCoords(x, y, (
            visual.display.item_width, visual.display.item_height), start_pos)
        visual.display.canvas.rectangle(
            [x_start - .11 * visual.display.item_width, y_start - .13 * visual.display.item_height,
             x_start + .11 * visual.display.item_width, y_start + .13 * visual.display.item_height],
            fill="white",
            outline="white")
        start_text = '' if not annotations else 'S'
        visual.display.canvas.text(
            (x_start - .06 * visual.display.item_width, y_start - .13 * visual.display.item_height),
            text=start_text, font=font_start, fill="black")
    def usage_transform(self,usage):
        usage = max(0, -0.3 + usage * 1.3)
        return usage

    def get_Usage_Correctness_And_Netcorrectness_Map(self, envir, arrow_map_file, maze_dir, annotations=False):
        from StatsAndVisualisation.visual import Visual
        t = min(4,envir.timePolicy.getStatCounter(envir))

        # create the map
        mazefile = maze_dir + self.mazefilename
        envir.generateMap(mazefile)
        ambiguities = self.ambiguities(envir, t)
        f = open(self.file.name, "wb")
        f.write("Maze " + str(mazefile) + "\n")
        slice_end = (t + 1) * envir.statfreq
        f.write("t <= " + str(slice_end) + "\n")
        size = 0
        total_steps = 0
        for coord in self.distribution[t]:
            if sum(self.distribution[t][coord]) > 0:
                size += 1

            total_steps += sum(self.distribution[t][coord])
        #     total_net_steps+=self.network_usage_distribution[t][0][coord]
        # assert total_net_steps <= total_steps
        # overlay the arrows on the current pictures

        # pic = visual.getPicture("Arrow.png", visual.canvas, (visual.display.item_width, visual.display.item_height))
        # ["north","west","east","south"]
        # ["north", "west", "east", "south"]
        # north --> 0: 90
        # west --> 1 : 180
        # east --> 2 : 0
        # south ---> 3: 270
        # if t==0:
        #     angledict={"north":90,"west":90,"east":90,"south":90}
        # elif t==1:
        #     angledict = {"north": 180, "west": 180, "east": 180, "south": 180}
        # elif t==2:
        #     angledict = {"north": 270, "west": 270, "east": 270, "south": 270}
        # elif t==3:
        #     angledict = {"north": 0, "west": 0, "east": 0, "south": 0}
        # else:
        angledict = {"north": 90, "west": 180, "east": 0, "south": 270}
        arrowimages = []

        ambiguities_list = []

        heatlist = []
        correctnesslist = []

        envir.agent.learner.stats.correctness.append(0)
        if envir.agent.learner.network_usage:
            envir.agent.learner.stats.correctnessNP.append(0)
            network_list = [[] for _ in range(self.max_nets)]
            ambiguities_listNP = [[] for _f in range(self.max_nets)]
            correctnesslistNP = [[] for _ in range(self.max_nets)]
            heatlistNP = [[] for _ in range(self.max_nets)]
            totalNPusage = [0 for i in range(self.max_nets)]

        experience_consistency_count = {}

        # font_arrowfq=("Arial",6)
        # font_start=("Arial",18)
        # font_ambiguity=("Arial",11)
        # font_networkproportion=("Arial",6)
        # font_correctness=("Arial",6)
        # font_correctnessNP=("Arial", 6)
        dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
        font_arrowfq = ImageFont.truetype(dir + "Times New Roman.ttf", size=18)
        font_start = ImageFont.truetype(dir + "Times New Roman.ttf", size=26)
        font_goal = ImageFont.truetype(dir + "Times New Roman.ttf", size=52)
        font_ambiguity = ImageFont.truetype(dir + "Times New Roman.ttf", size=18 if annotations else 26)
        font_networkproportion = ImageFont.truetype(dir + "Times New Roman.ttf", size=18)
        font_correctness = ImageFont.truetype(dir + "Times New Roman.ttf", size=18)
        font_correctnessNP = ImageFont.truetype(dir + "Times New Roman.ttf", size=18)

        arrow_pos = (0.30, 0.30)
        arrowfq_pos = (0.02, 0.20)
        start_pos = (0.90, 0.20)
        ambiguity_pos = (0.650 if annotations else .45, 0.80 if annotations else .70)
        network_pos = (0.02, 0.02)
        correctness_pos = (0.02, 0.80)

        network_yinc = .10  # use when having two networks
        correctnessNP_pos = (0.02, 0.65)
        if envir.reset_type == ResetType.longterm:
            goals = envir.goal_coords
            optimalresponses = dill.load(open(mazefile + "_optimalResponsesMultigoal"))
        else:
            goals = [envir.foodCoord]
            optimalresponses = dill.load(open(mazefile + "_optimalResponses", "rb"))
        g_index = 0
        for goal in goals:
            g_index += 1



            correctness_visual=self.init_visual(envir)
            if envir.agent.learner.network_usage:
                usage_visual = self.init_visual(envir)
                netcorrectness_visual=self.init_visual(envir)
            heat_visual=self.init_visual(envir)
            for x in range(envir.sizeX):
                for y in range(envir.sizeY):
                    coord = ((x, y), goal) if envir.reset_type == ResetType.longterm else (x, y)
                    if coord in self.distribution[t] and sum(self.distribution[t][coord]) > 0:
                        # print("%d,%d,%d" % (x, y, t))
                        # if (x,y) == envir.foodCoord:
                        #     x_g, y_g = visual.display.setCoords(x, y, (
                        #         visual.display.item_width, visual.display.item_height), start_pos)
                        #     visual.display.canvas.text(
                        #         (x_g * visual.display.item_width, y_g * visual.display.item_height),
                        #         text='G', font=font_goal, fill="black")
                        # get max prob action
                        coord = ((x, y), goal) if envir.reset_type == ResetType.longterm else (x, y)



                        # print("netwdistr="+str(self.network_usage_distribution[(x,y,t)]))
                        # m=np.argmax(self.network_usage_distribution[(x, y, t)])
                        # visual.display.canvas.text((x_x, y_y), text=m, font=font_n)
                        if envir.agent.learner.network_usage:
                            print(self.max_nets)

                            for netind in range(self.max_nets):
                                net_usage=sum(self.net_output_distribution[t][netind][coord])
                                if coord in self.network_usage_distribution[t][netind] and \
                                        self.network_usage_distribution[t][netind][coord] > 0:
                                    choices=self.network_usage_distribution[t][netind][coord]
                                    sumact=sum(self.distribution[t][coord])
                                    #visitation = MazeStatistics.getVisitationRate(envir.statfreq, size, sumact)
                                    usage=choices / float(sumact)
                                    usage=self.usage_transform(usage)
                                    rgb = MazeStatistics.getRGB(usage)
                                    print(rgb)
                                    col = '#%02x%02x%02x' % rgb
                                    x_arrow, y_arrow = usage_visual.display.setCoords(x, y, (
                                        usage_visual.display.item_width, usage_visual.display.item_height), arrow_pos)
                                    usage_visual.display.canvas.rectangle(
                                        [x_arrow - .30 * usage_visual.display.item_width, y_arrow - .30 * usage_visual.display.item_height,
                                         x_arrow + .70 * usage_visual.display.item_width, y_arrow + .70 * usage_visual.display.item_height],
                                        fill=col, outline="black")


                                if coord in self.net_output_distribution[t][netind] and sum(
                                        self.net_output_distribution[t][netind][coord]) > 0:
                                    print("net output distr")
                                    print(self.net_output_distribution[t][netind][coord])
                                    correctchoices = 0
                                    totalchoices = sum(self.net_output_distribution[t][netind][coord])

                                    for action in optimalresponses[coord]:
                                        index = self.externalactions[action]
                                        correctchoices += self.net_output_distribution[t][netind][coord][index]
                                    prop_correct = correctchoices / float(totalchoices)


                                    rgb = MazeStatistics.getRGB(prop_correct)
                                    col = '#%02x%02x%02x' % rgb

                                    x_arrow, y_arrow = netcorrectness_visual.display.setCoords(x, y, (                                       netcorrectness_visual.display.item_width, netcorrectness_visual.display.item_height), arrow_pos)
                                    netcorrectness_visual.display.canvas.rectangle(
                                        [x_arrow - .30 * netcorrectness_visual.display.item_width, y_arrow - .30 * netcorrectness_visual.display.item_height,
                                         x_arrow + .70 * netcorrectness_visual.display.item_width, y_arrow + .70 * netcorrectness_visual.display.item_height],
                                        fill=col, outline="black")

                        if coord in self.distribution[t] and sum(self.distribution[t][coord]) > 0:
                                print("yes")
                                correctchoices = 0
                                totalchoices = sum(self.distribution[t][coord])

                                for action in optimalresponses[coord]:
                                    index = self.externalactions[action]
                                    correctchoices += self.distribution[t][coord][index]
                                prop_correct = correctchoices / float(totalchoices)
                                rgb = MazeStatistics.getRGB(prop_correct)
                                col = '#%02x%02x%02x' % rgb
                                x_arrow, y_arrow = correctness_visual.display.setCoords(x, y, (
                                    correctness_visual.display.item_width, correctness_visual.display.item_height),
                                                                                           arrow_pos)
                                correctness_visual.display.canvas.rectangle(
                                    [x_arrow - .30 * correctness_visual.display.item_width,
                                     y_arrow - .30 * correctness_visual.display.item_height,
                                     x_arrow + .70 * correctness_visual.display.item_width,
                                     y_arrow + .70 * correctness_visual.display.item_height],
                                    fill=col, outline="black")
                                visitationrate=MazeStatistics.getVisitationRate(envir.statfreq,size,totalchoices)
                                heatrgb = MazeStatistics.getRGB(visitationrate)
                                heatcol = '#%02x%02x%02x' % heatrgb

                                heat_visual.display.canvas.rectangle(
                                    [x_arrow - .30 * heat_visual.display.item_width,
                                     y_arrow - .30 * heat_visual.display.item_height,
                                     x_arrow + .70 * heat_visual.display.item_width,
                                     y_arrow + .70 * heat_visual.display.item_height],
                                    fill=heatcol, outline="black")

                        if (envir.reset_type == ResetType.random and coord in envir.start_coords) or (
                                        envir.reset_type == ResetType.fixed and coord == envir.startCoord)\
                                or (envir.reset_type == ResetType.longterm and coord[0] == envir.startCoord):
                            if envir.agent.learner.network_usage:
                                self.create_starting_pic(usage_visual,x,y,start_pos,font_start)
                                self.create_starting_pic(netcorrectness_visual,x,y,start_pos,font_start)
                            self.create_starting_pic(correctness_visual, x, y, start_pos, font_start)
                            self.create_starting_pic(heat_visual, x, y, start_pos, font_start)

            envir.setGoal([g for g in goals if g != goal], [goal])
            correctness_visual.display.save(arrow_map_file + "t" + str(t) + "g" + str(g_index) + "correctness", extension=".png")

            if envir.agent.learner.network_usage:
                netcorrectness_visual.display.save(
                    arrow_map_file + "t" + str(t) + "g" + str(g_index) + "net_correctness",
                    extension=".png")

                usage_visual.display.save(arrow_map_file + "t" + str(t) + "g" + str(g_index) + "net_usage",
                                            extension=".png")
            heat_visual.display.save(arrow_map_file + "t" + str(t) + "g" + str(g_index) + "heat",
                                            extension=".png")


    def getStats(self,envir,t,arrow_map_file,maze_dir,annotations=False, pol_type=PolType.GLOBAL):
        from StatsAndVisualisation.visual import Visual

        #create the map
        mazefile = maze_dir+self.mazefilename
        print(mazefile)
        envir.generateMap(mazefile)
        ambiguities = self.ambiguities(envir,t)
        f=open(self.file.name,"wb")
        f.write("Maze "+str(mazefile) +"\n")
        slice_end = (t + 1) * envir.statfreq
        f.write("t <= "+str(slice_end)+ "\n")
        size = 0
        total_steps=0
        for coord in self.distribution[t]:
            if sum(self.distribution[t][coord]) > 0:
               size += 1


            total_steps+=sum(self.distribution[t][coord])
        #     total_net_steps+=self.network_usage_distribution[t][0][coord]
        # assert total_net_steps <= total_steps
        # overlay the arrows on the current pictures

        # pic = visual.getPicture("Arrow.png", visual.canvas, (visual.display.item_width, visual.display.item_height))
        # ["north","west","east","south"]
        # ["north", "west", "east", "south"]
        # north --> 0: 90
        # west --> 1 : 180
        # east --> 2 : 0
        # south ---> 3: 270
        # if t==0:
        #     angledict={"north":90,"west":90,"east":90,"south":90}
        # elif t==1:
        #     angledict = {"north": 180, "west": 180, "east": 180, "south": 180}
        # elif t==2:
        #     angledict = {"north": 270, "west": 270, "east": 270, "south": 270}
        # elif t==3:
        #     angledict = {"north": 0, "west": 0, "east": 0, "south": 0}
        # else:
        angledict = {"north": 90, "west": 180, "east": 0, "south":270}
        arrowimages = []


        ambiguities_list=[]


        heatlist=[]
        correctnesslist=[]

        envir.agent.learner.stats.correctness.append(0)
        if envir.agent.learner.network_usage:
            envir.agent.learner.stats.correctnessNP.append(0)
            network_list = [[] for _ in range(self.max_nets)]
            ambiguities_listNP = [[] for _f in range(self.max_nets)]
            correctnesslistNP = [[] for _ in range(self.max_nets)]
            heatlistNP = [[] for _ in range(self.max_nets)]
            totalNPusage = [0 for i in range(self.max_nets)]

        experience_consistency_count={}

        # font_arrowfq=("Arial",6)
        # font_start=("Arial",18)
        # font_ambiguity=("Arial",11)
        # font_networkproportion=("Arial",6)
        # font_correctness=("Arial",6)
        # font_correctnessNP=("Arial", 6)
        dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
        font_arrowfq=ImageFont.truetype(dir+"Times New Roman.ttf",size=18)
        font_start = ImageFont.truetype(dir+"Times New Roman.ttf",size=26)
        font_goal = ImageFont.truetype(dir + "Times New Roman.ttf", size=52)
        font_ambiguity = ImageFont.truetype(dir+"Times New Roman.ttf",size=18 if annotations else 26)
        font_networkproportion = ImageFont.truetype(dir+"Times New Roman.ttf",size=18)
        font_correctness = ImageFont.truetype(dir+"Times New Roman.ttf",size=18)
        font_correctnessNP = ImageFont.truetype(dir+"Times New Roman.ttf",size=18)

        arrow_pos=(0.30,0.30)
        arrowfq_pos=(0.02,0.20)
        start_pos=(0.90, 0.20)
        ambiguity_pos=(0.650 if annotations else .45, 0.80 if annotations else .70)
        network_pos= (0.02, 0.02)
        correctness_pos=(0.02,0.80)

        network_yinc=.10 # use when having two networks
        correctnessNP_pos = (0.02, 0.65)
        if envir.reset_type == ResetType.longterm:
            goals=envir.goal_coords
            optimalresponses= dill.load(open(mazefile + "_optimalResponsesMultigoal"))
        else:
            goals=[envir.foodCoord]
            optimalresponses = dill.load(open(mazefile + "_optimalResponses", "rb"))
        g_index=0
        for goal in goals:
            g_index+=1
            visual = Visual(envir.sizeX, envir.sizeY, (envir.observation_size, envir.observation_size), frameX=1600,
                            frameY=1600, buttons=False,
                            recording_intervals=None, videofile=None, tk_canvas=False)
            visual.display.createCanvas()
            envir.setGoal([g for g in goals if g != goal], [goal])
            visual.display.setEnvir(envir)
            for x in range(envir.sizeX):
                for y in range(envir.sizeY):

                    # print("%d,%d,%d" % (x, y, t))
                    # if (x,y) == envir.foodCoord:
                    #     x_g, y_g = visual.display.setCoords(x, y, (
                    #         visual.display.item_width, visual.display.item_height), start_pos)
                    #     visual.display.canvas.text(
                    #         (x_g * visual.display.item_width, y_g * visual.display.item_height),
                    #         text='G', font=font_goal, fill="black")
                    # get max prob action
                    coord=((x,y),goal) if envir.reset_type == ResetType.longterm else (x,y)
                    if coord in self.distribution[t] and sum(self.distribution[t][coord]) > 0:
                        ambiguities_list.append(ambiguities[coord])
                        choices = self.distribution[t][coord]
                        # fill the map
                        sumact = sum(choices)
                        heatlist.append(sumact)
                        # print("statfreq=" + str(envir.statfreq))
                        # print("sumact=" + str(sumact))
                        # print("size=" + str())
                        visitation = MazeStatistics.getVisitationRate(envir.statfreq, size, sumact)
                        rgb = MazeStatistics.getRGB(visitation)
                        col = '#%02x%02x%02x' % rgb

                        if pol_type==PolType.GLOBAL:
                            arrow_choices=choices
                        elif pol_type==PolType.NETWORK:
                            arrow_choices=self.net_output_distribution[t][0][coord]
                        else:
                            arrow_choices=[choices[i] - self.net_output_distribution[t][0][coord][i] for i in range(len(choices))]
                        maxact = np.argmax(arrow_choices)
                        action = self.actionslist[maxact]
                        angle = angledict[action]

                        arrowimages.append(
                            # getPicture(self, filename, (item_width, item_height), shrinkFactor=(1, 1), angle=0)
                            visual.display.getPicture("arrow.png",
                                                      (visual.display.item_width, visual.display.item_height),
                                                      shrinkFactor=(0.40, 0.40), angle=angle,
                                                      fill=rgb))  # turn all white into col

                        # ambiguityimages.append(Label(visual.canvas, text=ambiguities[(x,y,t)]))
                        x_arrow, y_arrow = visual.display.setCoords(x, y, (
                            visual.display.item_width, visual.display.item_height), arrow_pos)
                        x_arrowfq, y_arrowfq = visual.display.setCoords(x, y, (
                            visual.display.item_width, visual.display.item_height), arrowfq_pos)
                        x_ambi, y_ambi = visual.display.setCoords(x, y, (
                            visual.display.item_width, visual.display.item_height), ambiguity_pos)

                        visual.display.canvas.rectangle(
                            [x_arrow - .30 * visual.display.item_width, y_arrow - .30 * visual.display.item_height,
                             x_arrow + .70 * visual.display.item_width, y_arrow + .70 * visual.display.item_height],
                            fill=col, outline="black")

                        # visual.display.canvas.bitmap((x_arrow,y_arrow),arrowimages[-1])
                        visual.display.mainimage.paste(arrowimages[-1], (x_arrow, y_arrow))
                        if annotations:
                            visual.display.canvas.text((x_arrowfq, y_arrowfq),
                                                       text="MAX=%.2f" % (choices[maxact] / float(sumact)),
                                                       font=font_arrowfq, fill="black")
                            visual.display.canvas.text((x_ambi, y_ambi), text="A=%d" % (ambiguities[coord]),
                                                       font=font_ambiguity, fill="black")

                        # set correct responses
                        correctchoices = 0

                        totalchoices = sum(choices)
                        for action in optimalresponses[coord]:
                            index = self.externalactions[action]
                            correctchoices += choices[index]
                        prop_correct = correctchoices / float(totalchoices)
                        correctnesslist.append(prop_correct)
                        envir.agent.learner.stats.correctness[t] += correctchoices
                        if annotations:
                            correctness_string = "C=%.2f" % (prop_correct)
                            x_corr, y_corr = visual.display.setCoords(x, y, (
                                visual.display.item_width, visual.display.item_height), correctness_pos)

                            visual.display.canvas.text((x_corr, y_corr), text=correctness_string, font=font_correctness,
                                                       fill="black")

                        if (envir.reset_type == ResetType.random and coord in envir.start_coords) or (
                                        envir.reset_type == ResetType.fixed and coord == envir.startCoord)\
                                or (envir.reset_type == ResetType.longterm and coord[0] == envir.startCoord):

                            x_start, y_start = visual.display.setCoords(x, y, (
                                visual.display.item_width, visual.display.item_height), start_pos)
                            visual.display.canvas.rectangle(
                                [x_start - .11 * visual.display.item_width, y_start - .13 * visual.display.item_height,
                                 x_start + .11 * visual.display.item_width, y_start + .13 * visual.display.item_height],
                                fill="white",
                                outline="white")
                            start_text='' if not annotations else 'S'
                            visual.display.canvas.text(
                                (x_start - .06 * visual.display.item_width, y_start - .13 * visual.display.item_height),
                                text=start_text, font=font_start, fill="black")

                        # add network proportion to the map, and add network_list and ambiguities for calculating correlation

                        if envir.agent.learner.network_usage:

                            # print("netwdistr="+str(self.network_usage_distribution[(x,y,t)]))
                            # m=np.argmax(self.network_usage_distribution[(x, y, t)])
                            # visual.display.canvas.text((x_x, y_y), text=m, font=font_n)
                            print(self.max_nets)

                            for netind in range(self.max_nets):
                                # if coord in self.network_usage_distribution[t][netind] and \
                                #                 self.network_usage_distribution[t][netind][coord] > 0:
                                #     network_list[netind].append(
                                #         self.network_usage_distribution[t][netind][coord] / float(sumact))
                                #     new_network_pos = (network_pos[0], network_pos[1] + netind * network_yinc)
                                #     x_net, y_net = visual.display.setCoords(x, y, (
                                #         visual.display.item_width, visual.display.item_height),
                                #                                             new_network_pos)
                                #     totalNPusage[netind] += self.network_usage_distribution[t][netind][coord]
                                #     if annotations:
                                #         if self.max_nets > 1:
                                #             n_use_text="USE%d=%.2f" % (netind, network_list[netind][-1])
                                #         else:
                                #             n_use_text="USE=%.2f" % (network_list[netind][-1])
                                #         visual.display.canvas.text((x_net, y_net),
                                #                                    text=n_use_text,
                                #                                    font=font_networkproportion, fill="black")
                                #     ambiguities_listNP[netind].append(ambiguities[coord])
                                #     heatlistNP[netind].append(heatlist[-1])

                                if coord in self.net_output_distribution[t][netind] and sum(
                                        self.net_output_distribution[t][netind][coord]) > 0:
                                    print("yes")
                                    correctchoices = 0
                                    totalchoices = sum(self.net_output_distribution[t][netind][coord])

                                    for action in optimalresponses[coord]:
                                        index = self.externalactions[action]
                                        correctchoices += self.net_output_distribution[t][netind][coord][index]
                                    prop_correct = correctchoices / float(totalchoices)
                                    correctnesslistNP[netind].append(prop_correct)
                                    envir.agent.learner.stats.correctnessNP[t] += correctchoices
                                    correctnessNP_string = "PC=%.2f" % (prop_correct)

                                    x_corrNP, y_corrNP = visual.display.setCoords(x, y, (
                                        visual.display.item_width, visual.display.item_height),
                                                                                  correctnessNP_pos)
                                    if annotations:
                                        visual.display.canvas.text((x_corrNP, y_corrNP), text=correctnessNP_string,
                                                                   font=font_correctnessNP, fill="black")

                                    network_list[netind].append(
                                        totalchoices / float(sumact))
                                    new_network_pos = (network_pos[0], network_pos[1] + netind * network_yinc)
                                    x_net, y_net = visual.display.setCoords(x, y, (
                                        visual.display.item_width, visual.display.item_height),
                                                                            new_network_pos)
                                    totalNPusage[netind] += totalchoices
                                    if annotations:
                                        if self.max_nets > 1:
                                            n_use_text = "USE%d=%.2f" % (netind, network_list[netind][-1])
                                        else:
                                            n_use_text = "USE=%.2f" % (network_list[netind][-1])
                                        visual.display.canvas.text((x_net, y_net),
                                                                   text=n_use_text,
                                                                   font=font_networkproportion, fill="black")
                                    ambiguities_listNP[netind].append(ambiguities[coord])
                                    heatlistNP[netind].append(heatlist[-1])

            print("making picture")
            if pol_type==PolType.GLOBAL:
                poltypestring=""
            elif pol_type==PolType.NETWORK:
                poltypestring="N"
            elif pol_type==PolType.P_MATRIX:
                poltypestring="P"
            else:
                raise Exception("")
            visual.display.save(arrow_map_file+"t"+str(t)+"g"+str(g_index)+poltypestring,extension=".png")


        if t>0:
            envir.agent.learner.stats.problem_specific['best_path'].append(self.best[t])
            envir.agent.learner.stats.problem_specific['worst_path'].append(self.worst[t])
            envir.agent.learner.stats.problem_specific['sd_path'].append(self.sd[t])
            envir.agent.learner.stats.problem_specific['mean_path'].append(self.mean[t])
            envir.agent.learner.stats.problem_specific['goals'].append(self.goals_achieved[t])
        else:
            envir.agent.learner.stats.problem_specific['best_path']=[self.best[t]]
            envir.agent.learner.stats.problem_specific['worst_path'] = [self.worst[t]]
            envir.agent.learner.stats.problem_specific['mean_path']=[self.mean[t]]
            envir.agent.learner.stats.problem_specific['sd_path'] = [self.sd[t]]
            envir.agent.learner.stats.problem_specific['goals'] = [self.goals_achieved[t]]

        envir.agent.learner.stats.correctness[t]/=float(envir.statfreq) #record the total correctness proportion

        # correlation between heatmap and ambiguity

        #correlation = np.corrcoef(ambiguities_list, heatlist)
        # key0 = "heat_ambiguity"
        # if key0 in envir.agent.learner.stats.correlation:
        #     envir.agent.learner.stats.correlation[key0].append(correlation[0, 1])
        # else:
        #     envir.agent.learner.stats.correlation[key0] = [correlation[0, 1]]
        # f.write("\n\n --------------- \n \n ambiguity/heat correlation : \n" + str(
        #     envir.agent.learner.stats.correlation[key0]))
        key0 = "heat_correctness"
        correlation = np.corrcoef(heatlist,correctnesslist)
        if key0 in envir.agent.learner.stats.correlation:
            envir.agent.learner.stats.correlation[key0].append(correlation[0, 1])
        else:
            envir.agent.learner.stats.correlation[key0] = [correlation[0, 1]]
        f.write("\n\n --------------- \n \n heat_correctness correlation : \n" + str(
            envir.agent.learner.stats.correlation[key0]))
        # compute correlation between ambiguity score and frequency of network usage: hypothesis: is negative
        if envir.agent.learner.network_usage: # multi-network correlations not yet supported
            # correlation between correctness and ambiguity
            if totalNPusage[0]==0:
                envir.agent.learner.stats.correctnessNP[t]==np.nan
            else:
                envir.agent.learner.stats.correctnessNP[t]/= float(totalNPusage[0])
            envir.agent.learner.stats.totalNPusage.append([])
            for net in range(self.max_nets):
                envir.agent.learner.stats.totalNPusage[t].append(totalNPusage[net])
                if correctnesslistNP[net]:
                    print(len(correctnesslistNP[net]))
                    print(len(ambiguities_listNP[net]))
                    # correlationNP = np.corrcoef(ambiguities_listNP[net], correctnesslistNP[net])
                    # key = "correctnessNP_ambiguity"+str(net)
                    # if key in envir.agent.learner.stats.correlation:
                    #     envir.agent.learner.stats.correlation[key].append(correlationNP[0, 1])
                    # else:
                    #     envir.agent.learner.stats.correlation[key] = [correlationNP[0, 1]]
                    # f.write("\n\n --------------- \n \n ambiguity/correctness correlation of net "+str(net)+": \n" +
                    #                 str(envir.agent.learner.stats.correlation[key]))

                    # key0 = "heat_correctnessNP"+str(net)
                    # correlation = np.corrcoef(heatlistNP[net], correctnesslistNP[net])
                    # if key0 in envir.agent.learner.stats.correlation:
                    #     envir.agent.learner.stats.correlation[key0].append(correlation[0, 1])
                    # else:
                    #     envir.agent.learner.stats.correlation[key0] = [correlation[0, 1]]
                    # f.write("\n\n --------------- \n \n heat_correctnessNP correlation : \n" + str(
                    #     envir.agent.learner.stats.correlation[key0]))

                else:
                    print("WARNING: empty correctnesslistNP for net %d. maybe jump network only ?"%(net))

            # correlation between network usage and ambiguity
            for net in range(self.max_nets):
                if network_list[net]:
                    # print("net " + str(net))
                    # print("ambi" + str(ambiguities_listNP[net]))
                    # print("netlist" + str(network_list[net]))
                    correlation = np.corrcoef(ambiguities_listNP[net], network_list[net])
                    key2 = "advice_ambiguity"+str(net)
                    if key2 in envir.agent.learner.stats.correlation:
                        envir.agent.learner.stats.correlation[key2].append(correlation[0, 1])
                    else:
                        envir.agent.learner.stats.correlation[key2]=[correlation[0, 1]]
                    f.write("\n\n --------------- \n \n ambiguity/n_usage correlation of net "+str(net)+": \n"
                                    + str(envir.agent.learner.stats.correlation[key2]))
                else:
                    print("WARNING: network %d not used at all?"%(net))



            f.write("\n\n --------------- \n \n compressStats : \n" + str(envir.agent.learner.stats.compressStats))

            f.write("\n\n --------------- \n \n modularityStats : \n" + str(envir.agent.learner.stats.modularityStats))
        if hasattr(envir.agent.learner,"IP"):
            IPdict = "{" + "\n".join("{}: {}".format(k, v) for k, v in self.IPdistribution[t].items()) + "}"
            f.write("\n\n --------------- \n \n IP-distribution over (x,y,t) : \n" + str(IPdict))
        # if self.IS_stats is not None:
        #     self.IS_stats.print_dictionaries(f,t)

        f.close()


class POcmanMeltingPotStatistics(object):
    """
    mazes specific to a particular task
    statistics such as:
        -pacman's heatmap
        -ghost's heatmap
        -frequency of hitting particular object on a particular location
    """
    def __init__(self,start,task,environment):
        self.task=task
        self.start=start
        self.pacman_location={}
        self.distance={}  # distance between object and pacman
        self.object_location={}
        self.number_of_hits={}
        self.chosen_actions={}
        reward, dynamic, topology=task
        self.maze_map=type(environment).get_maze(topology)


        # remove the object from the maze
        for x in range(self.maze_map.shape[0]):
            for y in range(self.maze_map.shape[1]):
                if self.maze_map[x,y] not in [POcmanFlags.E_OBSTACLE, POcmanFlags.E_FREE]:
                    self.maze_map[x,y]=POcmanFlags.E_FREE
                if self.maze_map[x,y]==POcmanFlags.E_FREE:
                    self.pacman_location[(x,y)] = 0
                    self.object_location[(x,y)] = 0
                    self.number_of_hits[(x,y)] = 0
                    self.chosen_actions[(x,y)] = {}
                    self.distance[(x,y)] = 0
    # def get_goals_and_objecttype(self,environment,task):
    #     topology, dynamic, reward = task
    #     if not PacmanDynamic.is_dynamic(dynamic):
    #         goals = environment.melting_pot_foodlocations(topology)
    #     self.add_other_agents()
    def add_object(self,vis,x,y,object_visitation,task,max_size,offset):
        sizex,sizey=max_size
        if int(object_visitation * sizex) == 0 or int(object_visitation * sizey)==0:
            return
        reward, dynamic, topology = task
        if PacmanDynamic.is_dynamic(dynamic):
            filename = "pacman_ghost.jpg" if reward > 0 else "pacman_normalghost.png"
        else:
            filename = "pacman_food.jpeg"
        vis.display.addPicture(x, y, shrink_factor=(object_visitation, object_visitation),offset=offset,
                               filename=filename)


    def update(self,pacman,object,action):
        """

        :param pacman: current location of pacman
        :param object: current location of the main object
        :param action: currently chosen action
        :return:
        """
        self.pacman_location[pacman] += 1
        self.object_location[object] += 1
        if pacman == object:
            self.number_of_hits[pacman] += 1
        if str(action) not in self.chosen_actions[pacman]:
            self.chosen_actions[pacman][str(action)]=0
        self.chosen_actions[pacman][str(action)] += 1
        self.distance[pacman]+=manhattan_dist(pacman,object)

    def getStats(self,filename):
        """
        heatmap of pacman and the object;
        -red/blue for frequency of pacman visit
        -bigger or smaller object depending on frequency of the object
        :param e:
        :param filename:
        :param totaltime:
        :return:
        """

        angledict = {"north": 90, "west": 180, "east": 0, "south":270}
        arrowimages = []

        dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
        font_stay = ImageFont.truetype(dir + "Times New Roman.ttf", size=56)
        font_distance = ImageFont.truetype(dir + "Times New Roman.ttf", size=45)


        arrow_pos=(0.30,0.30)
        distance_pos=(.55,.05)
        object_offset=(.10,.10)
        sizeX=self.maze_map.shape[0]
        sizeY=self.maze_map.shape[1]

        visual = Visual(sizeX,sizeY , (None,None), frameX=1600,
                        frameY=1600, buttons=False,
                        recording_intervals=None, videofile=None, tk_canvas=False)
        visual.display.createCanvas()

        statfreq=sum(freq for (x,y),freq in self.pacman_location.items()) # total number of visitations (not the same for different policies)
        size=len(self.pacman_location)
        for x in range(sizeX):
            for y in range(sizeY):

                if self.maze_map[x,y]==POcmanFlags.E_OBSTACLE:
                    visual.display.addPicture(x, y, shrink_factor=(1., 1.),
                                           filename="Obstacle.jpeg")

                if (x,y) in self.pacman_location and self.pacman_location[(x,y)] > 0:
                    self.distance[(x,y)]/=float(self.pacman_location[(x,y)]) # average distance

                    pacman_visitationrate = MazeStatistics.getVisitationRate(statfreq, size,
                                                                             self.pacman_location[(x, y)])
                    rgb = MazeStatistics.getRGB(pacman_visitationrate)
                    col = '#%02x%02x%02x' % rgb
                    choices=self.chosen_actions[(x,y)]
                    maxact = max(choices, key=choices.get)

                    x_arrow, y_arrow = visual.display.setCoords(x, y, (
                        visual.display.item_width, visual.display.item_height), arrow_pos)
                    rectangle_coords=[x_arrow - .30 * visual.display.item_width, y_arrow - .30 * visual.display.item_height,
                             x_arrow + .70 * visual.display.item_width, y_arrow + .70 * visual.display.item_height]
                    x_dist,y_dist = visual.display.setCoords(x,y,(visual.display.item_width,visual.display.item_height),distance_pos)
                    visual.display.canvas.rectangle(rectangle_coords, fill=col, outline="black")
                    if (x,y)==self.start:
                        visual.display.fat_rectangle(rectangle_coords,width=3,color="green")

                    if maxact=="stay":
                        # display stay
                        visual.display.canvas.text((x_arrow+.10, y_arrow+.10), text="X",
                                                   font=font_stay, fill="black")
                    else:
                        angle = angledict[maxact]

                        arrowimages.append(
                            # getPicture(self, filename, (item_width, item_height), shrinkFactor=(1, 1), angle=0)
                            visual.display.getPicture("arrow.png",
                                                      (visual.display.item_width, visual.display.item_height),
                                                      shrinkFactor=(0.40, 0.40), angle=angle,
                                                      fill=rgb))  # turn all white into col

                        # visual.display.canvas.bitmap((x_arrow,y_arrow),arrowimages[-1])
                        visual.display.mainimage.paste(arrowimages[-1], (x_arrow, y_arrow))
                    visual.display.canvas.text((x_dist,y_dist), text="d=%d"%(int(round(self.distance[(x,y)]))),
                                               font=font_distance, fill="black")

                if (x,y) in self.object_location and self.object_location[(x,y)] > 0:
                    obj_visitationrate = MazeStatistics.getVisitationRate(statfreq, 0.05*size, self.object_location[(x,y)])
                    self.add_object(visual,x,y,obj_visitationrate,self.task,
                                    max_size=(visual.display.item_width,visual.display.item_height),offset=object_offset)

        visual.display.save(filename, extension=".png")


    @classmethod
    def getAllStats(cls,filename,stats):
        slices=len(stats)

        for tt in range(slices):
            for key in stats[tt]:
                stats[tt][key].getStats(filename + "heatmap_%s_slice%d"%(str(key),tt))

def getAllNetworkStats(learner):
    NP=learner.NP
    print("num nets = %d"%(NetworkAnalysis.numNetworks(NP)))
    for i in range(NetworkAnalysis.numNetworks(NP)):
        print("network %d :"%(i))
        print("num nodes = %d" %(NetworkAnalysis.numNodes(NP.representation[i])))
        print("num conns = %d"%(NetworkAnalysis.numConns(NP.representation[i])))
        print("compressionstats ="+str(NetworkAnalysis.compressionStats(NP.representation[i])))
        print("modularityQ = " +str(NetworkAnalysis.structuralmodularityQ(learner, i)))
def getAllMazeStats(learner):
    if isinstance(learner.statistics, list):
        for stat in learner.statistics:
            print(stat.networkdistribution)
            print(stat.IPdistribution)
    else:
        print(learner.statistics.networkdistribution)
        print(learner.statistics.IPdistribution)


def create_legend(num_rectangles,filename):
    visual = Visual(num_rectangles, 1, (None,None), frameX=1600,
                    frameY=1600/num_rectangles, buttons=False,
                    recording_intervals=None, videofile=None, tk_canvas=False)
    visual.display.createCanvas()
    visitation_rate = 0
    next_visitation_rate=0
    dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
    font = ImageFont.truetype(dir + "Times New Roman.ttf", size=56)
    for rectangle in range(num_rectangles): #from visitation rate [0,1/num_r] to [visitation rate > 1]
        next_visitation_rate = min(1,visitation_rate+1 / float(num_rectangles-1))
        avg_visitationrate = (next_visitation_rate + visitation_rate)/2.0
        col=MazeStatistics.getRGB(avg_visitationrate)
        xmin=rectangle * visual.display.item_width
        ymin=0
        xmax=(rectangle+1) * visual.display.item_width
        ymax=visual.display.item_height/2
        visual.display.canvas.rectangle([xmin, ymin, xmax , ymax], fill=col, outline="black")
        if visitation_rate==1:
            text=">1"
        else:
            text="[%.2f,%.2f]" % (visitation_rate,next_visitation_rate)
        visual.display.canvas.text((xmin+.15*visual.display.item_width, ymin+.15*visual.display.item_height), fill="black",text=text, font=font)
        visitation_rate =next_visitation_rate
    visual.display.save(filename)
def create_legend_multilabel(num_rectangles,filename):
    """
    create a legend with at differnt heights different annotatiions
    """
    from collections import OrderedDict
    proportions=['=[0,.20]','=[0.20,0.40]','=[0.40,0.60]','=[0.60,0.80]','=[0.80,1]']
    visitations=['=[0,.25]','=[0.25,0.50)','=[0.50,0.75]','=[0.75,1.0]','>1']
    epsilons=['=[0,.04]','=[0.04,0.08]','=[0.08,0.12]','=[0.12,0.16]','=[0.16,0.20]']
    annotations=OrderedDict({'Vis':visitations,'Eps':epsilons,'Pr':proportions})

    visual = Visual(num_rectangles, 1, (None,None), frameX=1600,
                    frameY=600, buttons=False,
                    recording_intervals=None, videofile=None, tk_canvas=False)
    visual.display.createCanvas()
    visitation_rate = 0
    next_visitation_rate=0
    dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
    font = ImageFont.truetype(dir + "Times New Roman.ttf", size=35)
    for rectangle in range(num_rectangles): #from visitation rate [0,1/num_r] to [visitation rate > 1]
        next_visitation_rate = min(1,visitation_rate+1 / float(num_rectangles-1))
        avg_visitationrate = (next_visitation_rate + visitation_rate)/2.0
        col=MazeStatistics.getRGB(avg_visitationrate)
        xmin=rectangle * visual.display.item_width
        ymin=-0.10
        xmax=(rectangle+1) * visual.display.item_width
        ymax=visual.display.item_height/2
        visual.display.canvas.rectangle([xmin, ymin, xmax , ymax], fill=col, outline="black")

        for key,value in annotations.items():
            visual.display.canvas.text((xmin+.15*visual.display.item_width, (ymin+.15)*visual.display.item_height), fill="black",text="%s%s"%(key,value[rectangle]), font=font)
            ymin+=.15

        visitation_rate =next_visitation_rate
    visual.display.save(filename)
def create_labelbased_legend(filename,labels):
    num_rectangles=len(labels)
    visual = Visual(num_rectangles, 1, (None,None), frameX=1600,
                    frameY=1600/num_rectangles, buttons=False,
                    recording_intervals=None, videofile=None, tk_canvas=False)
    visual.display.createCanvas()
    visitation_rate = 0
    next_visitation_rate=0
    dir = str(os.environ['HOME']) + "/PycharmProjects/PhD/StatsAndVisualisation/"
    font = ImageFont.truetype(dir + "Times New Roman.ttf", size=56)
    for rectangle in range(num_rectangles): #from visitation rate [0,1/num_r] to [visitation rate > 1]
        min=rectangle*1/float(num_rectangles)
        max=(rectangle+1)/float(num_rectangles)
        mean=(min+max)/float(2)
        col=MazeStatistics.getRGB(mean)
        xmin=rectangle * visual.display.item_width
        ymin=0
        xmax=(rectangle+1) * visual.display.item_width
        ymax=visual.display.item_height/2
        visual.display.canvas.rectangle([xmin, ymin, xmax , ymax], fill=col, outline="black")
        if visitation_rate==1:
            text="[%.2f,%.2f]" % (min,max)
        visual.display.canvas.text((xmin+.15*visual.display.item_width, ymin+.15*visual.display.item_height), fill="black",text=text, font=font)
        visitation_rate =next_visitation_rate
    visual.display.save(filename)
def main():
    #create_legend(5,"legend")
    statsObject=dill.load(open('POmazeFinal_SSA_WMwm120p100ff21actions_noreset_stats_object',"rb"))
    statsObject.getStats()



if __name__ == '__main__':
    create_legend_multilabel(5, "legend_6figs.eps")
