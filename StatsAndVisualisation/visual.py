
import sys
if sys.version_info[0] == 2:
    import Tkinter as tk
    import tkFont
else:
    from tkinter import font as tkFont
    import tkinter as tk

import time

import numpy as np
from PIL import ImageTk as itk

from PIL import Image, ImageDraw
#from pyscreenshot import grab
import os
import sys

#import cv2
try:
    import cv2
    print("cv2 version:"+str(cv2.__version__))
except:
    print("WARNING: ignoring cv2")

import multiprocessing

e = multiprocessing.Event()
p = None

FRAME_RATE=10.0




class IS_PILImageDisplay:
    def __init__(self,use_NP,learner,rows,numToDisplay,frameX,frameY,root):
        self.root=root

        self.use_NP=use_NP

        self.sizeX =learner.n_ops + 1# num ops
        self.num_to_display=numToDisplay
        self.sizeY = rows # num IPs to display + cRows
        self.frameX = frameX
        self.frameY = frameY
        self.item_width = self.frameX / self.sizeX
        self.item_height = self.frameY / self.sizeY
        self.item_height = self.frameY / self.sizeY
        self.createCanvas()
        if self.use_NP:
            self.createNPcanvas()
    def fillItems(self,environment):
        self.setEnvir(environment)
        environment.agent.addAgentToCanvas(environment)
        #environment.setAgentMapPIL()   not needed
    def _createCanvas(self,image):
        self.mainimage=Image.new('RGBA', (self.frameX, self.frameY), (255, 255, 255))
        self.canvas = ImageDraw.Draw(image)
    def createCanvas(self):
        self._createCanvas(self.mainimage)
    def createNPCanvas(self):
        self._createCanvas(self.NPwindow)

    def create_rectangle(self, cor, width, color):
            line = (cor[0], cor[1], cor[0], cor[3])
            self.canvas.line(line, fill=color, width=width)
            line = (cor[0], cor[1], cor[2], cor[1])
            self.canvas.line(line, fill=color, width=width)
            line = (cor[0], cor[3], cor[2], cor[3])
            self.canvas.line(line, fill=color, width=width)
            line = (cor[2], cor[1], cor[2], cor[3])
            self.canvas.line(line, fill=color, width=width)
    def shorthand(self,instruction_name):
        if len(instruction_name) < 7:
            return instruction_name
        else:
            return instruction_name[:7]
    def create_rect(self,xx,yy,color,text,font):
        x0 = xx - 0.5 * self.item_width
        x1 = xx + 0.5 * self.item_width
        y0 = yy - 0.5 * self.item_height
        y1 = yy + 0.5 * self.item_height
        self.create_rectangle((x0, y0, x1, y1), width=self.item_width,fill=color,font=font)
        self.canvas.text((xx, yy), text=text, fill=(0,0,0,0))
    def getCrows(self,learner):
        # how many rows needed to store Max-Min rectangles ?
        # --> (Max-Min)*item_width/sizeX
        return (learner.Max-learner.Min)*(self.item_width)
    def fill_canvas(self,learner):
        print("action=" + str(learner.chosenAction.function.__name__))
        # print("args=" + str(num_args))
        # print("i=%d" % (i))
        # print("relativeIP=%d" % (relativeIP))
        print("instr=" + str(learner.currentInstruction))
        """ create matrix of rectangles and fill with info"""

        font = tkFont.Font(family="Arial",size=8)
        for i in range(len(learner.actions)):
            xx, yy = self.setCoords(i+1, 0, (self.item_width, self.item_height))
            x0 = xx - 0.5 * self.item_width
            x1 = xx + 0.5 * self.item_width
            y0 = yy - 0.5 * self.item_height
            y1 = yy + 0.5 * self.item_height
            self.create_rectangle((x0, y0, x1, y1), width=self.item_width, fill=(255,255,255,255))
            self.canvas.text((xx, yy), text=self.shorthand(learner.actions[i].function.__name__), fill=(0,0,0,0),
                             font=font)

        num_args = learner.chosenAction.n_args
        startIP=learner.IP



        IPmin=max(learner.ProgramStart,learner.IP- self.num_to_display/2)
        IPmax=min(learner.Max+1,learner.IP+self.num_to_display/2 +num_args)

        for IP in range(IPmin,IPmax):
            row=IP - IPmin + 1
            xx, yy = self.setCoords(0, row , (self.item_width, self.item_height))
            self.create_rect(xx,yy,"white","IP=%d"%(IP),font)

            #print(learner.chosenAction.function.__name__)
            for i in range(len(learner.actions)):
                xx, yy = self.setCoords(i+1,row, (self.item_width, self.item_height))

                instructionprob = learner.Pol[IP][i]
                relativeIP=IP-startIP

                if self.use_NP and learner.usedNPinstruction is not None: #used instruction has form [NPinstr0,..,NPinstr_n]
                    if relativeIP <= num_args and relativeIP >= 0 and i == learner.currentInstruction[relativeIP]:
                        color = "green"
                    elif relativeIP < len(learner.usedNPinstruction) and relativeIP >= 0 and i == learner.usedNPinstruction[relativeIP]:
                        color = "green"
                    else:
                        color = "white"
                elif relativeIP <= num_args and relativeIP >=0 and i == learner.currentInstruction[relativeIP]:
                        color = "red"
                else:
                    color="white"
                if instructionprob > 0.50:
                    ffont=tkFont.Font(family="Arial",size=11)
                else:
                    ffont=font
                self.create_rect(xx,yy,color,"%.3f"%(instructionprob),ffont)

            for IP in range(IPmin, IPmax):
                row = IP - IPmin + 1
                xx, yy = self.setCoords(0, row, (self.item_width, self.item_height))
                self.create_rect(xx,yy,"white",text="%d" % (IP), font=font)

                #print(learner.chosenAction.function.__name__)
                for i in range(len(learner.actions)):
                    xx, yy = self.setCoords(i + 1, row, (self.item_width, self.item_height))

                    instructionprob = learner.Pol[IP][i]
                    relativeIP = IP - startIP

                    if self.use_NP and learner.usedNPinstruction is not None:  # used instruction has form [NPinstr0,..,NPinstr_n]
                        if relativeIP <= num_args and relativeIP >= 0 and i == learner.currentInstruction[relativeIP]:
                            color = "green"
                        elif relativeIP < len(learner.usedNPinstruction) and relativeIP >= 0 and i == \
                                learner.usedNPinstruction[relativeIP]:
                            color = "green"
                        else:
                            color = "white"
                    elif relativeIP <= num_args and relativeIP >= 0 and i == learner.currentInstruction[relativeIP]:
                        color = "red"
                    else:
                        color = "white"
                    if instructionprob > 0.50:
                        ffont = tkFont.Font(family="Arial", size=11)
                    else:
                        ffont = font

                    self.create_rect(xx, yy,color, text="%.3f" % (instructionprob), font=ffont)

                # now the storage
                input_cells= range(learner.Min,learner.Min + 4 +learner.num_inputs)
                working_cells=range(learner.Min + 4 +learner.num_inputs,0)
                register_cells=range(0,learner.ProgramStart)
                program_cells=range(learner.ProgramStart,learner.Max+1)
                col=-1
                row=IPmax-IPmin + 2
                cmin = learner.Min
                cmax = learner.Max
                # print("learner c" + str(learner.c))
                # print("c"+str(cmin+2)+" = " + str(learner.c[cmin+2]))
                # raw_input()
                for i in range(cmin,cmax+1):
                    index=i-cmin
                    if index>0 and index % self.sizeX == 0:
                        col=0
                        row+=1
                    else:
                        col+=1
                    if i in input_cells:
                        color="red"
                    elif i in working_cells:
                        color="blue"
                    elif i in register_cells:
                        color="yellow"
                    else:
                        color="green"


                    text="%d"%(learner.c[i])
                    xx, yy = self.setCoords(col, row, (self.item_width, self.item_height))
                    self.create_rect(xx, yy, color,text=text, font=font)

    def fill_NPcanvas(self, learner):
        """ create matrix of rectangles and fill with info"""
        if not learner.action_is_set: return  # can't display instructions if there are none
        font = tkFont.Font(family="Arial", size=8)
        instrs=learner.NP.representation[learner.net_key].instruction_set
        for i in range(len(instrs)): #ffirst the labels
            xx, yy = self.setCoords(i + 1, 0, (self.item_width, self.item_height))
            x0 = xx - 0.5 * self.item_width
            x1 = xx + 0.5 * self.item_width
            y0 = yy - 0.5 * self.item_height
            y1 = yy + 0.5 * self.item_height
            self.NPcanvas.create_rectangle(x0, y0, x1, y1, fill="white")
            self.NPcanvas.text((xx, yy), text=self.shorthand(instrs[i].function.__name__), font=font)
        for i in range(len(instrs)): #now fill the rectangles and possibly color one of them
            xx, yy = self.setCoords(i + 1, 1, (self.item_width, self.item_height))
            x0 = xx - 0.5 * self.item_width
            x1 = xx + 0.5 * self.item_width
            y0 = yy - 0.5 * self.item_height
            y1 = yy + 0.5 * self.item_height
            if self.use_NP and learner.usedNPinstruction is not None:  # used instruction has form [NPinstr0,..,NPinstr_n]
                if instrs[i] == learner.next_chosenAction:
                    color="green"
                else:
                    color="white"
                self.NPcanvas.create_rectangle(x0, y0, x1, y1, fill=color)

    def iteration(self,learner):
        """main function which displays policy matrix and action chosen"""

        self.canvas.delete(tk.ALL)
        self.fill_canvas(learner)
        self.canvas.pack(side='right')
        if self.use_NP:
            self.NPcanvas.delete(tk.ALL)
            self.fill_NPcanvas(learner)
            self.NPcanvas.pack(side='right')
        self.root.update()

    def setCoords(self,x,y,item_shape, offset):
        (item_width, item_height)=item_shape
        xx = (offset[0] + x) *item_width
        yy = (offset[1] + y) *item_height
        return xx,yy


class IS_Visualisation:
    """ visualises the IS learner at every cycle"""
    def __init__(self,use_NP,learner,rows,numToDisplay,frameX,frameY,root):
        self.root=root

        self.use_NP=use_NP

        self.sizeX =learner.n_ops + 1# num ops
        self.num_to_display=numToDisplay
        self.sizeY = rows # num IPs to display + cRows
        self.frameX = frameX
        self.frameY = frameY
        self.item_width = self.frameX / self.sizeX
        self.item_height = self.frameY / self.sizeY
        self.item_height = self.frameY / self.sizeY
        self.createCanvas()
        if self.use_NP:
            self.createNPcanvas()

    def createCanvas(self):
        self.window = tk.Toplevel()
        self.canvas = tk.Canvas(self.window, width=self.frameX, height=self.frameY, bg='white')

    def createNPcanvas(self):
        self.NPwindow = tk.Toplevel()
        self.NPcanvas = tk.Canvas(self.NPwindow, width=self.frameX, height=self.frameY, bg='white')
    def shorthand(self,instruction_name):
        if len(instruction_name) < 7:
            return instruction_name
        else:
            return instruction_name[:7]
    def create_rect(self,xx,yy,color,text,font):
        x0 = xx - 0.5 * self.item_width
        x1 = xx + 0.5 * self.item_width
        y0 = yy - 0.5 * self.item_height
        y1 = yy + 0.5 * self.item_height
        self.canvas.create_rectangle(x0, y0, x1, y1, fill=color)
        self.canvas.create_text((xx, yy), text=text, font=font)
    def getCrows(self,learner):
        # how many rows needed to store Max-Min rectangles ?
        # --> (Max-Min)*item_width/sizeX
        return (learner.Max-learner.Min)*(self.item_width)
    def fill_canvas(self,learner):
        if not learner.action_is_set: return # can't display instructions if there are none
        print("action=" + str(learner.chosenAction.function.__name__))
        # print("args=" + str(num_args))
        # print("i=%d" % (i))
        # print("relativeIP=%d" % (relativeIP))
        print("instr=" + str(learner.currentInstruction))
        """ create matrix of rectangles and fill with info"""

        font = tkFont.Font(family="Arial",size=8)
        for i in range(len(learner.actions)):
            xx, yy = self.setCoords(i+1, 0, (self.item_width, self.item_height))
            x0 = xx - 0.5 * self.item_width
            x1 = xx + 0.5 * self.item_width
            y0 = yy - 0.5 * self.item_height
            y1 = yy + 0.5 * self.item_height
            self.canvas.create_rectangle(x0, y0, x1, y1, fill="white")
            self.canvas.create_text((xx, yy), text=self.shorthand(learner.actions[i].function.__name__), font=font)

        num_args = learner.chosenAction.n_args
        startIP=learner.IP



        IPmin=max(learner.ProgramStart,learner.IP- self.num_to_display/2)
        IPmax=min(learner.Max+1,learner.IP+self.num_to_display/2 +num_args)

        for IP in range(IPmin,IPmax):
            row=IP - IPmin + 1
            xx, yy = self.setCoords(0, row , (self.item_width, self.item_height))
            self.create_rect(xx,yy,"white","IP=%d"%(IP),font)

            #print(learner.chosenAction.function.__name__)
            for i in range(len(learner.actions)):
                xx, yy = self.setCoords(i+1,row, (self.item_width, self.item_height))

                instructionprob = learner.Pol[IP][i]
                relativeIP=IP-startIP

                if self.use_NP and learner.usedNPinstruction is not None: #used instruction has form [NPinstr0,..,NPinstr_n]
                    if relativeIP <= num_args and relativeIP >= 0 and i == learner.currentInstruction[relativeIP]:
                        color = "green"
                    elif relativeIP < len(learner.usedNPinstruction) and relativeIP >= 0 and i == learner.usedNPinstruction[relativeIP]:
                        color = "green"
                    else:
                        color = "white"
                elif relativeIP <= num_args and relativeIP >=0 and i == learner.currentInstruction[relativeIP]:
                        color = "red"
                else:
                    color="white"
                if instructionprob > 0.50:
                    ffont=tkFont.Font(family="Arial",size=11)
                else:
                    ffont=font
                self.create_rect(xx,yy,color,"%.3f"%(instructionprob),ffont)

            for IP in range(IPmin, IPmax):
                row = IP - IPmin + 1
                xx, yy = self.setCoords(0, row, (self.item_width, self.item_height))
                self.create_rect(xx,yy,"white",text="%d" % (IP), font=font)

                #print(learner.chosenAction.function.__name__)
                for i in range(len(learner.actions)):
                    xx, yy = self.setCoords(i + 1, row, (self.item_width, self.item_height))

                    instructionprob = learner.Pol[IP][i]
                    relativeIP = IP - startIP

                    if self.use_NP and learner.usedNPinstruction is not None:  # used instruction has form [NPinstr0,..,NPinstr_n]
                        if relativeIP <= num_args and relativeIP >= 0 and i == learner.currentInstruction[relativeIP]:
                            color = "green"
                        elif relativeIP < len(learner.usedNPinstruction) and relativeIP >= 0 and i == \
                                learner.usedNPinstruction[relativeIP]:
                            color = "green"
                        else:
                            color = "white"
                    elif relativeIP <= num_args and relativeIP >= 0 and i == learner.currentInstruction[relativeIP]:
                        color = "red"
                    else:
                        color = "white"
                    if instructionprob > 0.50:
                        ffont = tkFont.Font(family="Arial", size=11)
                    else:
                        ffont = font

                    self.create_rect(xx, yy,color, text="%.3f" % (instructionprob), font=ffont)

                # now the storage
                input_cells= range(learner.Min,learner.Min + 4 +learner.num_inputs)
                working_cells=range(learner.Min + 4 +learner.num_inputs,0)
                register_cells=range(0,learner.ProgramStart)
                program_cells=range(learner.ProgramStart,learner.Max+1)
                col=-1
                row=IPmax-IPmin + 2
                cmin = learner.Min
                cmax = learner.Max
                # print("learner c" + str(learner.c))
                # print("c"+str(cmin+2)+" = " + str(learner.c[cmin+2]))
                # raw_input()
                for i in range(cmin,cmax+1):
                    index=i-cmin
                    if index>0 and index % self.sizeX == 0:
                        col=0
                        row+=1
                    else:
                        col+=1
                    if i in input_cells:
                        color="red"
                    elif i in working_cells:
                        color="blue"
                    elif i in register_cells:
                        color="yellow"
                    else:
                        color="green"


                    text="%d"%(learner.c[i])
                    xx, yy = self.setCoords(col, row, (self.item_width, self.item_height))
                    self.create_rect(xx, yy, color,text=text, font=font)

    def fill_NPcanvas(self, learner):
        """ create matrix of rectangles and fill with info"""
        if not learner.action_is_set: return  # can't display instructions if there are none
        font = tkFont.Font(family="Arial", size=8)
        instrs=learner.NP.representation[learner.net_key].instruction_set
        for i in range(len(instrs)): #ffirst the labels
            xx, yy = self.setCoords(i + 1, 0, (self.item_width, self.item_height))
            x0 = xx - 0.5 * self.item_width
            x1 = xx + 0.5 * self.item_width
            y0 = yy - 0.5 * self.item_height
            y1 = yy + 0.5 * self.item_height
            self.NPcanvas.create_rectangle(x0, y0, x1, y1, fill="white")
            self.NPcanvas.create_text((xx, yy), text=self.shorthand(instrs[i].function.__name__), font=font)
        for i in range(len(instrs)): #now fill the rectangles and possibly color one of them
            xx, yy = self.setCoords(i + 1, 1, (self.item_width, self.item_height))
            x0 = xx - 0.5 * self.item_width
            x1 = xx + 0.5 * self.item_width
            y0 = yy - 0.5 * self.item_height
            y1 = yy + 0.5 * self.item_height
            if self.use_NP and learner.usedNPinstruction is not None:  # used instruction has form [NPinstr0,..,NPinstr_n]
                if instrs[i] == learner.next_chosenAction:
                    color="green"
                else:
                    color="white"
                self.NPcanvas.create_rectangle(x0, y0, x1, y1, fill=color)

    def iteration(self,learner):
        """main function which displays policy matrix and action chosen"""

        self.canvas.delete(tk.ALL)
        self.fill_canvas(learner)
        self.canvas.pack(side='right')
        if self.use_NP:
            self.NPcanvas.delete(tk.ALL)
            self.fill_NPcanvas(learner)
            self.NPcanvas.pack(side='right')
        self.root.update()

    def setCoords(self,x,y,item_shape, offset):
        (item_width, item_height)=item_shape
        xx = (offset[0] + x) *item_width
        yy = (offset[1] + y) *item_height
        return xx,yy

class BaseVisual(object):
    show_intermediate_actions = False

    on = False
    record_counter = 0
    mapobjfolder = str(os.environ['HOME']) + '/LifelongRL/StatsAndVisualisation/Mapobjects/'
    file=None #file f
    def __init__(self,sizeX,sizeY,observationShape, frameX,frameY,recording_intervals,videofile):
        (observationSizeX, observationSizeY)= observationShape
        self.recording_intervals=recording_intervals
        if recording_intervals:
            self.current_recording_interval = self.recording_intervals.pop(0)
        else:
            self.current_recording_interval=None
        self.videofile=videofile
        self.framerate = float(FRAME_RATE)
        self.sizeX = sizeX
        self.sizeY = sizeY
        print("sizes=%d,%d" % (self.sizeX, self.sizeY))

        self.item_width = frameX / sizeX
        self.item_height = frameY / sizeY

        self.frameX = sizeX*self.item_width
        self.frameY = sizeY*self.item_height
        print("frame sizes=%d,%d" % (self.frameX, self.frameY))



        print("item sizes=%d,%d" % (self.item_width, self.item_height))
        if observationSizeX is not None:
            self.obs_width = frameX / (2 * observationSizeX + 1)
            self.obs_height = frameY / (2 * observationSizeY + 1)

        self.visualmap = {}
        self.visualobsmap = {}

    def setCoords(self,x,y,item_shape, offset=(0.5,0.5)):
        (item_width, item_height)=item_shape
        xx = (offset[0] + x) *item_width
        yy = (offset[1] + y) *item_height
        return int(xx),int(yy)

    def checkPicture(self,pic,obj):
        if pic == None:
            return False
        return True

class CanvasDisplay(BaseVisual):
    waitForSMS = False
    IS_vis = None
    on=True
    def __init__(self,sizeX,sizeY,observationShape,frameX,frameY,buttons,recording_intervals,videofile):
        BaseVisual.__init__(self,sizeX,sizeY,observationShape, frameX,frameY,recording_intervals,videofile)
        self.root = tk.Tk()
        self.canvas = tk.Canvas(self.root, width=frameX, height=frameY, bg='white')
        self.canvas2 = tk.Canvas(self.root, width=frameX, height=frameY, bg='white')
        self.base_offset = (0.50,0.50)
        if buttons:
            upspeedbutton = tk.Button(self.root, width=10, height=1, text='FASTER', command=self.faster)
            downspeedbutton = tk.Button(self.root, width=10, height=1, text='SLOWER', command=self.slower)
            stopbutton = tk.Button(self.root, width=10, height=1, text='ON/OFF', command=self.on_off)
            waitbutton = tk.Button(self.root, width=10, height=1, text='WAIT', command=self.wait)
            nextSMSbutton = tk.Button(self.root,width=10,height=1,text='NEXT_SMS',command=self.nextSMS)
            networkbutton = tk.Button(self.root, width=10, height=1, text="TEST NETWORK", command=self.test_network())
            self.wait_until=0
            #switch_maze_button = tk.Button(self.root,width=10,height=1,text='Wait_Switch_maze',command=self.next_maze)
            upspeedbutton.pack()
            downspeedbutton.pack()
            stopbutton.pack()
            waitbutton.pack()
            nextSMSbutton.pack()
            networkbutton.pack()


    def tick(self,environment,intermediate=False):
        print(self.on)
        if not self.on:
            if self.current_recording_interval:
                if environment.t >= self.current_recording_interval[0]:
                    self.record_counter+=1
                    self.on=True
                    self.startrecording()
            else:
                if environment.t >= self.wait_until :
                    self.on = True
                if hasattr(environment.agent.learner,"polChanged") and environment.agent.learner.beginOfSMS and self.waitForSMS:
                    print("wait false")
                    self.waitForSMS=False
        if self.on:
            if not intermediate:
                self.iteration(environment)
            elif intermediate and self.show_intermediate_actions:
                self.iteration(environment)
        if self.current_recording_interval and environment.t >= self.current_recording_interval[1]:
            self.stoprecording()
            try:
                self.current_recording_interval = self.recording_intervals.pop(0)
            except:
                # run is over
                import sys
                sys.exit()
            self.on = False
    def iteration(self,environment):
        environment.convert_to_map()
        self.canvas.delete(tk.ALL)
        self.canvas2.delete(tk.ALL)
        self.fillItems(environment)
        self.canvas.pack(side='left')
        self.canvas2.pack(side='right')
        self.root.update()
        time.sleep(1 / self.framerate)
    def fillItems(self,environment):
        self.setEnvironmentMap(environment)
        environment.setAgentMap()
        if self.IS_vis is not None:
            self.IS_vis.iteration(environment.agent.learner)
    def setEnvironmentMap(self,environment):
        self.setEnvir(environment)
        self.ag_imgs = []
        environment.agent.addAgentToCanvas(environment)
    def addPicture(self,x,y,filename,offset=(0.5,0.5),shrink_factor=(1,1)):

        xx, yy = self.setCoords(x,y, (self.item_width, self.item_height), offset=offset)
        self.ag_imgs.append(self.getPicture(filename, self.canvas,
                                                 (self.item_width, self.item_height), shrink_factor))
        if self.checkPicture(self.ag_imgs[-1], self):
            self.canvas.create_image(xx, yy, image=self.ag_imgs[-1])

    def addObsPicture(self,x,y,filename,offset=(0.5,0.5),shrink_factor=(1,1)):
        xx, yy = self.setCoords(x,y, (self.obs_width, self.obs_height), offset)
        im = self.getPicture(filename, self.canvas2,
                                                 (self.obs_width, self.obs_height), shrink_factor)
        if self.checkPicture(im, self):
            self.visualobsmap[(x, y)]=im
            self.canvas2.create_image(xx, yy, image=self.visualobsmap[(x,y)])
    def addText(self,x,y, offset, txt, font_t,font_size):
        xx, yy = self.setCoords(x, y,(self.item_width, self.item_height), offset)
        #
        font = tkFont.Font(family=font_t,size=font_size)
        self.canvas.create_text((xx, yy), text=txt, font=font)
    def getPicture(self, filename, canvas, item_shape, shrinkFactor=(1, 1), angle=0):
        (item_width, item_height)=item_shape
        if filename == None:
            return None
        return itk.PhotoImage(Image.open(fp=self.mapobjfolder + filename).resize(
            (int(shrinkFactor[0] * item_width), int(shrinkFactor[1] * item_height))).rotate(angle), master=canvas)
    def setEnvir(self,environment):

        for x in range(environment.sizeX):
            for y in range(environment.sizeY):
                #print(environment.map[x][y].filename)
                obj=environment.map[x][y]
                rot=0
                if hasattr(obj,"rotation"):
                    rot=obj.rotation
                pic = self.getPicture(obj.filename,self.canvas,(self.item_width,self.item_height),angle=rot)
                if self.checkPicture(pic,obj):
                    self.visualmap[(x,y)]=pic
                    xx,yy = self.setCoords(x,y,(self.item_width,self.item_height))
                    self.canvas.create_image(xx, yy, image=self.visualmap[(x, y)])
    def save(self,filename):
        img = grab(bbox=(0, 0, self.frameX, self.frameY))
        img.save(filename + '.png')
        print(filename +".png saved")

    #  buttons
    def faster(self):
        print("framerate = %d" % (self.framerate))
        self.framerate *= 2

    def slower(self):
        print("framerate = %d" % (self.framerate))
        self.framerate /= 2

    def on_off(self):
        self.on = True if not self.on else False
        self.wait_until = float("inf")

    def wait(self):
        self.on = False
        print("waiting. Wait until what time ? ? ")
        self.wait_until = input()

    def nextSMS(self):
        self.on = False
        self.waitForSMS = True
    def test_network(self):
        # do some tests on a realistic sequence of observation
        pass
    def record_video(self,video,filename):
        while True:

            #capture screen and save it to file
            self.save(filename)
            frame=cv2.imread(filename+'.png')
            video.write(frame)
            if e.is_set():
                video.release()
                cv2.destroyAllWindows()
                e.clear()
                break
    def record(self):

        print("recording")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        file=self.videofile+str(self.record_counter)
        print(str(os.environ['HOME']) +"/Videos/"+file+'.avi')
        video = cv2.VideoWriter(str(os.environ['HOME']) +"/Videos/"+file+'.avi', fourcc, 20.0, (self.frameX, self.frameY))
        #ret, frame = cap.read()


        self.record_video(video,file)
    def startrecording(self):
        print("preparing recording")
        global p
        p = multiprocessing.Process(target=self.record, args=())
        p.start()

    #-------end video capture and stop tk
    def stoprecording(self):
        e.set()
        p.join()

        # self.root.quit()
        # self.root.destroy()
        print("stopped recording")

class PILImageDisplay(BaseVisual):
    def __init__(self, sizeX, sizeY, observationShape, frameX, frameY,recording_intervals,videofile):
        BaseVisual.__init__(self, sizeX, sizeY, observationShape, frameX, frameY,recording_intervals,videofile)
        self.mainimage= Image.new('RGBA', (int(self.frameX), int(self.frameY)), (255, 255, 255))
        self.canvas = ImageDraw.Draw(self.mainimage)
        self.base_offset = (0.0, 0.0)
    def fillItems(self,environment):
        self.setEnvir(environment)
        environment.agent.addAgentToCanvas(environment)
        #environment.setAgentMapPIL()   not needed
    def createCanvas(self):
        self.mainimage=Image.new('RGBA', (self.frameX, self.frameY), (255, 255, 255))
        self.canvas = ImageDraw.Draw(self.mainimage)
    def iteration(self,environment):
        self.createCanvas()
        self.fillItems(environment)
        self.canvas = ImageDraw.Draw(self.mainimage)
        self.save(self.file)
        frame = cv2.imread(self.file + '.png')
        self.video.write(frame)
        #self.mainimage.show()
        time.sleep(1 / self.framerate)
    def fat_rectangle(self,cor,width,color):
        line = (cor[0], cor[1], cor[0], cor[3])
        self.canvas.line(line, fill=color, width=width)
        line = (cor[0], cor[1], cor[2], cor[1])
        self.canvas.line(line, fill=color, width=width)
        line = (cor[0], cor[3], cor[2], cor[3])
        self.canvas.line(line, fill=color, width=width)
        line = (cor[2], cor[1], cor[2], cor[3])
        self.canvas.line(line, fill=color, width=width)
    def save(self,filename,extension=".png"):
        del self.canvas
        self.mainimage.save(filename + extension)
        print(filename+extension+" saved")
    # def savePS(self,filename):
    #     del self.canvas
    #     self.mainimage.save(filename + '.ps')
    # def getPicture(self, filename,(item_width, item_height), shrinkFactor=(1, 1), angle=0):
    #     if filename == None:
    #         return None
    #     return PIL.Image.open(fp=self.mapobjfolder + filename).resize(
    #         (int(shrinkFactor[0] * item_width), int(shrinkFactor[1] * item_height))).rotate(angle)
    def record(self):
        print("recording")
        fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
        self.file = self.videofile + str(self.record_counter)
        self.video = cv2.VideoWriter(self.file + ".avi", fourcc, 20.0, (self.frameX, self.frameY))

        #self.video.open()
        # ret, frame = cap.read()
    def stoprecord(self):


        self.video.release()

        #cv2.destroyWindow("/home/david/Videos/"+self.file + ".avi")
        print("stop recording")

    def setEnvir(self,environment):

        for x in range(environment.sizeX):
            for y in range(environment.sizeY):
                #print(environment.map[x][y].filename)
                obj=environment.get_map(x,y)
                rot=0
                if hasattr(obj,"rotation"):
                    rot=obj.rotation
                pic = self.getPicture(obj.filename,(self.item_width,self.item_height),angle=rot)

                if self.checkPicture(pic,obj):
                    self.visualmap[(x,y)]=pic
                    xx,yy = self.setCoords(x,y,(self.item_width,self.item_height),offset=(0,0))
                    #print(str((xx, yy, xx + self.item_width, yy + self.item_height)))
                    self.mainimage.paste(self.visualmap[(x, y)], (xx,yy))
                    #self.canvas.bitmap((xx, yy), self.visualmap[(x,y)])




    def addPicture(self,x,y,filename,offset=(0,0),shrink_factor=(1,1)):
        xx, yy = self.setCoords(x,y, (self.item_width, self.item_height), offset)
        self.ag_img = self.getPicture(filename, (self.item_width, self.item_height), shrink_factor)
        if self.checkPicture(self.ag_img, self):
            #print(str((xx, yy+self.item_height, xx + self.item_width, yy)))
            self.mainimage.paste(self.ag_img, (xx, yy))
            #self.canvas.bitmap((xx, yy),self.ag_img)
    def addObsPicture(self,x,y,filename,offset=(0,0),shrink_factor=(1,1)):
        pass
    def addText(self,x,y, offset, txt,font_t,font_size):
        xx, yy = self.setCoords(x, y,(self.item_width, self.item_height), offset=(0.1,0.1))
        #
        #font = tkFont.Font(family=font_t)
        self.canvas.text((xx, yy), txt,fill=(0,0,0,0))

    def getPicture(self, filename, item_shape, shrinkFactor=(1, 1), angle=0, fill=None):
        (item_width, item_height)=item_shape
        if filename == None:
            return None

        image = Image.open(fp=self.mapobjfolder + filename).resize((int(shrinkFactor[0] * item_width),
                                                                    int(shrinkFactor[1] * item_height))).rotate(angle)
        if fill is not None:
            rgb_im = image.convert('RGBA')
            width,height=image.size

            pix = rgb_im.load()
            oldcol=255,255,255,255
            for x in range(0, width):
                for y in range(0, height):
                    r, g, b, a = rgb_im.getpixel((x, y))
                    #print((r,g,b))

                    if (r,g,b,a)==oldcol:
                        #print("fill="+str(fill))
                        pix[x,y]= fill
            image = rgb_im
            # data = np.array(image)
            #
            # r1, g1, b1 = 255,255,255  # we will replace all white values
            #
            # red, green, blue = data[:, :, 0], data[:, :, 1], data[:, :, 2]
            # mask = (red == r1) & (green == g1) & (blue == b1)
            # data[:, :, :3][mask] = fill
            #
            # image = Image.fromarray(data)
        return image.convert("RGBA")



    def tick(self,environment,intermediate=False):
        if not self.on:
            if self.current_recording_interval:
                if environment.t >= self.current_recording_interval[0]:
                    self.record_counter+=1
                    self.on=True
                    self.record()
            else:
                raise Exception()
        if self.on:
            if not intermediate:
                self.iteration(environment)
            elif intermediate and self.show_intermediate_actions:
                self.iteration(environment)
        if environment.t >= self.current_recording_interval[1]:
            self.stoprecord()
            try:
                self.current_recording_interval = self.recording_intervals.pop(0)
            except:
                # run is over
                import sys
                sys.exit()
            self.on = False

class Visual(object):
    """ visualises the agent in the environment at every real time step"""



    def __init__(self,sizeX,sizeY,observationShape, frameX=500,frameY=500,buttons=True,
                 recording_intervals=None,videofile=None, tk_canvas=False):
        (observationSizeX, observationSizeY)=observationShape

        if not tk_canvas:
            #assert recording_intervals is not None
            self.display = PILImageDisplay(sizeX,sizeY,(observationSizeX,observationSizeY), frameX=frameX,frameY=frameY,
                                           recording_intervals=recording_intervals,videofile=videofile)
        else:
            self.display = CanvasDisplay(sizeX,sizeY,(observationSizeX,observationSizeY), frameX=frameX,frameY=frameY,buttons=buttons,
                                         recording_intervals=recording_intervals,videofile=videofile)


        #self.root.withdraw()
        # self.startframe = tk.Frame(self.root)

    def tick(self,environment,intermediate=False):
        self.display.tick(environment,intermediate)

        #self.record=record
        # if self.record:

        #switch_maze_button.pack()







    # def next_maze(self):
    #     self.on = False
    #     global p
    #     p = multiprocessing.Process(target=self.listen_maze, args=())
    #     p.start()
    #     e.set()
    # def listen_maze(self):
    #
    #     print("listening")
    #     if e.is_set():
    #         while True:
    #             #print("looping")
    #             #print(self.t)
    #             if self.t % SWITCHING_FREQ == 0:
    #                 print("switching maze")
    #                 self.on=True
    #                 break
    #     print("final")
    #     p.join()






# class EfficientScreenRecording(object):
#     """ visualises the agent in the environment at every real time step"""
#     on=True
#
#     mapobjfolder=str(os.environ['HOME']) + '/PycharmProjects/PhD/StatsAndVisualisation/Mapobjects/'
#
#
#     def __init__(self,sizeX,sizeY,(observationSizeX,observationSizeY), frameX=500,frameY=500,buttons=True, recording_intervals=None,videofile=None):
#         BaseVisual.__init__(self, sizeX, sizeY, (observationSizeX, observationSizeY), frameX, frameY)
#         self.root = tk.Tk()
#         self.sizeX=sizeX
#         self.sizeY=sizeY
#         print("sizes=%d,%d"%(self.sizeX,self.sizeY))
#         self.frameX=frameX
#         self.frameY=frameY
#         print("frame sizes=%d,%d" % (self.frameX, self.frameY))
#         self.item_width = frameX/sizeX
#         self.item_height = frameY/sizeY
#         print("item sizes=%d,%d" % (self.item_width, self.item_height))
#         self.obs_width = frameX/(2*observationSizeX+1)
#         self.obs_height=frameY/(2*observationSizeY+1)
#         #self.root.withdraw()
#         # self.startframe = tk.Frame(self.root)
#
#         white = (255, 255, 255)
#         image1 = Image.new("RGB", (self.frameX, self.frameY), white)
#         self.draw = ImageDraw.Draw(image1)
#
#         # do the PIL image/draw (in memory) drawings
#         self.visualmap = {}
#         self.visualobsmap = {}
#
#         self.recording_intervals=recording_intervals
#         if self.recording_intervals is not None:
#             self.on=False
#             self.record_counter=0
#             self.videofile=videofile
#             self.current_recording_interval=self.recording_intervals.pop(0)
#         #self.record=record
#         #switch_maze_button.pack()
#     def tick(self,environment,intermediate=False):
#
#         if not self.on:
#             if environment.t >= self.current_recording_interval[0]:
#                     self.record_counter+=1
#                     self.on=True
#                     self.startrecording()
#         if self.on:
#             if not intermediate:
#                 self.iteration(environment)
#             elif intermediate and self.show_intermediate_actions:
#                 self.iteration(environment)
#
#             if environment.t >= self.current_recording_interval[1]:
#                 self.stoprecording()
#                 self.current_recording_interval=self.recording_intervals.pop()
#                 if self.current_recording_interval is None:
#                     import sys
#                     sys.exit()
#                 self.on=False
#
#
#     def getPicture(self,filename,canvas,(item_width,item_height),shrinkFactor=(1,1),angle=0):
#         if filename == None:
#             return None
#         return itk.PhotoImage(PIL.Image.open(fp=self.mapobjfolder+filename).resize((int(shrinkFactor[0]*item_width), int(shrinkFactor[1]*item_height))).rotate(angle),master=canvas)
#
#     def setCoords(self,x,y,(item_width,item_height), offset=(0.5,0.5)):
#         xx = (offset[0] + x) *item_width
#         yy = (offset[1] + y) *item_height
#         return xx,yy
#
#     def checkPicture(self,pic,obj):
#         if pic == None:
#             return False
#         return True
#     def setEnvir(self,environment):
#
#         for x in range(environment.sizeX):
#             for y in range(environment.sizeY):
#                 #print(environment.map[x][y].filename)
#                 obj=environment.map[x][y]
#                 rot=0
#                 if hasattr(obj,"rotation"):
#                     rot=obj.rotation
#                 pic = self.getPicture(obj.get_filename,self.canvas,(self.item_width,self.item_height),angle=rot)
#                 if self.checkPicture(pic,obj):
#                     self.visualmap[(x,y)]=pic
#                     xx,yy = self.setCoords(x,y,(self.item_width,self.item_height))
#                     self.canvas.create_image(xx, yy, image=self.visualmap[(x, y)])
#                 # else don't add empty object
#     def setEnvironmentMap(self,environment):
#         self.setEnvir(environment)
#         environment.agent.addAgentToCanvas(environment)
#     def fillItems(self,environment):
#         self.setEnvironmentMap(environment)
#         environment.setAgentMap()
#         if self.IS_vis is not None:
#             self.IS_vis.iteration(environment.agent.learner)
#
#     def iteration(self,environment):
#         self.canvas.delete(tk.ALL)
#         self.canvas2.delete(tk.ALL)
#         self.fillItems(environment)
#         self.canvas.pack(side='left')
#         self.canvas2.pack(side='right')
#         self.root.update()
#         time.sleep(1 / self.framerate)
#
#     def record(self):
#         from pyscreenshot import grab
#         import numpy as np
#         print("recording")
#         cap = cv2.VideoCapture(0)
#         fourcc = cv2.CV_FOURCC('M','J','P','G')
#         filename=self.videofile+str(self.record_counter)+'.jpg'
#         video = cv2.VideoWriter(filename+'.avi', fourcc, 20.0, (self.frameX, self.frameY))
#         ret, frame = cap.read()
#
#
#         while (cap.isOpened()):
#             #ret, frame = cap.read()
#             # if ret == True:
#             #     frame = cv2.flip(frame, 0)
#             #
#             #     # write the flipped frame
#             #     video.write(frame)
#             #
#             #     # cv2.imshow('frame', frame)
#             #     # if cv2.waitKey(1) & 0xFF == ord('q'):
#             #     #     break
#             # if e.is_set():
#             #     cap.release()
#             #     video.release()
#             #     cv2.destroyAllWindows()
#             #     e.clear()
#             #     break
#             #capture screen and save it to file
#             img = grab(bbox=(0,0 , self.frameX,self.frameY))
#             img.save(filename)
#             frame=cv2.imread(filename)
#             video.write(frame)
#             if e.is_set():
#                 cap.release()
#                 video.release()
#                 cv2.destroyAllWindows()
#                 e.clear()
#                 break
#
#
#     def startrecording(self):
#         print("preparing recording")
#         global p
#         p = multiprocessing.Process(target=self.record, args=())
#         p.start()
#
#     #-------end video capture and stop tk
#     def stoprecording(self):
#         e.set()
#         p.join()
#
#         # self.root.quit()
#         # self.root.destroy()
#         print("stopped recording")
