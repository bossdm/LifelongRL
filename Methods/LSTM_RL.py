

from random import random, randint

from Methods.Learner import CompleteLearner

import numpy as np
from pybrain.structure.networks.feedforward import FeedForwardNetwork
from pybrain.structure.networks.recurrent import RecurrentNetwork
from pybrain.structure.modules import BiasUnit, SigmoidLayer, LinearLayer, LSTMLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.datasets import SupervisedDataSet,SequentialDataSet
from pybrain.structure.connections import FullConnection, IdentityConnection
from pybrain.auxiliary.gradientdescent import GradientDescent
from overrides import overrides
import copy
FREEZE_INTERVAL=1000
def clip(a,min,max):
    if a < min:
        return min
    elif a > max:
        return max
    else:
        return a
def safe_exp(list):
    list=[clip(l,-500,500) for l in list]
    return np.exp(np.array(list))
def safe_division(nums,num2):
    return  np.log(x.clip(min=minval))


    #     Skip
    #     to
    #     content
    #     Features
    #     Business
    #     Explore
    #     Marketplace
    #     Pricing
    #     This
    #     repository
    #     Sign in or Sign
    #     up
    #
    #     218
    #     2, 319
    #
    #     699
    #
    # pybrain / pybrain
    # Code
    # Issues
    # 109
    # Pull
    # requests
    # 24
    # Projects
    # 0
    # Wiki
    # pybrain / pybrain / supervised / trainers / backprop.py
    # 24
    # d39f9
    # on
    # 17
    # May
    # 2015
    #
    # @theno
    #
    # theno
    # Fix
    # incompatibility
    # with Python 3 using //
    #
    # @schaul
    # @bayerj
    # @abbgrade
    # @rueckstiess
    # @mmaker
    # @wernight
    # @theno
    # @ranma42
    #
    # 261
    # lines(235
    # sloc) 10.6
    # KB
    # from __future__ import print_function
    #
    # __author__ = 'Daan Wierstra and Tom Schaul'
    #
    # from scipy import dot, argmax
    # from random import shuffle
    # from math import isnan
    # from pybrain.supervised.trainers.trainer import Trainer
    # from pybrain.utilities import fListToString
    # from pybrain.auxiliary import GradientDescent
    #
    # class BackpropTrainer(Trainer):
    #     """Trainer that trains the parameters of a module according to a
    #     supervised dataset (potentially sequential) by backpropagating the errors
    #     (through time)."""
    #
    #     def __init__(self, module, dataset=None, learningrate=0.01, lrdecay=1.0,
    #                  momentum=0., verbose=False, batchlearning=False,
    #                  weightdecay=0.):
    #         """Create a BackpropTrainer to train the specified `module` on the
    #         specified `dataset`.
    #         The learning rate gives the ratio of which parameters are changed into
    #         the direction of the gradient. The learning rate decreases by `lrdecay`,
    #         which is used to to multiply the learning rate after each training
    #         step. The parameters are also adjusted with respect to `momentum`, which
    #         is the ratio by which the gradient of the last timestep is used.
    #         If `batchlearning` is set, the parameters are updated only at the end of
    #         each epoch. Default is False.
    #         `weightdecay` corresponds to the weightdecay rate, where 0 is no weight
    #         decay at all.
    #         """
    #         Trainer.__init__(self, module)
    #         self.setData(dataset)
    #         self.verbose = verbose
    #         self.batchlearning = batchlearning
    #         self.weightdecay = weightdecay
    #         self.epoch = 0
    #         self.totalepochs = 0
    #         # set up gradient descender
    #         self.descent = GradientDescent()
    #         self.descent.alpha = learningrate
    #         self.descent.momentum = momentum
    #         self.descent.alphadecay = lrdecay
    #         self.descent.init(module.params)
    #
    #     def train(self):
    #         """Train the associated module for one epoch."""
    #         assert len(self.ds) > 0, "Dataset cannot be empty."
    #         self.module.resetDerivatives()
    #         errors = 0
    #         ponderation = 0.
    #         shuffledSequences = []
    #         for seq in self.ds._provideSequences():
    #             shuffledSequences.append(seq)
    #         shuffle(shuffledSequences)
    #         for seq in shuffledSequences:
    #             e, p = self._calcDerivs(seq)
    #             errors += e
    #             ponderation += p
    #             if not self.batchlearning:
    #                 gradient = self.module.derivs - self.weightdecay * self.module.params
    #                 new = self.descent(gradient, errors)
    #                 if new is not None:
    #                     self.module.params[:] = new
    #                 self.module.resetDerivatives()
    #
    #         if self.verbose:
    #             print("Total error: {z: .12g}".format(z=errors / ponderation))
    #         if self.batchlearning:
    #             self.module._setParameters(self.descent(self.module.derivs))
    #         self.epoch += 1
    #         self.totalepochs += 1
    #         return errors / ponderation
    #
        # def _calcDerivs(self, seq):
        #     """Calculate error function and backpropagate output errors to yield
        #     the gradient."""
        #     self.module.reset()
        #     for sample in seq:
        #         self.module.activate(sample[0])
        #     error = 0
        #     ponderation = 0.
        #     for offset, sample in reversed(list(enumerate(seq))):
        #         # need to make a distinction here between datasets containing
        #         # importance, and others
        #         target = sample[1]
        #         outerr = target - self.module.outputbuffer[offset]
        #         if len(sample) > 2:
        #             importance = sample[2]
        #             error += 0.5 * dot(importance, outerr ** 2)
        #             ponderation += sum(importance)
        #             self.module.backActivate(outerr * importance)
        #
        #         else:
        #             error += 0.5 * sum(outerr ** 2)
        #             ponderation += len(target)
        #             # FIXME: the next line keeps arac from producing NaNs. I don't
        #             # know why that is, but somehow the __str__ method of the
        #             # ndarray class fixes something,
        #             str(outerr)
        #             self.module.backActivate(outerr)
        #
        #     return error, ponderation
    #
    #     def _onlineCalcDerivs(self, target):
    #         """Calculate error function given that the module activated on the most recent sample"""
    #         #self.module.reset() no reset
    #         # for sample in seq:
    #         #     self.module.activate(sample[0])
    #         error = 0
    #         ponderation = 0.
    #
    #         # need to make a distinction here between datasets containing
    #         # importance, and others
    #         outerr = target - self.module.outputbuffer[self.module.maxoffset]
    #         error += 0.5 * sum(outerr ** 2)
    #         self.module.backActivate(outerr)
    #
        # def _checkGradient(self, dataset=None, silent=False):
        #     """Numeric check of the computed gradient for debugging purposes."""
        #     if dataset:
        #         self.setData(dataset)
        #     res = []
        #     for seq in self.ds._provideSequences():
        #         self.module.resetDerivatives()
        #         self._calcDerivs(seq)
        #         e = 1e-6
        #         analyticalDerivs = self.module.derivs.copy()
        #         numericalDerivs = []
        #         for p in range(self.module.paramdim):
        #             storedoldval = self.module.params[p]
        #             self.module.params[p] += e
        #             righterror, dummy = self._calcDerivs(seq)
        #             self.module.params[p] -= 2 * e
        #             lefterror, dummy = self._calcDerivs(seq)
        #             approxderiv = (righterror - lefterror) / (2 * e)
        #             self.module.params[p] = storedoldval
        #             numericalDerivs.append(approxderiv)
        #         r = list(zip(analyticalDerivs, numericalDerivs))
        #         res.append(r)
        #         if not silent:
        #             print(r)
        #     return res
    #
    #     def testOnData(self, dataset=None, verbose=False):
    #         """Compute the MSE of the module performance on the given dataset.
    #         If no dataset is supplied, the one passed upon Trainer initialization is
    #         used."""
    #         if dataset == None:
    #             dataset = self.ds
    #         dataset.reset()
    #         if verbose:
    #             print('\nTesting on data:')
    #         errors = []
    #         importances = []
    #         ponderatedErrors = []
    #         for seq in dataset._provideSequences():
    #             self.module.reset()
    #             e, i = dataset._evaluateSequence(self.module.activate, seq, verbose)
    #             importances.append(i)
    #             errors.append(e)
    #             ponderatedErrors.append(e / i)
    #         if verbose:
    #             print(('All errors:', ponderatedErrors))
    #         assert sum(importances) > 0
    #         avgErr = sum(errors) / sum(importances)
    #         if verbose:
    #             print(('Average error:', avgErr))
    #             print(('Max error:', max(ponderatedErrors), 'Median error:',
    #                    sorted(ponderatedErrors)[len(errors) // 2]))
    #         return avgErr
    #
    #     def testOnClassData(self, dataset=None, verbose=False,
    #                         return_targets=False):
    #         """Return winner-takes-all classification output on a given dataset.
    #         If no dataset is given, the dataset passed during Trainer
    #         initialization is used. If return_targets is set, also return
    #         corresponding target classes.
    #         """
    #         if dataset == None:
    #             dataset = self.ds
    #         dataset.reset()
    #         out = []
    #         targ = []
    #         for seq in dataset._provideSequences():
    #             self.module.reset()
    #             for input, target in seq:
    #                 res = self.module.activate(input)
    #                 out.append(argmax(res))
    #                 targ.append(argmax(target))
    #         if return_targets:
    #             return out, targ
    #         else:
    #             return out
    #
    #     def trainUntilConvergence(self, dataset=None, maxEpochs=None, verbose=None,
    #                               continueEpochs=10, validationProportion=0.25,
    #                               trainingData=None, validationData=None,
    #                               convergence_threshold=10):
    #         """Train the module on the dataset until it converges.
    #         Return the module with the parameters that gave the minimal validation
    #         error.
    #         If no dataset is given, the dataset passed during Trainer
    #         initialization is used. validationProportion is the ratio of the dataset
    #         that is used for the validation dataset.
    #
    #         If the training and validation data is already set, the splitPropotion is ignored
    #         If maxEpochs is given, at most that many epochs
    #         are trained. Each time validation error hits a minimum, try for
    #         continueEpochs epochs to find a better one."""
    #         epochs = 0
    #         if dataset is None:
    #             dataset = self.ds
    #         if verbose is None:
    #             verbose = self.verbose
    #         if trainingData is None or validationData is None:
    #             # Split the dataset randomly: validationProportion of the samples for
    #             # validation.
    #             trainingData, validationData = (
    #                 dataset.splitWithProportion(1 - validationProportion))
    #         if not (len(trainingData) > 0 and len(validationData)):
    #             raise ValueError("Provided dataset too small to be split into training " +
    #                              "and validation sets with proportion " + str(validationProportion))
    #         self.ds = trainingData
    #         bestweights = self.module.params.copy()
    #         bestverr = self.testOnData(validationData)
    #         bestepoch = 0
    #         self.trainingErrors = []
    #         self.validationErrors = [bestverr]
    #         while True:
    #             trainingError = self.train()
    #             validationError = self.testOnData(validationData)
    #             if isnan(trainingError) or isnan(validationError):
    #                 raise Exception("Training produced NaN results")
    #             self.trainingErrors.append(trainingError)
    #             self.validationErrors.append(validationError)
    #             if epochs == 0 or self.validationErrors[-1] < bestverr:
    #                 # one update is always done
    #                 bestverr = self.validationErrors[-1]
    #                 bestweights = self.module.params.copy()
    #                 bestepoch = epochs
    #
    #             if maxEpochs != None and epochs >= maxEpochs:
    #                 self.module.params[:] = bestweights
    #                 break
    #             epochs += 1
    #
    #             if len(self.validationErrors) >= continueEpochs * 2:
    #                 # have the validation errors started going up again?
    #                 # compare the average of the last few to the previous few
    #                 old = self.validationErrors[-continueEpochs * 2:-continueEpochs]
    #                 new = self.validationErrors[-continueEpochs:]
    #                 if min(new) > max(old):
    #                     self.module.params[:] = bestweights
    #                     break
    #                 lastnew = round(new[-1], convergence_threshold)
    #                 if sum(round(y, convergence_threshold) - lastnew for y in new) == 0:
    #                     self.module.params[:] = bestweights
    #                     break
    #         # self.trainingErrors.append(self.testOnData(trainingData))
    #         self.ds = dataset
    #         if verbose:
    #             print(('train-errors:', fListToString(self.trainingErrors, 6)))
    #             print(('valid-errors:', fListToString(self.validationErrors, 6)))
    #         return self.trainingErrors[:bestepoch], self.validationErrors[:1 + bestepoch]





    #default_alpha_schedule = AlphaSchedule(.10,[2*10**6,20*10**6])
class RL_LSTM(CompleteLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """
    temp_scale=1.
    def __init__(self, n_input,actions,file,epsilon=.05, kappa=.1, alpha=0.0002, horizon=100,gamma=0.98, qlambda=0.8,reward_as_input=False):
        self.epsilon=epsilon
        self.horizon=horizon

        self.alpha = alpha
        self.gamma = gamma
        self.qlambda = qlambda
        self.kappa=kappa
        CompleteLearner.__init__(self,actions,file)
        self.numActions=len(self.actions)
        self.laststate=None
        self.state=None
        self.dataset = {'state':[],'action':[],'reward': []}
        self.nn = None
        self.target_nn=None
        self.lastaction=None
        self.t=0
        self.reward_as_input=reward_as_input
        self.n_input = n_input+1 if self.reward_as_input else n_input
        self.next_advantages=[0 for i in range(self.numActions)]

        self.nn = self.buildLSTMNetwork()
        self.td_nn=self.buildTDnetwork()
        self.td_trainer = BackpropTrainer(self.td_nn)
        self.TDnext=[0]


    def buildLSTMNetwork(self):
        # create network and modules
        net = RecurrentNetwork()
        inp = LinearLayer(self.n_input,name="Input")
        h1 = LSTMLayer(3,name='LSTM')
        h2 = SigmoidLayer(10,name='sigm')
        outp = LinearLayer(self.numActions,name='output')
        # add modules
        net.addOutputModule(outp)
        net.addInputModule(inp)
        net.addModule(h1)
        net.addModule(h2)
        # create connections from input
        net.addConnection(FullConnection(inp, h1,name="input_LSTM"))
        net.addConnection(FullConnection(inp, h2,name="input_sigm"))
        # create connections from LSTM
        net.addConnection(FullConnection(h1,h2,name="LSTM_sigm"))

        # add whichever recurrent connections
        net.addRecurrentConnection(FullConnection(h1, h1, name='LSTM_rec'))
        net.addRecurrentConnection(FullConnection(h2, h1, name='sigm_LSTM_rec'))
        # create connections to output
        net.addConnection(FullConnection(h1, outp, name="LSTM_outp"))
        net.addConnection(FullConnection(h2,outp,name="sigm_outp"))



        # finish up
        net.sortModules()
        net.randomize()
        self.printModules(net)
        self.e=[0 for param in range(len(net.params))]
        # for each action, need to accumulate the gradient
        self.accumulated_gradients = [[0 for param in range(len(net.params))] for i in range(self.numActions)]
        return net
    def buildTDnetwork(self):
        # create network and modules
        net = FeedForwardNetwork()
        inp = LinearLayer(self.n_input,name="Input")
        h1 = SigmoidLayer(10,name='sigm')
        outp = LinearLayer(1,name='output')
        # add modules
        net.addOutputModule(outp)
        net.addInputModule(inp)
        net.addModule(h1)
        # create connections from input
        net.addConnection(FullConnection(inp, h1,name="input_LSTM"))

        # create connections to output
        net.addConnection(FullConnection(h1, outp, name="LSTM_outp"))

        # finish up
        net.sortModules()
        net.randomize()

        return net
    def select_Egreedy(self):
        if random() < self.epsilon or self.lastaction is None:
             self.lastaction = randint(0,self.numActions-1)
        #      self.e = [0 for param in range(len(self.nn.params))] #exploratory action taken --> reset eligibilitie
        else:
             self.lastaction = self.maxind

    def selectBoltz(self):
        if self.lastaction is None:
            self.lastaction=randint(0,self.numActions-1)
            return
        if self.temperature == 0:
            probs=[1/float(self.numActions) for i in range(self.numActions)]
        else:
            probs = [adv/ self.temperature for adv in self.current_advantages]
        probs = safe_exp(probs)
        C = sum(probs)
        probs = probs / C
        print(probs)
        r = random()
        prob = 0
        for i in range(len(probs)):
            prob += probs[i]
            if r <= prob:
                self.lastaction=i
                return
    def setAction(self): #determine self.chosenAction
        self.selectBoltz()
        self.chosenAction = self.actions[self.lastaction]
        print("chosenAction="+str(self.chosenAction.function.__name__))

    def Advantages(self):
        self.current_advantages = copy.copy(self.next_advantages)
        self.next_advantages=self.nn.activate(self.state)
        print("curr A"+str(self.current_advantages))
        print("next A"+str(self.next_advantages))

    def predictTD(self):
        """ predict the TD"""
        self.TDcurr=self.TDnext[0]
        self.temperature=abs(self.TDcurr)*self.temp_scale
        self.TDnext=self.td_nn.activate(self.state)
        print("curr TD" + str(self.TDcurr))
        print("next TD" + str(self.TDnext))
        print("temp " + str(self.temperature))
    def getTD_Error(self,terminal):
        self.current=self.current_advantages[self.lastaction]
        self.maxind=np.argmax(self.current_advantages)
        self.maxnext=max(self.next_advantages) if not terminal else 0
        return self.current_advantages[self.maxind] + (self.r+self.gamma*self.maxnext - self.current_advantages[self.maxind])/self.kappa - self.current

    #
    # for offset, sample in reversed(list(enumerate(seq))):
    #     # need to make a distinction here between datasets containing
    #     # importance, and others
    #     target = sample[1]
    #     outerr = target - self.module.outputbuffer[offset]
    #     if len(sample) > 2:
    #         importance = sample[2]
    #         error += 0.5 * dot(importance, outerr ** 2)
    #         ponderation += sum(importance)
    #         self.module.backActivate(outerr * importance)
    #     else:
    #         error += 0.5 * sum(outerr ** 2)
    #         ponderation += len(target)
    #         # FIXME: the next line keeps arac from producing NaNs. I don't
    #         # know why that is, but somehow the __str__ method of the
    #         # ndarray class fixes something,
    #         str(outerr)
    #         self.module.backActivate(outerr)
    def onlineCalcDerivs(self):
        """Calculate derivative of the output given that the module activated on the most recent sample"""
        # self.module.reset() no reset
        # for sample in seq:
        #     self.module.activate(sample[0])
        # need to make a distinction here between datasets containing
        # importance, and others
        #outerr = target - self.nn.outputbuffer[self.nn.maxoffset
        self.nn.backActivate(self.next_advantages)
        # self.accumulated_gradients[self.lastaction] += self.nn.derivs
    def getDerivFor(self,p):
        # get the derivative of d_output[lastaction]/d_param[p]
        return self.nn.derivs[p]
    def eligibility_traces(self,terminal):
        # assumes error was backpropagated, but  weights not updated yet
        self.TD=self.getTD_Error(terminal)
        for p in range(len(self.nn.params)):
            param=self.nn.params[p]
            self.e[p] = self.gamma*self.qlambda*self.e[p] + self.getDerivFor(p)
            param += self.alpha*self.TD*self.e[p]
        self.nn._setParameters(self.nn.params)

    def updateValue(self,terminal=False):
        if self.lastaction is not None:
            self.Advantages()
            self.onlineCalcDerivs()
            self.eligibility_traces(terminal)

    def updateTemperature(self):
        if self.lastaction is not None:
            self.predictTD()
            target=abs(self.TD)+self.gamma*self.TDnext
            outerr = target - self.td_nn.outputbuffer[-1]
            self.td_nn.backActivate(outerr)
            self.td_trainer.ds=SupervisedDataSet(self.n_input, 1)
            self.td_trainer.ds.addSample(self.laststate,[outerr])
            self.td_trainer.train()

    @overrides
    def learn(self):
        self.updateValue()
        self.updateTemperature()

    @overrides
    def setObservation(self,agent,environment):
        environment.setObservation(agent)
        self.t = environment.t
        self.laststate = copy.copy(self.state)
        added = [self.r] if self.reward_as_input else []
        self.state = tuple(self.observation +  added)

        print("last"+str(self.laststate))
        print("current"+str(self.state))


        # if len(self.dataset['state']) >= self.horizon:
        #     self.dataset['state']=self.dataset['state'][1:]
        #     #self.dataset['action'] = self.dataset['action'][1:]
        #     self.dataset['reward'] = self.dataset['reward'][1:]
        # self.dataset['state'].append(self.state)
        # if self.laststate:
        #     self.dataset['reward'].append(self.r)
        #     #self.dataset['action'].append(self.lastaction)
        # if self.nn and self.t >= self.horizon :
        #     self.states=np.array(self.dataset['state'][:-1]).reshape(1,self.horizon-1,self.n_input)
        #     self.next_states=np.array(self.dataset['state'][1:]).reshape(1,self.horizon-1,self.n_input)

    @overrides
    def cycle(self,agent,environment):
        self.setObservation(agent,environment)
        if self.laststate != None:
            self.learn()
        self.setAction()
        self.performAction(agent,environment)
        self.setReward(environment)
    @overrides
    def printPolicy(self):
        pass



    def performAction(self, agent, environment):
        self.chosenAction.perform([agent,environment])
        #self.alpha = self.alpha_schedule.updateAlpha(self.alpha, self.t)

    def reset(self,terminal=False):
        # initialise states to the null observation, the rewards to zero
        # self.dataset['state']=[[0 for j in range(self.n_input)] for i in range(self.horizon)]
        # self.dataset['reward']=[0 for i in range(self.horizon)]
        self.updateValue(terminal)
        self.state=[0 for j in range(self.n_input)]
        self.nn.reset()
    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r
    def printModules(self,net):
        for mod in net.modules:
            print("Module:"+str(mod.name)+" with "+str(mod.dim) + " units")
            if mod.paramdim > 0:
                print("--parameters:", mod.params)
            for conn in net.connections[mod]:
                print("-connection to", conn.outmod.name)
                if conn.paramdim > 0:
                    self.printConnections(conn)
                print("Recurrent connections")
                for conn in net.recurrentConns:
                    print("-", conn.inmod.name, " to", conn.outmod.name)
                    self.printConnections(conn)
    def printConnections(self,conn):
        for con in range(len(conn)):
            print(conn.whichBuffers(con), conn.params[con])
if __name__ == '__main__':
    learner=QLambdaLearner(5, [1,2,3], "",network=True)
