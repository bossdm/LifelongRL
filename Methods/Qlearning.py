

from random import random, randint

from Methods.Learner import CompleteLearner

import numpy as np
import theano
from keras.models import Sequential
from keras.layers.core import Activation, Dense
from keras.layers.recurrent import LSTM
from keras.optimizers import RMSprop, Adadelta, SGD
from overrides import overrides
import copy

FREEZE_INTERVAL=1000

class AlphaSchedule(object):
    def __init__(self, decay, times):
        self.decay =decay
        self.times = times
    def updateAlpha(self, alpha, t):
        if t in self.times:
            return alpha*self.decay
        else:
            return alpha
#default_alpha_schedule = AlphaSchedule(.10,[2*10**6,20*10**6])
class QLambdaLearner(CompleteLearner):
    """ Q-lambda is a variation of Q-learning that uses an eligibility trace. """

    def __init__(self, n_input,actions,file,states=None, epsilon=.05, alpha=0.02, horizon=100,gamma=0.95, qlambda=0.9,network=False,alpha_schedule=None,
                 batch_size=1000,reward_as_input=False,stateful=False):
        self.epsilon=epsilon
        self.horizon=horizon
        self.batch_size=batch_size
        self.n_input=n_input
        self.alpha = alpha
        self.gamma = gamma
        self.qlambda = qlambda
        self.stateful=stateful
        CompleteLearner.__init__(self,actions,file)
        self.numActions=len(self.actions)
        self.laststate=None
        self.state=None
        self.reward_as_input=reward_as_input
        if self.reward_as_input:
            self.n_input+=1
        self.dataset = {'state':[],'action':[],'reward': []}
        self.nn = None
        self.target_nn=None
        self.t=0
        self.states_batch=[]
        self.targets_batch=[]
        self.alpha_schedule=alpha_schedule
        if network:
            self.nn , self.params= self.initNetwork()
            self.target_nn = self.nn
            self.target_params=self.params
        else:
            self.states=states
            self.Q = {}
            for state in self.states:
                self.Q[state]=[]
                for a in range(self.numActions):
                    self.Q[state].append(self.gamma + random())
    def initNetwork(self):
        nn = Sequential()
        nn.add(Dense(self.n_input,batch_input_shape=(1,self.horizon-1,self.n_input,)))
        # self.nn.add(Activation('relu'))
        # model.add(Dropout(0.2)) I'm not using dropout, but maybe you waself.nna give it a try?
        # nn.add(Dense(20))
        # nn.add(Activation('relu'))
        nn.add(LSTM(20,stateful=self.stateful))
        nn.add(Activation('relu'))
        # model.add(Dropout(0.2))
        nn.add(Dense(20))
        nn.add(Activation('relu'))
        nn.add(Dense(output_dim=self.numActions, init='lecun_uniform'))
        nn.add(Activation('linear'))  # linear output so we can have range of real-valued outputs
        #opt = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        opt = Adadelta(lr=1.0, rho=0.95, epsilon=1e-08, decay=0.0) #== default settings
        nn.compile(loss='mse', optimizer=opt)

        layers = nn.layers
        # Grab all the parameters together.
        params = [param
                       for layer in layers
                       for param in layer.trainable_weights]
        return nn, params

    def getAllParams(self):
        params_value=[]
        for i,p in enumerate(self.params):
            params_value.append(p.get_value())
        return params_value

    def setAllParams(self, list_of_values):
        for i,p in enumerate(self.params):
            p.set_value(list_of_values[i])
    def _resetQHat(self):
        for i, (param, target_param) in enumerate(zip(self.params, self.target_params)):
            K.set_value(target_param, K.get_value(param))
    def setAction(self): #determine self.chosenAction
        if self.nn and self.t % FREEZE_INTERVAL == 0:
            self._resetQHat()
        if random() < self.epsilon or self.t <= self.horizon:
             self.lastaction = randint(0,self.numActions-1)
        else:
             self.lastaction = np.argmax(self.getValues())
        self.chosenAction = self.actions[self.lastaction]
    def getValues(self):
        if self.nn:
            return self.nn.predict(self.states, batch_size=1).flatten()
        else:
            return self.Q[tuple(self.laststate)]
    def getTargetValues(self):
        if self.target_nn:
            return self.target_nn.predict(self.next_states, batch_size=1).flatten()
        else:
            return self.Q[tuple(self.state)]
    def getAllActivations(self):
        activations=[]
        for layer in self.nn.layers:
            activations.append(layer.output)
        return activations
    def setAllActivations(self,activations):
        i=0
        for layer in self.nn.layers:
            layer.output=activations[i]
            i+=1
    def updateValue(self,laststate, state, lastaction,lastreward,terminal=False,lbda=1):
        # if  len(self.states) % self.batch_size != 0: # self.t < self.horizon or
        #
        #     return
        qvalues = self.getValues()
        #print("curr vals" + str(qvalues))
        if qvalues is None: return
        if terminal:
            maxnext=0
        else:
            next_qvalues = self.getTargetValues()
            #max_q_index = np.argmax(next_qvalues)
            maxnext = max(next_qvalues)

            #print("next vals" + str(next_qvalues))


        if self.nn:
            update = (lastreward + (self.gamma * maxnext))
            qvalues[lastaction] = update
            #old_activations=self.getAllActivations()
            self.nn.train_on_batch(self.states,
                                   qvalues.reshape(1, self.numActions))
            # self.states_batch.append(self.state)
            # self.targets_batch.append(qvalues)
            #self.setAllActivations(old_activations)

        else:
            qvalue = qvalues[lastaction]
            self.Q[laststate][lastaction]=qvalue + self.alpha * lbda * (lastreward + self.gamma * maxnext - qvalue)
    # def train_batch(self):
    #     for i in range(len(self.batch_size)):
    #         self.nn.train_on_batch(self.states_batch[i],
    #                                self.targets_batch[i].reshape(1, self.numActions))
    #     self.nn.reset_states()
    #     # self.nn.fit(self.states_batch, self.targets_batch.reshape(self.batch_size,1,self.numActions), batch_size=1, nb_epoch=1, verbose=0)
    #     self.states_batch = []
    #     self.targets_batch = []
    def Qlambda_learn(self):

        states = self.dataset['state']
        if (len(states) < self.horizon):
            return
        assert len(states)==self.horizon
        actions = self.dataset['action']
        rewards = self.dataset['reward']

        for i in range(self.horizon - 1, 0, -1):
            lbda = self.qlambda ** (self.horizon - 1 - i)
            # if eligibility trace gets too long, break
            if lbda < 0.0001:
                break

            state = states[i]
            laststate = states[i - 1]
            # action = int(actions[i])
            lastaction = int(actions[i - 1])
            lastreward = rewards[i - 1]
            self.updateValue(laststate, state, lastaction, lastreward, lbda)
    def Qlearn(self):
        laststate = self.states if self.nn else self.laststate
        state = self.next_states if self.nn else self.state
        self.updateValue(laststate, state,self.lastaction, self.r)

    @overrides
    def learn(self):
        if self.qlambda > 0:
            self.Qlambda_learn()
        else:
            if self.t >= self.horizon:
                self.Qlearn()
    @overrides
    def setObservation(self,agent,environment):
        environment.setObservation(agent)
        self.t = environment.t
        self.laststate = copy.copy(self.state)
        add=[self.r] if self.reward_as_input else []
        self.state = tuple(self.observation + add)




        # print("last"+str(self.laststate))
        # print("current"+str(self.state))
        if len(self.dataset['state']) >= self.horizon:
            self.dataset['state']=self.dataset['state'][1:]
            #self.dataset['action'] = self.dataset['action'][1:]
            self.dataset['reward'] = self.dataset['reward'][1:]
        self.dataset['state'].append(self.state)
        if self.laststate != None:
            self.dataset['reward'].append(self.r)
            #self.dataset['action'].append(self.lastaction)
        if self.nn and self.t >= self.horizon :
            self.states=np.array(self.dataset['state'][:-1]).reshape(1,self.horizon-1,self.n_input)
            self.next_states=np.array(self.dataset['state'][1:]).reshape(1,self.horizon-1,self.n_input)

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
        pass
    def setReward(self,environment):
        self.r = environment.currentTask.reward_fun(environment.agent, environment)
        self.R +=self.r

