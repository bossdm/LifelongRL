"""
Q-learning with LSTM controller
"""
import lasagne
import numpy as np
import theano
import theano.tensor as T
from deep_q_rl.updates import deepmind_rmsprop
from overrides import overrides
from random import random
from Learner import CompleteLearner
from lasagne.layers.shape import ReshapeLayer

class LSTM_QLearner(CompleteLearner):



    def __init__(self, n_time, input_width, input_height, num_hidden, num_LSTM_units,
                 discount, learning_rate, rho,
                 rms_epsilon, momentum,
                 batch_size, update_rule,
                  actions, file='',clip_delta=0, input_scale=1.0):
        CompleteLearner.__init__(self,actions,file)
        self.input_width = input_width
        self.input_height = input_height
        self.num_actions =len(actions)

        self.batch_size = batch_size
        self.discount = discount
        self.rho = rho
        self.lr = learning_rate
        self.rms_epsilon = rms_epsilon
        self.momentum = momentum
        self.clip_delta = clip_delta

        self.rng = lasagne.random.get_rng()

        self.cycles=0
        self.batch_size=batch_size

        self.n_time = n_time
        #lasagne.random.set_rng(self.rng)

        self.update_counter = 0

        self.network = self.build_network((n_time,batch_size,input_width, input_height),
                                         num_hidden, num_LSTM_units, self.num_actions)

        states = T.tensor4('states')
        next_states = T.tensor4('next_states')
        rewards = T.col('rewards')
        actions = T.icol('actions')


        # Shared variables for training from a minibatch of replayed
        # state transitions, each consisting of num_frames + 1 (due to
        # overlap) images, along with the chosen action and resulting
        # reward (no terminal state)
        self.obss_shared = theano.shared(
            np.zeros((batch_size, n_time, input_height, input_width),
                     dtype=theano.config.floatX))
        self.rewards_shared = theano.shared(
            np.zeros((batch_size, 1), dtype=theano.config.floatX),
            broadcastable=(False, True))
        self.actions_shared = theano.shared(
            np.zeros((batch_size, 1), dtype='int32'),
            broadcastable=(False, True))
       # no terminal states

        # Shared variable for a single state, to calculate q_vals.
        self.state_shared = theano.shared(
            np.zeros((input_height, input_width),
                     dtype=theano.config.floatX))

        q_vals = lasagne.layers.get_output(self.network, states / input_scale)


        next_q_vals = lasagne.layers.get_output(self.network,
                                                    next_states / input_scale)
        next_q_vals = theano.gradient.disconnected_grad(next_q_vals)

        actionmask = T.eq(T.arange(self.num_actions).reshape((1, -1)),
                          actions.reshape((-1, 1))).astype(theano.config.floatX)

        target = (rewards +  self.discount * T.max(next_q_vals, axis=1, keepdims=True))
        output = (q_vals * actionmask).sum(axis=1).reshape((-1, 1))
        diff = target - output

        if self.clip_delta > 0:
            # If we simply take the squared clipped diff as our loss,
            # then the gradient will be zero whenever the diff exceeds
            # the clip bounds. To avoid this, we extend the loss
            # linearly past the clip point to keep the gradient constant
            # in that regime.
            #
            # This is equivalent to declaring d loss/d q_vals to be
            # equal to the clipped diff, then backpropagating from
            # there, which is what the DeepMind implementation does.
            quadratic_part = T.minimum(abs(diff), self.clip_delta)
            linear_part = abs(diff) - quadratic_part
            loss = 0.5 * quadratic_part ** 2 + self.clip_delta * linear_part
        else:
            loss = 0.5 * diff ** 2

        batch_accumulator='mean'

        if batch_accumulator == 'sum':
            loss = T.sum(loss)
        elif batch_accumulator == 'mean':
            loss = T.mean(loss)
        else:
            raise ValueError("Bad accumulator: {}".format(batch_accumulator))

        params = lasagne.layers.helper.get_all_params(self.network)
        train_givens = {
            states: self.obss_shared[:, :-1], #get all except the last
            next_states: self.obss_shared[:, 1:], #get all except the first
            rewards: self.rewards_shared,
            actions: self.actions_shared,
        }
        if update_rule == 'deepmind_rmsprop':
            updates = deepmind_rmsprop(loss, params, self.lr, self.rho,
                                       self.rms_epsilon)
        elif update_rule == 'rms_prop':
            updates = lasagne.updates.rmsprop(loss, params, self.lr, self.rho,
                                              self.rms_epsilon)
        elif update_rule == 'adam':
            updates = lasagne.updates.adam(loss,params,self.lr,epsilon=self.rms_epsilon)
        else:
            raise ValueError("Unrecognized update: {}".format(update_rule))

        if self.momentum > 0:
            updates = lasagne.updates.apply_momentum(updates, None,
                                                     self.momentum)

        self._train = theano.function([], [loss], updates=updates,
                                      givens=train_givens)
        q_givens = {
            states: self.state_shared.reshape((self.input_height,self.input_width))
        }
        self._q_vals = theano.function([], q_vals[0], givens=q_givens)

    def train(self):
        """
        Train one batch.

        Arguments:

        obss - b x  h x w numpy array, where b is batch size, h is height and w is width.
        actions - b x 1 numpy array of integers
        rewards - b x 1 numpy array

        Returns: average loss
        """

        self.obss_shared.set_value(self.data_set.obss)
        self.actions_shared.set_value(self.data_set.actions)
        self.rewards_shared.set_value(self.data_set.rewards)
        loss = self._train()
        self.update_counter += 1
        return np.sqrt(loss)

    def q_vals(self, state):
        self.state_shared.set_value(state)
        return self._q_vals()

    def reset_q_hat(self):
        all_params = lasagne.layers.helper.get_all_param_values(self.network)
        lasagne.layers.helper.set_all_param_values(self.next_network, all_params)

    def build_network(self, input_shape,num_hidden, num_LSTM_units, output_dim,truncation_steps=-1):
        """
        Build a large network
        """
        # gate_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        #                                                 b=lasagne.init.Constant(0.))
        # cell_parameters = lasagne.layers.recurrent.Gate(W_in=lasagne.init.Orthogonal(), W_hid=lasagne.init.Orthogonal(),
        #                                                 # Setting W_cell to None denotes that no cell connection will be used. W_cell=None, b=lasagne.init.Constant(0.), # By convention, the cell nonlinearity is tanh in an LSTM. nonlinearity=lasagne.nonlinearities.tanh)
        #

        #shape = (BATCH_SIZE, length, input_width,input_height)
        l_in = lasagne.layers.InputLayer(input_shape)

        l_lstm = lasagne.layers.LSTMLayer(l_in, num_units=num_LSTM_units)

        l_out = lasagne.layers.DenseLayer(l_lstm,output_dim,nonlinearity=None)

        return l_out

    @overrides
    def setAction(self):

        if self.step_counter  >= self.batch_size + self.n_time:
            self.epsilon = max(self.epsilon_min,self.epsilon - self.epsilon_rate)

            if self.step_counter % self.update_frequency == 0:
                    loss = self.train()
                    self.batch_counter += 1

        if self.rng.rand() < self.epsilon:
            self.action = self.rng.randint(0, self.num_actions)
        else:
            q_vals = self.q_vals(self.observation)
            self.action=np.argmax(q_vals)
        self.chosenAction = self.actions[self.action]


    def add_sample(self, obs, action, reward):
        """Add a time step record.

        Arguments:
            obs -- observed image
            action -- action chosen by the agent
            reward -- reward received after taking the action
            terminal -- boolean indicating whether the episode ended
            after this time step
        """
        self.data_set.obss.append(obs)
        self.data_set.actions.append(action)
        self.data_set.rewards.append(reward)


#     def get_batch(self, batch_size):
#         """Return corresponding imgs, actions, rewards, and terminal status for
# batch_size randomly chosen state transitions.
#
#         """
#         # Allocate the response.
#         imgs = np.zeros((batch_size,
#                          self.height,
#                          self.width),
#                         dtype='uint8')
#         actions = np.zeros((batch_size, 1), dtype='int32')
#         rewards = np.zeros((batch_size, 1), dtype='floatX')
#
#         count = 0
#         while count < batch_size:
#             # Randomly choose a time step from the replay memory.
#             index = self.rng.randint(self.bottom,
#                                      self.bottom + self.size - self.phi_length)
#
#             # Both the before and after states contain phi_length
#             # frames, overlapping except for the first and last.
#             all_indices = np.arange(index, index + self.phi_length + 1)
#             end_index = index + self.phi_length - 1
#
#
#             # Add the state transition to the response.
#             imgs[count] = self.imgs.take(all_indices, axis=0, mode='wrap')
#             actions[count] = self.actions.take(end_index, mode='wrap')
#             rewards[count] = self.rewards.take(end_index, mode='wrap')
#             count += 1
#
#         return imgs, actions, rewards


    @overrides
    def learn(self):
        self.step_counter += 1


    @overrides
    def setObservation(self, agent, environment):
        environment.setObservation(agent)

    @overrides
    def cycle(self, agent, environment):
       self.setObservation(agent,environment)
       if len(self.last_observation) >= self.n_time: self.last_observation = self.last_observation[1:]
       self.last_observation.append(self.observation)
       self.setAction()
       self.last_action = self.chosenAction
       self.setReward(environment.currentTask.reward_fun(agent, environment))
       self.add_sample(self.last_observation, self.last_action, self.r)
       self.cycles+=1

    @overrides
    def performAction(self, agent, environment):
        self.chosenAction.perform(agent,environment)


    def reset(self):
        pass


    def printPolicy(self):
        pass







if __name__ == '__main__':
    main()
