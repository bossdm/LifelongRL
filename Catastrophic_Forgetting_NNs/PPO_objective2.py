"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np
import tensorflow as tf

import keras.backend as K

from keras.layers import LSTM,Dense

from Catastrophic_Forgetting_NNs.CustomNetworks import CustomNetworks
from keras.models import load_model, Model
from keras.layers import Input
DEBUG_MODE=True

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim,neurons,learning_rate,clipping,epochs,c1,c2,filename=None):
        """
        Args:
            obs_dim: num observation dimensions (int)
            act_dim: num action dimensions (int)
            kl_targ: target KL divergence between pi_old and pi_new
            hid1_mult: size of first hidden layer, multiplier of obs_dim
            policy_logvar: natural log of initial policy variance
        """
        self.c_1=c1
        self.c_2=c2
        self.epochs = epochs
        self.lr = learning_rate
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.clipping = clipping
        if filename is None:
            self._build_graph(neurons)
            self._init_session()
        else:
            self._restore_session(neurons,filename)


    def save(self,name):
        with self.g.as_default():
            model_saver = tf.train.Saver()
            # Train the model and save it in the end
            model_saver.save(self.sess, name+"_session")

        print("save session")
        # print("model:")
        # self.print_some_parameters()


    def _restore_session(self,neurons,filename):
        # self.sess = tf.Session()
        # saver = tf.train.Saver()
        # saver.restore(self.sess, filename)
        # self.g=self.sess.graph
        # with self.g.as_default():
        #     self._placeholders()
        #     self._restore_network()
        #     self._procedures()
        self._build_graph(neurons)
        self._init_session()
        with self.g.as_default():
            model_saver = tf.train.Saver()
            model_saver.restore(self.sess, filename)
        print("Model restored.")
        # print("model:")
        # self.print_some_parameters()

    def _procedures(self):
        self._logprob()
        self._entropy()
        self._loss_train_op()

    def _restore_procedures(self):
        self._logprob()
        self._entropy()
        self._loss_train_op()
    def _build_graph(self,neurons):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn(neurons)
            self._procedures()
            self.init = tf.global_variables_initializer()




    def _placeholders(self):
        """ Input placeholders"""
        # observations, actions and advantages:
        self.obs_ph = tf.placeholder(tf.float32, (None,)+ self.obs_dim, 'obs_ph')
        self.act_ph = tf.placeholder(tf.float32, (None,self.act_dim), 'act_ph')
        self.advantages_ph = tf.placeholder(tf.float32, (None,), 'advantages_ph')
        # # strength of D_KL loss terms:
        # self.beta_ph = tf.placeholder(tf.float32, (), 'beta')
        # self.eta_ph = tf.placeholder(tf.float32, (), 'eta')
        # # learning rate:
        # self.lr_ph = tf.placeholder(tf.float32, (), 'eta')
        # log_vars and means with pi_old (previous step's policy parameters):
        self.logp_old = tf.placeholder(tf.float32, (None, 1), 'logp_old')
        self.observed_value = tf.placeholder(tf.float32, (None, 1), 'observed_value')

    def small_scale_ppo_lstm(self,num_neurons,label):

        #
        # x=Input(tensor=self.obs_ph)
        # x = Dense(output_dim=num_neurons, activation='relu', batch_input_shape=(None,)+self.obs_dim)(x)
        #
        # # Use all traces for training
        # # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        # # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
        #
        # # Use last trace for training
        # x = LSTM(num_neurons, activation='tanh')(x)
        # # if task_features:
        # #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
        # # else:
        # # Actor Stream
        # actor = Dense(self.act_dim, activation=tf.nn.softmax)(x)
        #
        # # Critic Stream
        # critic = Dense(1, activation='linear')(x)
        #
        # model = Model(input=self.obs_ph, output=[actor, critic])

        hidden1=tf.layers.dense(inputs=self.obs_ph, units=num_neurons, name=label+"1",activation=tf.nn.relu)

        # Use all traces for training
        # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        hidden2=tf.keras.layers.LSTM(units=num_neurons, name=label+"2",activation=tf.tanh)(hidden1)
        # if task_features:
        #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
        # else:
        # Actor Stream
        actor = tf.layers.dense(inputs=hidden2,units=self.act_dim, name=label+"actor", activation=tf.nn.softmax)

        # Critic Stream
        critic = tf.layers.dense(inputs=hidden2,units=1, name=label+"3",activation=None)  # None-> linear

        return actor,critic
    def _policy_nn(self,num_neurons):
        """ Neural net for policy approximation function

        Policy parameterized by Gaussian means and variances. NN outputs mean
         action based on observation. Trainable variables hold log-variances
         for each action dimension (i.e. variances not determined by NN).
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        self.prob,self.value=self.small_scale_ppo_lstm(num_neurons,label="model")




    def _logprob(self):
        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """

        self.get_log_p = -K.categorical_crossentropy(self.act_ph,self.prob)

    def _entropy(self):
        self.get_entropy = K.mean(K.categorical_crossentropy(self.prob,self.prob))

    # def _kl_entropy(self):
    #     """
    #     Add to Graph:
    #         1. KL divergence between old and new distributions
    #         2. Entropy of present policy given states and actions
    #
    #     https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Kullback.E2.80.93Leibler_divergence
    #     https://en.wikipedia.org/wiki/Multivariate_normal_distribution#Entropy
    #     """
    #     log_det_cov_old = tf.reduce_sum(self.old_theta_ph)
    #     log_det_cov_new = tf.reduce_sum(self.theta)
    #     tr_old_new = tf.reduce_sum(tf.exp(self.old_theta_ph - self.theta))
    #
    #     self.kl = 0.5 * tf.reduce_mean(log_det_cov_new - log_det_cov_old + tr_old_new +
    #                                    tf.reduce_sum(tf.square(self.means - self.old_means_ph) /
    #                                                  tf.exp(self.theta), axis=1) -
    #                                    self.act_dim)
    #     self.entropy = 0.5 * (self.act_dim * (np.log(2 * np.pi) + 1) +
    #                           tf.reduce_sum(self.theta))

    # def _sample(self):
    #     """ Sample from distribution, given observation """
    #     self.sampled_act = (self.means +
    #                         tf.exp(self.theta / 2.0) *
    #                         tf.random_normal(shape=(self.act_dim,)))

    def _loss_train_op(self):
        """
        Three loss terms:
            1) standard policy gradient
            2) D_KL(pi_old || pi_new)
            3) Hinge loss on [D_KL - kl_targ]^2

        See: https://arxiv.org/pdf/1707.02286.pdf
        """

        print('setting up loss with clipping objective')
        pg_ratio = tf.exp(self.get_log_p - self.logp_old)
        clipped_pg_ratio = tf.clip_by_value(pg_ratio, 1 - self.clipping, 1 + self.clipping)
        surrogate_loss = tf.minimum(self.advantages_ph * pg_ratio,
                                    self.advantages_ph * clipped_pg_ratio)
        value_loss = tf.losses.mean_squared_error(self.value,self.observed_value)
        self.loss = -tf.reduce_mean(surrogate_loss) + self.c_1*value_loss -self.c_2*self.get_entropy

        optimizer = tf.train.AdamOptimizer(self.lr)  # we will use fixed learning rate
        self.train_op = optimizer.minimize(self.loss)

    def _init_session(self):
        """Launch TensorFlow session and initialize variables"""
        self.sess = tf.Session(graph=self.g)
        self.sess.run(self.init)

    def sample(self, obs):
        """Draw sample from policy distribution"""
        feed_dict = {self.obs_ph: obs}

        return self.sess.run(self.sampled_act, feed_dict=feed_dict)

    def get_values(self,observes):
        feed_dict = {self.obs_ph: observes}
        return self.sess.run(self.value,
                      feed_dict)
    def get_probability(self,observes):
        feed_dict = {self.obs_ph: observes}
        return self.sess.run(self.prob,
                      feed_dict)

    def get_all_weights(self):
        vars = self.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        tvars_vals = self.sess.run(vars)
        weights=[]
        for val in tvars_vals:
            weights=np.append(weights,val)
        return weights

    def update_batch(self, observes, actions, advantages, discounted_rewards):
        """ Update policy based on observations, actions and advantages

        Args:
            observes: observations, shape = (N, obs_dim)
            actions: actions, shape = (N, act_dim)
            advantages: advantages, shape = (N,)

        """
        feed_dict = {self.obs_ph: observes,
                     self.act_ph: actions,
                     self.advantages_ph: advantages,
                     self.observed_value: discounted_rewards}
        logp_old = self.sess.run(self.get_log_p,
                                                      feed_dict)
        feed_dict[self.logp_old]=np.expand_dims(logp_old,axis=1)
        self.sess.run(self.train_op, feed_dict)
        loss = self.sess.run(self.loss, feed_dict)
        return loss
        # # TODO: too many "magic numbers" in next 8 lines of code, need to clean up
        # if kl > self.kl_targ * 2:  # servo beta to reach D_KL target
        #     self.beta = np.minimum(35, 1.5 * self.beta)  # max clip beta
        #     if self.beta > 30 and self.lr_multiplier > 0.1:
        #         self.lr_multiplier /= 1.5
        # elif kl < self.kl_targ / 2:
        #     self.beta = np.maximum(1 / 35, self.beta / 1.5)  # min clip beta
        #     if self.beta < (1 / 30) and self.lr_multiplier < 10:
        #         self.lr_multiplier *= 1.5
        #
        # logger.log({'PolicyLoss': loss,
        #             'PolicyEntropy': entropy,
        #             'KL': kl,
        #             'Beta': self.beta,
        #             '_lr_multiplier': self.lr_multiplier})

    def close_sess(self):
        """ Close TensorFlow session """
        self.sess.close()

