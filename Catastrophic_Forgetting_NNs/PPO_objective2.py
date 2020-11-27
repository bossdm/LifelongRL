"""
NN Policy with KL Divergence Constraint (PPO / TRPO)

Written by Patrick Coady (pat-coady.github.io)
"""
import numpy as np

import tensorflow.compat.v1 as tf

import keras.backend as K

PARALLEL=False

if PARALLEL:
    import ray.experimental.tf_utils




DEBUG_MODE=True

class Policy(object):
    """ NN-based policy approximation """
    def __init__(self, obs_dim, act_dim,neurons,learning_rate,clipping,epochs,c1,c2,filename=None,w=None,large_scale=False):
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
        self.large_scale=large_scale
        if filename is None:
            self._build_graph(neurons,w)
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
        #self.print_parameters()
        with self.g.as_default():
            model_saver = tf.train.Saver()
            model_saver.restore(self.sess, filename)
        print("Model restored.")
        # print("model:")
        #self.print_parameters()
    def print_parameters(self):
        for W in self.get_all_weights():
            print(W)
    def _procedures(self):
        self._logprob()
        self._entropy()
        self._loss_train_op()

    def _restore_procedures(self):
        self._logprob()
        self._entropy()
        self._loss_train_op()
    def _build_graph(self,neurons,w=None):
        """ Build and initialize TensorFlow graph """
        self.g = tf.Graph()
        with self.g.as_default():
            self._placeholders()
            self._policy_nn(neurons,w)
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

    def small_scale_ppo_lstm(self,num_neurons,w,label):

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
        self.num_neurons=num_neurons
        # (pid=28537)(11, 80)
        # (pid=28537)(80, )
        # (pid=28537)(80, 320)
        # (pid=28537)(80, 320)
        # (pid=28537)(320, )
        # (pid=28537)(80, 5)
        # (pid=28537)(5, )
        # (pid=28537)(80, 1)
        # (pid=28537)(1, )
        if w is None:
            hidden1=tf.layers.dense(inputs=self.obs_ph, units=num_neurons, name=label+"1",activation=tf.nn.relu)
        else:
            initw1 = tf.constant_initializer(w[0])
            #initb1 = tf.constant_initializer(w[1])
            #print("bias_W1 =", w[1])
            hidden1 = tf.layers.dense(inputs=self.obs_ph, units=num_neurons, name=label + "1", activation=tf.nn.relu,
                                      kernel_initializer=initw1)
        # Use all traces for training
        # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        if w is None:
            hidden2=tf.keras.layers.LSTM(units=num_neurons, name=label+"2",activation=tf.tanh)(hidden1)
        else:
            initw2 = tf.constant_initializer(w[2])
            initr2 = tf.constant_initializer(w[3])
            #initb2 = tf.constant_initializer(w[4])
            #print("bias LSTM =", w[4])
            hidden2 = tf.keras.layers.LSTM(units=num_neurons, name=label + "2", activation=tf.tanh,
                                      kernel_initializer=initw2,recurrent_initializer=initr2)(hidden1)
        # if task_features:
        #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
        # else:
        # Actor Stream
        if w is None:
            actor = tf.layers.dense(inputs=hidden2,units=self.act_dim, name=label+"actor", activation=tf.nn.softmax)
        else:
            initw3 = tf.constant_initializer(w[5])
            #initb3 = tf.constant_initializer(w[6])
            #print("bias_act =", w[6])
            actor = tf.layers.dense(inputs=hidden2, units=self.act_dim, name=label + "actor", activation=tf.nn.softmax,
                                    kernel_initializer=initw3)

        # Critic Stream
        if w is None:
            critic = tf.layers.dense(inputs=hidden2,units=1, name=label+"3",activation=None)  # None-> linear
        else:
            initw4 = tf.constant_initializer(w[7])
            #initb4 = tf.constant_initializer(w[8])
            #print("bias critic", w[8])
            critic = tf.layers.dense(inputs=hidden2, units=1, name=label + "3", activation=None,
                                    kernel_initializer=initw4)

        return actor,critic

    def large_scale_ppo_lstm(self,num_neurons,w,label):
        # replace dense layer by:
        # model.add(TimeDistributed(Conv2D(32, (8, 8), strides=4, activation='relu'),
        #                           batch_input_shape=(input_shape)))
        # model.add(TimeDistributed(Conv2D(64, (4, 4), strides=2, activation='relu')))
        # model.add(TimeDistributed(Conv2D(64, (3, 3), strides=1, activation='relu')))

        if w is None:
            hidden1=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=8,
                                    strides=4 ,name=label+"1",activation=tf.nn.relu,
                                  batch_input_shape=self.obs_ph.shape))
        else:
            initw1 = tf.constant_initializer(w[0])
            #initb1 = tf.constant_initializer(w[1])
            #print("bias_W1 =", w[1])
            hidden1=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=32, kernel_size=8 , strides=4 ,
                                     name=label+"1",activation=tf.nn.relu,kernel_initializer=initw1,
                                  batch_input_shape=self.obs_ph.shape))

        if w is None:
            hidden1b=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(
                filters=64, kernel_size=(4, 4),
                strides=2, name=label + "1b",
                activation=tf.nn.relu))
        else:
            initw1b = tf.constant_initializer(w[2])
            #initb1 = tf.constant_initializer(w[1])
            #print("bias_W1 =", w[1])
            hidden1b=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(input_shape=hidden1.output_shape,filters=64, kernel_size=(4,4) , strides=2 ,
                                     name=label+"1b",activation=tf.nn.relu,kernel_initializer=initw1b))
        if w is None:
            hidden1c=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3),
                                                                            strides=1 ,name=label+"1c",activation=tf.nn.relu))
        else:
            initw1c = tf.constant_initializer(w[4])
            #initb1 = tf.constant_initializer(w[1])
            #print("bias_W1 =", w[1])
            hidden1c=tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(filters=64, kernel_size=(3,3) , strides=1 ,
                                     name=label+"1c",activation=tf.nn.relu,kernel_initializer=initw1c))
        if w is None:
            hidden2=tf.keras.layers.LSTM(units=256, name=label+"2",activation=tf.tanh)
        else:
            initw2 = tf.constant_initializer(w[6])
            initr2 = tf.constant_initializer(w[7])
            #initb2 = tf.constant_initializer(w[4])
            #print("bias LSTM =", w[4])
            hidden2 = tf.keras.layers.LSTM(units=512, name=label + "2", activation=tf.tanh,
                                      kernel_initializer=initw2,recurrent_initializer=initr2)

        h1 = hidden1(self.obs_ph)
        h1b = hidden1b(h1)
        h1c=hidden1c(h1b)
        h1c = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())(h1c)
        h2=hidden2(h1c)
        # if task_features:
        #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
        # else:
        # Actor Stream
        if w is None:
            actor = tf.layers.dense(inputs=h2,units=self.act_dim, name=label+"actor", activation=tf.nn.softmax)
        else:
            initw3 = tf.constant_initializer(w[8])
            #initb3 = tf.constant_initializer(w[6])
            #print("bias_act =", w[6])
            actor = tf.layers.dense(inputs=h2, units=self.act_dim, name=label + "actor", activation=tf.nn.softmax,
                                    kernel_initializer=initw3)

        # Critic Stream
        if w is None:
            critic = tf.layers.dense(inputs=h2,units=1, name=label+"3",activation=None)  # None-> linear
        else:
            initw4 = tf.constant_initializer(w[10])
            #initb4 = tf.constant_initializer(w[8])
            #print("bias critic", w[8])
            critic = tf.layers.dense(inputs=h2, units=1, name=label + "3", activation=None,
                                    kernel_initializer=initw4)

        return actor,critic

    def _policy_nn(self,num_neurons,w=None):
        """ Neural net for policy approximation function
        """
        # hidden layer sizes determined by obs_dim and act_dim (hid2 is geometric mean)
        if self.large_scale:
            self.prob, self.value = self.large_scale_ppo_lstm(num_neurons, w, label="model")
        else:
            self.prob,self.value=self.small_scale_ppo_lstm(num_neurons,w,label="model")




    def _logprob(self):

        """ Calculate log probabilities of a batch of observations & actions

        Calculates log probabilities using previous step's model parameters and
        new parameters being trained.
        """

        self.get_log_p = -K.categorical_crossentropy(self.act_ph+1e-10,self.prob+1e-10) # avoid any numerical issues by e-10

    def _entropy(self):

        self.get_entropy = K.mean(K.categorical_crossentropy(self.prob+1e-10,self.prob+1e-10)) # avoid any numerical issues by e-10

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

        if PARALLEL:
            with self.g.as_default():
                self.variables = ray.experimental.tf_utils.TensorFlowVariables(self.loss, self.sess)
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
    def get_all_weights_list(self):
        # vars = self.g.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # tvars_vals = self.sess.run(vars)
        # return tvars_vals
        return self.variables.get_weights()

    def set_all_weights(self,weights):
        print("set weights")
        # self._build_graph(self.num_neurons,tvars_vals)
        # self._init_session()
        self.variables.set_weights(weights)
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

