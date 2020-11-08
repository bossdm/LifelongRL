#!/usr/bin/env python
from __future__ import print_function

# import skimage as skimage
# from skimage import transform, color, exposure
# from skimage.viewer import ImageViewer
import random
from random import choice
import numpy as np
from collections import deque
import time

import json
from keras.models import model_from_json
from keras.models import Sequential, load_model, Model
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.wrappers import TimeDistributed
from keras.layers import Conv2D, Dense, Flatten, merge, MaxPooling2D, Input, AveragePooling2D, Lambda, merge, Activation, Embedding
from keras.optimizers import SGD, Adam, rmsprop, Adadelta
from keras.layers.recurrent import LSTM, GRU
from keras.layers.normalization import BatchNormalization
from keras import backend as K
import tensorflow.compat.v1 as tf

#tf.python.control_flow_ops = tf

from  Catastrophic_Forgetting_NNs.TaskSpecificLayer import TaskSpecificLayer


beta=0.01 # entropy regularisation constant




def entropy_regularisation(y_true,y_pred):
    """
    move y_pred towards y_true, but add
    :param y_true:
    :param y_pred:
    :return:
    """
    #return K.sum(K.prod(K.not_equal(y_pred,0),K.prod(K.log(y_pred),y_true))) + beta*entropy(y_pred)
    return -(K.categorical_crossentropy(y_true,y_pred) + beta*K.categorical_crossentropy(y_pred,y_pred))
def entropy_bonus(prob):
    return -beta*K.categorical_crossentropy(prob,prob)

def ppo_objective(y_true,y_pred):
    pass

class CustomNetworks(object):

    @staticmethod
    def value_distribution_network(input_shape, num_atoms, action_size, learning_rate):
        """Model Value Distribution

        With States as inputs and output Probability Distributions for all Actions
        """

        state_input = Input(shape=(input_shape))
        cnn_feature = Convolution2D(32, 8, 8, subsample=(4,4), activation='relu')(state_input)
        cnn_feature = Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')(cnn_feature)
        cnn_feature = Convolution2D(64, 3, 3, activation='relu')(cnn_feature)
        cnn_feature = Flatten()(cnn_feature)
        cnn_feature = Dense(512, activation='relu')(cnn_feature)

        distribution_list = []
        for i in range(action_size):
            distribution_list.append(Dense(num_atoms, activation='softmax')(cnn_feature))

        model = Model(input=state_input, output=distribution_list)

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    @staticmethod
    def actor_network(input_shape, action_size, learning_rate):
        """Actor Network for A2C
        """

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=action_size, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    @staticmethod
    def critic_network(input_shape, value_size, learning_rate):
        """Critic Network for A2C
        """

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=value_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod
    def policy_reinforce(input_shape, action_size, learning_rate):
        """
        Model for REINFORCE
        """

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), input_shape=(input_shape)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2)))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Convolution2D(64, 3, 3))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(64))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(32))
        model.add(BatchNormalization())
        model.add(Activation('relu'))
        model.add(Dense(output_dim=action_size, activation='softmax'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='categorical_crossentropy',optimizer=adam)

        return model

    @staticmethod
    def dqn(input_shape, action_size, learning_rate):

        model = Sequential()
        model.add(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu', input_shape=(input_shape)))
        model.add(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))
        model.add(Convolution2D(64, 3, 3, activation='relu'))
        model.add(Flatten())
        model.add(Dense(output_dim=512, activation='relu'))
        model.add(Dense(output_dim=action_size, activation='linear'))

        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod
    def dueling_dqn(input_shape, action_size, learning_rate):

        state_input = Input(shape=(input_shape))
        x = Convolution2D(32, 8, 8, subsample=(4, 4), activation='relu')(state_input)
        x = Convolution2D(64, 4, 4, subsample=(2, 2), activation='relu')(x)
        x = Convolution2D(64, 3, 3, activation='relu')(x)
        x = Flatten()(x)

        # state value tower - V
        state_value = Dense(256, activation='relu')(x)
        state_value = Dense(1, init='uniform')(state_value)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], dim=-1), output_shape=(action_size,))(state_value)

        # action advantage tower - A
        action_advantage = Dense(256, activation='relu')(x)
        action_advantage = Dense(action_size)(action_advantage)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_size,))(action_advantage)

        # merge to state-action value function Q
        state_action_value = merge([state_value, action_advantage], mode='sum')

        model = Model(input=state_input, output=state_action_value)
        #model.compile(rmsprop(lr=learning_rate), "mse")
        adam = Adam(lr=learning_rate)
        model.compile(loss='mse',optimizer=adam)

        return model

    @staticmethod
    def drqn(input_shape, action_size, learning_rate,task_features,use_task_bias,use_task_gain):

        model = Sequential()
        # model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'), batch_input_shape=(input_shape)))
        # model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')))
        # model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
        model.add(TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
                                  batch_input_shape=(input_shape)))
        model.add(TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation='relu')))
        model.add(TimeDistributed(Conv2D(64, (3, 3), strides=(1,1), activation='relu')))
        model.add(TimeDistributed(Flatten()))

        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        model.add(LSTM(512,  activation='tanh'))
        if task_features:
            model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain))
        else:

            model.add(Dense(units=action_size, activation='linear'))

        #adam = Adam(lr=learning_rate)
        ada_delta = Adadelta(lr=learning_rate, rho=0.95, clipvalue=10.0)
        model.compile(loss='mse', optimizer=ada_delta)
        return model

    @staticmethod
    def feature_drqn(num_game_features,input_shape, action_size, learning_rate,task_features,use_task_bias,use_task_gain):

        x = Input(shape=input_shape, name='input')


        # model.add(TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'), batch_input_shape=(input_shape)))
        # model.add(TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu')))
        # model.add(TimeDistributed(Convolution2D(64, 3, 3, activation='relu')))
        y=TimeDistributed(Conv2D(32, (8, 8), strides=(4, 4), activation='relu'),
                                  batch_input_shape=(input_shape))(x)
        y=TimeDistributed(Conv2D(64, (4, 4), strides=(2, 2), activation='relu'))(y)
        #y=TimeDistributed(Conv2D(64, (3, 3), activation='relu'))(y)


        y1 = TimeDistributed(Dense(512))(y)  # main branch



        # Use last trace for training
        y1=TimeDistributed(Flatten())(y1)
        y1 = LSTM(512, activation='tanh')(y1)
        y1 = Dense(action_size,activation="linear")(y1)

        y2 = Flatten()(y)
        y2=Dense(512)(y2)

        y2 = Dense(num_game_features,activation="sigmoid")(y2)

        optimiser=rmsprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)

        model = Model(inputs=[x], outputs=[y1, y2])
        model.compile(loss=['mse','binary_crossentropy'],optimizer=optimiser)


        return model

    @staticmethod
    def small_scale_drqn(input_shape, action_size, task_features,use_task_bias,use_task_gain, num_neurons=80,learning_rate=.10):

        print("learning rate " +str(learning_rate))
        model = Sequential()

        model.add(Dense(output_dim=num_neurons, activation='relu',batch_input_shape=input_shape))

        # Use all traces for training
        #model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        #model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        model.add(LSTM(num_neurons,  activation='tanh'))
        # if task_features:
        #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
        # else:
        model.add(Dense(output_dim=action_size, activation='linear'))

        #cf DRQN paper
        ada_delta=Adadelta(lr=learning_rate, rho=0.95,clipvalue=10.0)
        model.compile(loss='mse',optimizer=ada_delta)

        return model

    @staticmethod
    def a2c_lstm(input_shape, action_size, value_size, learning_rate):
        """Actor and Critic Network share convolution layers with LSTM
        """

        state_input = Input(shape=(input_shape)) # 4x64x64x3
        x = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'))(state_input)
        x = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))(x)
        x = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)

        x = LSTM(512, activation='tanh')(x)

        # Actor Stream
        actor = Dense(action_size, activation='softmax')(x)

        # Critic Stream
        critic = Dense(value_size, activation='linear')(x)

        model = Model(input=state_input, output=[actor, critic])

        adam = Adam(lr=learning_rate, clipnorm=1.0)
        model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=adam, loss_weights=[1., 1.])

        return model


    @staticmethod
    def a2c_lstm(input_shape, action_size, value_size, learning_rate):
        """Actor and Critic Network share convolution layers with LSTM
        """

        state_input = Input(shape=(input_shape)) # 4x64x64x3
        x = TimeDistributed(Convolution2D(32, 8, 8, subsample=(4,4), activation='relu'))(state_input)
        x = TimeDistributed(Convolution2D(64, 4, 4, subsample=(2,2), activation='relu'))(x)
        x = TimeDistributed(Convolution2D(64, 3, 3, activation='relu'))(x)
        x = TimeDistributed(Flatten())(x)

        x = LSTM(512, activation='tanh')(x)

        # Actor Stream
        actor = Dense(action_size, activation='softmax')(x)

        # Critic Stream
        critic = Dense(value_size, activation='linear')(x)

        model = Model(input=state_input, output=[actor, critic])

        adam = Adam(lr=learning_rate, clipnorm=1.0)
        model.compile(loss=['categorical_crossentropy', 'mse'], optimizer=adam, loss_weights=[1., 1.])

        return model

    @staticmethod
    def small_scale_a2c_lstm(input_shape, action_size, value_size,num_neurons=80,learning_rate=.01):

        state_input = Input(shape=(input_shape))

        x=Dense(output_dim=num_neurons, activation='relu', batch_input_shape=input_shape)(state_input)

        # Use all traces for training
        # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
        # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))

        # Use last trace for training
        x=LSTM(num_neurons, activation='tanh')(x)
        # if task_features:
        #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
        # else:
        # Actor Stream
        actor = Dense(action_size, activation=tf.nn.softmax)(x)

        # Critic Stream
        critic = Dense(value_size, activation='linear')(x)

        model = Model(input=state_input, output=[actor, critic])

        #adam = Adam(lr=learning_rate, clipnorm=1.0)
        opt = rmsprop(lr=learning_rate,decay=.99,clipnorm=1.0)

        model.compile(optimizer=opt,loss=[entropy_regularisation,'mse'],loss_weights=[1.,1.])


        #model.compile(loss=['mse', 'mse'], optimizer=opt, loss_weights=[1., 1.])

        return model


    #
    # @staticmethod
    # def small_scale_ppo_lstm(input_shape, action_size, value_size,num_neurons=80):
    #
    #     state_input = Input(shape=(input_shape))
    #
    #     x=Dense(output_dim=num_neurons, activation='relu', batch_input_shape=input_shape)(state_input)
    #
    #     # Use all traces for training
    #     # model.add(LSTM(512, return_sequences=True,  activation='tanh'))
    #     # model.add(TimeDistributed(Dense(output_dim=action_size, activation='linear')))
    #
    #     # Use last trace for training
    #     x=LSTM(num_neurons, activation='tanh')(x)
    #     # if task_features:
    #     #     model.add(TaskSpecificLayer(task_features,use_task_bias,use_task_gain,units=action_size))
    #     # else:
    #     # Actor Stream
    #     actor = Dense(action_size, activation=tf.nn.softmax)(x)
    #
    #     # Critic Stream
    #     critic = Dense(value_size, activation='linear')(x)
    #
    #     # model = Model(input=state_input, output=[actor, critic])
    #     # target_model = Model(input=state_input, output=[actor, critic])
    #     #adam = Adam(lr=learning_rate, clipnorm=1.0)
    #
    #     #model.compile(optimizer=adam,loss=[ppo_objective,'mse'],loss_weights=[1.,1.])
    #
    #
    #     #model.compile(loss=['mse', 'mse'], optimizer=opt, loss_weights=[1., 1.])
    #
    #     return actor,critic

