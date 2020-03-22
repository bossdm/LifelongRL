from keras import backend as K
from keras.layers import Dense
import numpy as np
from overrides import overrides

class TaskSpecificLayer(Dense):



    def __init__(self,
                 task_features,
                 use_task_bias,
                 use_task_gain,
                 units,
                 activation=None,
                 use_bias=False, # default false, since can be achieved by task-specific gains
                 kernel_initializer='glorot_uniform',
                 bias_initializer='zeros',
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 activity_regularizer=None,
                 kernel_constraint=None,
                 bias_constraint=None,
                 **kwargs):
        self.current_task_bias=None
        self.current_task=None
        self.current_task_gain=None
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        Dense.__init__(self,units,
                 activation=activation,
                 use_bias=use_bias,
                 kernel_initializer=kernel_initializer,
                 bias_initializer=bias_initializer,
                 kernel_regularizer=kernel_regularizer,
                 bias_regularizer=bias_regularizer,
                 activity_regularizer=activity_regularizer,
                 kernel_constraint=kernel_constraint,
                 bias_constraint=bias_constraint,
                 **kwargs)
        self.task_features=task_features
        self.use_task_bias=use_task_bias
        self.use_task_gain=use_task_gain

    @overrides
    def build(self, input_shape):



        if self.use_task_bias:
            self.task_bias={}
            for F in self.task_features:
                self.task_bias[F] = self.add_weight(shape=(self.units,),
                                        initializer=self.bias_initializer,
                                        name='task_bias%s'%(str(F)),
                                        regularizer=self.bias_regularizer,
                                        constraint=self.bias_constraint)
        else:
            self.task_bias = None

        if self.use_task_gain:
            self.task_gain = {}
            for F in self.task_features:
                self.task_gain[F] = self.add_weight(shape=(self.units,),
                                            initializer=self.bias_initializer,
                                            name='task_gain%s'%(str(F)),
                                            regularizer=self.bias_regularizer,
                                            constraint=self.bias_constraint)
        else:
            self.task_gain = None
        Dense.build(self,input_shape)


    def call(self, inputs):
        output = K.dot(inputs, self.kernel)
        if self.use_bias:
            output = K.bias_add(output, self.bias)
        if self.use_task_bias:
            output = K.bias_add(output,self.current_task_bias)
        if self.use_task_gain:
            output = output * self.current_task_gain
        if self.activation is not None:
            output = self.activation(output)
        return output
    def set_current_task(self,F):
        self.current_task=F
        if self.task_bias:
            self.current_task_bias=self.task_bias[F]
        if self.task_gain:
            self.current_task_gain=self.task_gain[F]
    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) >= 2
        assert input_shape[-1]
        output_shape = list(input_shape)
        output_shape[-1] = self.units
        return tuple(output_shape)