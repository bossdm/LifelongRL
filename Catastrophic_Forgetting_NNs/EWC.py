"""

Make the loss function of Elastic Weight Consolidation;


L(theta) = L_b(theta) +


Fischer matrix calculated as in :
Pascanu, R., & Bengio, Y. (2013). Revisiting Natural Gradient for Deep Networks

Key assumption:
true value t_i ~ N(t|y_i,\beta**2) for all outputs i


---> p(D|theta) = density of the data given the parameters
                = prod_i N(t_i | y_i, \beta**2)

---> log p(D|theta) = sum_i log N(t_i|y_i, \beta**2)




TODO:

    -very small number of neurons-> gradients well-defined, large number --> gradients 0 or nan
    --> to do with log-likelihoods (log(0)=inf)
    -EWC penalty has no effect



"""
import os,sys
os.environ["CUDA_VISIBLE_DEVICES"]="-1"

import keras

import tensorflow as tf

from keras.models import Model, Sequential
from keras.layers import Dense, InputLayer, Input

import numpy as np

from Catastrophic_Forgetting_NNs.gradient_calcs import *
from copy import deepcopy

from keras.optimizers import RMSprop

from scipy.stats import norm
DEBUG_MODE=False
class OutputType:
    linear=0
    softmax=1
    sigmoid=2
class ObjectiveType:

    likelihood=0
    EWC=1
    EWC_predEval=2


class GaussianLikelihood(keras.layers.Layer):
    def __init__(self, y_true, n_out, beta, **kwargs):
        super(GaussianLikelihood, self).__init__(**kwargs)
        self.beta = beta
        self.n_out= n_out
        self.y_true = y_true
    def build(self, input_shape):
        pass
    def gaussian(self,z , mean):
        """
        multivariate gaussian each with the same variance but different mean
        :param z:
        :param mean:
        :param var:
        :param z_dim:
        :return:
        """
        var = self.beta
        z_dim = self.n_out
        return 1/np.sqrt(var*(2*np.pi)**z_dim)*K.exp(-0.5*var**(-1)*K.square(z-mean))

    def log_likelihood(self,y_true,y_pred):
        """
        toy function, code works like this
        :param y_true:
        :param y_pred:
        :return:
        """
        temp=K.log(self.gaussian(z=y_true,mean=y_pred))
        return K.sum(temp,axis=[0,1])

    def call(self, inputs):

        return self.log_likelihood(self.y_true,inputs)

    def get_config(self):
        # Implement get_config to enable serialization. This is optional.
        pass




class EWC_objective(object):
    n = 10000
    def __init__(self,lbda_task,learning_rate,batch_size,n_in, n_out,lbda=400.0,output_type=OutputType.linear,epochs=1,
                 objective_type=ObjectiveType.EWC,occurrence_weights=None):
        """

        :param out: output tensor
        :param y_pred:
        :param y_true:
        :param lbda:
        """
        self.current_task=None
        self.training_epochs=epochs
        self.learning_rate=learning_rate
        self.batch_size=batch_size
        self.n_in = n_in
        self.n_out = n_out
        #self.gradient_funs = init_gradient_calcs(self.model,self.theta,self.n_out)
        self.task_t={}

        self.lbda = lbda
        # self.x = K.placeholder(shape=(None,n_in))
        # self.y = self.model(self.x)


        # standard deviation used to compute the hessian ????
        self.output_type=output_type
        self.theta_star={}
        self.lbda_task=lbda_task
        self.occurrence_weights=occurrence_weights

        self.objective_type = objective_type


    def print_current_theta_and_thetastars(self):
        for F in self.theta_star:
            for v in range(len(self.theta)):
                theta_val = K.eval(self.theta[v])
                print("theta%d = %s" % (v, theta_val))
                print("theta_star%s,%d = %s" % (str(F),v, K.eval(self.theta_star[F][v])))
    def update_thetastar(self,F):
        # used for saving optimal weights after most recent task training
        self.theta_star[F] = []
        if DEBUG_MODE:
            print("updating theta_star for feature %s"%(str(F)))

        for v in range(len(self.theta)):
            theta_val = K.eval(self.theta[v])

            self.theta_star[F].append(theta_val)
            if DEBUG_MODE:
                self.prnt(self.theta_star[F][v],message="theta_starF%d"%(v))
                self.prnt(self.theta[v],message="theta%d"%(v))

        if F not in self.lbda_task: # two modes: specify all beforehand vs not all specified beforehand--> give equal weight
            self.lbda_task[F]=0.0
            for f in self.lbda_task:
                self.lbda_task[f]=self.lbda


    def restore(self,F):
        # reassign optimal weights for latest task
        #if hasattr(self, "star_vars"):
            for v in range(len(self.theta_star[F])):
                self.sess.run(self.theta[v].assign(self.theta_star[F][v]))
    def end_task(self,delta_t,time=0):
        """
        when task ends, add the found parameters for the previous task
        (creates a new entry if the task feature was not seen before)

        :param F_prev:
        :param task_importance: how much should we weight this task compared to others
        :return:
        """
        if self.objective_type==ObjectiveType.likelihood:
            return
        if self.current_task is not None and self.current_task not in self.theta_star:
            self.task_t[self.current_task]+=delta_t
            if self.task_t[self.current_task]  >= time:
                self.update_thetastar(self.current_task)




    def start_task(self,F):
        self.current_task=tuple(F)
        if self.current_task not in self.task_t:
            self.task_t[self.current_task] = 0
        self.previous_Fs = [F for F in self.theta_star if F != self.current_task]
        if self.objective_type==ObjectiveType.EWC_predEval:
            self.current_weight=self.occurrence_weights[F]
            # previous_weights = np.array([self.occurrence_weights[F] for F in self.previous_Fs])
            # C=self.current_weight+sum(previous_weights)
            # self.current_weight/=C
            self.previous_weights={F:self.occurrence_weights[F] for F in self.previous_Fs}


        if DEBUG_MODE:
            print("start task %s" % (F))
            for v in range(len(self.theta)):
                self.theta[v] = self.prnt(self.theta[v], message="theta%d" % (v))
                if F in self.theta_star:
                    self.theta_star[F][v] = self.prnt(self.theta[v], message="theta_starF%d" % (v))
        # based on these new values, you can check what happens to previous task loss
        #self.check_previous_task_loss()
    # def fit(self,x,y,batch_size,verbose,epochs, validation_split):
    #     K.set_value(self.x,x)
    #     self.model.fit(x,y,batch_size=batch_size,verbose=verbose,epochs=epochs,validation_split=validation_split)
    def get_lik(self,y_t,y_p,scale):
        return norm.pdf(y_t, loc=y_p, scale=scale)
    def get_numpy_likelihood(self,y_true,y_pred):
        return sum(np.log(self.get_lik(y_true[j][i],y_pred[j][i],self.beta[i])) for j in range(self.batch_size) for i in range(self.n_out))
    # def log_likelihood(self,y_true,y_pred):
    #     y_true=K.eval(y_true)
    #     y_pred=K.eval(y_pred)
    #     return K._to_tensor(sum(np.log(self.get_lik(y_true[i],y_pred[i])) for i in range(self.n_out)))
    # def gaussian(self,z, mean, var, z_dim):
    #
    #         return 1 / (K.pow(2 * np.pi, z_dim/2) * K.prod(var, 1)) * K.exp(
    #             - 1 / 2 * K.sum(K.square(z - mean) / var, 1))
    def gaussian(self,z , mean, var, z_dim):
        """
        multivariate gaussian each with the same variance but different mean
        :param z:
        :param mean:
        :param var:
        :param z_dim:
        :return:
        """
        return 1/np.sqrt(var*(2*np.pi)**z_dim)*K.exp(-0.5*var**(-1)*K.square(z-mean))

    def log_likelihood(self,y_true,y_pred):
        """
        toy function, code works like this
        :param y_true:
        :param y_pred:
        :return:
        """
        temp=K.log(self.gaussian(z=y_true,mean=y_pred,var=self.beta,z_dim=self.n_out))
        return K.sum(temp,axis=[0,1])


    def loss(self,y_true,y_pred):
        return  -self.log_likelihood(y_true,y_pred)
    def objective(self, y_true, y_pred):
        return self.loss(y_true,y_pred)  #+ self.previous_tasks_loss()
    def objective_EWC(self):
        def ob(y_true,y_pred):
            #return self.loss(y_true,y_pred) +
            if not self.theta_star:
                return self.loss(y_true,y_pred)
            # return self.previous_task_loss(self.model.output)
            #self.loss(y_true, self.model.output) +
            l= self.loss(y_true,y_pred) + self.previous_task_loss(self.theta)
            return l
        return ob
    def objective_EWC_mse(self):
        def ob(y_true,y_pred):
            #return self.loss(y_true,y_pred) +
            if not self.theta_star:
                return keras.losses.mse(y_true,y_pred)
            # return self.previous_task_loss(self.model.output)
            #self.loss(y_true, self.model.output) +
            l= keras.losses.mse(y_true,y_pred) + self.previous_task_loss(self.theta)
            return l
        return ob
    def objective_EWC_predEval(self, weight,previous_weights):
        def ob(y_true,y_pred):
            #return self.loss(y_true,y_pred) +
            if not self.theta_star:
                return self.loss(y_true,y_pred)
            # return self.previous_task_loss(self.model.output)
            #self.loss(y_true, self.model.output) +
            l= weight*self.loss(y_true,y_pred) + self.previous_task_loss_predictiveEval(self.theta,previous_weights)
            return l
        return ob

    def train(self,x,y,batch_size,epochs,verbose,validation_split):

        if self.objective_type==ObjectiveType.EWC:
            self.fit_EWC_classification(x, y, batch_size,epochs, verbose, validation_split,predEval=False)
            #l=self.loss(y,self.model.layers[-1].output)
        elif self.objective_type==ObjectiveType.EWC_predEval:
            self.fit_EWC_classification(x, y, batch_size, epochs, verbose, validation_split, predEval=True)
        else:
            self.beta = np.std(y, axis=0)
            self.model.compile(loss=self.objective, optimizer=Adadelta(lr=self.learning_rate, rho=0.95,
                                                             clipvalue=10.0))  # compile with new information

            self.model.fit(x, y, batch_size=batch_size,epochs=epochs, verbose=verbose, validation_split=validation_split)

    def compile_EWC(self,minibatches,model,loss_fun):
        # x and y are large number of random samples of inputs and outputs (100 minibatches)
        self.model = model
        self.theta=self.model.trainable_weights
        self.F_accum = [None for l in self.theta]


        if minibatches:
            # compute beta for likelihood
            x,y = minibatches[0]
            appended_y = y
            for (x, y) in minibatches[1:]:
                appended_y = np.concatenate((appended_y, y), axis=0)
            self.beta = np.std(appended_y, axis=0)/2
        else:
            self.beta = np.zeros(self.n_out) + 0.25

        print("previous Fs: %s" % (str(self.previous_Fs)))
        print("lambda's: %s" % (str(self.lbda_task)))
        print("current F: %s" % (str(self.current_task)))
        print("beta=", self.beta)
        if self.theta_star:
            # if there is no theta_stars then no need to compute fisher
            x, y = minibatches[0]
            self.compute_fisher_linear(x, y, self.theta)
            for (x,y) in  minibatches[1:]:
                #print("y shape ", y.shape)
                #self.print_current_theta_and_thetastars()
                self.compute_fisher_linear(x,y,self.theta,first=False)
            for i in range(len(self.F_accum)):
                self.F_accum[i]/=len(minibatches)
        #print("theta shape ", self.theta)
        #print("fisher shape ", self.F_accum)
        #print("will now compile EWC with Fisher=", self.F_accum)
        if loss_fun=="mse":
            loss=self.objective_EWC_mse()
        else:
            #print("appended_y ", appended_y)
            #print("beta ", self.beta)
            #print("beta shape ", self.beta.shape)
            loss=self.objective_EWC()
        self.model.compile(loss=loss, optimizer=RMSprop(0.001, rho=0.90,
                                                                  clipvalue=10.0))  # compile with new information  # compile with new information) # compile w
        return self.model
    def fit_EWC_classification(self,x,y,batch_size,epochs,verbose,validation_split,predEval):

        self.batch_size=x.size
        self.theta=self.model.trainable_weights
        self.beta = np.std(y,axis=0)
        print("previous Fs: %s"%(str(self.previous_Fs)))
        print("lambda's: %s"%(str(self.lbda_task)))
        print("current F: %s" % (str(self.current_task)))
        #self.print_current_theta_and_thetastars()
        if validation_split>0:
            split_at = int(x.shape[0] * (1 - validation_split))
            fisher_x = x[split_at:]
            fisher_y = y[split_at:]
        else:
            fisher_x = x
            fisher_y = y

        self.compute_fisher_linear(fisher_x,fisher_y,self.theta)

        if predEval:

            loss=self.objective_EWC_predEval(self.current_weight,self.previous_weights)
        else:
            loss=self.objective_EWC()
        self.model.compile(loss=loss,optimizer=Adadelta(lr=self.learning_rate, rho=0.95,clipvalue=10.0)) # compile with new information
        self.model.fit(x,y,batch_size=batch_size,epochs=epochs,verbose=verbose,validation_split=validation_split)

    def prepare_previous_task_loss(self,y,theta):
        """
        get the previous task loss based on current parameters:
         -theta
         -theta_star
        :return:
        """
        # update the most recent eel,self.model.trainable_weights)
        if self.output_type==OutputType.linear:
            self.compute_fisher_linear(y,theta)
        elif self.output_type==OutputType.softmax:
            self.compute_fisher_softmax(y)
        else:
            raise Exception('not supported')

    def prnt(self,tensor,message):
        return tf.Print(tensor,[tensor,tf.shape(tensor)],message=message)

    def check_previous_task_loss(self):
        """
        check the loss, given simple values (one previous task)

        Fisher=all ones
        theta=one time the same as the only theta_star ---> 0
        theta=one time always theta_star+1 ---> lbda_task * parameters
        :return:
        """
        self.theta=self.model.trainable_weights
        trainable_count = int(
            np.sum([K.count_params(p) for p in set(self.theta)]))
        if len(self.theta_star) == 1:
            for F in self.theta_star:
                theta_star_numpy=[K.eval(W) for W in self.theta_star[F]]
                same_theta = deepcopy(theta_star_numpy)
            for v in range(len(self.theta)):
                K.set_value(self.theta[v],same_theta[v])

            self.F_accum = [np.ones(K.eval(K.shape(W))) for W in self.theta]
            self.previous_Fs = [F for F in self.theta_star if F != self.current_task]
            loss=K.eval(self.previous_task_loss(self.theta))
            assert loss==0.0, str(loss)
            different_theta=[same_theta[v]+1 for v in range(len(same_theta))]
            for v in range(len(self.theta)):
                K.set_value(self.theta[v],different_theta[v])
            loss=K.eval(self.previous_task_loss(self.theta))
            assert loss==sum([self.lbda_task[F]/2.0 * trainable_count for F in self.theta_star if F != self.current_task])
            print("previous task loss correct")
            different_theta = [different_theta[v] -1.01 for v in range(len(same_theta))] #make a reasonable value, otherwise nans
            for v in range(len(self.theta)):
                K.set_value(self.theta[v],different_theta[v])
            if self.previous_Fs:
                self.model.compile(loss=self.previous_taskloss_wrapper(self.theta),optimizer=keras.optimizers.RMSprop(lr=self.learning_rate, rho=0.9, epsilon=None, decay=0.0)) # compile with new information
                x,y=generate_random_data(self.batch_size,fun1)
                self.model.fit(x,y,epochs=5,verbose=0,validation_split=.10)
                print("result is reasonable")
            # now back to original values
            for v in range(len(self.theta)):
                K.set_value(self.theta[v], same_theta[v])
    def previous_taskloss_wrapper(self,theta):
        """
        to check whether learning based on previous task loss is possible
        :param theta:
        :return:
        """
        def ob(y_true,y_pred):
            return self.previous_task_loss(theta)
        return ob
    def previous_task_loss(self,theta):
        """
        note: theta needs to be tensor to take into account parameter changes, theta_star[F] is fixed list of numpy arrays
        :return:
        """
        l=0.0
        for F in self.previous_Fs:
            for v in range(len(theta)):
                l = l + K.sum(self.lbda_task[F] / 2. * self.F_accum[v]*K.square(theta[v] - self.theta_star[F][v]))


                if DEBUG_MODE:
                    diff = self.prnt(diff,"diff")
        return l
    def previous_task_loss_predictiveEval(self,theta,weights):
        """
        note: theta needs to be tensor to take into account parameter changes, theta_star[F] is fixed list of numpy arrays
        :return:
        """
        l=K.variable(0.0)

        for F in self.previous_Fs:
            for v in range(len(theta)):
                l = l + weights[F]*K.sum(self.lbda_task[F] / 2. * self.F_accum[v]*K.square(theta[v] - self.theta_star[F][v]))


                if DEBUG_MODE:
                    diff = self.prnt(diff,"diff")

        return l


    def compute_fisher_linear(self, xx, yy, theta, first=True):
        self.model.add(GaussianLikelihood(yy,self.n_out,self.beta))
        for v in range(len(theta)):
            f = K.function([self.model.inputs], K.square(K.gradients(self.model.outputs, theta[v])))
            F = f(xx)
            if first:
                self.F_accum[v] = F
            else:
                self.F_accum[v] += F

        if DEBUG_MODE:
            print(self.F_accum)

        self.model = Sequential(self.model.layers[:-1])
        #print()

    # def compute_fisher_direct(self, xx, yy, theta):
    #
    #     # gs = init_gradient_calcs2(self.ll,self.model, theta, self.n_out)
    #     ll=self.log_likelihood(yy,self.model.output)
    #     gs = init_gradient_calcs(self.model.input,ll,theta, self.n_out)
    #     self.F_accum = square_deriv(theta, self.n_out, xx, gs)
    #
    #     if DEBUG_MODE:
    #         print(self.F_accum)

    def check_fisher(self):
        """
        perturb the parameters according to different shapes and see the effect
        :return:
        """
        raise NotImplementedError()

def init_small_network(N,inputs,outputs):

    from keras.initializers import glorot_normal,glorot_uniform
    n_out = 1
    model = Sequential()
    model.add(InputLayer(input_shape=(inputs,)))
    model.add(Dense(3, batch_input_shape=(None, inputs), activation='relu',
                    kernel_initializer=glorot_uniform(),
                    bias_initializer='zeros'))
    model.add(Dense(units=outputs, activation="linear"))
    return model

def compile(network, objective,lr):

    ada_delta=Adadelta(lr=lr, rho=0.95,clipvalue=10.0)
    network.compile(optimizer=ada_delta,loss=objective)
def fun1(x):
    return np.array([(x[0])])
def fun2(x):
    return np.array([x[0]+0.01*x[1]+0.01*x[2]])
def fun3(x):
    """

    hypothesis:
    if fisher correct, then this is much easier for EWC to learn multi-tasks
    :return:
    """

    return np.array([x[0]+0.10*x[1]+0.10*x[2]])


def fun4(x):
    """

    hypothesis:
    if fisher correct, then this is much easier for EWC to learn multi-tasks
    :return:
    """
    return np.array([1.05*x[0]+0.10*x[1]+0.10*x[2]])

def fun5(x):
    """

    hypothesis:
    if fisher correct, then this is much easier for EWC to learn multi-tasks
    :return:
    """
    return np.array([0.95*x[0]+0.10*x[1]+0.10*x[2]])

def generate_random_data(N,low,high,func):
    x = np.random.uniform(low=low,high=high,size=(N,3))
    y = np.array([func(x[i]) for i in range(N)])
    y_test=func(x[0])
    np.testing.assert_allclose(y[0],y_test)
    return x,y
def trainTask(N,net,ewc,likelihood,func,low,high, compare=False):
    x,y=generate_random_data(N,low,high,func)

    F=func.__name__
    print("EWC network training")
    ewc.start_task(F)

    ewc.train(x,y,batch_size=N,verbose=0,epochs=100,validation_split=.00)
    ewc.end_task(delta_t=len(x))
    if compare:
        print("loglikelihood network training")
        likelihood.train(x,y,batch_size=N,verbose=0,epochs=100,validation_split=.00)
        print("normal network training")
        net.fit(x, y, batch_size=N, verbose=0, epochs=100, validation_split=.00)
def range_mse(y,y_pred):
    return ((y - y_pred) ** 2).mean()
def afterTask(N,net,ewc,likelihood,funcs,lows,highs, compare=False):
    for i in range(len(funcs)):
        x, y = generate_random_data(N,lows[i],highs[i], funcs[i])
        print("data set initialised")
        print("on function %s" % (funcs[i].__name__))
        y_ewc = ewc.model.predict(x, batch_size=N)
        ewc_err = range_mse(y,y_ewc)

        print("ewc_err=%.4f"%(ewc_err))

        if compare:
            y_pred = net.predict(x, batch_size=N)
            err = range_mse(y,y_pred)
            y_lik = likelihood.model.predict(x, batch_size=N)
            lik_err = range_mse(y,y_lik)
            print(" err=%.4f; \n \n lik_err=%.4f "% (err,lik_err))
def toy_regression_problems():
    """
    simple regression problem:

    one dataset: (x,y,z) --> (x+y+z)

    another one: (x,y,z) --> -(x+y+z)


    :return:
    """
    from keras.objectives import mse



    # initialise data
    N = 1000
    n_in=3
    n_out=1
    function_set = [fun1, fun2, fun3, fun4,fun5]
    lows=[0.0,0.10,0.00,0.50,0]
    highs=[0.10,0.15,1.0,1.0,1.0]

    # initialise networks

    net=init_small_network(N,n_in, n_out)
    lr=.00025
    compile(net, mse, lr)

    #
    lbda_task={}
    net_likelihood = init_small_network(N,n_in, n_out)
    likelihood=EWC_objective(lbda_task,lr,N,net_likelihood,n_in,n_out, lbda=1.0,objective_type=ObjectiveType.likelihood)
    net_EWC= init_small_network(N, n_in, n_out)
    ewc=EWC_objective(lbda_task,lr,N,net_EWC,n_in,n_out, lbda=4.0,objective_type=ObjectiveType.EWC)
    #test_gaussian(ewc)

    compare=True


    for i in range(len(function_set)):
        trainTask(N,net,ewc,likelihood,function_set[i],lows[i],highs[i],compare)
        print("after function %s: "%function_set[i].__name__)
        afterTask(N,net,ewc,likelihood,function_set,lows,highs,compare)

    check_fisher([ewc])



def test_gaussian(ewc):
    print(ewc.model.summary())
    x, y = generate_random_data(ewc.batch_size,0.0,1.0, fun3 )
    y_pred=ewc.model.predict(x, batch_size=ewc.batch_size)


    result = K.eval(ewc.log_likelihood(y,y_pred))
    expected = ewc.get_numpy_likelihood(y,y_pred)
    #np.testing.assert_allclose(result, expected, rtol=1e-05)

def test_gradient_calc():
    """
    check the gradient calculation:

    simple case: y =w*x univariate ---> dy/dw = x

    for a batch of several inputs this is the sum(x)
    :return:
    """
    from keras.models import Model
    from keras.layers import Dense, Input

    n_out = 1
    x=Input(shape=(1,))
    y = Dense(1,batch_input_shape=(5,1),use_bias=False)(x)
    model = Model(x,y)
    theta = model.trainable_weights
    xx=np.array([[0.0],[1.0],[0.5],[1.5],[2.5]])
    gs = init_gradient_calcs(model.input, model.output, theta, n_out)
    grads = gs[0]([xx])

    assert grads[0]==sum(xx)

    print(grads)

def check_fisher(methods):
    for method in methods:
        method.check_fisher()

if __name__ == '__main__':
    #test_gradient_calc()
    toy_regression_problems()