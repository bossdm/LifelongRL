
import keras.backend as K
import numpy as np
# function implementation

def get_functions(model,grads):
    return K.function([model.input], grads)
def init_gradient_calcs(input_tensor,output_tensor,theta,n_out):
    g=[]
    for o in range(n_out):
        grads = K.gradients(output_tensor[:,o], theta)
        g.append(K.function([input_tensor], grads))
    return g
def init_gradient_calcs2(loglikelihood,model,theta,n_out):
    g=[]
    for o in range(n_out):
        grads = K.gradients(loglikelihood, theta)
        g.append(K.function([model.layers[0].input], grads))
    return g

def square_deriv_plus_init(model,theta,n_out,xx):
    """
    compute the square deriv (useful for linear output fisher matrix)
    for a given dataset
    :param model:
    :param theta:
    :param n_out:
    :param xx:
    :return:
    """
    gs = init_gradient_calcs(model,theta, n_out)
    return square_deriv(theta,n_out,xx,gs)
def square_deriv(theta,n_out,xx,gs):
    ds = [np.zeros(theta[i].get_shape().as_list()) for i in range(len(theta))]
    for o in range(n_out):
        added = gs[o]([xx])
        for i in range(len(theta)):
            ds[i] += np.square(added[i])
    return [d/xx.shape[0] for d in ds]


# tensor implementation: differentiable

def init_gradient_calcs_T(y,theta,n_out):
    return [K.gradients(y[0,o], theta) for o in range(n_out)]
def square_deriv_T(theta,n_out,gs):
    ds = [K.variable(np.zeros(theta[i].get_shape().as_list())) for i in range(len(theta))]
    for o in range(n_out):
        grad=gs[o]
        for v in range(len(theta)):
            ds[v] = ds[v] + K.square(grad[v])
    return ds
