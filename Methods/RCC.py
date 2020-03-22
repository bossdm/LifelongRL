from math import tanh
from random import random
import math
import sys

import numpy

from pso import particle_swarm_optimize


def d_tanh(y):
    x=math.tanh(y)
    return 1.0 - x * x


def logistic(y):
    if y < -15.0:
        return 0.0
    elif y > 15.0:
        return 1.0


    return 1.0 / (1.0 + math.exp(-y))


def d_logistic(y):
    x = logistic(y)
    return x * (1 - x)


class _CandidateNode(object):
    def __init__(self, connection_weights):
        self.weights = connection_weights
        self.score = 0.0
        self.activations = []


class CascadeNet(object):
    def __init__(self,
                 input_nodes,
                 output_nodes,
                 function=tanh,
                 d_function=d_tanh,
                 weight_initialization_func=lambda: (random() - 0.3) * 0.6,
                 num_candidate_nodes=8):
        # CC starts with a minimal network consisting only of an input and an output layer.
        # Both layers are fully connected.
        self.output_connection_dampening = 0.7
        self.BIAS_UNITS = 1
        self.weight_initialization_func = weight_initialization_func
        self.output_nodes = output_nodes
        self.input_nodes = input_nodes
        self.num_candidate_nodes = num_candidate_nodes
        self._output_connections = numpy.array(
            [[weight_initialization_func() for _ in range(input_nodes + self.BIAS_UNITS)] for _ in
             range(output_nodes)])  # inputs*outputs
        self._cascade_connections = []
        self.function = function
        self.d_function = d_function
        self._output_activations = numpy.zeros(self.output_nodes)
        self._activations = numpy.zeros(self.input_nodes+self.BIAS_UNITS)
        self._activations[0] = 1.0  # bias node
        self.use_quick_prop = False

        self.learn_rate = 0.1
        self._momentums = numpy.zeros((output_nodes, input_nodes + self.BIAS_UNITS))

        self._last_weight_change = numpy.ones((output_nodes, input_nodes + self.BIAS_UNITS))
        self._last_weight_derivative = numpy.ones((output_nodes, input_nodes + self.BIAS_UNITS))

        self.momentum_coefficent = 0.5
        self.train_candidates_max_epochs = 200

    def _total_nodes(self):
        return self.input_nodes + self.output_nodes + len(self._cascade_connections) + self.BIAS_UNITS

    def _get_output_node_activation(self, connections):
        return self._get_output_node_activation_from_activations(connections, self._activations[:len(connections)])

    def _get_output_node_activation_from_activations(self, connections, activations):
        return self.function(numpy.inner(connections, activations))

    def _feed_forward_hidden_nodes(self, input):
        assert len(input) == self.input_nodes
        for i, x in enumerate(input):
            self._activations[i + self.BIAS_UNITS] = x

        for i, hidden_connections in enumerate(self._cascade_connections):
            activation_index = i + self.input_nodes + self.BIAS_UNITS
            self._activations[activation_index] = self._get_output_node_activation(hidden_connections)

    def _feed_forward_output_nodes(self):
        for i, output_connections in enumerate(self._output_connections):
            self._output_activations[i] = self._get_output_node_activation(output_connections)

        return self._output_activations

    def _feed_forward(self, inputs):
        self._feed_forward_hidden_nodes(inputs)
        return self._feed_forward_output_nodes()

    def _quick_prop(self, derivatives):
        quick_prop_change = (derivatives / (self._last_weight_derivative - derivatives)) * self._last_weight_change
        self._last_weight_derivative = derivatives
        self._last_weight_change = quick_prop_change
        return quick_prop_change

    def _train_single(self, input, target):
        result = self._feed_forward(input)
        derivatives = self._back_propagation(result, target)

        if self.use_quick_prop:
            quick_prop_change = self._quick_prop(derivatives)
            self._output_connections = numpy.add(self._output_connections, quick_prop_change)
        else:
            self._momentums = self._momentums * self.momentum_coefficent + numpy.multiply(self.learn_rate, derivatives)
            self._output_connections = numpy.add(self._output_connections, self._momentums)
        return CascadeNet._train_error_sum(result, target)

    @staticmethod
    def _train_error_sum(result, target):
        return numpy.sum(numpy.subtract(result, target) ** 2)

    @staticmethod
    def _train_error(result, target):
        return numpy.subtract(result, target) ** 2

    def _train_batch(self, inputs, targets):
        training_error = 0.0
        derivatives = numpy.zeros(self._output_connections.size)
        for i, input in enumerate(inputs):
            result = self._feed_forward(input)
            derivatives = numpy.add(derivatives, self._back_propagation(result, targets[i]))
            training_error += CascadeNet._train_error_sum(result, targets[i])

        if self.use_quick_prop:
            quick_prop_change = self._quick_prop(derivatives)
            self._output_connections = numpy.add(self._output_connections, quick_prop_change)
        else:
            self._momentums = self._momentums * self.momentum_coefficent + derivatives * self.learn_rate
            for i in range(self._momentums.shape[0]):
                self._momentums[i][0] *= 0.4  # reduce learning rate for bias
            self._output_connections = numpy.add(self._output_connections, self._momentums)
        return training_error

    def _get_errors(self, inputs, targets):
        errors = []

        for i, input in enumerate(inputs):
            result = self._feed_forward(input)
            error = CascadeNet._train_error(result, targets[i])
            errors.append(error)

        return errors

    def _get_errors_and_candidate_activations(self, candidates, inputs, targets):
        for candidate in candidates:
            candidate.activations.clear()

        errors = []
        results = []

        for i, input in enumerate(inputs):
            result = self._feed_forward(input)
            for candidate in candidates:
                candidate.activations.append(self._get_output_node_activation(candidate.weights))
            results.append(result)
            error = CascadeNet._train_error(result, targets[i])
            errors.append(error)

        return errors, results, candidates

    def _generate_candidate(self, inputs, targets):
        errors = self._get_errors(inputs, targets)
        mean_errors = [sum([x[y] for x in errors]) / len(errors) for y in range(len(errors[0]))]

        activations = []
        for input in inputs:
            self._feed_forward_hidden_nodes(input)
            activations.append(numpy.copy(self._activations[:self._non_output_nodes()]))

        # Generate the so-called candidate units. Every candidate unit is connected with all input units and with
        # all existing hidden units. Between the pool of candidate units and the output units there are no weights.
        candidates = [self._init_candidate() for _ in range(self.num_candidate_nodes)]

        best_correlation_score = sys.float_info.min

        momentums = {x: numpy.zeros(len(x.weights)) for x in candidates}

        epochs = 0
        iterations_without_improvement = 0

        while epochs < self.train_candidates_max_epochs and iterations_without_improvement < 3:
            epochs += 1
            last_best_correlation_score = best_correlation_score

            for candidate in candidates:
                candidate.activations.clear()

            for input in inputs:
                self._feed_forward_hidden_nodes(input)
                for candidate in candidates:
                    candidate.activations.append(self._get_output_node_activation(candidate.weights))

            for candidate in candidates:
                correlations = CascadeNet._correlations(candidate.activations, errors, mean_errors)
                correlation_signs = [math.copysign(1, x) for x in correlations]
                candidate.score = sum(correlations)
                derivatives = self._correlation_derivative(candidate.weights,
                                                            errors,
                                                            mean_errors,
                                                            candidate.activations,
                                                            correlation_signs)

                #update = momentums[candidate] * self.momentum_coefficent + numpy.multiply(derivatives, self.learn_rate)
                update = numpy.multiply(derivatives, self.learn_rate)
                #momentums[candidate] = update
                candidate.weights = numpy.add(candidate.weights, -update)

            best_candidate = max(candidates, key=lambda x: x.score)
            best_correlation_score = best_candidate.score
            if epochs % 4 == 0:
                print("best corr %s %s %s" % (best_correlation_score, best_candidate.activations, errors))
            if best_correlation_score <= last_best_correlation_score:
                iterations_without_improvement += 1

        return max(candidates, key=lambda x: x.score)

    def _add_hidden_node(self, candidate_weights):
        self._cascade_connections.append(candidate_weights)
        self._activations = numpy.zeros(self._total_nodes())
        self._activations[0] = 1.0
        self._output_connections = numpy.concatenate(
            [self._output_connections * self.output_connection_dampening,
             [[self.weight_initialization_func()] for x in range(self.output_nodes)]], axis=1)
        self._momentums = numpy.zeros((self.output_nodes, self._non_output_nodes()))
        self._last_weight_change = numpy.ones((self.output_nodes, self._non_output_nodes()))
        self._last_weight_derivative = numpy.ones((self.output_nodes, self._non_output_nodes()))

    def train(self, inputs, targets,
              stop_error_threshold=-sys.float_info.max,
              max_hidden_nodes=10,
              mini_batch_size=10,
              max_iterations_per_epoch=12):
        # Train all the connections ending at an output unit with a usual learning algorithm until the error of the net
        # no longer decreases.

        sum_of_error = self.backprop_train_till_convergence(inputs, targets, mini_batch_size=mini_batch_size,
                                                            stop_error_threshold=stop_error_threshold,
                                                            max_iterations_per_epoch=max_iterations_per_epoch)
        print("Sum of error %s" % sum_of_error)

        while sum_of_error > stop_error_threshold \
                and len(self._cascade_connections) < max_hidden_nodes:
            # Try to maximize the correlation between the activation of the candidate units and the residual error of
            # the net by training all the links leading to a candidate unit. Learning takes place with an ordinary
            # learning algorithm. The training is stopped when the correlation scores no longer improves.
            winning_candidate = self._generate_candidate(inputs, targets)

            print("winning candidate %s" % str(winning_candidate.weights))
            self._add_hidden_node(winning_candidate.weights)

            sum_of_error = self.backprop_train_till_convergence(inputs, targets,
                                                                max_iterations_per_epoch=max_iterations_per_epoch)
            print("Sum of error %s" % sum_of_error)
            #print_results(self, inputs, targets)

        return sum_of_error

    def _error_derivatives(self, result, target):
        output_errors = numpy.subtract(result, target)
        return [x * self.d_function(result[i]) for i, x in enumerate(output_errors)]

    def _non_output_nodes(self):
        return self.input_nodes + len(self._cascade_connections) + self.BIAS_UNITS

    def _back_propagation(self, result, target):
        derivatives = [numpy.array([0.0 for _ in range(self._non_output_nodes())]) for _ in
                       range(self.output_nodes)]

        output_errors = numpy.subtract(target, result)

        for input in range(self._non_output_nodes()):

            for output in range(len(result)):
                act =  self._activations[input]
                dedw = act * self.d_function(result[output]) * output_errors[output]
                derivatives[output][input] = dedw
        return derivatives
    def _back_propagation_through_time(self,result,target):
        derivatives = [numpy.array([0.0 for _ in range(self._non_output_nodes())]) for _ in
                       range(self.output_nodes)]
        self_derivatives = [numpy.array([0.0 for _ in range(self._non_output_nodes())]) for _ in
                            range(self.output_nodes)]
        output_errors = numpy.subtract(target, result)

        t=0
        while t < self.horizon:
            for input in range(self._non_output_nodes()):

                for output in range(len(result)):
                    act =self._activations[input] if t == 0 else self._activations[input] + self._self_weights[input]*derivatives[output][input][t-1]
                    dedw = act * self.d_function(result[output]) * output_errors[output]
                    derivatives[output][input] = dedw
                    if t>0:
                        act = self._activations[input][t-1] + self._self_weights[input] * self_derivatives[input][t-1]
                        self_derivatives[input] = act * self.d_function(result[output]) * output_errors[output]
            t-=1
        return derivatives, self_derivatives
    @staticmethod
    def _real_correlations(candidate_activation, errors) :
        correlations = [0.0 for _ in range(len(errors[0]))]
        std_dev_activation = [0.0 for _ in range(len(errors[0]))]
        std_dev_errors = [0.0 for _ in range(len(errors[0]))]
        error_means = [sum([x[i] for x in errors]) / len(errors) for i in range(len(errors[0]))]
        activation_mean = sum(candidate_activation) / len(candidate_activation)
        for p in range(len(candidate_activation)):
            for o in range(len(errors[0])):
                a = candidate_activation[p] - activation_mean
                b = errors[p][o] - error_means[o]
                std_dev_activation[o] += a * a
                std_dev_errors[o] += b * b
                correlations[o] += a * b

        for o in range(len(errors[0])):
            std_dev_activation[o] = math.sqrt(std_dev_activation[o] / len(errors))
            std_dev_errors[o] = math.sqrt(std_dev_errors[o] / len(errors))
            correlations[o] = correlations[o] / len(errors)
            correlations[o] = correlations[o] / (std_dev_activation[o] * std_dev_errors[o])

        return correlations

    @staticmethod
    def _correlations(candidate_activation, errors, mean_errors) :
        sums = numpy.zeros(len(errors[0]))
        for o in range(len(errors[0])):
            for p in range(len(candidate_activation)):
                x = candidate_activation[p] * (errors[p][o] - mean_errors[o])
                sums[o] += x

        return sums

    @staticmethod
    def _correlation(candidate_activation, errors, mean_errors) :
        sum = 0.0
        for p in range(len(candidate_activation)):
            for o in range(len(errors[0])):
                x = candidate_activation[p] * (errors[p][o] - mean_errors[o])
                sum += x

        return sum

    def _correlation_derivative(self, candidate_weights,
                                 errors,
                                 mean_errors,
                                 candidate_activations,
                                 sign_correlation_with_outputs):
        derrivatives = []
        for i in range(len(candidate_weights)):
            dcdw = 0.0
            for p in range(len(errors)):
                thetaO = 0.0
                for o in range(len(mean_errors)):  # output units
                    if self.recurrent:
                        act = candidate_activations[i] + self.self_weights[i]*self.derivatives
                    else:
                        act = candidate_activations[i]
                    thetaO += sign_correlation_with_outputs[o] * (errors[p][o] - mean_errors[o]) \
                              * self.d_function(candidate_activations[p]) \
                              *
                    dcdw += thetaO
            derrivatives.append(dcdw)
        return derrivatives

    def get_error(self, inputs, targets):
        error = 0.0
        for i, input in enumerate(inputs):
            result = self._feed_forward(input)
            error += CascadeNet._train_error_sum(result, targets[i])
        return error

    def get_result(self, input):
        return self._feed_forward(input)

    def backprop_train_till_convergence(self, inputs, targets,
                                        mini_batch_size=10,
                                        stop_error_threshold=0.05,
                                        max_iterations_per_epoch=2000):
        print("error pre=%s" % self.get_error(inputs, targets))

        iterations_without_improvement = 0
        previous_training_error = sys.float_info.max
        i = 0
        while True:
            if iterations_without_improvement > 1:
                print("stopped backprop because got 2 iterations without improvement")
                break
            if previous_training_error < stop_error_threshold:
                print("stopped backprop because got hit stop error threshold")
                break
            if i > max_iterations_per_epoch:
                print("stopped backprop because got max iterations")
                break

            training_error = 0.0
            for j in range(0, len(inputs), mini_batch_size):
                training_error += self._train_batch(inputs[j:j + mini_batch_size], targets[j:j + mini_batch_size])
            if i % 4 == 0:
                print("Trained %s" % training_error)
            if training_error + 0.000001 > previous_training_error:
                iterations_without_improvement += 1
            previous_training_error = training_error
            i += 1

        print("error post=%s" % self.get_error(inputs, targets))

        return training_error

    def _get_candidate_correlations(self, candidates, inputs, targets):
        errors, results, candidates = self._get_errors_and_candidate_activations(candidates, inputs, targets)
        mean_errors = [sum([x[y] for x in errors]) / len(errors) for y in range(len(errors[0]))]

        for candidate in candidates:
            print(candidate.activations)

        return [CascadeNet._real_correlations(candidate.activations, errors) for candidate in candidates]

    def _init_candidate(self):
        candidate_connections = numpy.array([self.weight_initialization_func()
                                             for _ in range(self._non_output_nodes())])
        return _CandidateNode(candidate_connections)


def print_results(network, inputs, targets):
    for i in range(len(inputs)):
        result = network._feed_forward(inputs[i])
        print("input %s target %s result %s", inputs[i], targets[i], result)
