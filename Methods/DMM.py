# attempt at Dataflow Matrix Machines, wish me luck !

class Transformation(object):
    def __init__(self):


class NeuronType(object):
    def __init__(self, n_inputs, n_outputs, input_stream_types, output_stream_types, F):
        self.n_inputs=n_inputs
        self.n_outputs=n_outputs
        self.input_stream_types=input_stream_types
        assert len(input_stream_types)==n_inputs
        self.output_stream_types=output_stream_types
        assert len(output_stream_types) == n_outputs
        self.F = F # the transformation
class StreamType(object):
    def __init__(self, length):
        self.length=length
class Signature(object):
    def __init__(self,neuronTypes,streamTypes):
        self.neuronTypes=neuronTypes
        self.streamTypes=streamTypes
class Neuron(object):
    def __init__(self, f, type):
        self.input
        self.output
        self.type=type
        self.f=f

class DMM(object):
    def __init__(self,neurons):
        self.neurons=neurons
        self.W=np.zeros(len(self.neurons),len(self.neurons)) # matrix of weights

    def compatibleNeurons(self,output_neuron,input_neuron):
        return self.neurons[output_neuron].input.streamType == self.neurons[input_neuron].output.streamType

    def setWeight(self,i,j,value):
        if self.compatibleNeurons(i,j):
            self.W[i][j] = value
        else:
            self.W[i][j] = 0