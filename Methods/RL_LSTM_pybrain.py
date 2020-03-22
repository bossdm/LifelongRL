from __future__ import print_function

#!/usr/bin/env python
# Example script for recurrent network usage in PyBrain.
__author__ = "Martin Felder"
__version__ = '$Id$'

from pylab import plot, hold, show
from scipy import sin, rand, arange
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.validation    import testOnSequenceData
from pybrain.tools.shortcuts     import buildNetwork



class RL_LSTM_Pybrain(object):

    def __init__(self,indim,outdim):
        # construct LSTM network - note the missing output bias
        rnn = buildNetwork( indim, 5,  outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, outputbias=False, recurrent=True)
        rnn2 = buildNetwork
        # define a training method
        trainer = RPropMinusTrainer( rnn )
    # instead, you may also try
