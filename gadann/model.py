#
# GADANN - GPU Accelerated Deep Artificial Neural Network
#
# Copyright (C) 2014 Daniel Pineo (daniel@pineo.net)
# 
# Permission is hereby granted, free of charge, to any person obtaining a 
# copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation 
# the rights to use, copy, modify, merge, publish, distribute, sublicense, 
# and/or sell copies of the Software, and to permit persons to whom the 
# Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Software
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING 
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
# DEALINGS IN THE SOFTWARE.

import numpy
import copy

from .tensor import Tensor
from .updater import SgdUpdater


# --------------  NeuralNetworkModel  ----------------
class NeuralNetworkModel(object):
    def __init__(self, layers, input_shape=None, updater=SgdUpdater()):
        self.updater = updater
        self.layers = []

        for layer_n, layer in enumerate(layers):
            if 'shape' not in layer:
                layer['shape'] = input_shape[1:]
            if 'input_shape' not in layer:
                layer['input_shape'] = input_shape
            if 'name' not in layer:
                layer['name'] = 'Layer ' + str(layer_n)
            if 'updater' not in layer:
                layer['updater'] = copy.copy(updater)
            self.layers.append(layers[layer_n]['layer'](**layer))
            input_shape = self.layers[-1].output_shape
            pass

        for n, (layer, prev) in enumerate(zip(self.layers+[None], [None]+self.layers)):
            if layer:
                layer.prev = prev
                layer.name = "Layer "+str(n)
            if prev:
                prev.next = layer

    '''
    def add(self, layer, **kwargs):
        if 'shape' not in kwargs:
            layer['shape'] = input_shape[1:]
        if 'input_shape' not in kwargs:
            layer['input_shape'] = input_shape
        if 'name' not in layer:
            layer['name'] = 'Layer ' + str(layer_n)
        if 'updater' not in layer:
            layer['updater'] = copy.copy(updater)
        self.layers.append(layer(**kwargs))
        input_shape = self.layers[-1].output_shape
    '''

    def classify(self, features):
        probabilities = self.probability(features)
        return Tensor(numpy.argmax(probabilities.get(), axis=1))
    '''
    def probability(self, features):
        predictions = self.predict(features)
        return tensor.Tensor(numpy.argmax(predictions.get(), axis=1))
        confidences = numpy.max(predictions.get(), axis=1)
    '''

    def evaluate(self, features, labels):
        classifications = (self.classify(f) for f in features)
        return numpy.fromiter(((l.get().flatten() == c.get().flatten()).mean() for (l,c) in zip(labels, classifications)),float).mean()

    def probability(self, input):
        for layer in self.layers:
            input = layer.fprop(input)
            assert( not numpy.isnan(input.get()).any())
        return input
        # return reduce(lambda x,l: l.fprop(x), self.layers, source)

    def __str__(self):
        return self.__class__.__name__ + '\n' + '\n'.join([str(l) for l in self.layers])

    def traverse(self, f):
        for layer in self.layers:
            layer.f()
