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
import logging
import time
import cv2

from .tensor import Tensor
from .updater import SgdUpdater
from . import kernels

logger = logging.getLogger(__name__)


# --------------  NeuralNetworkModel  ----------------
class NeuralNetworkModel(object):
    def __init__(self, layers, input_shape=None, updater=SgdUpdater()):
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
        return numpy.fromiter(((l.get().flatten() == c.get().flatten()).mean()
                               for (l, c) in zip(labels, classifications)), float).mean()

    def probability(self, input):
        for layer in self.layers:
            input = layer.fprop(input)
            assert(not numpy.isnan(input.get()).any())
        return input
        # return reduce(lambda x,l: l.fprop(x), self.layers, input)

    def __str__(self):
        return self.__class__.__name__ + '\n' + '\n'.join([str(l) for l in self.layers])

    def show(self):
        for layer_n, layer in enumerate(self.layers):
            if not layer.params:
                continue

            weights = layer.params['w']
            for deconv_layer in reversed(self.layers[:layer_n]):
                weights = deconv_layer.bprop(weights)

            cv2.imshow(layer.name, weights.mosaic().get() + .5)
            cv2.waitKey(1)

    def train_backprop(self, features, labels, n_epochs):
        logger.info("Training (batch gradient descent)")
        for epoch in range(n_epochs):
            logger.info("Epoch " + str(epoch),)
            start_time = time.time()
            for n, (batch_features, batch_targets) in enumerate(zip(features, labels)):

                # Save the activations during the forward pass, they will be used to
                # compute the gradients during the backward pass
                layer_activations = [batch_features]

                # Forward pass
                for layer in self.layers:
                    layer_activations.append(layer.fprop(layer_activations[-1]))

                # Error delta
                output_error = layer_activations[-1] - batch_targets

                # Backward pass
                for layer in reversed(self.layers):
                    input_error = layer.bprop(output_error, layer_activations.pop())
                    grads = layer.gradient(layer_activations[-1], output_error)
                    layer.update(grads)
                    output_error = input_error

            logger.info(' Epoch {}  Time={:.3f}'.format(epoch, time.time()-start_time))
            self.show()

    def train_contrastive_divergence(self, features, n_epochs):
        logger.info("Training (contrastive divergence)")

        for layer in self.layers[:-2]:  # don't train the last linear & softmax layers
            logger.info("training " + layer.name)

            # skip the layer if it has no parameters to train
            if not layer.params:
                continue

            for epoch in range(n_epochs):
                reconstruction_error_avg = 0
                start_time = time.time()
                for batch_n, v in enumerate(features):

                    # Gibbs sampling
                    p_h_given_v = kernels.logistic(layer.fprop(v))
                    h_sample = kernels.sample(p_h_given_v)
                    pos_grads = layer.gradient(v, h_sample)
                    p_v_given_h = kernels.logistic(layer.bprop(h_sample))
                    v_sample = kernels.sample(p_v_given_h)

                    ph = kernels.logistic(layer.fprop(v_sample))
                    h = kernels.sample(ph)
                    neg_grads = layer.gradient(v, h)

                    # Gradiant of log likelihood wrt the parameters
                    grads = {k: (neg_grads[k] - pos_grads[k]) / features.batch_size for k in pos_grads.keys()}

                    # Update parameters wrt the gradients
                    layer.update(grads)

                    # Running average of reconstruction error
                    reconstruction_error = ((v-p_v_given_h)**2).sum()/v.size
                    reconstruction_error_avg = .1*reconstruction_error + .9*reconstruction_error_avg

                    self.show()

                # print model.updater.status()
                logger.info(' Epoch {}  Time={:.3f}  Error={:.6f}'.format(epoch, time.time()-start_time, reconstruction_error))
                self.show()

            # propgate input data through the layer
            features = features.apply_batchwise(layer.fprop)


