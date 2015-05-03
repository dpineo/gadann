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

import sys
import pycuda
import pycuda.curandom
import pycuda.compiler
import pycuda.autoinit
import numpy
import gzip
import cPickle
import time
import math
import cv2
import operator
import itertools
import logging
import copy
import collections

import kernels
import tensor
import stream
import layer

logger = logging.getLogger(__name__)

#--------------------  Trainer  ---------------------
class Trainer(object):
    def __init__(self):
        pass

#--------------------  BatchGradientDescentTrainer  ---------------------
class BatchGradientDescentTrainer(Trainer):
    def __init__(self, model, updater):
        self.model = model
        self.updaters = collections.defaultdict(lambda: copy.deepcopy(updater))

    def train(self, dataset, n_epochs):
        logger.info("Training (batch gradient descent)")
        for epoch in xrange(n_epochs):
            logger.info("Epoch " + str(epoch),)
            start_time = time.time()
            for n, batch in enumerate(dataset):
                input_batch = batch['features']
                target_batch = batch['labels']
                assert( not numpy.isnan(input_batch.get()).any())
                assert( not numpy.isnan(target_batch.get()).any())
                #self.backprop(self.model.layers[0], input_batch, target_batch)
                self.model.layers[0].backprop(input_batch, target_batch)

            #print self.model.updater.status()
            logger.info('  Time={:.3f}'.format(time.time()-start_time))
            for layer in self.model.layers:
                layer.show()

    def backprop(self, layer, input, target):
        output = layer.fprop(input)

        if layer.next:
            output_error = self.backprop(layer.next, output, target)
        else:
            output_error = output - target

        input_error = layer.bprop(output_error, output)
        grads = layer.error_gradient(input, output_error)
        self.updaters[layer].update(layer.params, grads)

        return input_error

#--------------------  ContrastiveDivergenceTrainer  ---------------------
class ContrastiveDivergenceTrainer(Trainer):
    def __init__(self, model, updater):
        self.model = model
        self.updaters = collections.defaultdict(lambda: copy.deepcopy(updater))

    def train(self, inputs, n_epochs):
        logger.info("Training (contrastive divergence)")

        for layer in self.model.layers:
            if layer.params == []:
                logger.info("training " + layer.name)
                continue
            for epoch in xrange(n_epochs):
                logger.info("Epoch "+str(epoch))
                reconstruction_error_avg = 0
                start_time = time.time()
                for batch_n, batch in enumerate(inputs):
                    v = batch['features']

                    '''
                    foo = v.get()[0,...]
                    cv2.imshow('batch', numpy.transpose(v.get()[0,...]))
                    cv2.waitKey(100)
                    '''

                    error = tensor.Tensor(v.shape)

                    # Gibbs sampling
                    ph = kernels.logistic(layer.fprop(v))
                    h = kernels.sample(ph)
                    pos_grads = layer.loglikelihood_gradient(v,h)
                    pv = kernels.logistic(layer.bprop(h))
                    v = kernels.sample(pv)
                    ph = kernels.logistic(layer.fprop(v))
                    h = kernels.sample(ph)
                    neg_grads = layer.loglikelihood_gradient(v,h)
    
                    # Gradiant of log likelihood wrt the parameters
                    grads = [(neg_grad-pos_grad)/inputs.batch_size for (pos_grad, neg_grad) in zip(pos_grads,neg_grads)]

                    # Update parameters wrt the gradients
                    self.updaters[layer].update(layer.params, grads)

                    # Running average of reconstruction error
                    reconstruction_error = ((v-pv)**2).sum()/v.size
                    reconstruction_error_avg = .1*reconstruction_error + .9*reconstruction_error_avg

                #print model.updater.status()
                logger.info('  Time={:.3f}  Error={:.6f}'.format(time.time()-start_time, reconstruction_error))
                layer.show()

            inputs = inputs.fprop(layer)