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
import numpy
import gzip
import time
import math
import pycuda.autoinit
import operator
import itertools
import copy

import kernels
import tensor
import data_source
import layer
import trainer

#--------------  NeuralNetwork  ----------------
class NeuralNetwork(object):
	def __init__(self, layers, weight_cost=0.02):
		self.layers = layers
		for n, (layer, prev) in enumerate(itertools.izip(self.layers+[None],[None]+self.layers)):
			if layer:
				layer.prev = prev
				layer.name = "Layer "+str(n)
			if prev:
				prev.next = layer

	def classify(self, features):
		predictions = self.predict(features)
		return tensor.Tensor(numpy.argmax(predictions.get(), axis=1))

	def evaluate(self, features, labels):
		classifications = self.classify(features)
		return (labels.tensor.get().flatten() == classifications.get().flatten()).mean()

	def predict(self, dataset):
		for layer in self.layers:
			dataset = dataset.fprop(layer)
		return dataset.tensor

	def __str__(self):
		return self.__class__.__name__ + '\n' + '\n'.join([str(l) for l in self.layers])