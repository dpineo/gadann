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

import kernels
import tensor
import data_source
import layer

logger = logging.getLogger(__name__)

#--------------------  Trainer  ---------------------
class Trainer(object):
	def __init__(self):
		pass

#--------------------  BatchGradientDescentTrainer  ---------------------
class BatchGradientDescentTrainer(Trainer):
	def train(self, model, inputs, targets, n_epochs):
		logger.info("Training (batch gradient descent)")
		for epoch in xrange(n_epochs):
			logger.info("Epoch " + str(epoch),)
			start_time = time.time()
			for n, (input_batch, target_batch) in enumerate(itertools.izip(inputs, targets)):
				assert( not numpy.isnan(input_batch.get()).any())
				assert( not numpy.isnan(target_batch.get()).any())
				model.layers[0].backprop(input_batch, target_batch)
			logger.info('  Time={:.3f}'.format(time.time()-start_time))

#--------------------  ContrastiveDivergenceTrainer  ---------------------
class ContrastiveDivergenceTrainer(Trainer):
	def train(self, model, inputs, n_epochs, learning_rate=.01, update='basic'):
		logger.info("Training (contrastive divergence)")
		error = Tensor(inputs.tensor.shape)

		for layer in model.layers:
			logger.info(layer.name)
			velocity_w = tensor.zeros(layer.w.shape)
			velocity_v = tensor.zeros(layer.v_bias.shape)
			velocity_h = tensor.zeros(layer.h_bias.shape)

			for epoch in xrange(n_epochs):
				logger.info("Epoch "+str(epoch))
				start_time = time.time()
				for batch_n, v in enumerate(inputs):
					ph = logistic(layer.fprop(v))
					h = sample(ph)

					pos_grad_w, pos_grad_v, pos_grad_h = layer.grad(v,h)

					pv = logistic(layer.bprop(h))
					v = sample(pv)

					ph = logistic(layer.fprop(v))
					h = sample(ph)

					neg_grad_w, neg_grad_v, neg_grad_h = layer.grad(v,h)

					grad_w = (pos_grad_w - neg_grad_w)/inputs.batch_size
					grad_v = (pos_grad_v - neg_grad_v)/inputs.batch_size
					grad_h = (pos_grad_h - neg_grad_h)/inputs.batch_size

					if update == 'basic':
						layer.w = layer.w + grad_w * learning_rate
						layer.v_bias = layer.v_bias + grad_v * learning_rate
						layer.h_bias = layer.h_bias + grad_h * learning_rate
					elif update == 'momentum':
						velocity_w = velocity_w*.95 + grad_w*learning_rate
						layer.w = layer.w + velocity_w - layer.w*layer.weight_cost

						velocity_v = velocity_v*.95 + grad_v*learning_rate
						layer.v_bias = layer.v_bias + velocity_v - layer.v_bias*layer.weight_cost

						velocity_h = velocity_h*.95 + grad_h*learning_rate
						layer.h_bias = layer.h_bias + velocity_h - layer.h_bias*layer.weight_cost

					elif update == 'rmsprop':
						pass

					#layer.w = layer.w + (pos_grad_w - neg_grad_w)*(learning_rate/inputs.batch_size) - layer.w*layer.weight_cost
					#layer.v_bias = layer.v_bias + (pos_grad_v - neg_grad_v)*(learning_rate/inputs.batch_size) - layer.v_bias*layer.weight_cost
					#layer.h_bias = layer.h_bias + (pos_grad_h - neg_grad_h)*(learning_rate/inputs.batch_size) - layer.h_bias*layer.weight_cost

					# Save pv for calculating error
					pycuda.driver.memcpy_dtod(int(error.gpuarray.gpudata)+batch_n*pv.gpuarray.size*pv.gpuarray.dtype.itemsize, pv.gpuarray.gpudata, pv.gpuarray.mem_size*pv.gpuarray.dtype.itemsize)

				# Sum error
				error = (inputs.tensor-error)**2   # Revisit: make TensorSource implement Tensor interface

				avg_error = float(pycuda.gpuarray.sum(error.gpuarray).get())/inputs.tensor.shape[0]
				logger.info('  Time={:.3f}  Error={:.6f}'.format(time.time()-start_time, avg_error))

				cv2.imshow(model.layers[0].name, model.layers[0].w.mosaic().get()/10+.5)
				cv2.waitKey(1)

			inputs = inputs.fprop(layer)

			fo = gzip.GzipFile('weights.gz', 'wb')
			cPickle.dump((layer.w.get()), fo)
			fo.close()
