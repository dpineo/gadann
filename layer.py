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
import numpy
import math
import operator
import itertools
import scipy
import cudnn
import ctypes
import logging
import cv2

import kernels
import tensor

logger = logging.getLogger(__name__)

#-----------------------  Layer  --------------------------
class Layer(object):
	def __init__(self, feature_shape, n_features, learning_rate=0.01, weight_cost=0.002):
		pass

	def backprop(self, input, target):
		return self.bprop(self.next.backprop(self.fprop(input), target))

	def fprop(self, input):
		return input

	def bprop(self, input):
		return input

#-----------------------  Layer  --------------------------
class LinearLayer(Layer):
	def __init__(self, feature_shape, n_features, learning_rate=0.1, weight_cost=0.0002, init=0.01, visualize=False):
		self.learning_rate = learning_rate
		self.weight_cost = weight_cost
		self.visualize = visualize

		# model parameters
		self.name = None
		self.feature_shape = feature_shape
		self.feature_size = reduce(operator.__mul__, self.feature_shape, 1)
		self.shape = (n_features,)+ self.feature_shape
		self.n_features = n_features

		self.prev = self.next = None
		self.w = tensor.Tensor(init*numpy.random.randn(*self.shape))
		self.v_bias = tensor.zeros((1,self.feature_size))
		self.h_bias = tensor.zeros((1,self.n_features))

	def fprop(self, input):
		result = input.dot(self.w.T()) + input.ones_vector.dot(self.h_bias)
		assert( not numpy.isnan(result.get()).any())
		return result

	def bprop(self, input):
		result = input.dot(self.w) + input.ones_vector.dot(self.v_bias)
		assert( not numpy.isnan(result.get()).any())
		return result

	def backprop(self, input, target):
		output = self.fprop(input)
		output_error = self.next.backprop(output, target)
		input_error = self.bprop(output_error)
		tensor.sgemm(output_error.T(), input, self.w, alpha=-self.learning_rate, beta=1.0)
		tensor.sgemv(self.h_bias, output_error.T(), input.ones_vector.T(), alpha=-self.learning_rate, beta=1.0)
		assert( not numpy.isnan(self.w.get()).any())
		assert( not numpy.isnan(self.h_bias.get()).any())

		if self.visualize:
			cv2.imshow(self.name, self.w.mosaic().get()/10 + .5)
			cv2.waitKey(1)
		return input_error

	def grad(self, v, h):
		return h.T().dot(v), v.T().dot(v.ones), h.T().dot(h.ones)

	def __str__(self):
		return self.name + " - " + self.__class__.__name__ + "(shape=" + str(self.shape) + ")"

class ActivationLayer(Layer):
	def __init__(self, activation, **kwargs):
		#super(ActivationLayer, self).__init__(**kwargs)
		self.activation = getattr(kernels, activation)
		self.d_activation = getattr(kernels, 'd'+activation)

	def fprop(self, input):
		result = self.activation(input)
		assert( not numpy.isnan(result.get()).any())
		return result

	def bprop(self, input):
		result = self.d_activation(input)
		assert( not numpy.isnan(result.get()).any())
		return result

	def backprop(self, input, target):
		output = self.fprop(input)
		if self.next:
			output_error = self.next.backprop(output, target)
		else:
			output_error = output - target
		input_error = self.d_activation(output_error, output)
		assert( not numpy.isnan(input_error.get()).any())
		return input_error

	def __str__(self):
		return self.name + " - " + self.__class__.__name__ + "(activation='" + self.activation.__name__ + "')"

class DropoutLayer(Layer):
	def __init__(self, p):
		self.p_exclude = float(p)
		self.p_include = float(1-p)

	def backprop(self, input, target):
		bernoulli = tensor.bernoulli(input.shape, prob=self.p_include)
		input = (input*bernoulli)/self.p_include
		input_error = self.bprop(self.next.backprop(self.fprop(input), target))*bernoulli*self.p_include
		return input_error

	def __str__(self):
		return self.name + " - " + self.__class__.__name__ + "(p=" + str(self.p_exclude) + ")"

class ReshapeLayer(Layer):
	def __init__(self, **kwargs):
		self.input_shape  = kwargs['input_shape']
		self.output_shape = kwargs['output_shape']

	def fprop(self, input):
		input.shape = self.output_shape
		return input

	def bprop(self, input):
		input.shape = self.input_shape
		return input


#-----------------------  ConvLayer  --------------------------
class ConvLayer(LinearLayer):
	def __init__(self, conv_step=1, **kwargs):
		super(ConvLayer, self).__init__(**kwargs)
		self.conv_step = conv_step
		self.w.axes = ['features_out', 'features_in', 'height', 'width']

	def fprop(self, input):
		result = Tensor((input.shape[0],self.w.shape[0])+tuple([x-y for x,y in zip(input.shape[-2:],self.w.shape[-2:])]))
		grid = (result.shape[-2]/16,result.shape[-1]/16,result.shape[0])
		conv2d_16x16_kernel(input.gpuarray, self.w.gpuarray, self.h_bias.gpuarray, result.gpuarray, numpy.uint32(self.w.shape[1]), numpy.uint32(self.w.shape[0]), block=(16,16,1), grid=grid)
		assert( not numpy.isnan(result.get()).any())
		return result

	def bprop(self, input):
		result = tensor.zeros((input.shape[0],self.w.shape[1])+tuple([x+y for x,y in zip(input.shape[-2:],self.w.shape[-2:])]))
		grid = (input.shape[3]/16,input.shape[2]/16,1)
		bconv2d_16x16_kernel(input.gpuarray, self.w.gpuarray, self.v_bias.gpuarray, result.gpuarray, numpy.uint32(self.w.shape[0]), numpy.uint32(self.w.shape[1]), block=(16,16,1), grid=grid)
		assert( not numpy.isnan(result.get()).any())
		return result

	def grad(self, v, h):
		w_update = tensor.zeros(self.w.shape)
		v_bias_update = tensor.zeros(self.v_bias.shape)
		h_bias_update = tensor.zeros(self.h_bias.shape)
		grid = (h.shape[-2]/16,h.shape[-1]/16) # revisit: grid rounds down to nearest 16

		iconv2d_16x16_kernel(v.gpuarray, h.gpuarray, w_update.gpuarray, numpy.uint32(w_update.shape[1]), numpy.uint32(w_update.shape[0]), block=(16,16,1), grid=grid)
		iconv2d_h_bias_16x16_naive_kernel(h.gpuarray, h_bias_update.gpuarray, numpy.uint32(reduce(operator.__mul__, h.shape[-2:], 1)), block=(w_update.shape[0],1,1), grid=grid)

		v_bias_block = self.w.shape[-2:] + (1,)
		v_bias_grid = (1,1,1)
		iconv2d_v_bias_16x16_naive_kernel(v.gpuarray, v_bias_update.gpuarray, numpy.uint32(v.shape[1]-16), numpy.uint32(v.shape[2]-16), block=v_bias_block, grid=v_bias_grid)

		assert( not numpy.isnan(w_update.get()).any()) 
		assert( not numpy.isnan(v_bias_update.get()).any())
		assert( not numpy.isnan(h_bias_update.get()).any())
		return w_update, v_bias_update, h_bias_update

#-----------------------  NumpyConvLayer  --------------------------

class NumpyConvLayer(ConvLayer):
	def fprop(self, input):
		result = []
		for n in range(self.w.shape[0]):
			result.append(scipy.signal.convolve(input.get()[0,...], self.w.get()[n,...], mode='valid'))
		return Tensor(numpy.ascontiguousarray(numpy.vstack(result)[numpy.newaxis,...]))

	def bprop(self, input):
		result = []
		for n in range(self.w.shape[0]):
			w = numpy.fliplr(numpy.flipud(self.w.get()[n,...]))
			result.append(scipy.signal.convolve(input.get()[n,...], w, mode='full'))
		result = numpy.vstack(result).mean(axis=0)
		return Tensor(result)

#-----------------------  CudnnLayer  --------------------------
# Create a cuDNN context

if pycuda.autoinit.device.compute_capability() >= (3, 0):
	cudnn_context = cudnn.cudnnCreate()

class CudnnConvLayer(ConvLayer):
	def __init__(self, **kwargs):
		super(CudnnConvLayer, self).__init__(**kwargs)
		self.learning_rate = self.learning_rate/10
		saxpy(self.w, self.w, 1000)

		self.filter_descriptor = cudnn.cudnnCreateFilterDescriptor()
		self.bias_descriptor = cudnn.cudnnCreateTensorDescriptor()
		self.convolution_descriptor = cudnn.cudnnCreateConvolutionDescriptor()
		self.input_descriptor = cudnn.cudnnCreateTensorDescriptor()
		self.output_descriptor = cudnn.cudnnCreateTensorDescriptor()

		cudnn.cudnnSetFilter4dDescriptor(
			self.filter_descriptor,
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*self.w.shape
		)

		logger.info('filter_descriptor:', cudnn.cudnnGetFilter4dDescriptor(self.filter_descriptor))

		cudnn.cudnnSetTensor4dDescriptor(
			self.bias_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*(self.h_bias.shape + (1,1))
		)
		logger.info('bias_descriptor:', cudnn.cudnnGetTensor4dDescriptor(self.bias_descriptor))

	def fprop(self, input):
		assert len(input.shape) == 4

		cudnn.cudnnSetTensor4dDescriptor(
			self.input_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*input.shape
		)

		logger.info('input_descriptor:', cudnn.cudnnGetTensor4dDescriptor(self.input_descriptor))

		cudnn.cudnnSetConvolution2dDescriptor(
			self.convolution_descriptor,
			0, 0, 1, 1, 1, 1,
			cudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION'])

		logger.info('convolution_descriptor:',  cudnn.cudnnGetConvolution2dDescriptor(self.convolution_descriptor))

		# Get output dimensions (first two values are n_input and filters_out)
		batch, channels, height_output, width_output = cudnn.cudnnGetConvolution2dForwardOutputDim(
			self.convolution_descriptor,
			self.input_descriptor,
			self.filter_descriptor
		)

		# Output tensor
		output = Tensor((batch, self.n_features, height_output, width_output))

		cudnn.cudnnSetTensor4dDescriptor(
			self.output_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*output.shape
		)
		logger.info('output_descriptor:',  cudnn.cudnnGetTensor4dDescriptor(self.output_descriptor))
		
		workspace_size = cudnn.cudnnGetConvolutionForwardWorkspaceSize(
			cudnn_context,
			self.input_descriptor,
			self.filter_descriptor,
			self.convolution_descriptor,
			self.output_descriptor,
			cudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'],
		).value
		workspace = Tensor((workspace_size,))
		logger.info('workspace_size:', workspace_size)

		algo = cudnn.cudnnGetConvolutionForwardAlgorithm(
			cudnn_context,
			self.input_descriptor,
			self.filter_descriptor,
			self.convolution_descriptor,
			self.output_descriptor,
			cudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'],
			0
		)
		
		assert( not numpy.isnan(input.get()).any())
		assert( not numpy.isnan(self.w.get()).any())

		# Perform convolution
		cudnn.cudnnConvolutionForward(
			cudnn_context,
			1,
			self.input_descriptor,
			input.data(),
			self.filter_descriptor,
			self.w.data(),
			self.convolution_descriptor,
			algo,
			workspace.data(),
			workspace_size,
			0,
			self.output_descriptor,
			output.data()
		)

		assert( not numpy.isnan(output.get()).any())

		cudnn.cudnnAddTensor(
			cudnn_context,
			cudnn.cudnnAddMode['CUDNN_ADD_SAME_C'],
			1,
			self.bias_descriptor,
			self.h_bias.data(),
			1,
			self.output_descriptor,
			output.data()
		)

		assert( not numpy.isnan(output.get()).any())
		return output

	def bprop(self, input):
		assert len(input.shape) == 4
		cudnn.cudnnSetTensor4dDescriptor(
			self.input_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*input.shape
		)

		cudnn.cudnnSetConvolution2dDescriptor(
			self.convolution_descriptor,
			0, 0, 1, 1, 1, 1,
			cudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION'])

		# Output tensor
		output = Tensor((input.shape[0], self.w.shape[1], input.shape[2]+self.w.shape[2]-1, input.shape[3]+self.w.shape[3]-1))

		cudnn.cudnnSetTensor4dDescriptor(
			self.output_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*output.shape
		)

		# Perform convolution
		cudnn.cudnnConvolutionBackwardData(
			cudnn_context,
			1,
			self.filter_descriptor,
			self.w.data(),
			self.input_descriptor,
			input.data(),
			self.convolution_descriptor,
			0,
			self.output_descriptor,
			output.data()
		)

		assert( not numpy.isnan(output.get()).any())
		return output


	def backprop(self, input, target):
		assert len(input.shape) == 4

		output = self.fprop(input)
		output_error = self.next.backprop(output, target)
		input_error = self.bprop(output_error)

		cudnn.cudnnSetTensor4dDescriptor(
			self.input_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*input.shape
		)

		cudnn.cudnnSetTensor4dDescriptor(
			self.output_descriptor,
			cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
			cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
			*output_error.shape
		)
		'''
		print '--backprop--'
		print cudnn.cudnnGetTensor4dDescriptor(self.input_descriptor)
		print cudnn.cudnnGetFilter4dDescriptor(self.filter_descriptor)
		print cudnn.cudnnGetConvolution2dDescriptor(self.convolution_descriptor)
		print cudnn.cudnnGetTensor4dDescriptor(self.output_descriptor)
		'''
		# Perform convolution
		cudnn.cudnnConvolutionBackwardFilter(
			cudnn_context,
			-self.learning_rate,
			self.input_descriptor,
			input.data(),
			self.output_descriptor,
			output_error.data(),
			self.convolution_descriptor,
			1,
			self.filter_descriptor,
			self.w.data()
		)

		cudnn.cudnnConvolutionBackwardBias(
			cudnn_context,
			-self.learning_rate,
			self.output_descriptor,
			output_error.data(),
			1,
			self.bias_descriptor,
			self.h_bias.data()
		)

		assert( not numpy.isnan(self.w.get()).any())
		assert( not numpy.isnan(self.h_bias.get()).any())
		return input_error
