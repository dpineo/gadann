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

import gadann
import unittest
import numpy
import cv2
import scipy.signal


class TestConv(unittest.TestCase):
	def setUp(self):
		self.cuda_layer = gadann.ConvLayer(feature_shape=(1,16,16), n_features=3, activation='logistic')
		self.numpy_layer = gadann.NumpyConvLayer(feature_shape=(1,16,16), n_features=3, activation='logistic')
		self.cudnn_layer = gadann.CudnnConvLayer(feature_shape=(1,16,16), n_features=3, activation='logistic')

		self.w_numpy = numpy.zeros(self.cuda_layer.shape)
		self.w_numpy[0,0,0,0] = 1

		self.cuda_layer.w = gadann.Tensor(self.w_numpy)
		self.numpy_layer.w = gadann.Tensor(self.w_numpy)
		self.cudnn_layer.w = gadann.Tensor(self.w_numpy, axes='nchw')


	def test_fprop(self):
		v_numpy = numpy.zeros((1,1,32,32))
		v_numpy[0,0,16,16] = 1
		v = gadann.Tensor(v_numpy)
		#v_cudnn = gadann.CudnnTensor(v_numpy)

		h_cuda = self.cuda_layer.fconv(v)
		h_numpy = self.numpy_layer.fconv(v)
		h_cudnn = self.cudnn_layer.fconv(v_cudnn)

		print h_cuda.shape
		print h_cuda.get()
		print h_numpy.shape
		print h_numpy.get()
		print h_cudnn.shape
		print h_cudnn.get()

		cv2.imshow('v', v.mosaic().get())
		#cv2.imshow('cuda_layer.w', self.cuda_layer.w.mosaic().get())
		#cv2.imshow('numpy_layer.w', self.numpy_layer.w.mosaic().get())
		#cv2.imshow('h_cuda', h_cuda.mosaic().get())
		cv2.imshow('h_numpy', h_numpy.mosaic().get())
		cv2.imshow('h_cudnn', h_cudnn.mosaic().get())
		cv2.waitKey(-1)

	def test_bprop(self):
		h_numpy = numpy.zeros((3,1,16,16))
		h_numpy[0,0,0,0] = 1
		h_numpy[0,0,8,8] = 1
		h = gadann.Tensor(h_numpy)
		#h_cudnn = gadann.CudnnTensor(h_numpy)

		v_cuda = self.cuda_layer.bconv(h)
		v_numpy = self.numpy_layer.bconv(h)
		v_cudnn = self.cudnn_layer.bconv(h_cudnn)

		print v_cuda.shape
		print v_cuda.get()
		print v_numpy.shape
		print v_numpy.get()
		print v_cudnn.shape
		print v_cudnn.get()

		cv2.imshow('h', h.mosaic().get())
		#cv2.imshow('cuda_layer.w', self.cuda_layer.w.mosaic().get())
		#cv2.imshow('numpy_layer.w', self.numpy_layer.w.mosaic().get())
		#cv2.imshow('v_cuda', v_cuda.mosaic().get())
		cv2.imshow('v_numpy', v_numpy.mosaic().get())
		cv2.imshow('v_cudnn', v_cudnn.mosaic().get())
		cv2.waitKey(-1)

TestConv("test_bprop").debug()