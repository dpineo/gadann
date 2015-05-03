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

import pycuda
import pycuda.gpuarray
import pycuda.compiler
import pycuda.autoinit
import numpy
from kernels import *
import unittest
from tensor import *
import time
from scipy.ndimage.filters import convolve


'''
input = numpy.random.randint(0,9,(2,2,3,3))
#input = numpy.random.randint(0,9,(5,3))
naive_input = input.copy()

input = gadann.Tensor(input)
naive_input = gadann.Tensor(naive_input)

result = gadann.kernels.softmax(input)
print 'softmax result'
print result.get()

naive_result = gadann.kernels.naive_softmax(naive_input)
print 'naive result'
print naive_result.get()
'''

class TestMatrixMultiply(unittest.TestCase):
	def setUp(self):
		self.startTime = time.time()

	def tearDown(self):
		t = time.time() - self.startTime
		print " %.3f seconds" % (t)

	def test_multiply(self):
		self.A_numpy = numpy.random.randn(100,784).astype(numpy.float32)
		self.B_numpy = numpy.random.randn(784,500).astype(numpy.float32)
		self.A = Tensor(self.A_numpy)
		self.B = Tensor(self.B_numpy)

		numpy.testing.assert_almost_equal(
			(self.A*self.B).get(),
			numpy.dot(self.A_numpy,self.B_numpy),
			decimal=3
		)

	def test_multiply_transposed(self):
		self.A_numpy = numpy.random.randn(784,100).astype(numpy.float32)
		self.B_numpy = numpy.random.randn(784,500).astype(numpy.float32)
		self.A = Tensor(self.A_numpy)
		self.B = Tensor(self.B_numpy)

		numpy.testing.assert_almost_equal(
			(self.A.T()*self.B).get(),
			numpy.dot(self.A_numpy.transpose(),self.B_numpy),
			decimal=3
		)

	def test_multiply_allocator(self):
		self.A_numpy = numpy.random.randn(100,784).astype(numpy.float32)
		self.B_numpy = numpy.random.randn(784,500).astype(numpy.float32)
		self.A = Tensor(self.A_numpy)
		self.tmp = Tensor(self.B_numpy)
		self.B = Tensor(self.tmp.shape, gpudata=self.tmp.gpudata)

		numpy.testing.assert_almost_equal(
			(self.A*self.B).get(),
			numpy.dot(self.A_numpy,self.B_numpy),
			decimal=3
		)

	def test_sgemv(self):
		A_numpy = numpy.random.randn(784,100).astype(numpy.float32)
		x_numpy = numpy.random.randn(100,1).astype(numpy.float32)
		A = Tensor(A_numpy)
		x = Tensor(x_numpy)

		y = Tensor((784,1))
		sgemv(y, A, x)
		numpy.testing.assert_almost_equal(
			y.get(),
			numpy.dot(A_numpy,x_numpy),
			decimal=3
		)

	
	def test_sgemv_transposed(self):
		A_numpy = numpy.random.randn(100,784).astype(numpy.float32)
		x_numpy = numpy.random.randn(100,1).astype(numpy.float32)
		A = Tensor(A_numpy)
		x = Tensor(x_numpy)

		y = Tensor(numpy.zeros((784,1)))
		sgemv(y, A.T(), x)

		numpy.testing.assert_almost_equal(
			y.get(),
			numpy.dot(A_numpy.transpose(),x_numpy),
			decimal=3
		)

			
	def test_broadcast(self):
		A_numpy = numpy.random.randn(100,784).astype(numpy.float32)
		x_numpy = numpy.random.randn(784,).astype(numpy.float32)
		A = Tensor(A_numpy)
		x = Tensor(x_numpy)

		numpy.testing.assert_almost_equal(
			(A+x).get(),
			(A_numpy+x_numpy),
			decimal=3
		)

			
	def test_mult_conv(self):
		A_numpy = numpy.random.randn(32,32).astype(numpy.float32)
		B_numpy = numpy.random.randn(16,16).astype(numpy.float32)
		A = Tensor(A_numpy)
		B = Tensor(B_numpy)
		print convolve(A_numpy,B_numpy)

		numpy.testing.assert_almost_equal(
			(A*B).get(),
			convolve(A_numpy,B_numpy),
			decimal=3
		)

		
	def test_fprop(self):
		import itertools
		import gzip
		import cPickle
		with gzip.open('mnist.pkl.gz','rb') as file:
			((self.train_features, self.train_labels),
			(self.valid_features, self.valid_labels),
			(self.test_features, self.test_labels)) = cPickle.load(file)

		self.train_features = Tensor(self.train_features, batch_size=100)
		self.train_labels_onehot = Tensor(self.train_labels, batch_size=100).as_onehot()
		self.train_labels = Tensor(self.train_labels, batch_size=100)

		self.test_features = Tensor(self.test_features, batch_size=100)
		self.test_labels_onehot = Tensor(self.test_labels, batch_size=100).as_onehot()
		self.test_labels = Tensor(self.test_labels, batch_size=100)


		W_numpy = numpy.random.randn(784,500).astype(numpy.float32)
		x_numpy = numpy.random.randn(100,784).astype(numpy.float32)

		W = Tensor(W_numpy)
		x = Tensor(x_numpy)

		for input_batch, target_batch in itertools.izip(self.train_features, self.train_labels):
			result = logistic(input_batch*W + input_batch*W)


TestMatrixMultiply("test_mult_conv").debug()

#if __name__ == '__main__':
#	unittest.main(exit=False, verbosity=2)
