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
import operator
import ctypes
import math

import kernels
import cublas

mempool = pycuda.tools.DeviceMemoryPool()

# y <- a*x + y
def saxpy(y, x, alpha=1):
	assert y.gpuarray.size == x.gpuarray.size
	cublas.cublasSaxpy(y.gpuarray.size, alpha, x.gpuarray, 1, y.gpuarray, 1)
	assert( not numpy.isnan(y.get()).any())

# A <- alpha*x*yT + A
def sger(y, x, AT, alpha=1):
	at_rows, at_cols = AT.shape
	lda = AT.gpuarray.shape[1]
	status = cublasSger(at_rows, at_cols, alpha, x.gpuarray, 1, y.gpuarray, 1, AT.gpuarray, lda)
	assert(status == cublas.CUBLAS_STATUS_SUCCESS)
	assert( not numpy.isnan(AT.get()).any())

# y <- alpha*A*x + beta*y
def sgemv(y, AT, x, alpha=1, beta=0):
	at_rows, at_cols = AT.shape
	at_gpuarray_rows, at_gpuarray_cols = AT.gpuarray.shape

	y_rows, y_cols = y.shape
	x_rows, x_cols = x.shape
	lda = AT.gpuarray.shape[1]
	trans = 'T' if not AT.transposed else 'N'
	'''
	assert at_rows == y_rows
	assert at_cols == x_rows
	assert x_cols == 1
	assert y_cols == 1
	'''
	assert( not numpy.isnan(x.get()).any())
	assert( not numpy.isnan(AT.get()).any())

	cublas.cublasSgemv(trans, at_gpuarray_cols, at_gpuarray_rows, alpha, AT.gpuarray, lda, x.gpuarray, 1, beta, y.gpuarray, 1)
	assert( not numpy.isnan(y.get()).any())

# C = alpha*A*B + beta*C
def sgemm(AT, BT, CT, alpha=1, beta=0):
	at_cols, at_rows = AT.reduced_shape()
	bt_cols, bt_rows = BT.reduced_shape()
	ct_cols, ct_rows = CT.reduced_shape()
	lda = reduce(operator.__mul__, AT.gpuarray.shape[1:], 1)
	ldb = reduce(operator.__mul__, BT.gpuarray.shape[1:], 1)
		 
	transa = 'T' if AT.transposed else 'N'
	transb = 'T' if BT.transposed else 'N'

	#print at_cols, ct_cols  # unique dim of first array
	#print at_rows, bt_cols  # common dimension
	#print bt_rows, ct_rows  # unique dim of second array
	'''
	assert at_rows == bt_cols  # common dimension
	assert bt_rows == ct_rows  # unique dim of first array
	assert at_cols == ct_cols  # unique dim of second array
	'''

	cublas.cublasSgemm(transb, transa, bt_rows, at_cols, bt_cols, alpha, BT.gpuarray, ldb, AT.gpuarray, lda, beta, CT.gpuarray, bt_rows)
	assert( not numpy.isnan(CT.get()).any())


def zeros(shape):
	result = Tensor(shape)
	result.gpuarray.fill(0)
	assert( not numpy.isnan(result.get()).any())
	return result

def ones(shape):
	result = Tensor(shape)
	result.gpuarray.fill(1)
	assert( not numpy.isnan(result.get()).any())
	return result

def bernoulli(shape, prob):
	result = Tensor(shape)
	result.gpuarray.fill(prob)
	result = kernels.sample(result)
	assert( not numpy.isnan(result.get()).any())
	return result

#-----------------------  Tensor  --------------------------
class Tensor(object):
	def __init__(self, arg, axes=None, gpudata=None):
		if isinstance(arg, tuple):
			if gpudata:
				self.gpuarray = pycuda.gpuarray.GPUArray(arg, dtype='float32', gpudata=gpudata)
			else:
				self.gpuarray = pycuda.gpuarray.GPUArray(arg, dtype='float32', allocator=mempool.allocate)
				if __debug__:
					self.gpuarray.fill(numpy.nan)
		else:
			self.gpuarray = pycuda.gpuarray.GPUArray(arg.shape, dtype='float32', allocator=mempool.allocate)
			self.gpuarray.set(arg.astype('float32'))

		#self.shape = self.gpuarray.shape
		self.transposed = False
		self.axes = axes

	def dim(self, name):
		return self.shape[self.axis(name)]

	def axis(self, name):
		return self.axes.index(name)

	# Returns a pointer to the data in GPU device memory
	def data(self):
		return ctypes.c_void_p(int(self.gpuarray.gpudata))

	def get_shape(self):
		if not self.transposed:
			return self.gpuarray.shape
		else:
			return self.gpuarray.shape[1:] + (self.gpuarray.shape[0],)

	def set_shape(self, shape):
		if not self.transposed:
			self.gpuarray.shape = shape
		else:
			self.gpuarray.shape = shape[1:] + (shape[0],)

	shape = property(get_shape, set_shape)

	# Gets the shape with the convolutional dims collapsed to 1D
	def reduced_shape(self):
		if not self.transposed:
			x= (self.gpuarray.shape[0], reduce(operator.__mul__, self.gpuarray.shape[1:], 1))
		else:
			x= (reduce(operator.__mul__, self.gpuarray.shape[1:], 1), self.gpuarray.shape[0])
		#print self.shape, self.gpuarray.shape, x
		return x

	def __sub__(self, other):
		result = Tensor(self.shape)
		pycuda.driver.memcpy_dtod(result.gpuarray.gpudata, self.gpuarray.gpudata, self.gpuarray.nbytes)
		saxpy(result, other, alpha=-1.0)
		assert( not numpy.isnan(result.get()).any())
		return result

	def __add__(self, other):
		result = Tensor(self.shape)
		pycuda.driver.memcpy_dtod(result.gpuarray.gpudata, self.gpuarray.gpudata, self.gpuarray.nbytes)
		assert( not numpy.isnan(result.get()).any())
		saxpy(result, other, alpha=1)
		assert( not numpy.isnan(result.get()).any())
		return result

	def __mul__(self, other):
		if type(other) == float:
			result = zeros(self.shape)
			saxpy(result, self, alpha=other)
			return result
		else:
			result = kernels.mult(self, other)
			return result

	def dot(self, other):
		#assert ( self.shape[:1] == other.shape[:1] )
		if self.transposed:
			result = Tensor(self.gpuarray.shape[1:] + other.gpuarray.shape[1:])
		elif other.transposed:
			result = Tensor((self.gpuarray.shape[0],other.gpuarray.shape[0]))
		else:
			result = Tensor((self.gpuarray.shape[0],)+other.gpuarray.shape[1:])
		#print 'gpuarray:    %s %d * %s %d -> %s %d' % (self.gpuarray.shape, self.transposed, other.gpuarray.shape, other.transposed, result.gpuarray.shape, result.transposed)
		#print 'interpreted: %s * %s -> %s' % (self.shape, other.shape, result.shape)
		#print 'reduced:     %s * %s -> %s' % (self.reduced_shape(), other.reduced_shape(), result.reduced_shape())
		sgemm(self, other, result)
		assert( not numpy.isnan(result.get()).any())
		return result

	def __div__(self, other):
		if type(other) == float:
			result = zeros(self.shape)
			saxpy(result, self, alpha=1.0/other)
			return result
		else:
			result = kernels.div(self, other)
			return result

	def __pow__(self, scalar):
		return Tensor((self.gpuarray**scalar).get())

	def outer(self, other):
		#revisit
		raise NotImplementedError
		assert self.shape[1] == 1 and other.shape[0] == 1
		result = Tensor((other.shape[1],self.shape[0])).T()
		sger(self, other, result)
		assert( not numpy.isnan(result.get()).any())
		return result

	def broadcast(self, shape):
		# revisit
		raise NotImplementedError
		result = Tensor(shape)
		broadcast_kernel(self, result)
		assert( not numpy.isnan(result.get()).any())
		return result

	def __rmul__(self, scalar):
		result = Tensor(self.shape)
		saxpy(result, self, alpha=scalar)
		assert( not numpy.isnan(result.get()).any())
		return result

	def sample(self):
		result = sample(self)
		assert( not numpy.isnan(self.get()).any())
		return result.gpuarray.get()

	def get(self):
		if not self.transposed:
			return self.gpuarray.get()
		else:
			return self.gpuarray.get().transpose()
	
	def as_onehot(self):
		data = self.get()
		onehot = numpy.zeros((data.shape[0],max(data)+1), dtype=numpy.float32)
		for i in range(data.shape[0]):
			onehot[i,data[i]] = 1
		return Tensor(onehot)

	def T(self):
		result = Tensor(self.shape, gpudata=self.gpuarray.gpudata)
		#result.shape = tuple(reversed(result.shape))
		result.transposed = True
		return result

	def mosaic(self):
		result = kernels.mosaic(self)
		assert( not numpy.isnan(result.get()).any())
		return result

