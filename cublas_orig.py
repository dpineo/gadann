"""Wrappers around CUDA CUBLAS library.

For functions with names starting with cublas, please consult CUDA
BLAS (CUBLAS) library documentation.

Note
----
Since GPUArray stores matrix entries in row-major ordering, but CUBLAS
assumes column-major ordering, caution need to be taken when passing
GPUArray objects as arguments.

* No change need to be made for BLAS 1 functions.

* For BLAS 2 functions, the matrix is interpreted as transposed
  matrix, so the transp flag need to be set accordingly.
  
* For BLAS 3 functions, the input matrices and output matrix are
  interpreted as transposed matrices, so the order of matrix
  multiplication need to switched, while the transp flags should
  remain unchanged.

I have wrapped the most commonly used CUBLAS functions in saxpy, sgemv,
sgemm, etc. The precautions mentioned above have been taken for the
users. One can use these wrapped functions as templates to wrap other
cublas* functions.

Example
-------
>>> from numpy import arange, reshape, ones
>>> from pycuda.gpuarray import to_gpu, empty_like
>>> from parret.cublas import cublasSaxpy, saxpy, cublasSgemv, sgemv, cublasSgemm, sgemm
>>> alpha = 2.0
>>> x = to_gpu(arange(4, dtype='float32'))
>>> y = to_gpu(ones(4, dtype='float32'))
>>> 
>>> ## saxpy : y = alpha * x + y
>>> print 'alpha = ', alpha
alpha =  2.0
>>> print 'x = ',x
x =  [ 0.  1.  2.  3.]
>>> print 'y = ',y
y =  [ 1.  1.  1.  1.]
>>> cublasSaxpy(x.size, alpha, x, 1, y, 1)
>>> y
array([ 1.,  3.,  5.,  7.], dtype=float32)
>>> y = to_gpu(ones(4, dtype='float32'))
>>> saxpy(alpha, x, y)
array([ 1.,  3.,  5.,  7.], dtype=float32)
>>> 
>>> ## sgemv : y = A * x
>>> A = to_gpu(reshape(arange(16, dtype='float32'), (4,4)))
>>> print 'A =',A 
A = [[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]]
>>> print 'x = ',x
x =  [ 0.  1.  2.  3.]
>>> cublasSgemv('T', 4, 4, 1.0, A, 4, x, 1, 0.0, y, 1)
>>> y
array([ 14.,  38.,  62.,  86.], dtype=float32)
>>> y = to_gpu(ones(4, dtype='float32'))
>>> sgemv(A, x, y)
array([ 14.,  38.,  62.,  86.], dtype=float32)
>>> 
>>> ## sgemm : C = A * B
>>> B = to_gpu(reshape(arange(1,17, dtype='float32'), (4,4)))
>>> C = empty_like(B)
>>> print 'A =',A 
A = [[  0.   1.   2.   3.]
 [  4.   5.   6.   7.]
 [  8.   9.  10.  11.]
 [ 12.  13.  14.  15.]]
>>> print 'B =',B
B = [[  1.   2.   3.   4.]
 [  5.   6.   7.   8.]
 [  9.  10.  11.  12.]
 [ 13.  14.  15.  16.]]
>>> cublasSgemm('N', 'N', 4, 4, 4, 1.0, B, 4, A, 4, 0.0, C, 4)
>>> C
array([[  62.,   68.,   74.,   80.],
	   [ 174.,  196.,  218.,  240.],
	   [ 286.,  324.,  362.,  400.],
	   [ 398.,  452.,  506.,  560.]], dtype=float32)
>>> sgemm(A, B, C)
array([[  62.,   68.,   74.,   80.],
	   [ 174.,  196.,  218.,  240.],
	   [ 286.,  324.,  362.,  400.],
	   [ 398.,  452.,  506.,  560.]], dtype=float32)
"""

import _cublas
from _cublas import *

import util  # to add ctypes support to GPUArray objects

## Obtain the list of exported names
import re
__all__ = re.findall('cublas\w*' , ' '.join(dir(_cublas)))
__all__.remove('cublasStatus')
__all__ += [] #'scopy', 'sswap', 'sscal', 'sdot','snrm2', 'saxpy', 'sgemv', 'sgemm']

####################
# Useful utilities #
####################
CUBLAS_STATUS = {
	CUBLAS_STATUS_SUCCESS:          'CUBLAS_STATUS_SUCCESS',
	CUBLAS_STATUS_NOT_INITIALIZED:  'CUBLAS_STATUS_NOT_INITIALIZED',
	CUBLAS_STATUS_ALLOC_FAILED:     'CUBLAS_STATUS_ALLOC_FAILED',
	CUBLAS_STATUS_INVALID_VALUE:    'CUBLAS_STATUS_INVALID_VALUE', 
	CUBLAS_STATUS_MAPPING_ERROR:    'CUBLAS_STATUS_MAPPING_ERROR', 
	CUBLAS_STATUS_EXECUTION_FAILED: 'CUBLAS_STATUS_EXECUTION_FAILED', 
	CUBLAS_STATUS_INTERNAL_ERROR:   'CUBLAS_STATUS_INTERNAL_ERROR',
}
CUBLAS_STATUS_TYPE = ctypes.c_uint

def check_cublas_status(status):
	if status != CUBLAS_STATUS_SUCCESS:
	  raise Exception(CUBLAS_STATUS[status])

# cublasInit
def cublasInit():
	status = _cublas.cublasInit()
	check_cublas_status(status)

# cublasShutdown
def cublasShutdown():
	status = _cublas.cublasShutdown()
	check_cublas_status(status)    

def cublasGetError():
	status = _cublas.cublasGetError()
	check_cublas_status(status)

#########
# BLAS1 #
#########
'''
def scopy(x, y):
	cublasScopy(x.size, x, 1, y, 1)
	return y

def sswap(x, y):
	cublasSswap(x.size, x, 1, y, 1)
	return y

def sscal(alpha, x):
	cublasSscal(x.size, alpha, x, 1)
	return x

def sdot(x, y):
	return cublasSdot(x.size, x, 1, y, 1)

def snrm2(x):
	return cublasSnrm2(x.size, x, 1)

def saxpy(alpha, x, y):
	cublasSaxpy(x.size, alpha, x, 1, y, 1)
	return y
'''

#########
# BLAS2 #
#########
'''
def sgemv(A, x, y, trans='N', alpha=1.0, beta=0.0):
	## Because GPUArray is stored in row-major, while CUBLAS assumes
	## column-major, we have to do a transpose to make it right
	trans = 'T' if trans == 'N' else 'N'
	n, m = A.shape
	cublasSgemv(trans, m, n, alpha, A, m, x, 1, beta, y, 1)
	return y

'''
#########
# BLAS3 #
#########
#  The matricies are passed in as row-major, we interpret them 
# here as transposed column major to use with cublas.  We calculate CT in colum-major
# with the following formula:
#
#  C^T = (A*B)^T = B^T * A^T
#
#
# which gives us C in row-major
#
# http://stackoverflow.com/questions/14595750/transpose-matrix-multiplication-in-cublas-howto
#
import pycuda
import numpy

def test_segmm():
	# mxk * kxn ->mxn
	# 2x2 * 2x1 -> 2x1
	A = numpy.asarray([[.23,-.79],[.1,.21]], dtype=numpy.float32)
	A = pycuda.gpuarray.to_gpu(A)
	B = numpy.asarray([[.3],[.7]], dtype=numpy.float32)
	B = pycuda.gpuarray.to_gpu(B)
	C = pycuda.gpuarray.GPUArray((2,1), dtype=numpy.float32)
	cublas.sgemm(A, B, C, transa=False, transb=False)
	a = A.get()
	b = B.get()
	c = C.get()
	print "basic: ", numpy.linalg.norm(c-numpy.asarray([[-.484],[.177]]))
	print c
	pass

	# mxk * kxn ->mxn
	# 1x2 * 2x2 -> 1x2
	A = numpy.asarray([[.3,.7]], dtype=numpy.float32)
	A = pycuda.gpuarray.to_gpu(A)
	#B = numpy.asarray([[.23,.1],[-.79,.21]], dtype=numpy.float32)
	B = numpy.asarray([[.23,-.79],[.1,.21]], dtype=numpy.float32)
	B = pycuda.gpuarray.to_gpu(B)
	C = pycuda.gpuarray.GPUArray((1,2), dtype=numpy.float32)
	#cublas.sgemm(A, B, C, transa=False, transb=False)
	cublas.sgemm(A, B, C, transa=False, transb=True)
	a = A.get()
	b = B.get()
	c = C.get()
	print "swapped: ", numpy.linalg.norm(c-numpy.asarray([[-.484,.177]]))
	print c
	pass

	A = numpy.asarray([[.23,-.79],[.1,.21]], dtype=numpy.float32)
	A = pycuda.gpuarray.to_gpu(A)
	B = numpy.asarray([[.3,.7]], dtype=numpy.float32)
	B = pycuda.gpuarray.to_gpu(B)
	C = pycuda.gpuarray.GPUArray((2,1), dtype=numpy.float32)
	C.fill(0)
	cublas.sgemm(A, B, C, transa=False, transb=True)
	a = A.get()
	b = B.get()
	c = C.get()
	print "transposed b: ", numpy.linalg.norm(c-numpy.asarray([[-.484],[.177]]))
	print c
	pass

	A = numpy.asarray([[.23,-.79,0],[.1,.21,0],[0,0,0]], dtype=numpy.float32)
	A = pycuda.gpuarray.to_gpu(A)
	B = numpy.asarray([[.3,.7]], dtype=numpy.float32)
	B = pycuda.gpuarray.to_gpu(augment(B))
	C = pycuda.gpuarray.GPUArray((3,1), dtype=numpy.float32)
	C.fill(.12345)
	cublas.sgemm(A, B, C, transa=False, transb=True, augmented=True)
	a = A.get()
	b = B.get()
	c = C.get()
	print "augmented: ", numpy.linalg.norm(c-numpy.asarray([[-.484],[.177],[.12345]]))
	print c

	A = numpy.asarray([[.23,-.79,0],[.1,.21,0],[0,0,0]], dtype=numpy.float32)
	A = pycuda.gpuarray.to_gpu(A)
	B = numpy.asarray([[.3,.7]], dtype=numpy.float32)
	B = pycuda.gpuarray.to_gpu(augment(B))
	C = pycuda.gpuarray.GPUArray((1,3), dtype=numpy.float32)
	C.fill(.12345)
	cublas.sgemm(B, A, C, transa=False, transb=True, augmented=True)
	a = A.get()
	b = B.get()
	c = C.get()
	print "augmented: ", numpy.linalg.norm(c-numpy.asarray([[-.484,.177,.12345]]))
	print c

	A = numpy.asarray([[.3813,.54413,1]], dtype=numpy.float32)
	A = pycuda.gpuarray.to_gpu(A)
	B = numpy.asarray([[-.12,-.88,0]], dtype=numpy.float32)
	B = pycuda.gpuarray.to_gpu(augment(B))
	C = pycuda.gpuarray.GPUArray((1,1), dtype=numpy.float32)
	C.fill(.12345)
	cublas.sgemm(A, B, C, transa=False, transb=True)
	a = A.get()
	b = B.get()
	c = C.get()
	print "row weights: ", numpy.linalg.norm(c-numpy.asarray([[-.484],[.177],[0]]))
	print c

	print '---'
	pass



def sgemm(AT, BT, CT, transa=False, transb=False, alpha=1.0, beta=0.0, augmented=False):
	assert not isinstance(transa, str) and not isinstance(transb, str)

	# leading dimension is independent of whether transposed
	lda = AT.shape[1]
	ldb = BT.shape[1]
	ldc = CT.shape[1]

	((op_a_cols, op_a_rows), transa) = (AT.shape, 'T') if transa else (reversed(AT.shape), 'N')
	((op_b_cols, op_b_rows), transb) = (BT.shape, 'T') if transb else (reversed(BT.shape), 'N')    
	(ct_cols, ct_rows) = CT.shape

	k = op_a_cols
	m = ct_rows
	n = ct_cols

	# The last column of the output is a fixed column of ones, don't change them
	if augmented == True:
		m=m-1   # for row vectors on left
	if augmented == "bprop":
		n=n-1  # for column vectors on left

	cublasSgemm(transb, transa, m, n, k, alpha, BT, ldb, AT, lda, beta, CT, ldc)

	#c = pycuda.gpuarray.GPUArray(CT.shape, dtype=numpy.float32)
	#c.fill(0)
	#cublasSgemm(transb, transa, m, n, k, alpha, BT, ldb, AT, lda, beta, c, ldc)
	#c = c.get()
	#pass


def sgemv(A, x, y, trans='N', alpha=1.0, beta=0.0):
    ## Because GPUArray is stored in row-major, while CUBLAS assumes
    ## column-major, we have to do a transpose to make it right
    trans = 'T' if trans == 'N' else 'N'
    n, m = A.shape
    cublasSgemv(trans, m, n, alpha, A, m, x, 1, beta, y, 1)
    return y

if __name__ == "__main__":
	import doctest
	doctest.testmod()