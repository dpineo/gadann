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
import numpy

from . import tensor


class Source(object):
    pass


class TensorSource(object):
    def __init__(self, t, batch_size=100):
        self.tensor = t
        self.batch_size = batch_size
        tensor.Tensor.ones_vector = tensor.ones((self.batch_size, 1))
        assert( not numpy.isnan(self.tensor.get()).any())

    def fprop(self, layer):
        for n, batch in enumerate(self):
            assert(not numpy.isnan(batch.get()).any())
            batch = layer.fprop(batch)
            if 'result' not in locals():
                result = tensor.Tensor((self.tensor.shape[0],)+batch.shape[1:])

            pycuda.driver.memcpy_dtod(int(result.gpuarray.gpudata) + n*batch.gpuarray.nbytes, batch.gpuarray.gpudata, batch.gpuarray.nbytes)
        assert(not numpy.isnan(result.get()).any())
        return TensorSource(result)

    def __iter__(self):
        self.batch = tensor.Tensor((self.batch_size,) + self.tensor.shape[1:], gpudata=self.tensor.gpuarray.gpudata)
        self.batch_n = 0
        return self

    def next(self):
        assert(not numpy.isnan(self.tensor.get()).any())
        if self.batch_n >= self.tensor.gpuarray.nbytes/self.batch.gpuarray.nbytes:
            # if (self.batch_n+1)*self.batch.gpuarray.nbytes > self.tensor.gpuarray.nbytes:
            raise StopIteration
        self.batch.gpuarray.gpudata = int(self.tensor.gpuarray.gpudata) + self.batch_n*self.batch.gpuarray.nbytes
        self.batch_n += 1
        assert( not numpy.isnan(self.batch.get()).any())
        return self.batch


class ImageDirectorySource(object):
    def __init__(self, directory):
        self.directory = directory

    def transform(self, function):
        tensor = function(tensor)
        return TensorSource(tensor)

    def __iter__(self):
        self.batch = Tensor((self.batch_size,) + self.tensor.shape[1:], gpudata=self.tensor.gpuarray.gpudata)
        self.batch_n = 0
        return self

    def next(self):
        if self.batch_n >= self.gpuarray.nbytes/self.batch.gpuarray.nbytes:
            raise StopIteration
        self.batch.gpuarray.gpudata = int(self.gpuarray.gpudata) + self.batch_n*self.batch.gpuarray.nbytes
        self.batch_n += 1
        return self.batch


class MessageQueueSource(object):
    def __init__(self, message_queue):
        pass

    def transform(self, function):
        tensor = function(tensor)
        return TensorSource(tensor)

    def __iter__(self):
        self.batch = Tensor((self.batch_size,) + self.tensor.shape[1:], gpudata=self.tensor.gpuarray.gpudata)
        self.batch_n = 0
        return self

    def next(self):
        if self.batch_n >= self.gpuarray.nbytes/self.batch.gpuarray.nbytes:
            raise StopIteration
        self.batch.gpuarray.gpudata = int(self.gpuarray.gpudata) + self.batch_n*self.batch.gpuarray.nbytes
        self.batch_n += 1
        return self.batch