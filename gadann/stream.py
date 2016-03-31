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


class Stream(object):
    pass


class TransformStream(object):
    def __init__(self, input, func):
        self.input = input
        self.func = func

    def __iter__(self):
        return map(self.func, self.input)


class TensorStream(object):
    def __init__(self, t, batch_size=100):
        self.tensor = t
        self.batch_size = batch_size
        tensor.Tensor.ones_vector = tensor.ones((self.batch_size,1))
        assert(not numpy.isnan(self.tensor.get()).any())

    def fprop(self, layer):
        for n, batch in enumerate(self):
            assert(not numpy.isnan(batch.get()).any())
            batch = layer.fprop(batch)
            if 'result' not in locals():
                result = tensor.Tensor((self.tensor.shape[0],)+batch.shape[1:])

            pycuda.driver.memcpy_dtod(int(result.gpuarray.gpudata) + n*batch.gpuarray.nbytes, batch.gpuarray.gpudata, batch.gpuarray.nbytes)
        assert(not numpy.isnan(result.get()).any())
        return TensorStream(result)

    def __iter__(self):
        self.batch = tensor.Tensor((self.batch_size,) + self.tensor.shape[1:], gpudata=self.tensor.gpuarray.gpudata)
        self.batch_n = 0
        return self

    def __next__(self):
        assert(not numpy.isnan(self.tensor.get()).any())
        if self.batch_n >= self.tensor.gpuarray.nbytes/self.batch.gpuarray.nbytes:
            raise StopIteration
        self.batch.gpuarray.gpudata = int(self.tensor.gpuarray.gpudata) + self.batch_n*self.batch.gpuarray.nbytes
        self.batch_n += 1
        assert(not numpy.isnan(self.batch.get()).any())
        return self.batch


class ImageDirectoryStream(object):
    def __init__(self, directory):
        self.directory = directory

    def transform(self, function):
        t = function(self.tensor)
        return TensorStream(t)

    def __iter__(self):
        self.batch = tensor.Tensor((self.batch_size,) + self.tensor.shape[1:], gpudata=self.tensor.gpuarray.gpudata)
        self.batch_n = 0
        return self

    def next(self):
        if self.batch_n >= self.gpuarray.nbytes/self.batch.gpuarray.nbytes:
            raise StopIteration
        self.batch.gpuarray.gpudata = int(self.gpuarray.gpudata) + self.batch_n*self.batch.gpuarray.nbytes
        self.batch_n += 1
        return self.batch


class ImagenetStream(object):
    def __init__(self, imagenet_filename, resize=256, batch_size=1):
        import cv2

        self.imagenet_filename = imagenet_filename
        self.resize = resize
        self.batch_size = batch_size
        self.batch = tensor.Tensor((self.batch_size,) + (3,resize,resize))
        tensor.Tensor.ones_vector = tensor.ones((self.batch_size,1))
        self.resize = resize

    def transform(self, function):
        tensor = function(tensor)
        return gadann.TensorDataStream(tensor)

    def __iter__(self):
        import tarfile
        imagenet_tarfile = tarfile.open(self.imagenet_filename)
        for category in imagenet_tarfile:
            if category.isfile():
                category = imagenet_tarfile.extractfile(category)
                category_tarfile = tarfile.open(fileobj=category)
                for image in category_tarfile:
                    if image.isfile():
                        image = category_tarfile.extractfile(image)
                        image = cv2.imdecode(numpy.fromstring(image.read(), dtype='uint8'),cv2.CV_LOAD_IMAGE_COLOR)
                        scale = float(self.resize)/min(image.shape[0], image.shape[1])+.01
                        image = cv2.resize(image, (int(scale*image.shape[1]), int(scale*image.shape[0])))
                        w0, h0 = [(x-self.resize)/2 for x in image.shape[:-1]]
                        image = image[w0:w0+self.resize,h0:h0+self.resize,:]
                        image = numpy.transpose(image)
                        image = numpy.reshape(image, (1,3,self.resize,self.resize))
                        image = numpy.ascontiguousarray(image)
                        self.batch_shape = image.shape
                        #yield tensor.Tensor(image)
                        yield {'features':tensor.Tensor(image), 'labels': 0}


class MessageQueueStream(object):
    def __init__(self, message_queue):
        pass

    def transform(self, function):
        tensor = function(tensor)
        return TensorDataStream(tensor)

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