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

import numpy
import gzip
import pickle
import os

from . import tensor

def load_mnist(path):
    # with gzip.open(os.path.join(gadann.module_dir, '..', 'data', 'mnist.pkl.gz'), 'rb') as file:
    with gzip.open(os.path.join(path, 'mnist.pkl.gz'), 'rb') as file:
        ((train_features, train_labels),
         (valid_features, valid_labels),
         (test_features, test_labels)) = pickle.load(file, encoding='latin1')

    self.train_features = self.train_features.reshape((50000, 1, 28, 28))
    self.test_features = self.test_features.reshape((10000, 1, 28, 28))

    self.train_features = gadann.Tensor(self.train_features, batch_size=100)
    self.train_labels = gadann.Tensor(self.train_labels, batch_size=100)
    self.train_labels_onehot = self.train_labels.apply_batchwise(tensor.onehot)

    self.test_features = gadann.Tensor(self.test_features, batch_size=100)
    self.test_labels = gadann.Tensor(self.test_labels, batch_size=100)
    self.test_labels_onehot = self.test_labels.apply_batchwise(tensor.onehot)


def load_norb(num_training_samples=1000, binarize_y=False):
    def read_bytes(file_handle, num_type, count):
        num_bytes = count * numpy.dtype(num_type).itemsize
        string = file_handle.read(num_bytes)
        return numpy.fromstring(string, dtype=num_type)

    def load(filename):
        file_handle = open(filename, encoding='latin1')
        type_key = read_bytes(file_handle, 'int32', 1)[0]
        num_dims = read_bytes(file_handle, 'int32', 1)[0]
        shape = numpy.fromfile(file_handle, dtype='int32', count=max(num_dims, 3))[:num_dims]
        num_elems = numpy.prod(shape)

        if type_key == 507333716:
            return numpy.fromfile(file_handle, dtype='int32', count=num_elems).reshape(shape)
        elif type_key == 507333717:
            return numpy.fromfile(file_handle, dtype='uint8', count=num_elems).reshape(shape).astype('float32')/256.0

    def binarize_labels(y, n_classes=10):
        new_y = numpy.zeros((n_classes, y.shape[0]))
        for i in range(y.shape[0]):
            new_y[y[i], i] = 1
        return new_y

    print(os.getcwd())
    features = load('data/smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat')
    labels = load('data/smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat')
    train_x = features[num_training_samples:]
    train_y = labels[num_training_samples:]
    test_x = features[:num_training_samples]
    test_y = labels[:num_training_samples]

    #reshape((50000, 1, 28, 28)

    if binarize_y:
        train_y = binarize_labels(train_y, n_classes=5)
        test_y = binarize_labels(train_y, n_classes=5)
    return ((train_x, train_y), (test_x, test_y))


class load_imagenet(object):
    def __init__(self, imagenet_filename, resize=256, batch_size=1):
        import cv2

        self.imagenet_filename = imagenet_filename
        self.resize = resize
        self.batch_size = batch_size
        self.batch = tensor.Tensor((self.batch_size,) + (3,resize,resize))
        tensor.Tensor.ones_vector = tensor.ones((self.batch_size,1))
        self.resize = resize

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
