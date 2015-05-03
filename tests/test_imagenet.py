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

import sys, os
sys.path.append(os.path.abspath(__file__)[:-len('\\tests\\test_imagenet.py')])

import cv2
import numpy
import gadann
import gzip
import cPickle
import numpy
import unittest
import cv2
import logging
import time
import csv


logger = logging.getLogger(__name__)

synset_labels = csv.DictReader(open('D:\\projects\\convnet\\data\\imagenet\\synset_labels.txt'), delimiter = '\t', fieldnames=['synset', 'label'])
synset_labels = { x['synset']:x['label'] for x in synset_labels }

def imagenet():
    import tarfile
    imagenet_tarfile = tarfile.open('D:\projects\convnet\\data\\imagenet\ILSVRC2014_DET_train.tar')
    for synset in imagenet_tarfile:
        if synset.isfile():
            synset = imagenet_tarfile.extractfile(synset)
            synset_name = os.path.splitext(os.path.basename(synset.name))[0]
            try:
                synset_label = synset_labels[synset_name]
                synset_tarfile = tarfile.open(fileobj=synset)
                for image in synset_tarfile:
                    if image.isfile():
                        image = synset_tarfile.extractfile(image)
                        yield synset_label, cv2.imdecode(numpy.fromstring(image.read(), dtype='uint8'),cv2.CV_LOAD_IMAGE_COLOR)
            except:
                pass

'''
synset_ids = set()
for label, image in imagenet():
    synset_ids.add(label)

    #cv2.imshow('image', image)
    #cv2.waitKey(1)
'''

class TestImagenet(unittest.TestCase):
    def setUp(self):
        logger.info('')
        logger.info("----- Test Case: %s -----" % (self._testMethodName))
        for line in self._testMethodDoc.replace('\t','').splitlines():
            logger.info(line)

        self.train = gadann.ImagenetStream('D:\projects\convnet\data\imagenet\ILSVRC2014_DET_train.tar', resize=256)
        numpy.random.seed(1234)
        self.start_time = time.time()

    def tearDown(self):
        logger.info("Elapsed time: " + str(time.time() - self.start_time))

    def test_imagenet(self):
        '''LinearLayer->Softmax.
        The Linear layers are fully connected and use cublas.
        Trained using batch gradient descent
        '''

        '''
        for image in self.train_features:
            cv2.imshow('image', image)
            cv2.waitKey(100)
            pass
        '''
        '''
        model = gadann.NeuralNetworkModel(
            layers=[
                gadann.DropoutLayer(.2),
                gadann.CudnnConvLayer(feature_shape=(3,11,11), n_features=16),
                gadann.ActivationLayer(activation='relu'),

                gadann.DropoutLayer(.5),
                gadann.CudnnConvLayer(feature_shape=(16,5,5), n_features=16),
                gadann.ActivationLayer(activation='relu'),

                gadann.DropoutLayer(.5),
                gadann.LinearLayer(feature_shape=(64*(256-11-5+2)*(256-11-5+2),), n_features=10),
                gadann.ActivationLayer(n_features=10, activation='softmax')
            ]
        )
        '''
        model = gadann.NeuralNetworkModel(
            input_shape = (3,256,256),
            layers = [
                {'layer':gadann.LinearLayer,     'n_features':32, 'shape':(32,32)},
                {'layer':gadann.ActivationLayer, 'activation':gadann.logistic},
                {'layer':gadann.LinearLayer,     'n_features':64},
                {'layer':gadann.ActivationLayer, 'activation':gadann.logistic},
                {'layer':gadann.LinearLayer,     'n_features':1000},
                {'layer':gadann.ActivationLayer, 'activation':gadann.softmax}
            ],
            updater = gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.0001)
        )


        #updater=gadann.SgdUpdater(learning_rate=0.01, weight_cost=0)
        #updater=gadann.MomentumUpdater(learning_rate=0.0001, inertia=0.0, weight_cost=0)
        updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.00)

        #gadann.ContrastiveDivergenceTrainer(model, model.updater).train(self.train_features, n_epochs=10)
        gadann.BatchGradientDescentTrainer(model, updater).train(self.train, n_epochs=10)

        train_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")


TestImagenet("test_imagenet").debug()


