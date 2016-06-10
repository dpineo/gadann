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

import os
import gzip
import pickle
import numpy
import unittest
import cv2
import logging
import time

import gadann

logger = logging.getLogger(__name__)


class TestMnist(unittest.TestCase):
    def setUp(self):
        logger.info('')
        logger.info("----- Test Case: %s -----" % (self._testMethodName))
        for line in self._testMethodDoc.replace('\t','').splitlines():
            logger.info(line)

        with gzip.open(os.path.join(gadann.module_dir, '..', 'data', 'mnist.pkl.gz'),'rb') as file:
            ((self.train_features, self.train_labels),
            (self.valid_features, self.valid_labels),
            (self.test_features, self.test_labels)) = pickle.load(file, encoding='latin1')

        self.train_features = self.train_features.reshape((50000, 1, 28, 28))
        self.test_features = self.test_features.reshape((10000, 1, 28, 28))

        #self.train_features = self.train_features.reshape((50000, 784))
        #self.test_features = self.test_features.reshape((10000, 784))

        self.train_features = gadann.Tensor(self.train_features)
        self.train_labels_onehot = gadann.Tensor(self.train_labels).as_onehot()
        self.train_labels = gadann.Tensor(self.train_labels)

        self.test_features = gadann.Tensor(self.test_features)
        self.test_labels_onehot = gadann.Tensor(self.test_labels).as_onehot()
        self.test_labels = gadann.Tensor(self.test_labels)

        self.train_features = gadann.TensorStream(self.train_features)
        self.train_labels = gadann.TensorStream(self.train_labels)
        self.train_labels_onehot = gadann.TensorStream(self.train_labels_onehot)

        self.test_features = gadann.TensorStream(self.test_features)
        self.test_labels = gadann.TensorStream(self.test_labels)
        self.test_labels_onehot = gadann.TensorStream(self.test_labels_onehot)

        numpy.random.seed(1234)
        self.start_time = time.time()

    def tearDown(self):
        logger.info("Elapsed time: " + str(time.time() - self.start_time))

    def test_mnist_singlelayer(self):
        """ LinearLayer->Softmax.
        The Linear layers are fully connected and use cublas.
        Trained using batch gradient descent
        """

        model = gadann.NeuralNetworkModel(
            input_shape=(1, 28, 28),
            layers=[
                {'layer': gadann.LinearLayer,     'n_features': 10},
                {'layer': gadann.ActivationLayer, 'activation': gadann.softmax}
            ],
        )
        updater = gadann.SgdUpdater(learning_rate=0.01, weight_cost=0.00)

        #gadann.BatchGradientDescentTrainer(model, updater).train(self.train_features, self.train_labels_onehot, n_epochs=100)
        model.train_sgd(self.train_features, self.train_labels_onehot, updater=updater, n_epochs=100)

        train_accuracy = model.evaluate(self.test_features, self.test_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")
        logger.info("Test set accuracy = " + str(test_accuracy*100) + "%")

    def test_mnist_multilayer(self):
        """ LinearLayer->Logistic->Linear->Softmax.
        The Linear layers are fully connected and use cublas.
        Trained using batch gradient descent
        """

        model = gadann.NeuralNetworkModel(
            input_shape=(1, 28, 28),
            layers=[
                {'layer': gadann.LinearLayer,     'n_features': 300},
                {'layer': gadann.ActivationLayer, 'activation': gadann.logistic},
                {'layer': gadann.LinearLayer,     'n_features': 10},
                {'layer': gadann.ActivationLayer, 'activation': gadann.softmax}
            ],
            updater=gadann.SgdUpdater(learning_rate=0.01, weight_cost=0)
        )

        '''
        model = gadann.NeuralNetworkModel(shape = (1,28,28))
        model.add(gadann.DropoutLayer,    prob=.2)
        model.add(gadann.LinearLayer,     n_features=300)
        model.add(gadann.ActivationLayer, activation=gadann.logistic)
        model.add(gadann.DropoutLayer,    prob=.5)
        model.add(gadann.LinearLayer,     n_features=10)
        model.add(gadann.ActivationLayer, activation=gadann.softmax)
        '''
         
        #updater=gadann.SgdUpdater(learning_rate=0.01, weight_cost=0)

        #gadann.ContrastiveDivergenceTrainer(model, updater).train(self.train_features, n_epochs=10)
        #gadann.BatchGradientDescentTrainer(model, updater).train(self.train_features, self.train_labels_onehot, n_epochs=10)
        model.train_contrastive_divergence(self.train_features, n_epochs=10)
        model.train_backprop(self.train_features, self.train_labels_onehot, n_epochs=10)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")
        logger.info("Test set accuracy = " + str(test_accuracy*100) + "%")

    def test_mnist_pretrained(self):
        """ First pretrain a single LinearLayer, Then use in
        LinearLayer->Logistic->Softmax model.
        The Linear layers are fully connected and use cublas.
        Trained using batch gradient descent
        """
        '''
        model = gadann.NeuralNetworkModel(
            layers=[
                gadann.LinearLayer(feature_shape=(1,28,28), n_features=512),
            ]
        )
        #updater=gadann.SgdUpdater(learning_rate=0.1, weight_cost=0)
        #updater=gadann.MomentumUpdater(learning_rate=0.1, inertia=0.9, weight_cost=0.01)
        updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.0)

        gadann.ContrastiveDivergenceTrainer(model,updater).train(self.train_features, n_epochs=10)

        neural_net_model = gadann.NeuralNetworkModel(
            layers=[
                model.layers[0],
                gadann.ActivationLayer(activation='logistic'),
                gadann.LinearLayer(feature_shape=(512,), n_features=10),
                gadann.ActivationLayer(n_features=10, activation='softmax')
            ]
        )
        '''
        model = gadann.NeuralNetworkModel(
            input_shape=(1, 28, 28),
            layers=[
                {'layer': gadann.DropoutLayer,    'prob': .2},
                {'layer': gadann.LinearLayer,     'n_features': 300},
                {'layer': gadann.ActivationLayer, 'activation': gadann.logistic},
                {'layer': gadann.DropoutLayer,    'prob': .5},
                {'layer': gadann.LinearLayer,     'n_features': 10},
                {'layer': gadann.ActivationLayer, 'activation': gadann.softmax}
            ],
            updater=gadann.RmspropUpdater(learning_rate=0.02, inertia=0.0, weight_cost=0.00)
        )

        #updater=gadann.SgdUpdater(learning_rate=0.1, weight_cost=0)
        #updater=gadann.MomentumUpdater(learning_rate=0.1, inertia=0.9, weight_cost=0.01)
        updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.00)

        gadann.ContrastiveDivergenceTrainer(model,updater).train(self.train_features, n_epochs=10)

        gadann.BatchGradientDescentTrainer(neural_net_model, model.updater).train(self.train_features, self.train_labels_onehot, n_epochs=10)

        train_accuracy = neural_net_model.evaluate(self.train_features, self.train_labels)
        test_accuracy = neural_net_model.evaluate(self.test_features, self.test_labels)
        print("Training set accuracy = " + str(train_accuracy*100) + "%")
        print("Test set accuracy = " + str(test_accuracy*100) + "%")

    def test_mnist_singlelayer_cudnn(self):
        """ CudnnConvLayer->Softmax.
        The Linear layers are convolutional and use cuDNN.
        Trained using batch gradient descent
        """

        model = gadann.NeuralNetworkModel(
            layers=[
                gadann.CudnnConvLayer(feature_shape=(1,28,28), n_features=10),
                gadann.ActivationLayer(n_features=10, activation='softmax')
            ]
        )
        print(model)

        #updater=gadann.SgdUpdater(learning_rate=0.001, weight_cost=0)
        #updater=gadann.MomentumUpdater(learning_rate=0.0001, inertia=0.0, weight_cost=0)
        updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.00)

        #gadann.ContrastiveDivergenceTrainer(model,updater).train(self.train_features, n_epochs=10)
        gadann.BatchGradientDescentTrainer(model,updater).train(self.train_features, self.train_labels_onehot, n_epochs=10)

        print(model)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")
        logger.info("Test set accuracy = " + str(test_accuracy*100) + "%")

    def test_mnist_multilayer_cudnn(self):
        """ CudnnConvLayer->Logistic->CudnnConvLayer->Softmax.
        The Linear layers are convolutional and use cuDNN.
        Trained using batch gradient descent
        """

        model = gadann.NeuralNetworkModel(
            layers=[
                gadann.ConvLayer(feature_shape=(1,16,16), n_features=64),
                gadann.ActivationLayer(activation='logistic'),
                gadann.DenseLayer(feature_shape=(64,13,13), n_features=10),
                gadann.ActivationLayer(activation='softmax')
            ]
        )
        #updater=gadann.SgdUpdater(learning_rate=0.001, weight_cost=0)
        #updater=gadann.MomentumUpdater(learning_rate=0.0001, inertia=0.0, weight_cost=0)
        updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.00)

        gadann.ContrastiveDivergenceTrainer(model, updater).train(self.train_features, n_epochs=10)
        gadann.BatchGradientDescentTrainer(model, updater).train(self.train_features, self.train_labels_onehot, n_epochs=10)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")
        logger.info("Test set accuracy = " + str(test_accuracy*100) + "%")
        self.assertTrue(test_accuracy > .95)

    def test_mnist_pretrained_cudnn(self):
        """
        Uses NVIDIA cuDNN library for convolutions
        """
        self.train_features = self.train_features.tensor.get()
        self.train_features = numpy.reshape(self.train_features, (50000, 1, 28, 28))
        self.train_features = gadann.TensorStream(gadann.Tensor(self.train_features), batch_size=100)

        model = gadann.NeuralNetwork(
            layers=[
                gadann.CudnnConvLayer(feature_shape=(1,16,16), n_features=50),
                gadann.CudnnConvLayer(feature_shape=(50,13,13), n_features=10),
            ]
        )
        gadann.ContrastiveDivergenceTrainer().train(model, self.train_features, n_epochs=1, learning_rate=0.01)

        model = gadann.NeuralNetwork(
            layers=[
                gadann.CudnnConvLayer(feature_shape=(1,16,16), n_features=50),
                gadann.ActivationLayer(activation='logistic'),
                gadann.CudnnConvLayer(feature_shape=(50,13,13), n_features=10),
                gadann.ActivationLayer(activation='softmax')
            ]
        )

        gadann.BatchGradientDescentTrainer(learning_rate=.001).train(model, self.train_features, self.train_labels_onehot, n_epochs=10)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")
        logger.info("Test set accuracy = " + str(test_accuracy*100) + "%")
        print("Training set accuracy = " + str(train_accuracy*100) + "%")
        print("Test set accuracy = " + str(test_accuracy*100) + "%")
        self.assertTrue(test_accuracy > .95)

    def test_mnist_localization(self):
        """ CudnnConvLayer->Logistic->CudnnConvLayer->Softmax.
        The Linear layers are convolutional and use cuDNN.
        Trained using batch gradient descent
        """

        model = gadann.NeuralNetworkModel(
            input_shape=(1, 28, 28),
            layers=[
                {'layer': gadann.LinearLayer,     'n_features': 512},
                {'layer': gadann.ActivationLayer, 'activation': gadann.logistic},
                {'layer': gadann.LinearLayer,     'n_features': 10},
                {'layer': gadann.ActivationLayer, 'activation': gadann.softmax}
            ],
            updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.0001)
        )


        print(model)

        #updater=gadann.SgdUpdater(learning_rate=0.001, weight_cost=0)
        #updater=gadann.MomentumUpdater(learning_rate=0.0001, inertia=0.0, weight_cost=0)
        #updater=gadann.RmspropUpdater(learning_rate=0.01, inertia=0.0, weight_cost=0.00)

        #gadann.ContrastiveDivergenceTrainer(model,updater).train(self.train_features, n_epochs=10)
        gadann.BatchGradientDescentTrainer(model, model.updater).train(self.train_features, self.train_labels_onehot, n_epochs=10)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        logger.info("Training set accuracy = " + str(train_accuracy*100) + "%")
        logger.info("Test set accuracy = " + str(test_accuracy*100) + "%")

        mnist_mosaic_rgb = cv2.imread("D:\projects\convnet\data\mnist\mnist_mosaic_sparse.png")/256.0
        mnist_mosaic = mnist_mosaic_rgb[...,0]
        mnist_mosaic = numpy.reshape(mnist_mosaic,(1,1,480,640))
        mnist_mosaic = numpy.ascontiguousarray(mnist_mosaic)
        mnist_mosaic = gadann.Tensor(mnist_mosaic)

        model.layers = model.layers[:-1]
        start_time = time.time()
        probabilities = model.probability(mnist_mosaic)
        print("Mosaic classify time: " + str(time.time() - start_time))

        probabilities = probabilities.get()

        localization = numpy.argmax(probabilities, axis=1)

        localization = numpy.squeeze(localization)
        cv2.imshow("localization", localization/10.0)

        probabilities -= numpy.min(probabilities,axis=1)
        probabilities /= numpy.sum(probabilities, axis=1)


        probabilities_squeezed = numpy.max(probabilities, axis=1)
        probabilities_squeezed = numpy.squeeze(probabilities_squeezed)
        cv2.imshow("probabilities_squeezed", probabilities_squeezed)


        print(probabilities_squeezed.min(), probabilities_squeezed.max(), probabilities_squeezed.mean())
        mnist_mosaic = numpy.squeeze(mnist_mosaic.get())
        cv2.imshow("mnist_mosaic", mnist_mosaic)
        cv2.waitKey(1)
        
        for i in range(10):
            detections = (localization == i).astype('float32')
            detections = numpy.dstack([numpy.zeros_like(detections), numpy.zeros_like(detections), detections*probabilities_squeezed])
            detections = numpy.pad(detections, ((13,14),(13,14),(0,0)), mode='constant')
            cv2.imshow(str(i), .1*mnist_mosaic_rgb + detections)
            #cv2.imshow("prob "+str(i), .1*mnist_mosaic_rgb + probabilities)
        cv2.waitKey(-1)
        
        print(model)



#TestMnist("test_mnist_singlelayer").debug()
TestMnist("test_mnist_multilayer").debug()

#unittest.main(exit=False, verbosity=2)

#TestMnist("test_mnist_multilayer").debug()
#TestMnist("test_mnist_pretrained").debug()

#TestMnist("test_mnist_singlelayer_cudnn").debug()
#TestMnist("test_mnist_multilayer_cudnn").debug()
#TestMnist("test_mnist_pretrained_cudnn").debug()

#TestMnist("test_mnist_localization").debug()

#from gadann.utils import profile
#profile('TestMnist("test_mnist_multilayer").debug()', locals())
cv2.waitKey(-1)