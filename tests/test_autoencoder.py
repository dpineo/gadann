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

import os, sys
sys.path = [os.path.dirname(os.getcwd())] + sys.path

import gadann
import gzip
import pickle
import numpy
import unittest


class TestContrastiveDivergenceDescentTrainer(unittest.TestCase):
    def setUp(self):
        with gzip.open('mnist.pkl.gz', 'rb') as file:
            ((self.train_features, self.train_labels),
            (self.valid_features, self.valid_labels),
            (self.test_features, self.test_labels)) = pickle.load(file, encoding='latin1')

        self.train_features = gadann.Tensor(self.train_features)
        self.train_labels_onehot = gadann.Tensor(self.train_labels).as_onehot()
        self.train_labels = gadann.Tensor(self.train_labels)

        self.test_features = gadann.Tensor(self.test_features)
        self.test_labels_onehot = gadann.Tensor(self.test_labels).as_onehot()
        self.test_labels = gadann.Tensor(self.test_labels)

        numpy.random.seed(1234)

    def test_autoencoder_singlelayer(self):

        model = gadann.NeuralNetwork(
            layers=[
                # Layer(feature_shape=(1,1,784), n_features=10, activation='softmax'),
                gadann.Layer(feature_shape=(1,1,784), n_features=500, activation='logistic'),
                gadann.Layer(feature_shape=(500,1,1), n_features=10, activation='softmax')
            ]
        )

        gadann.ContrastiveDivergenceTrainer(learning_rate=.01).train(model, self.train_features,  n_epochs=10)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        print("Training set accuracy = " + str(train_accuracy*100) + "%")
        print("Test set accuracy = " + str(test_accuracy*100) + "%")

    def test_autoencoder_multilayer(self):

        model = gadann.NeuralNetwork(
            layers=[
                gadann.Layer(feature_shape=(1,1,784), n_features=500, activation='logistic'),
                gadann.Layer(feature_shape=(500,1,1), n_features=10, activation='softmax')
            ]
        )

        # model.train_rbm(self.train_features, n_epochs=10)
        # model.train_backprop(self.train_features, self.train_labels_onehot, n_epochs=101)

        train_accuracy = model.evaluate(self.train_features, self.train_labels)
        test_accuracy = model.evaluate(self.test_features, self.test_labels)
        print("Training set accuracy = " + str(train_accuracy*100) + "%")
        print("Test set accuracy = " + str(test_accuracy*100) + "%")



TestContrastiveDivergenceDescentTrainer("test_autoencoder_singlelayer").debug()