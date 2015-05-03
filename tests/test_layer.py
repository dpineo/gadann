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
import gzip
import cPickle
import numpy
import pycuda
import unittest

class TestLayer(unittest.TestCase):
	def setUp(self):
		with gzip.open('mnist.pkl.gz','rb') as file:
			self.train_set, self.valid_set, self.test_set = cPickle.load(file)

		self.data = numpy.asarray(self.train_set[0], dtype=numpy.float32)
		self.data = DataProvider(self.data, shape=(1,1,784), batch_size=100)

		numpy.random.seed(1234)
		self.model = NeuralNetwork(
			layers=[
				Layer(feature_shape=(1,1,784), n_features=500, activation='logistic'),
				Layer(feature_shape=(500,1,1), n_features=10, activation='softmax')
			]
		)


	def tearDown(self):
		pass

	def test_fprop_batch(self):
		self.model.train_rbm(self.data, n_epochs=10)
		#self.model.train_backprop(self.data, n_epochs=10)

if __name__ == '__main__':
	unittest.main(exit=False, verbosity=2, failfast=True)
