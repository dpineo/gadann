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
import unittest
import cv2

class TestImage(unittest.TestCase):
	def setUp(self):
		'''
		dir = 'D:\\projects\\prime\\data\\dc9\\day2_chan1_png\\'
		img_list = []
		for n in range(10):
			img_list.append(cv2.imread(dir+'sen0_000'+str(n)+'.png')[numpy.newaxis,:,:,:])
		img_array = numpy.concatenate(img_list, axis=0)
		img_array = numpy.ascontiguousarray(numpy.rollaxis(img_array,3,1))
		img_array = gadann.Tensor(img_array)
		self.train_features = gadann.TensorStream(img_array, batch_size=1)
		'''

		img_array = numpy.ones((1,32,32))
		img_array[:,:,16:] += 1
		img_array[:,16:,:] += 2
		#cv2.imshow("img", img_array)
		cv2.imshow("img", cv2.resize(cv2.merge(img_array)/10, (512,512), interpolation=cv2.INTER_NEAREST))
		cv2.waitKey(1)
		img_array = gadann.Tensor(img_array)
		self.train_features = gadann.TensorStream(img_array, batch_size=1)
		numpy.random.seed(1234)
		numpy.set_printoptions(threshold='nan')

	def test_grayscale_image(self):
		model = gadann.NeuralNetwork(
			layers=[
				gadann.Layer(feature_shape=(1,16,16), n_features=20, activation='logistic'),
				gadann.Layer(feature_shape=(20,13,13), n_features=20, activation='logistic'),
				gadann.Layer(feature_shape=(500,), n_features=20, activation='softmax')
			]
		)
		gadann.ContrastiveDivergenceTrainer().train(model, self.train_features, n_epochs=10, learning_rate=.0001)

	def test_rgb_image(self):
		model = gadann.NeuralNetwork(
			layers=[
				gadann.Layer(feature_shape=(3,16,16), n_features=20, activation='logistic'),
				gadann.Layer(feature_shape=(20,13,13), n_features=20, activation='logistic'),
				gadann.Layer(feature_shape=(500,), n_features=20, activation='softmax')
			]
		)

		gadann.ContrastiveDivergenceTrainer().train(model, self.train_features, n_epochs=10, learning_rate=.0001)

#unittest.main(exit=False, verbosity=2)

TestImage("test_grayscale_image").debug()

