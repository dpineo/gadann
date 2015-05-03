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

from deep_learning import Autoencoder, Layer, DataProvider
import gzip
import cPickle
import numpy

def image_2_design(image):
	image = pycuda.gpuarray.to_gpu(image.astype('float32'))
	design = pycuda.gpuarray.GPUArray(((image.shape[1]-16)*(image.shape[0]-16), 256), dtype=numpy.float32)
	image_2_design_kernel(image, design.gpudata, block=(16,16,1), grid=(image.shape[0]/16-1,image.shape[1]/16-1,1))
	design = design.get()
	return design

def design_2_image(design):
	design = pycuda.gpuarray.to_gpu(design)
	dim = int(math.sqrt(design.shape[0]))+16
	image = pycuda.gpuarray.zeros((dim,dim), dtype=numpy.float32)
	design_2_image_kernel(design, image.gpudata, block=(16,16,1), grid=(image.shape[0]/16-1,image.shape[1]/16-1,1))
	image = image.get()
	return image

def draw_design(design, file):
	# Plot filters from the weight matrix
	dim = int(math.sqrt(design.shape[1]))
	image = PIL.Image.fromarray(tile_raster_images( X = design,
				img_shape = (dim,dim),
				tile_shape = (dim,dim), 
				tile_spacing=(1,1),
				scale_rows_to_unit_interval=False))
	image.save(file)

def show_design(title, design):
	# Plot filters from the weight matrix
	dim = int(math.sqrt(design.shape[1]))
	image = tile_raster_images( X = design,
				img_shape = (dim,dim),
				tile_shape = (dim,dim), 
				tile_spacing=(1,1),
				scale_rows_to_unit_interval=False)
	cv2.imshow(title, image)

'''
# Get the dataset
f = gzip.open('mnist.pkl.gz','rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()
mnist_train_data = numpy.asarray(train_set[0], dtype=numpy.float32)

mnist_train_targets = numpy.zeros((len(train_set[1]),10), dtype=numpy.float32)
for i in range(len(train_set[0])):
	mnist_train_targets[int(i),train_set[1][int(i)]] = 1

# Create and train the model
NeuralNetwork = Autoencoder(
	layers=[
		Layer((mnist_train_data.shape[1], 500),activation='logistic', decoder_activation=None)
		#Layer((200,200),activation='logistic'),
		#Layer((500,mnist_train_targets.shape[1]),activation='logistic')
	]
)

mnist_train_targets = DataProvider(mnist_train_data, batch_size=100)
mnist_train_data = DataProvider(mnist_train_data, batch_size=100)

#NeuralNetwork.train_unsupervised(mnist_train_data, n_epochs=1)
#NeuralNetwork.layers[0].draw_layer("unsupervised")

NeuralNetwork.train_supervised(mnist_train_data, mnist_train_targets, n_epochs=1)
NeuralNetwork.layers[0].show_layer("layer0")

predictions = NeuralNetwork.predict(mnist_train_data)

accuracy = numpy.sqrt(((mnist_train_data.get() - predictions)**2).mean())
print "Train set accuracy = " + str(accuracy)

# Validate the model
mnist_test_data = numpy.asarray(test_set[0], dtype=numpy.float32)
mnist_test_data = DataProvider(mnist_test_data, batch_size=100)
predictions = NeuralNetwork.predict(mnist_test_data)
accuracy = numpy.sqrt(((mnist_test_data.get() - predictions)**2).mean())
print "Test set accuracy = " + str(accuracy)

show_design("predictions", predictions)
cv2.waitKey(-1)
pass
'''