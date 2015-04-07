import sys
import pycuda
import pycuda.curandom
import pycuda.compiler
from kernels import *
from mult import augment, unaugment, sgemv
import numpy
import gzip
import cPickle
from plot_utils import tile_raster_images
import PIL.Image
import time
import matplotlib.pyplot
import math
import pycuda.autoinit
import cv2
import operator
import itertools
import copy
from _cublas import cublasSgemm


sample = pycuda.elementwise.ElementwiseKernel(
	"float* x, float* p, float* out",
	"if (x[i] > p[i])  {out[i]=1.0;} else {out[i] = 0.0;}",
	"sample")

update_weights = pycuda.elementwise.ElementwiseKernel(
	"float* weights,  float batch_size, float learning_rate, float* pos_prods, float* neg_prods, float weight_cost",
	"weights[i] += 0.1 * ((pos_prods[i] - neg_prods[i])/batch_size - weight_cost * weights[i])",
	"update_weights")

calculate_error = pycuda.elementwise.ElementwiseKernel(
	"float* data, float* error_tmp",
	"float v = data[i]-error_tmp[i]; error_tmp[i] = v*v;",
	"calculate_error")


def seed_getter(count):
	return pycuda.gpuarray.arange(1, count, 1, dtype=numpy.float32)

numpy.random.seed(1234)
rand_gen = pycuda.curandom.XORWOWRandomNumberGenerator()
#rand_gen = pycuda.curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter)

#-----------------------  Dataset  --------------------------
class Dataset(object):
	def __init__(self, data, shape, batch_size, zero_mean=False, batch=True):
		self.zero_mean = zero_mean
		self.batch_size = batch_size

		# REVISIT: remove this and make it a decorator
		if self.zero_mean:
			self.means = data.sum(axis=1)/data.shape[1]
			data -= self.means[:,numpy.newaxis]

		self.output_shape = (batch_size,) + shape
		
		self.data = pycuda.gpuarray.to_gpu(data.astype('float32'))
		self.batch = pycuda.gpuarray.GPUArray(self.output_shape, dtype=numpy.float32, gpudata=self.data.gpudata)
		self.y = self.batch

	def get(self):
		if self.zero_mean:
			return self.data.get() + self.means
		else:
			return self.data.get()

	def __iter__(self):
		self.batch.gpudata = int(self.data.gpudata) - self.batch.nbytes
		return self

	def next(self):
		self.batch.gpudata += self.batch.nbytes
		if self.batch.gpudata + self.batch.nbytes > int(self.data.gpudata) + self.data.nbytes:
			raise StopIteration
		return self.batch

#-----------------------  Layer  --------------------------
class Layer(object):
	def __init__(self, feature_shape, n_features, activation='logistic', decoder_activation=None, name=None, learning_rate=0.001, weight_cost=0.0):

		self.weight_cost = weight_cost
		self.learning_rate = learning_rate

		# model parameters
		self.name = name
		self.feature_shape = feature_shape
		self.feature_size = reduce(operator.__mul__, self.feature_shape, 1)
		self.n_features = n_features

		self.prev = self.next = None
		self.w = numpy.zeros([self.feature_size, self.n_features]).astype(numpy.float32)
		self.w = 0.01*numpy.random.randn(*self.w.shape).astype(numpy.float32)
		self.w = pycuda.gpuarray.to_gpu(self.w)

		self.b = pycuda.gpuarray.zeros((1, self.n_features), dtype=numpy.float32)

		self.v_bias = pycuda.gpuarray.to_gpu(numpy.zeros([self.feature_size]).astype(numpy.float32))
		self.h_bias = pycuda.gpuarray.to_gpu(numpy.zeros([self.n_features]).astype(numpy.float32))

		self.activation = globals()[activation]
		self.d_activation = globals()['d'+activation]

	# Allocates the GPU memory needed to do a forward propogation
	def init_fprop(self):
		self.input_shape = self.prev.output_shape
		batch_size, feature_size, x_size, y_size = self.input_shape
		self.output_shape = (batch_size, self.n_features, self.input_shape[-2]-self.feature_shape[-2]+1, self.input_shape[-1]-self.feature_shape[-1]+1)

		self.bias_ones = pycuda.gpuarray.zeros((batch_size, 1, 1, 1), dtype=numpy.float32)+1

		self.x = pycuda.gpuarray.GPUArray(self.input_shape, dtype=numpy.float32)
		self.x.fill(1)
		self.z = pycuda.gpuarray.GPUArray(self.output_shape, dtype=numpy.float32)
		self.z.fill(1)
		self.y = pycuda.gpuarray.GPUArray(self.output_shape, dtype=numpy.float32)
		self.y.fill(1)
		if self.prev:
			self.x.gpudata = int(self.prev.y.gpudata)
			

	def init_bprop(self):
		self.init_fprop()
		self.de_dy = pycuda.gpuarray.GPUArray(self.output_shape, dtype=numpy.float32)
		self.de_dy.fill(1)


	def init_rbm(self):

		self.init_fprop()

		self.pv = pycuda.gpuarray.to_gpu(numpy.ones(self.input_shape, dtype=numpy.float32))
		self.v = pycuda.gpuarray.to_gpu(numpy.ones(self.input_shape, dtype=numpy.float32))
		self.ph = pycuda.gpuarray.to_gpu(numpy.ones(self.output_shape, dtype=numpy.float32))
		self.h = pycuda.gpuarray.to_gpu(numpy.ones(self.output_shape, dtype=numpy.float32))

		self.pos_grad_w = pycuda.gpuarray.zeros(self.w.shape, dtype=numpy.float32)
		self.neg_grad_w = pycuda.gpuarray.zeros(self.w.shape, dtype=numpy.float32)

		self.pos_grad_v = pycuda.gpuarray.zeros(self.v_bias.shape, dtype=numpy.float32)
		self.neg_grad_v = pycuda.gpuarray.zeros(self.v_bias.shape, dtype=numpy.float32)

		self.pos_grad_h = pycuda.gpuarray.zeros(self.h_bias.shape, dtype=numpy.float32)
		self.neg_grad_h = pycuda.gpuarray.zeros(self.h_bias.shape, dtype=numpy.float32)

		self.bias_ones = pycuda.gpuarray.zeros((self.h.shape[0],), dtype=numpy.float32)+1

		self.rng = pycuda.gpuarray.GPUArray(self.input_shape, dtype=numpy.float32)

	def plot_layer(self):
		filter_fig = matplotlib.pyplot.figure(self.name)
		weights = self.w.get()
		for n in range(min(100,weights.shape[1]-1)):
			filter_fig.add_subplot(10,10,n,frameon=False, xticks=[], yticks=[])
			filter = weights[:-1,n]
			filter = filter.reshape((int(filter.size**.5),int(filter.size**.5)))
			matplotlib.pyplot.imshow(filter)
		matplotlib.pyplot.tight_layout(pad=1, w_pad=1, h_pad=1)
		matplotlib.pyplot.show()

	def draw_layer(self, title):
		# Plot filters from the weight matrix
		dim = int(math.sqrt(self.w.shape[0]))
		image = PIL.Image.fromarray(tile_raster_images( X = self.w.get().T,
					img_shape = (dim,dim),tile_shape = (dim,dim), 
					tile_spacing=(1,1)))
		image.save(title+'.png')

	def show_layer(self, title):
		# Plot filters from the weight matrixas
		dim = int(math.sqrt(self.w.shape[0]))
		image = tile_raster_images( X = self.w.get()[:-1,:].T,
					img_shape = (dim,dim),tile_shape = (dim,dim), 
					tile_spacing=(1,1))
		cv2.imshow(title, image)

	# fprop_activation
	def fprop(self, input_dataset):
		output_dataset = Dataset(numpy.zeros((input_dataset.data.shape[0],)+self.output_shape[1:], dtype=numpy.float32), shape=self.output_shape[1:], batch_size=input_dataset.batch_size)
		for (input_batch, output_batch) in itertools.izip(input_dataset, output_dataset):
			assert( not numpy.isnan(input_batch.get()).any())
			assert( not numpy.isnan(output_batch.get()).any())

			#cublas.sgemm(input_batch, self.w, output_batch, transa=False, transb=False, augmented=True)
			#multiply(input_batch, self.w, output_batch)
			multiply_cublas_with_bias(input_batch, self.b, self.w, output_batch)

			assert( not numpy.isnan(output_batch.get()).any())

			self.activation(output_batch, output_batch)
			assert( not numpy.isnan(output_batch.get()).any())
		
		assert( not numpy.isnan(output_dataset.data.get()).any())
		return output_dataset


	def bprop(self, output_dataset):
		input_dataset = Dataset(numpy.zeros((output_dataset.data.shape[0], self.shape[0]), dtype=numpy.float32), batch_size=output_dataset.batch_size)
		for (input_batch, output_batch) in itertools.izip(input_dataset, output_dataset):
			assert( not numpy.isnan(input_batch.get()).any())
			assert( not numpy.isnan(output_batch.get()).any())

			#cublas.sgemm(output_batch, self.w, input_batch, transa=False, transb=True, augmented=True)
			multiply(output_batch, self.w, input_batch, transpose='B')
			assert( not numpy.isnan(output_batch.get()).any())

			self.activation(input_batch, input_batch)
			assert( not numpy.isnan(input_batch.get()).any())
		
		assert( not numpy.isnan(input_dataset.data.get()).any())
		return input_dataset

	def fprop_batch(self):
		if True: #self.feature_shape == (1,1):
			multiply_cublas_with_bias(self.v, self.v_bias, self.w, self.ph)
			self.activation(self.ph, self.ph)
			rand_gen.fill_uniform(self.rng)
			sample(self.ph, self.rng, self.h)
		else:
			conv2d(self.v, self.w, self.ph, self.output_shape)
			#self.activation(self.ph, self.ph, self.ph.shape)
			self.activation(self.ph, self.ph, self.output_shape) 
			rand_gen.fill_uniform(self.rng)
			sample(self.ph, self.rng, self.h)
		assert( not numpy.isnan(self.ph.get()).any())
		assert( not numpy.isnan(self.h.get()).any())

	def bprop_batch(self):
		if True: #self.feature_shape[1] == 1:

			#print self.h.get()
			#print self.h_bias.get()
			#print self.w.get()
			#print self.pv.get()

			multiply_cublas_with_bias(self.h, self.h_bias, transpose(self.w), self.pv)

			#print self.pv.get()

			self.activation(self.pv, self.pv)
			#print self.pv.get()
			rand_gen.fill_uniform(self.rng)
			sample(self.pv, self.rng, self.v)
			#print self.v.get()
			#pass

		else:
			conv2d(v_in, layer.w, v_out_prob)
			layer.activation(v_out_prob, v_out_prob, v_out_prop.shape)
			rand_gen.fill_uniform(layer.rng)
			sample(v_out_prob, layer.rng, v_out_sample)
		assert( not numpy.isnan(self.pv.get()).any())
		assert( not numpy.isnan(self.v.get()).any())
		
#--------------------  MLP  ---------------------
class MLP(object):
	def __init__(self, layers, learning_rate=0.1, weight_cost=0.02):
		self.layers = layers
		self.learning_rate = learning_rate
		for n, (layer, prev) in enumerate(itertools.izip(self.layers+[None],[None]+self.layers)):
			if layer:
				layer.prev = prev
				layer.name = "Layer "+str(n)
			if prev:
				prev.next = layer


	def classify(self, data):
		predictions = self.predict(data)
		classifications =  numpy.argmax(predictions, axis=1)
		return classifications

	def predict(self, dataset):
		for layer in self.layers:
			dataset = layer.fprop(dataset)
		return dataset.get()

	def draw_layers(self):
		for layer in self.layers:
			layer.draw_layer()

	def train_backprop(self, inputs, targets, n_epochs):
		self.layers[0].prev = inputs
		for layer in self.layers:
			layer.init_bprop()

		print "Supervised Training"
		for epoch in xrange(n_epochs):
			print "Epoch " + str(epoch),
			error_sum=[0]*len(self.layers)

			start_time = time.time()
			for (input_batch, target_batch) in itertools.izip(inputs, targets):

				# Forward pass, populates the input and output fieldsw
				self.layers[0].x.gpudata = input_batch.gpudata
				assert( not numpy.isnan(self.layers[0].x.get()).any())
				for layer in self.layers:
					assert( not numpy.isnan(layer.x.get()).any())
					assert( not numpy.isnan(layer.w.get()).any())
					multiply_cublas_with_bias(layer.x, layer.b, layer.w, layer.z)
					assert( not numpy.isnan(layer.z.get()).any())
					assert( not numpy.isnan(layer.b.get()).any())
					layer.activation(layer.z, layer.y)
					assert( not numpy.isnan(layer.z.get()).any())
					assert( not numpy.isnan(layer.y.get()).any())

				# Backward pass, populates the errors field
				for layer in reversed(self.layers):
					if layer.next:
						multiply(layer.next.de_dy, transpose(layer.next.w), layer.de_dy)
						assert( not numpy.isnan(layer.de_dy.get()).any() )
					else:
						# For squared error cost: E=1/2*(t-y)^2 => dE/dy=t-y
						diff(layer.y, target_batch, layer.de_dy)
						assert( not numpy.isnan(layer.de_dy.get()).any() )

					# multiply by dy/dz
					layer.d_activation(layer.de_dy, layer.y, layer.de_dy)
					assert( not numpy.isnan(layer.de_dy.get()).any() )					
					multiply_cublas(transpose(layer.bias_ones), layer.de_dy, layer.b, alpha=-layer.learning_rate, beta=1.0)
					multiply_cublas(transpose(layer.x), layer.de_dy, layer.w, alpha=-layer.learning_rate, beta=1.0)
					assert( not numpy.isnan(layer.w.get()).any())

				# Get loss
				for n, layer in enumerate(self.layers):
					error_sum[n] += numpy.absolute(layer.de_dy.get()).sum()/inputs.batch_size

			print('  Time={:.3f}   Error={}'.format(time.time()-start_time, error_sum))


	def prop(self, layer, v_in, v_out_prob, v_out_sample, transpose=''):
		if len(layer.shape) > 1:
			conv2d(v_in, layer.w, v_out_prob)
			layer.activation(v_out_prob, v_out_prob)
			rand_gen.fill_uniform(layer.rng)
			sample(v_out_prob, layer.rng, v_out_sample)
		else:
			matrix_multiply(v_in, layer.w, v_out_prob, transpose=transpose, preserve_last_column=True)
			layer.activation(v_out_prob, v_out_prob)
			rand_gen.fill_uniform(layer.rng)
			sample(v_out_prob, layer.rng, v_out_sample)

	def train_rbm(self, inputs, n_epochs):
		assert inputs.data.dtype == 'float32'
		batch_size=100
		self.layers[0].prev = inputs
		for layer in self.layers:
			layer.init_rbm()
		error = pycuda.gpuarray.zeros(inputs.data.shape, dtype=numpy.float32)

		for layer in self.layers[:-1]:
			print "Unsupervised Training " + layer.name

			for epoch in xrange(n_epochs):
				print "Epoch " + str(epoch),
				start_time = time.time()
				epoch_err = 0
				for batch_n, batch in enumerate(inputs):
					# Load the input data for this batch into v
					pycuda.driver.memcpy_dtod(layer.v.gpudata, batch.gpudata, batch.nbytes)
					assert( not numpy.isnan(layer.v.get()).any())

					layer.fprop_batch()

					#print layer.h.get()[0,:,0,0]

					multiply(transpose(layer.v), layer.h, layer.pos_grad_w)
					multiply(transpose(layer.v), layer.bias_ones, layer.pos_grad_v) # sum row
					multiply(transpose(layer.h), layer.bias_ones, layer.pos_grad_h) # sum row

					#print layer.pos_grad_w.get()
					#print layer.pos_grad_v.get()
					#print layer.pos_grad_h.get()

					
					#sgemv(layer.v, layer.bias_ones, layer.pos_grad_v, trans='T') # sum row
					#sgemv(layer.h, layer.bias_ones, layer.pos_grad_h, trans='T') # sum row
					assert( not numpy.isnan(layer.pos_grad_w.get()).any())
					assert( not numpy.isnan(layer.pos_grad_v.get()).any())
					assert( not numpy.isnan(layer.pos_grad_h.get()).any())

					layer.bprop_batch()


					#print layer.v.get()[0,0,0,:]

					layer.fprop_batch()


					#print layer.h.get()[0,:,0,0]





					multiply(transpose(layer.v), layer.h, layer.neg_grad_w)
					multiply(transpose(layer.v), layer.bias_ones, layer.neg_grad_v) # sum row
					multiply(transpose(layer.h), layer.bias_ones, layer.neg_grad_h) # sum row
					#sgemv(layer.v, layer.bias_ones, layer.neg_grad_v, trans='T') # sum row
					#sgemv(layer.h, layer.bias_ones, layer.neg_grad_h, trans='T') # sum row
					
					
					assert( not numpy.isnan(layer.neg_grad_w.get()).any())
					assert( not numpy.isnan(layer.neg_grad_v.get()).any())
					assert( not numpy.isnan(layer.neg_grad_h.get()).any())

					#print layer.pos_grad_w.get() - layer.neg_grad_w.get()
					#print layer.pos_grad_v.get() - layer.neg_grad_v.get()
					#print layer.pos_grad_h.get() - layer.neg_grad_h.get()

					update_weights(layer.w, batch_size, self.learning_rate, layer.pos_grad_w, layer.neg_grad_w, layer.weight_cost)
					update_weights(layer.v_bias, batch_size, self.learning_rate, layer.pos_grad_v, layer.neg_grad_v, layer.weight_cost)
					update_weights(layer.h_bias, batch_size, self.learning_rate, layer.pos_grad_h, layer.neg_grad_h, layer.weight_cost)
					assert( not numpy.isnan(layer.w.get()).any())
					assert( not numpy.isnan(layer.v.get()).any())
					assert( not numpy.isnan(layer.h.get()).any())

					#print layer.w.get()
					#print layer.v_bias.get()
					#print layer.h_bias.get()

					# Store error
					pycuda.driver.memcpy_dtod(int(error.gpudata)+batch_n*batch.size*layer.pv.dtype.itemsize, layer.pv.gpudata, batch.mem_size*layer.h.dtype.itemsize)

				# Sum error
				calculate_error(inputs.data, error)
				avg_error = float(pycuda.gpuarray.sum(error).get())/inputs.data.shape[0]
				print('  Time={:.3f}  Error={:.6f}'.format(time.time()-start_time, avg_error))
				#print('  Time={:.3f}'.format(time.time()-start_time))
				layer.draw_layer(layer.name + '_epoch' + str(epoch))

			inputs = layer.fprop(inputs)

			fo = gzip.GzipFile('weights.gz', 'wb')
			cPickle.dump((layer.w.get()), fo)
			fo.close()

			
def profile(command, locals):
	import cProfile
	import pstats
	import gprof2dot
	import subprocess
	cProfile.runctx(command, globals(), locals, 'profile.pstats')
	gprof2dot.Main().main(["-f", "pstats", "-o", "profile.dot", "profile.pstats"])
	subprocess.call(['dot', '-Tpdf', '-oprofile.pdf', 'profile.dot'])
	print command + " run time: " +  str(pstats.Stats('profile.pstats').total_tt) + " seconds"

device = pycuda.autoinit.device
mem_free, mem_total = pycuda.driver.mem_get_info()
print('Device %s (%.2f GiB free/%.2f GiB total)' % (device.name(), (float(mem_free)/(1024*1024*1024)),(float(mem_total)/(1024*1024*1024))))

print sys.executable
if __debug__:
	print('Running in debug mode')
else:
	print('Running in optimized mode')

cuda_src = open("kernels.cu", 'r').read()
sm = pycuda.compiler.SourceModule(cuda_src)

import re
p = re.compile('\w*_kernel')
kernels = p.findall(cuda_src)
for kernel in kernels:
	globals()[kernel] = sm.get_function(kernel)
