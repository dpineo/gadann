import pycuda
import pycuda.curandom
import pycuda.compiler
import numpy
import pycuda.autoinit
import itertools

from .kernels import *


# --------------------  Autoencoder  ---------------------
class Autoencoder(MLP):
    def __init__(self, layers, learning_rate=0.1, weight_cost=0.02):
        self.layers = layers
        for n, (layer, prev) in enumerate(itertools.izip(self.layers+[None],[None]+self.layers)):
            if layer:
                layer.prev = prev 
                layer.name = "Layer "+str(n)
            if prev:
                prev.next = layer


# revisit
class AutoencoderBackpropTrainer(object):
    def __init__(self, model):
        pass

    def train_supervised(self, inputs, targets, n_epochs, batch_size=100):
        # Allocate temp storage to run the algorithm
        for layer in self.layers:
            layer.enc_inputs = pycuda.gpuarray.GPUArray((batch_size, layer.n_vis+1), dtype=numpy.float32)
            layer.enc_inputs.fill(1)
            layer.enc_outputs = pycuda.gpuarray.GPUArray((batch_size, layer.n_hid+1), dtype=numpy.float32)
            layer.enc_outputs.fill(1)
            layer.enc_z = pycuda.gpuarray.GPUArray((batch_size, layer.n_hid+1), dtype=numpy.float32)
            layer.enc_z.fill(1)
            layer.enc_errors = pycuda.gpuarray.GPUArray((batch_size, layer.n_hid+1), dtype=numpy.float32)
            layer.enc_errors.fill(1)

            layer.dec_inputs = pycuda.gpuarray.GPUArray((batch_size, layer.n_hid+1), dtype=numpy.float32)
            layer.dec_inputs.fill(1)
            layer.dec_outputs = pycuda.gpuarray.GPUArray((batch_size, layer.n_vis+1), dtype=numpy.float32)
            layer.dec_outputs.fill(1)
            layer.dec_z = pycuda.gpuarray.GPUArray((batch_size, layer.n_vis+1), dtype=numpy.float32)
            layer.dec_z.fill(1)
            layer.dec_errors = pycuda.gpuarray.GPUArray((batch_size, layer.n_vis+1), dtype=numpy.float32)
            layer.dec_errors.fill(1)

        print("Supervised Training")
        for epoch in xrange(n_epochs):
            print("Epoch " + str(epoch))
            error_sum=[0]*len(self.layers)

            for (input_batch, target_batch) in itertools.izip(inputs, targets):

                # Encoder forward pass, populates the enc_input and enc_output fields
                for layer in self.layers:
                    if layer.prev:
                        layer.enc_inputs.gpudata = int(layer.prev.enc_outputs.gpudata)
                    else:
                        layer.enc_inputs.gpudata = input_batch.gpudata

                    cublas.sgemm(layer.enc_inputs, layer.w, layer.enc_z, transa=False, transb=False, augmented=True)
                    layer.activation(layer.enc_z, layer.enc_outputs)
                    assert( not numpy.isnan(layer.enc_z.get()).any())
                    assert( not numpy.isnan(layer.enc_outputs.get()).any())

                # Decoder forward pass, populates the dec_input and dec_output fields
                for layer in reversed(self.layers):
                    if layer.next:
                        layer.dec_inputs.gpudata = int(layer.next.dec_outputs.gpudata)
                    else:
                        layer.dec_inputs.gpudata = layer.enc_outputs.gpudata

                    cublas.sgemm(layer.dec_inputs, layer.w, layer.dec_z, transa=False, transb=True, augmented=True)
                    layer.activation(layer.dec_z, layer.dec_outputs)
                    assert( not numpy.isnan(layer.dec_z.get()).any())
                    assert( not numpy.isnan(layer.dec_outputs.get()).any())
                    
                # Decoder backward pass, populates the errors field
                for layer in self.layers:
                    if layer.prev:
                        cublas.sgemm(layer.prev.dec_errors, layer.prev.w, layer.dec_errors, transa=False, transb=False, augmented="bprop")
                    else: 
                        diff(layer.dec_outputs, target_batch, layer.dec_errors)
                        assert( not numpy.isnan(layer.dec_errors.get()).any() )

                    layer.d_activation(layer.dec_errors, layer.dec_outputs, layer.dec_errors)
                    assert( not numpy.isnan(layer.dec_errors.get()).any() )

                    cublas.sgemm(layer.dec_errors, layer.dec_inputs, layer.w, transa=False, transb=True, beta=1.0, alpha=-layer.learning_rate, augmented="bprop")
                    assert( not numpy.isnan(layer.w.get()).any())

                '''
                # Encoder backward pass, populates the errors field
                for layer in reversed(self.layers):
                    if layer.next:
                        cublas.sgemm(layer.next.enc_errors, layer.next.w, layer.enc_errors, transa=False, transb=True, augmented="bprop")
                    else:
                        cublas.sgemm(layer.dec_errors, layer.w, layer.enc_errors, transa=False, transb=False, augmented="bprop")
                        assert( not numpy.isnan(layer.enc_errors.get()).any() )

                    layer.d_activation(layer.enc_errors, layer.enc_outputs, layer.enc_errors)
                    assert( not numpy.isnan(layer.enc_errors.get()).any() )

                    cublas.sgemm(layer.enc_inputs, layer.enc_errors, layer.w, transa=True, transb=False, beta=1.0, alpha=-layer.learning_rate, augmented=True)
                    assert( not numpy.isnan(layer.w.get()).any())
                '''

                # Get loss
                #for n, layer in enumerate(self.layers):
                #    error_sum[n] += numpy.absolute(layer.e.get()).sum()/batch_size

            #print error_sum

        for layer in self.layers:
            del(layer.enc_inputs)
            del(layer.enc_outputs)
            del(layer.enc_z)
            del(layer.enc_errors)
            del(layer.dec_inputs)
            del(layer.dec_outputs)
            del(layer.dec_z)
            del(layer.dec_errors)

    def train_unsupervised(self, inputs, n_epochs=10):
        for layer in self.layers:
            print("Unsupervised Training " + layer.name)
            layer.train_unsupervised(inputs, n_epochs)
            inputs = layer.fprop(inputs)

    def classify(self, data):
        predictions = self.predict(data)
        classifications =  numpy.argmax(predictions, axis=1)
        return classifications

    def predict(self, dataset):
        for layer in self.layers:
            dataset = layer.fprop(dataset)
        for layer in reversed(self.layers):
            dataset = layer.bprop(dataset)
        return dataset.get()

    def draw_layers(self):
        for layer in self.layers:
            layer.draw_layer()
