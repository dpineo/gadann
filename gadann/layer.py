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
import operator
import scipy
import logging
import cv2
import functools
import pickle

from . import cudnn
from . import kernels
from . import tensor

logger = logging.getLogger(__name__)


# -----------------------  Layer  --------------------------
class Layer(object):
    def __init__(self, **kwargs):
        self.params = []
        self.name = kwargs['name']
        self.input_shape = kwargs['input_shape'] + (3-len(kwargs['input_shape']))*(1,)
        self.input_size = functools.reduce(operator.__mul__, self.input_shape, 1)
        self.updater = kwargs['updater']
        pass

    def fprop(self, input):
        return input

    def bprop(self, input, fprop_result=None):
        return input

    def error_gradient(self, input, error):
        return []

    def loglikelihood_gradient(self, v, h):
        return []

    def update(self, grads):
        self.updater.update(self.params, grads)

    def show(self):
        pass

    def save(self):
        # Save weights to file
        fo = gzip.GzipFile(self.name + '_weights.gz', 'wb')
        pickle.dump([param.get() for param in self.params], fo)
        fo.close()


# -----------------------  LinearLayer  --------------------------
class LinearLayer(Layer):
    def __init__(self, **kwargs):
        super(LinearLayer, self).__init__(**kwargs)

        if 'stride' in kwargs:
            self.stride = kwargs['stride']
        else:
            self.stride = (1, 1)

        if 'padding' in kwargs:
            self.padding = kwargs['padding']
        else:
            self.padding = (0, 0)

        self.n_features = kwargs['n_features']
        try:
            self.shape = (self.n_features, self.input_shape[0]) + kwargs['shape'] + (2-len(kwargs['shape']))*(1,)
        except:
            self.shape = (self.n_features,)+kwargs['input_shape']

        self.output_shape = (self.shape[0],)+tuple([(x-y)//s+1+2*p for x, y, s, p in zip(self.input_shape[-2:], self.shape[-2:], self.stride, self.padding)])

        init = kwargs.get('init', 0.01)
        w = tensor.Tensor(init*numpy.random.randn(*self.shape))
        v_bias = tensor.zeros((1, self.input_size))
        h_bias = tensor.zeros((1, self.n_features))
        self.params = [w, v_bias, h_bias]

        return

        self.filter_descriptor = cudnn.cudnnCreateFilterDescriptor()
        self.bias_descriptor = cudnn.cudnnCreateTensorDescriptor()
        self.convolution_descriptor = cudnn.cudnnCreateConvolutionDescriptor()
        self.input_descriptor = cudnn.cudnnCreateTensorDescriptor()
        self.output_descriptor = cudnn.cudnnCreateTensorDescriptor()

        cudnn.cudnnSetFilter4dDescriptor(
            self.filter_descriptor,
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *w.shape
        )

        logger.info('filter_descriptor:', cudnn.cudnnGetFilter4dDescriptor(self.filter_descriptor))

        cudnn.cudnnSetTensor4dDescriptor(
            self.bias_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *(h_bias.shape + (1,1))
        )

    def __str__(self):
        return self.name + " - " + self.__class__.__name__ + "(shape=" + str(self.input_shape) + ")"

    def show(self):
        cv2.imshow(self.name, .1*self.params[0].mosaic().get()+.5)
        cv2.waitKey(1)

    def fprop(self, input):
        #if self.shape[1:] == input.shape[1:]:
        if True:
            return self.fprop_dense(input)
        else:
            return self.fprop_conv(input)

    def bprop(self, input, fprop_result=None):
        #if self.shape[0] == input.shape[1]:
        if True:
            return self.bprop_dense(input, fprop_result)
        else:
            return self.bprop_conv(input, fprop_result)

    def error_gradient(self, input, error):
        #if self.shape[1:] == input.size:
        if True:
            return self.error_gradient_dense(input, error)
        else:
            return self.error_gradient_conv(input, error)

    def fprop_dense(self, input):
        w, v_bias, h_bias = self.params
        result = input.dot(w.T()) + input.ones_vector.dot(h_bias)
        assert not numpy.isnan(result.get()).any()
        return result

    def bprop_dense(self, input, fprop_result=None):
        w, v_bias, h_bias = self.params
        result = input.dot(w) + input.ones_vector.dot(v_bias)
        assert not numpy.isnan(result.get()).any()
        return result

    def fprop_conv(self, input):
        assert len(input.shape) == 4

        w, v_bias, h_bias = self.params

        assert not numpy.isnan(w.get()).any()
        assert not numpy.isnan(v_bias.get()).any()
        assert not numpy.isnan(h_bias.get()).any()

        cudnn.cudnnSetTensor4dDescriptor(
            self.input_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *input.shape
        )

        logger.info('input_descriptor:', cudnn.cudnnGetTensor4dDescriptor(self.input_descriptor))

        cudnn.cudnnSetConvolution2dDescriptor(
            self.convolution_descriptor,
            self.padding[0], self.padding[1], self.stride[0], self.stride[1], 1, 1,
            cudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION'])

        logger.info('convolution_descriptor:',  cudnn.cudnnGetConvolution2dDescriptor(self.convolution_descriptor))

        # Get output dimensions (first two values are n_input and filters_out)
        batch, channels, height_output, width_output = cudnn.cudnnGetConvolution2dForwardOutputDim(
            self.convolution_descriptor,
            self.input_descriptor,
            self.filter_descriptor
        )

        # Output tensor
        output = tensor.Tensor((batch, self.n_features, height_output, width_output))

        cudnn.cudnnSetTensor4dDescriptor(
            self.output_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *output.shape
        )
        logger.info('output_descriptor:',  cudnn.cudnnGetTensor4dDescriptor(self.output_descriptor))
        
        workspace_size = cudnn.cudnnGetConvolutionForwardWorkspaceSize(
            cudnn_context,
            self.input_descriptor,
            self.filter_descriptor,
            self.convolution_descriptor,
            self.output_descriptor,
            cudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'],
        ).value
        workspace = tensor.Tensor((workspace_size,))
        logger.info('workspace_size:', workspace_size)

        algo = cudnn.cudnnGetConvolutionForwardAlgorithm(
            cudnn_context,
            self.input_descriptor,
            self.filter_descriptor,
            self.convolution_descriptor,
            self.output_descriptor,
            cudnn.cudnnConvolutionFwdPreference['CUDNN_CONVOLUTION_FWD_PREFER_FASTEST'],
            0
        )
        
        assert(not numpy.isnan(input.get()).any())
        assert(not numpy.isnan(w.get()).any())

        # Perform convolution
        cudnn.cudnnConvolutionForward(
            cudnn_context,
            1,
            self.input_descriptor,
            input.data(),
            self.filter_descriptor,
            w.data(),
            self.convolution_descriptor,
            algo,
            workspace.data(),
            workspace_size,
            0,
            self.output_descriptor,
            output.data()
        )

        assert( not numpy.isnan(output.get()).any())

        cudnn.cudnnAddTensor(
            cudnn_context,
            cudnn.cudnnAddMode['CUDNN_ADD_SAME_C'],
            1,
            self.bias_descriptor,
            h_bias.data(),
            1,
            self.output_descriptor,
            output.data()
        )

        assert not numpy.isnan(output.get()).any()
        return output

    def bprop_conv(self, input, fprop_result=None):
        assert len(input.shape) == 4
        w, v_bias, h_bias = self.params

        cudnn.cudnnSetTensor4dDescriptor(
            self.input_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *input.shape
        )

        cudnn.cudnnSetConvolution2dDescriptor(
            self.convolution_descriptor,
            0, 0, 1, 1, 1, 1,
            cudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION'])

        # Output tensor
        output = tensor.Tensor((input.shape[0], w.shape[1], input.shape[2]+w.shape[2]-1, input.shape[3]+w.shape[3]-1))

        cudnn.cudnnSetTensor4dDescriptor(
            self.output_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *output.shape
        )

        # Perform convolution
        cudnn.cudnnConvolutionBackwardData(
            cudnn_context,
            1,
            self.filter_descriptor,
            w.data(),
            self.input_descriptor,
            input.data(),
            self.convolution_descriptor,
            0,
            self.output_descriptor,
            output.data()
        )

        assert not numpy.isnan(output.get()).any()
        return output

    def loglikelihood_gradient(self, v, h):
        return [h.T().dot(v),  v.T().dot(v.ones_vector), h.T().dot(h.ones_vector)]


        w, v_bias, h_bias = self.params

        cudnn.cudnnSetTensor4dDescriptor(
            self.input_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *v.shape
        )

        cudnn.cudnnSetTensor4dDescriptor(
            self.output_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *(h.shape + (1,1))
        )

        w_grad = tensor.zeros(w.shape)
        v_bias_grad = tensor.zeros(v_bias.shape)
        h_bias_grad = tensor.zeros(h_bias.shape)

        # Perform convolution
        cudnn.cudnnConvolutionBackwardFilter(
            cudnn_context,
            1,
            self.input_descriptor,
            v.data(),
            self.output_descriptor,
            h.data(),
            self.convolution_descriptor,
            1,
            self.filter_descriptor,
            w_grad.data()
        )

        cudnn.cudnnConvolutionBackwardBias(
            cudnn_context,
            1,
            self.output_descriptor,
            h.data(),
            1,
            self.bias_descriptor,
            h_bias_grad.data()
        )

        assert not numpy.isnan(w.get()).any()
        assert not numpy.isnan(h_bias.get()).any()
        return [w_grad, v_bias_grad, h_bias_grad]

    def error_gradient_dense(self, input, error):
        w, v_bias, h_bias = self.params
        w_grad = tensor.zeros(w.shape)
        v_bias_grad = tensor.zeros(v_bias.shape)
        h_bias_grad = tensor.zeros(h_bias.shape)
        tensor.sgemm(error.T(), input, w_grad, alpha=1, beta=0)
        tensor.sgemv(h_bias_grad, error.T(), input.ones_vector.T(), alpha=1, beta=0)
        grads = [w_grad, v_bias_grad, h_bias_grad]
        assert not numpy.isnan(w_grad.get()).any()
        assert not numpy.isnan(v_bias_grad.get()).any()
        assert not numpy.isnan(h_bias_grad.get()).any()
        return grads
        # return {w: w_grad, v_bias:v_bias_grad, h_bias:h_bias_grad}

    def error_gradient_conv(self, input, error):
        w, v_bias, h_bias = self.params

        cudnn.cudnnSetTensor4dDescriptor(
            self.input_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *input.shape
        )

        cudnn.cudnnSetTensor4dDescriptor(
            self.output_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *error.shape
        )

        w_grad = tensor.zeros(w.shape)
        v_bias_grad = tensor.zeros(v_bias.shape)
        h_bias_grad = tensor.zeros(h_bias.shape)

        # Perform convolution
        cudnn.cudnnConvolutionBackwardFilter(
            cudnn_context,
            1,
            self.input_descriptor,
            input.data(),
            self.output_descriptor,
            error.data(),
            self.convolution_descriptor,
            1,
            self.filter_descriptor,
            w_grad.data()
        )

        cudnn.cudnnConvolutionBackwardBias(
            cudnn_context,
            1,
            self.output_descriptor,
            error.data(),
            1,
            self.bias_descriptor,
            h_bias_grad.data()
        )

        assert not numpy.isnan(w_grad.get()).any()
        assert not numpy.isnan(h_bias_grad.get()).any()
        return [w_grad, v_bias_grad, h_bias_grad]


# -----------------------  DenseLayer  --------------------------
class DenseLayer(LinearLayer):
    def __init__(self, **kwargs):
        super(DenseLayer, self).__init__(**kwargs)

    def loglikelihood_gradient(self, v, h):
        return [h.T().dot(v),  v.T().dot(v.ones_vector), h.T().dot(h.ones_vector)]

    def __str__(self):
        return self.name + " - " + self.__class__.__name__ + "(shape=" + str(self.feature_shape) + ")"

    def show(self):
        w = self.params[0]
        if w.shape[1] not in (1, 3):
            return
        cv2.imshow(self.name, self.params[0].mosaic().get()/10+.5)
        cv2.moveWindow(self.name, 0, 0)
        cv2.waitKey(1)


class ActivationLayer(Layer):
    def __init__(self, activation, **kwargs):
        super(ActivationLayer, self).__init__(**kwargs)
        self.activation = activation
        self.d_activation = getattr(kernels, 'd'+activation.__name__)
        self.output_shape = self.input_shape

    def fprop(self, input):
        result = self.activation(input)
        assert not numpy.isnan(result.get()).any()
        return result

    def bprop(self, input, fprop_result):
        result = self.d_activation(input, fprop_result)
        assert not numpy.isnan(result.get()).any()
        return result

    def __str__(self):
        return self.name + " - " + self.__class__.__name__ + "(activation='" + self.activation.__name__ + "')"


class DropoutLayer(Layer):
    def __init__(self, **kwargs):
        super(DropoutLayer, self).__init__(**kwargs)
        self.p_exclude = float(kwargs.pop('prob'))
        self.p_include = float(1-self.p_exclude)
        self.output_shape = self.input_shape

    def __str__(self):
        return self.name + " - " + self.__class__.__name__ + "(p=" + str(self.p_exclude) + ")"

    
class MaxPoolingLayer(Layer):
    def __init__(self, **kwargs):
        super(MaxPoolingLayer, self).__init__(**kwargs)
        if 'padding' not in kwargs:
            kwargs['padding'] = (0, 0)
        if 'stride' not in kwargs:
            kwargs['stride'] = (1, 1)

        self.shape = kwargs['shape']
        self.padding = kwargs['padding']
        self.stride = kwargs['stride']

        self.output_shape = (self.input_shape[0], (self.input_shape[1]+2*self.padding[0])/self.stride[0], (self.input_shape[2]+2*self.padding[1])/self.stride[1])
        
        self.pooling_descriptor = cudnn.cudnnCreatePoolingDescriptor()
        self.input_descriptor = cudnn.cudnnCreateTensorDescriptor()
        self.output_descriptor = cudnn.cudnnCreateTensorDescriptor()

        cudnn.cudnnSetPooling2dDescriptor(
            self.pooling_descriptor,
            cudnn.cudnnPoolingMode['CUDNN_POOLING_MAX'],
            self.shape[0],
            self.shape[1],
            self.padding[0],
            self.padding[1],
            self.stride[0],
            self.stride[1]
        )
        pass

    def fprop(self, input):

        cudnn.cudnnSetTensor4dDescriptor(
            self.input_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *input.shape
        )

        # Output tensor
        output = tensor.Tensor((input.shape[0], input.shape[1], input.shape[2]/self.stride[0], input.shape[3]/self.stride[1]))

        cudnn.cudnnSetTensor4dDescriptor(
            self.output_descriptor,
            cudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW'],
            cudnn.cudnnDataType['CUDNN_DATA_FLOAT'],
            *output.shape
        )

        cudnn.cudnnPoolingForward(
            cudnn_context,
            self.pooling_descriptor,
            1,
            self.input_descriptor,
            input.data(),
            0,
            self.output_descriptor,
            output.data()
        )
        return output

    def __str__(self):
        return self.name + " - " + self.__class__.__name__ + "(p=" + str(self.p_exclude) + ")"


class ReshapeLayer(Layer):
    def __init__(self, **kwargs):
        super(ReshapeLayer, self).__init__(**kwargs)
        self.input_shape  = kwargs['input_shape']
        self.output_shape = kwargs['output_shape']
        assert(reduce(operator.__mul__, self.input_shape, 1) == reduce(operator.__mul__, self.output_shape, 1))

    def fprop(self, input):
        assert input.shape[1:] == self.input_shape
        input.shape = (input.shape[0],) + self.output_shape
        return input

    def bprop(self, input, fprop_result=None):
        assert input.shape[1:] == self.output_shape
        input.shape = (input.shape[0],) + self.input_shape
        return input


# -----------------------  ConvLayer  --------------------------
class GadannConvLayer(LinearLayer):
    def __init__(self, **kwargs):
        super(ConvLayer, self).__init__(**kwargs)
        # self.conv_step = kwargs['conv_step']
        # self.w.axes = ['features_out', 'features_in', 'height', 'width']

    def fprop(self, input):
        w, v_bias, h_bias = self.params
        result = Tensor((input.shape[0],w.shape[0])+tuple([x-y for x,y in zip(input.shape[-2:],w.shape[-2:])]))
        grid = (result.shape[-2]/16,result.shape[-1]/16,result.shape[0])
        conv2d_16x16_kernel(input.gpuarray, w.gpuarray, h_bias.gpuarray, result.gpuarray, numpy.uint32(self.w.shape[1]), numpy.uint32(w.shape[0]), block=(16,16,1), grid=grid)
        assert not numpy.isnan(result.get()).any()
        return result

    def bprop(self, input, fprop_result=None):
        w, v_bias, h_bias = self.params
        result = tensor.zeros((input.shape[0],w.shape[1])+tuple([x+y for x,y in zip(input.shape[-2:],w.shape[-2:])]))
        grid = (input.shape[3]/16,input.shape[2]/16,1)
        bconv2d_16x16_kernel(input.gpuarray, w.gpuarray, v_bias.gpuarray, result.gpuarray, numpy.uint32(w.shape[0]), numpy.uint32(w.shape[1]), block=(16,16,1), grid=grid)
        assert not numpy.isnan(result.get()).any()
        return result

    def loglikelihood_gradient(self, v, h):
        w, v_bias, h_bias = self.params
        w_update = tensor.zeros(w.shape)
        v_bias_update = tensor.zeros(v_bias.shape)
        h_bias_update = tensor.zeros(h_bias.shape)
        grid = (h.shape[-2]/16,h.shape[-1]/16) # revisit: grid rounds down to nearest 16

        kernels.iconv2d_16x16_kernel(v.gpuarray, h.gpuarray, w_update.gpuarray, numpy.uint32(w_update.shape[1]), numpy.uint32(w_update.shape[0]), block=(16,16,1), grid=grid)
        kernels.iconv2d_h_bias_16x16_naive_kernel(h.gpuarray, h_bias_update.gpuarray, numpy.uint32(reduce(operator.__mul__, h.shape[-2:], 1)), block=(w_update.shape[0],1,1), grid=grid)

        v_bias_block = w.shape[-2:] + (1,)
        v_bias_grid = (1,1,1)
        kernels.iconv2d_v_bias_16x16_naive_kernel(v.gpuarray, v_bias_update.gpuarray, numpy.uint32(v.shape[1]-16), numpy.uint32(v.shape[2]-16), block=v_bias_block, grid=v_bias_grid)

        assert not numpy.isnan(w_update.get()).any()
        assert not numpy.isnan(v_bias_update.get()).any()
        assert not numpy.isnan(h_bias_update.get()).any()
        return [w_update, v_bias_update, h_bias_update]


# -----------------------  NumpyConvLayer  --------------------------
class NumpyConvLayer(LinearLayer):
    def fprop(self, input):
        result = []
        for n in range(self.w.shape[0]):
            result.append(scipy.signal.convolve(input.get()[0,...], self.w.get()[n,...], mode='valid'))
        return Tensor(numpy.ascontiguousarray(numpy.vstack(result)[numpy.newaxis,...]))

    def bprop(self, input, fprop_result=None):
        result = []
        for n in range(self.w.shape[0]):
            w = numpy.fliplr(numpy.flipud(self.w.get()[n,...]))
            result.append(scipy.signal.convolve(input.get()[n,...], w, mode='full'))
        result = numpy.vstack(result).mean(axis=0)
        return Tensor(result)


cudnn_context = cudnn.cudnnCreate()
