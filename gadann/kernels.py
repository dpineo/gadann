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

import pycuda
import pycuda.curandom
import pycuda.compiler
import pycuda.autoinit
import numpy
import operator
import math
import os

import tensor

def seed_getter_func(count):
    return pycuda.gpuarray.arange(0, count, 1, dtype=numpy.int32)

#rand_gen = pycuda.curandom.XORWOWRandomNumberGenerator()
rand_gen = pycuda.curandom.XORWOWRandomNumberGenerator(seed_getter=seed_getter_func)

def mult(a,b):
    output = tensor.Tensor(a.shape)
    mult_kernel(a.gpuarray, b.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(a.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def div(a,b):
    output = tensor.Tensor(a.shape)
    div_kernel(a.gpuarray, b.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(a.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def sample(input):
    rand = tensor.Tensor(input.shape)
    rand_gen.fill_uniform(rand.gpuarray)
    output = tensor.Tensor(input.shape)
    sample_kernel(input.gpuarray, rand.gpuarray, output.gpuarray, block=(128,1,1), grid=(int(math.ceil(input.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output


def sqrt(input):
    output = tensor.Tensor(input.shape)
    sqrt_kernel(input.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(input.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def broadcast(input):
    result = tensor.Tensor(input.shape)
    sample_kernel(input.gpuarray, result.gpuarray, block=(shape[1],1,1), grid=(shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def logistic(input):
    output = tensor.Tensor(input.shape)
    logistic_kernel(input.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(input.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def dlogistic(input, activation):
    output = tensor.Tensor(input.shape)
    dlogistic_kernel(input.gpuarray, activation.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(input.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

# channelwise softmax
def softmax(input):
    output = tensor.Tensor(input.shape)
    if True:
        if len(input.shape) == 4:
            grid = (input.shape[2],input.shape[3],1)
            block = (input.shape[0],1,1)
            channel_sum = tensor.Tensor((input.shape[0],input.shape[2],input.shape[3]))
            channel_max = tensor.Tensor((input.shape[0],input.shape[2],input.shape[3]))
        elif len(input.shape) == 2:
            grid = (1,1,1)
            block = (input.shape[0],1,1)
            channel_sum = tensor.Tensor((input.shape[0],1,1))
            channel_max = tensor.Tensor((input.shape[0],1,1))
        assert( not numpy.isnan(input.get()).any())
        channel_max_kernel(input.gpuarray, channel_max.gpuarray, numpy.uint32(input.shape[1]), block=block, grid=grid)

        if len(input.shape) == 2:
            grid = (1,1,input.shape[0])
            block = (input.shape[1],1,1)
        elif len(input.shape) == 4:
            grid = (input.shape[2],input.shape[3],input.shape[0])
            block = (input.shape[1],1,1)
        channel_subtract_kernel(input.gpuarray, channel_max.gpuarray, output.gpuarray, numpy.uint32(input.shape[1]), block=block, grid=grid)

        if len(input.shape) == 4:
            grid = (input.shape[2],input.shape[3],1)
            block = (input.shape[0],1,1)
        elif len(input.shape) == 2:
            grid = (1,1,1)
            block = (input.shape[0],1,1)
        channel_sum_kernel(output.gpuarray, channel_sum.gpuarray, numpy.uint32(input.shape[1]), block=block, grid=grid)

        if len(input.shape) == 2:
            grid = (1,1,input.shape[0])
            block = (input.shape[1],1,1)
        elif len(input.shape) == 4:
            grid = (input.shape[2],input.shape[3],input.shape[0])
            block = (input.shape[1],1,1)
        softmax_divide_kernel(output.gpuarray, channel_sum.gpuarray, block=block, grid=grid)
    else:
        if len(input.shape) == 2:
            softmax_naive_kernel(input.gpuarray, output.gpuarray, block=(input.shape[1],1,1), grid=(input.shape[0],1,1))
        else:
            softmax_naive_kernel(input.gpuarray, output.gpuarray, block=(input.shape[1],1,1), grid=(input.shape[0],1,1))

    assert( not numpy.isnan(output.get()).any())
    return output

def naive_softmax(input):
	output = tensor.Tensor(input.shape)
	#shape = (input.shape[0], reduce(operator.__mul__, input.shape[1:], 1))
	#channel_stride = reduce(operator.__mul__, input.shape[2:], 1)
	assert( not numpy.isnan(input.get()).any())
	softmax_naive_kernel(input.gpuarray, output.gpuarray, block=(input.shape[1],1,1), grid=(input.shape[0],1,1))
	assert( not numpy.isnan(output.get()).any())
	return output

def dsoftmax(input, activation):
    output = tensor.Tensor(input.shape)
    #shape = (input.shape[0], reduce(operator.__mul__, input.shape[1:], 1))
    dsoftmax_kernel(input.gpuarray, activation.gpuarray, output.gpuarray, block=(input.shape[1],1,1), grid=(input.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def tanh(input, output):
    tanh_kernel(input, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def dtanh(input, activation, output):
    dtanh_kernel(input, activation, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def relu(input):
    output = tensor.Tensor(input.shape)
    relu_kernel(input.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(input.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def drelu(input, activation):
    output = tensor.Tensor(input.shape)
    drelu_kernel(input.gpuarray, activation.gpuarray, output.gpuarray, numpy.uint32(output.gpuarray.size), block=(128,1,1), grid=(int(math.ceil(input.gpuarray.size/128.0)),1,1))
    assert( not numpy.isnan(output.get()).any())
    return output

def softplus(input, output):
    softplus_kernel(input, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def dsoftplus(input, activation, output):
    dsoftplus_kernel(input, activation, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def gaussian(input, output):
    gaussian_kernel(input, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def dgaussian(input, activation, output):
    dgaussian_kernel(input, activation, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def identity(input, output):
    identity_kernel(input, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def didentity(input, activation, output):
    didentity_kernel(input, activation, output, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(output.get()).any())

def diff(output, target, result):
    diff_kernel(output, target, result, block=(output.shape[1],1,1), grid=(output.shape[0],1,1))
    assert( not numpy.isnan(result.get()).any())

def mosaic(input, border_size=1):
    block_shape = input.shape[-2:] + (1,)
    
    if input.shape[1] in (1,3):
        grid_size = input.shape[0]
        grid_shape = (int(math.ceil(math.sqrt(grid_size))), int(math.sqrt(grid_size)), input.shape[1])
        output = tensor.Tensor(((block_shape[1]+border_size)*grid_shape[1], (block_shape[0]+border_size)*grid_shape[0], input.shape[1]))
    else:
        grid_size = input.shape[0]*input.shape[1]
        grid_shape = (int(math.ceil(math.sqrt(grid_size))), int(math.sqrt(grid_size)), 1)
        output = tensor.Tensor(((block_shape[1]+border_size)*grid_shape[1], (block_shape[0]+border_size)*grid_shape[0], 1))
    
    output.gpuarray.fill(.2)
    mosaic_kernel(input.gpuarray, output.gpuarray, numpy.uint32(grid_size), numpy.uint32(border_size), block=block_shape, grid=grid_shape)
    return output

def image_2_design(image):
    output = tensor.Tensor((256,256))
    output.gpuarray.fill(0)
    image_2_design_kernel(image.gpuarray, output.gpuarray, block=(16,16,1), grid=(1,1,1))
    assert( not numpy.isnan(output.get()).any())
    return output


module_dir = os.path.dirname(os.path.abspath(__file__))
cuda_src = open(os.path.join(module_dir, "kernels.cu"), 'r').read()
sm = pycuda.compiler.SourceModule(cuda_src)

import re
p = re.compile('\w*_kernel')
kernels = p.findall(cuda_src)
for kernel in kernels:
    globals()[kernel] = sm.get_function(kernel)

