
import gadann
import pycuda
import pycuda.autoinit
from pycuda import gpuarray
import libcudnn, ctypes
import numpy as np
import cv2

# Create a cuDNN context
cudnn_context = libcudnn.cudnnCreate()

# Set some options and tensor dimensions
accumulate = libcudnn.cudnnAccumulateResults['CUDNN_RESULT_NO_ACCUMULATE']
tensor_format = libcudnn.cudnnTensorFormat['CUDNN_TENSOR_NCHW']
data_type = libcudnn.cudnnDataType['CUDNN_DATA_FLOAT']
convolution_mode = libcudnn.cudnnConvolutionMode['CUDNN_CONVOLUTION']
convolution_path = libcudnn.cudnnConvolutionPath['CUDNN_CONVOLUTION_FORWARD']

n_input = 1
filters_in = 3
height_in = 480
width_in = 640
filters_out = 10

height_filter = 16
width_filter = 16
pad_h = 0
pad_w = 0
vertical_stride = 1
horizontal_stride = 1
upscalex = 1
upscaley = 1

# Input tensor
dir = 'D:\\projects\\prime\\data\\dc9\\day2_chan1_png\\'
X = cv2.imread(dir+'sen0_0001.png')
X = gadann.Tensor(np.array(cv2.split(X)))

# Filter tensor
filters = gadann.Tensor(np.random.rand(filters_out, filters_in, height_filter, width_filter))

# Descriptor for input
X_desc = libcudnn.cudnnCreateTensor4dDescriptor()
libcudnn.cudnnSetTensor4dDescriptor(X_desc, tensor_format, data_type,
	n_input, filters_in, height_in, width_in)

# Filter descriptor
filters_desc = libcudnn.cudnnCreateFilterDescriptor()
libcudnn.cudnnSetFilterDescriptor(filters_desc, data_type, filters_out,
	filters_in, height_filter, width_filter)

# Convolution descriptor
conv_desc = libcudnn.cudnnCreateConvolutionDescriptor()
libcudnn.cudnnSetConvolutionDescriptor(conv_desc, X_desc, filters_desc,
	pad_h, pad_w, vertical_stride, horizontal_stride, upscalex, upscaley,
	convolution_mode)

# Get output dimensions (first two values are n_input and filters_out)
_, _, height_output, width_output = libcudnn.cudnnGetOutputTensor4dDim(
	conv_desc, convolution_path)

# Output tensor
Y = gadann.Tensor((n_input, filters_out, height_output, width_output))
Y_desc = libcudnn.cudnnCreateTensor4dDescriptor()
libcudnn.cudnnSetTensor4dDescriptor(Y_desc, tensor_format, data_type, n_input,
	filters_out, height_output, width_output)

# Get pointers to GPU memory
X_data = X.data()
filters_data = filters.data()
Y_data = Y.data()

# Perform convolution
libcudnn.cudnnConvolutionForward(cudnn_context, X_desc, X_data,
	filters_desc, filters_data, conv_desc, Y_desc, Y_data, accumulate)

Y=Y.get()[0,0,:,:]/(256*width_filter*height_filter)
cv2.imshow("Y", Y)
cv2.waitKey(-1)

# Clean up
libcudnn.cudnnDestroyTensor4dDescriptor(X_desc)
libcudnn.cudnnDestroyTensor4dDescriptor(Y_desc)
libcudnn.cudnnDestroyFilterDescriptor(filters_desc)
libcudnn.cudnnDestroyConvolutionDescriptor(conv_desc)
libcudnn.cudnnDestroy(cudnn_context)