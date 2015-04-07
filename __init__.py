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

import sys, os, logging.config

try:
	with open('logging.config') as config:
		logging.config.dictConfig(eval(config.read()))
except:
	with open(os.path.join(os.path.dirname(__file__),'logging.config')) as config:
		logging.config.dictConfig(eval(config.read()))

logger = logging.getLogger(__name__)

import pycuda.autoinit
import cublas
import cudnn

logger.info('Device: %s  (compute %d.%d)'% ((pycuda.autoinit.device.name(),) + pycuda.autoinit.device.compute_capability()))
logger.info('Memory: %.2f GiB free/%.2f GiB total' % tuple(map(lambda x:float(x)/(1024**3), pycuda.driver.mem_get_info())))
logger.info('CUDA version %d.%d.%d' % pycuda.driver.get_version())
logger.info('PyCUDA version %s' % (pycuda.VERSION_TEXT))
logger.info('cuDNN version %s.%s' % tuple([x for x in str(cudnn.cudnnGetVersion())]))
logger.info('Running in %s mode' % ("OPTIMIZED", "DEBUG")[__debug__])

from kernels import *
from tensor import *
from data_source import *
from layer import *
from trainer import *
from neuralnetwork import *