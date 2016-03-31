'''Wrapper for cublas.h.'''
import ctypes
from ctypes import Structure, c_char, c_float, c_int, c_double, c_char_p, c_uint, POINTER, cast

def pointer(gpuarray):
    """Return the pointer to the linear memory held by a GPUArray object."""
    return cast(int(gpuarray.gpudata), POINTER(c_float))

## Add _as_parameter_ attribute to GPUArray for it to be used as
## arguments to ctypes-wrapped functions.
import pycuda.gpuarray
pycuda.gpuarray.GPUArray._as_parameter_ = property(pointer)


# Begin preamble

def POINTER(obj):
    p = ctypes.POINTER(obj)

    # Convert None to a real NULL pointer to work around bugs
    # in how ctypes handles None on 64-bit platforms
    if not isinstance(p.from_param, classmethod):
        def from_param(cls, x):
            if x is None:
                return cls()
            else:
                return x
        p.from_param = classmethod(from_param)

    return p

# End preamble

_libs = {}
_libdirs = []

# Begin libraries
import platform
if platform.system()=='Microsoft':
    #_libs["cublas"] = ctypes.windll.LoadLibrary('cublas64_50_35.dll')
    _libs["cublas"] = ctypes.windll.LoadLibrary('cublas64_65.dll')
    #_libs["cublas"] = ctypes.windll.LoadLibrary('cublas.dll')
elif platform.system()=='Darwin':
    _libs["cublas"] = ctypes.cdll.LoadLibrary('/usr/local/cuda/lib/libcublas.dylib')
elif platform.system()=='Linux':
    _libs["cublas"] = ctypes.cdll.LoadLibrary('libcublas.so')
# else:                              _libs["cublas"] = ctypes.cdll.LoadLibrary('libcublas.so')

# End libraries

# /usr/local/cuda/include/vector_types.h: 365
class double2(Structure):
    pass

double2.__slots__ = [
    'x',
    'y',
]
double2._fields_ = [
    ('x', c_double),
    ('y', c_double),
]

# /usr/local/cuda/include/vector_types.h: 450
class float2(Structure):
    pass

float2.__slots__ = [
    'x',
    'y',
]
float2._fields_ = [
    ('x', c_float),
    ('y', c_float),
]

cuFloatComplex = float2 # /usr/local/cuda/include/cuComplex.h: 162

cuDoubleComplex = double2 # /usr/local/cuda/include/cuComplex.h: 272

cuComplex = cuFloatComplex # /usr/local/cuda/include/cuComplex.h: 387

cublasStatus = c_uint # /usr/local/cuda/include/cublas.h: 107


# /usr/local/cuda/include/cublas.h: 124
if hasattr(_libs['cublas'], 'cublasInit'):
    cublasInit = _libs['cublas'].cublasInit
    cublasInit.restype = cublasStatus
    cublasInit.argtypes = []

# /usr/local/cuda/include/cublas.h: 138
if hasattr(_libs['cublas'], 'cublasShutdown'):
    cublasShutdown = _libs['cublas'].cublasShutdown
    cublasShutdown.restype = cublasStatus
    cublasShutdown.argtypes = []

# /usr/local/cuda/include/cublas.h: 152
if hasattr(_libs['cublas'], 'cublasGetError'):
    cublasGetError = _libs['cublas'].cublasGetError
    cublasGetError.restype = cublasStatus
    cublasGetError.argtypes = []

# /usr/local/cuda/include/cublas.h: 172
if hasattr(_libs['cublas'], 'cublasAlloc'):
    cublasAlloc = _libs['cublas'].cublasAlloc
    cublasAlloc.restype = cublasStatus
    cublasAlloc.argtypes = [c_int, c_int, POINTER(POINTER(None))]

# /usr/local/cuda/include/cublas.h: 186
if hasattr(_libs['cublas'], 'cublasFree'):
    cublasFree = _libs['cublas'].cublasFree
    cublasFree.restype = cublasStatus
    cublasFree.argtypes = [POINTER(None)]

# /usr/local/cuda/include/cublas.h: 211
if hasattr(_libs['cublas'], 'cublasSetVector'):
    cublasSetVector = _libs['cublas'].cublasSetVector
    cublasSetVector.restype = cublasStatus
    cublasSetVector.argtypes = [c_int, c_int, POINTER(None), c_int, POINTER(None), c_int]

# /usr/local/cuda/include/cublas.h: 237
if hasattr(_libs['cublas'], 'cublasGetVector'):
    cublasGetVector = _libs['cublas'].cublasGetVector
    cublasGetVector.restype = cublasStatus
    cublasGetVector.argtypes = [c_int, c_int, POINTER(None), c_int, POINTER(None), c_int]

# /usr/local/cuda/include/cublas.h: 261
if hasattr(_libs['cublas'], 'cublasSetMatrix'):
    cublasSetMatrix = _libs['cublas'].cublasSetMatrix
    cublasSetMatrix.restype = cublasStatus
    cublasSetMatrix.argtypes = [c_int, c_int, c_int, POINTER(None), c_int, POINTER(None), c_int]

# /usr/local/cuda/include/cublas.h: 285
if hasattr(_libs['cublas'], 'cublasGetMatrix'):
    cublasGetMatrix = _libs['cublas'].cublasGetMatrix
    cublasGetMatrix.restype = cublasStatus
    cublasGetMatrix.argtypes = [c_int, c_int, c_int, POINTER(None), c_int, POINTER(None), c_int]

# /usr/local/cuda/include/cublas.h: 319
if hasattr(_libs['cublas'], 'cublasIsamax'):
    cublasIsamax = _libs['cublas'].cublasIsamax
    cublasIsamax.restype = c_int
    cublasIsamax.argtypes = [c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 348
if hasattr(_libs['cublas'], 'cublasIsamin'):
    cublasIsamin = _libs['cublas'].cublasIsamin
    cublasIsamin.restype = c_int
    cublasIsamin.argtypes = [c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 378
if hasattr(_libs['cublas'], 'cublasSasum'):
    cublasSasum = _libs['cublas'].cublasSasum
    cublasSasum.restype = c_float
    cublasSasum.argtypes = [c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 414
if hasattr(_libs['cublas'], 'cublasSaxpy'):
    cublasSaxpy = _libs['cublas'].cublasSaxpy
    cublasSaxpy.restype = None
    cublasSaxpy.argtypes = [c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 447
if hasattr(_libs['cublas'], 'cublasScopy'):
    cublasScopy = _libs['cublas'].cublasScopy
    cublasScopy.restype = None
    cublasScopy.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 481
if hasattr(_libs['cublas'], 'cublasSdot'):
    cublasSdot = _libs['cublas'].cublasSdot
    cublasSdot.restype = c_float
    cublasSdot.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 512
if hasattr(_libs['cublas'], 'cublasSnrm2'):
    cublasSnrm2 = _libs['cublas'].cublasSnrm2
    cublasSnrm2.restype = c_float
    cublasSnrm2.argtypes = [c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 550
if hasattr(_libs['cublas'], 'cublasSrot'):
    cublasSrot = _libs['cublas'].cublasSrot
    cublasSrot.restype = None
    cublasSrot.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, c_float]

# /usr/local/cuda/include/cublas.h: 592
if hasattr(_libs['cublas'], 'cublasSrotg'):
    cublasSrotg = _libs['cublas'].cublasSrotg
    cublasSrotg.restype = None
    cublasSrotg.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]

# /usr/local/cuda/include/cublas.h: 640
if hasattr(_libs['cublas'], 'cublasSrotm'):
    cublasSrotm = _libs['cublas'].cublasSrotm
    cublasSrotm.restype = None
    cublasSrotm.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int, POINTER(c_float)]

# /usr/local/cuda/include/cublas.h: 683
if hasattr(_libs['cublas'], 'cublasSrotmg'):
    cublasSrotmg = _libs['cublas'].cublasSrotmg
    cublasSrotmg.restype = None
    cublasSrotmg.argtypes = [POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float), POINTER(c_float)]

# /usr/local/cuda/include/cublas.h: 714
if hasattr(_libs['cublas'], 'cublasSscal'):
    cublasSscal = _libs['cublas'].cublasSscal
    cublasSscal.restype = None
    cublasSscal.argtypes = [c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 744
if hasattr(_libs['cublas'], 'cublasSswap'):
    cublasSswap = _libs['cublas'].cublasSswap
    cublasSswap.restype = None
    cublasSswap.argtypes = [c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 782
if hasattr(_libs['cublas'], 'cublasCaxpy'):
    cublasCaxpy = _libs['cublas'].cublasCaxpy
    cublasCaxpy.restype = None
    cublasCaxpy.argtypes = [c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 815
if hasattr(_libs['cublas'], 'cublasCcopy'):
    cublasCcopy = _libs['cublas'].cublasCcopy
    cublasCcopy.restype = None
    cublasCcopy.argtypes = [c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 848
if hasattr(_libs['cublas'], 'cublasZcopy'):
    cublasZcopy = _libs['cublas'].cublasZcopy
    cublasZcopy.restype = None
    cublasZcopy.argtypes = [c_int, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 879
if hasattr(_libs['cublas'], 'cublasCscal'):
    cublasCscal = _libs['cublas'].cublasCscal
    cublasCscal.restype = None
    cublasCscal.argtypes = [c_int, cuComplex, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 912
if hasattr(_libs['cublas'], 'cublasCrotg'):
    cublasCrotg = _libs['cublas'].cublasCrotg
    cublasCrotg.restype = None
    cublasCrotg.argtypes = [POINTER(cuComplex), cuComplex, POINTER(c_float), POINTER(cuComplex)]

# /usr/local/cuda/include/cublas.h: 951
if hasattr(_libs['cublas'], 'cublasCrot'):
    cublasCrot = _libs['cublas'].cublasCrot
    cublasCrot.restype = None
    cublasCrot.argtypes = [c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, c_float, cuComplex]

# /usr/local/cuda/include/cublas.h: 990
if hasattr(_libs['cublas'], 'cublasCsrot'):
    cublasCsrot = _libs['cublas'].cublasCsrot
    cublasCsrot.restype = None
    cublasCsrot.argtypes = [c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, c_float, c_float]

# /usr/local/cuda/include/cublas.h: 1021
if hasattr(_libs['cublas'], 'cublasCsscal'):
    cublasCsscal = _libs['cublas'].cublasCsscal
    cublasCsscal.restype = None
    cublasCsscal.argtypes = [c_int, c_float, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1054
if hasattr(_libs['cublas'], 'cublasCswap'):
    cublasCswap = _libs['cublas'].cublasCswap
    cublasCswap.restype = None
    cublasCswap.argtypes = [c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1088
if hasattr(_libs['cublas'], 'cublasZswap'):
    cublasZswap = _libs['cublas'].cublasZswap
    cublasZswap.restype = None
    cublasZswap.argtypes = [c_int, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1122
if hasattr(_libs['cublas'], 'cublasCdotu'):
    cublasCdotu = _libs['cublas'].cublasCdotu
    cublasCdotu.restype = cuComplex
    cublasCdotu.argtypes = [c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1157
if hasattr(_libs['cublas'], 'cublasCdotc'):
    cublasCdotc = _libs['cublas'].cublasCdotc
    cublasCdotc.restype = cuComplex
    cublasCdotc.argtypes = [c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1187
if hasattr(_libs['cublas'], 'cublasIcamax'):
    cublasIcamax = _libs['cublas'].cublasIcamax
    cublasIcamax.restype = c_int
    cublasIcamax.argtypes = [c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1216
if hasattr(_libs['cublas'], 'cublasIcamin'):
    cublasIcamin = _libs['cublas'].cublasIcamin
    cublasIcamin.restype = c_int
    cublasIcamin.argtypes = [c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1247
if hasattr(_libs['cublas'], 'cublasScasum'):
    cublasScasum = _libs['cublas'].cublasScasum
    cublasScasum.restype = c_float
    cublasScasum.argtypes = [c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1275
if hasattr(_libs['cublas'], 'cublasScnrm2'):
    cublasScnrm2 = _libs['cublas'].cublasScnrm2
    cublasScnrm2.restype = c_float
    cublasScnrm2.argtypes = [c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1313
if hasattr(_libs['cublas'], 'cublasZaxpy'):
    cublasZaxpy = _libs['cublas'].cublasZaxpy
    cublasZaxpy.restype = None
    cublasZaxpy.argtypes = [c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1347
if hasattr(_libs['cublas'], 'cublasZdotu'):
    cublasZdotu = _libs['cublas'].cublasZdotu
    cublasZdotu.restype = cuDoubleComplex
    cublasZdotu.argtypes = [c_int, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1378
if hasattr(_libs['cublas'], 'cublasZscal'):
    cublasZscal = _libs['cublas'].cublasZscal
    cublasZscal.restype = None
    cublasZscal.argtypes = [c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 1442
if hasattr(_libs['cublas'], 'cublasSgbmv'):
    cublasSgbmv = _libs['cublas'].cublasSgbmv
    cublasSgbmv.restype = None
    cublasSgbmv.argtypes = [c_char, c_int, c_int, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1504
if hasattr(_libs['cublas'], 'cublasSgemv'):
    cublasSgemv = _libs['cublas'].cublasSgemv
    cublasSgemv.restype = None
    cublasSgemv.argtypes = [c_char, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1552
if hasattr(_libs['cublas'], 'cublasSger'):
    cublasSger = _libs['cublas'].cublasSger
    cublasSger.restype = None
    cublasSger.argtypes = [c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1616
if hasattr(_libs['cublas'], 'cublasSsbmv'):
    cublasSsbmv = _libs['cublas'].cublasSsbmv
    cublasSsbmv.restype = None
    cublasSsbmv.argtypes = [c_char, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1669
if hasattr(_libs['cublas'], 'cublasSspmv'):
    cublasSspmv = _libs['cublas'].cublasSspmv
    cublasSspmv.restype = None
    cublasSspmv.argtypes = [c_char, c_int, c_float, POINTER(c_float), POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1719
if hasattr(_libs['cublas'], 'cublasSspr'):
    cublasSspr = _libs['cublas'].cublasSspr
    cublasSspr.restype = None
    cublasSspr.argtypes = [c_char, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float)]

# /usr/local/cuda/include/cublas.h: 1773
if hasattr(_libs['cublas'], 'cublasSspr2'):
    cublasSspr2 = _libs['cublas'].cublasSspr2
    cublasSspr2.restype = None
    cublasSspr2.argtypes = [c_char, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, POINTER(c_float)]

# /usr/local/cuda/include/cublas.h: 1832
if hasattr(_libs['cublas'], 'cublasSsymv'):
    cublasSsymv = _libs['cublas'].cublasSsymv
    cublasSsymv.restype = None
    cublasSsymv.argtypes = [c_char, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1887
if hasattr(_libs['cublas'], 'cublasSsyr'):
    cublasSsyr = _libs['cublas'].cublasSsyr
    cublasSsyr.restype = None
    cublasSsyr.argtypes = [c_char, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 1940
if hasattr(_libs['cublas'], 'cublasSsyr2'):
    cublasSsyr2 = _libs['cublas'].cublasSsyr2
    cublasSsyr2.restype = None
    cublasSsyr2.argtypes = [c_char, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2001
if hasattr(_libs['cublas'], 'cublasStbmv'):
    cublasStbmv = _libs['cublas'].cublasStbmv
    cublasStbmv.restype = None
    cublasStbmv.argtypes = [c_char, c_char, c_char, c_int, c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2062
if hasattr(_libs['cublas'], 'cublasStbsv'):
    cublasStbsv = _libs['cublas'].cublasStbsv
    cublasStbsv.restype = None
    cublasStbsv.argtypes = [c_char, c_char, c_char, c_int, c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2114
if hasattr(_libs['cublas'], 'cublasStpmv'):
    cublasStpmv = _libs['cublas'].cublasStpmv
    cublasStpmv.restype = None
    cublasStpmv.argtypes = [c_char, c_char, c_char, c_int, POINTER(c_float), POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2168
if hasattr(_libs['cublas'], 'cublasStpsv'):
    cublasStpsv = _libs['cublas'].cublasStpsv
    cublasStpsv.restype = None
    cublasStpsv.argtypes = [c_char, c_char, c_char, c_int, POINTER(c_float), POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2222
if hasattr(_libs['cublas'], 'cublasStrmv'):
    cublasStrmv = _libs['cublas'].cublasStrmv
    cublasStrmv.restype = None
    cublasStrmv.argtypes = [c_char, c_char, c_char, c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2282
if hasattr(_libs['cublas'], 'cublasStrsv'):
    cublasStrsv = _libs['cublas'].cublasStrsv
    cublasStrsv.restype = None
    cublasStrsv.argtypes = [c_char, c_char, c_char, c_int, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2340
if hasattr(_libs['cublas'], 'cublasZtrmv'):
    cublasZtrmv = _libs['cublas'].cublasZtrmv
    cublasZtrmv.restype = None
    cublasZtrmv.argtypes = [c_char, c_char, c_char, c_int, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 2403
if hasattr(_libs['cublas'], 'cublasZgemv'):
    cublasZgemv = _libs['cublas'].cublasZgemv
    cublasZgemv.restype = None
    cublasZgemv.argtypes = [c_char, c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 2467
if hasattr(_libs['cublas'], 'cublasCgemv'):
    cublasCgemv = _libs['cublas'].cublasCgemv
    cublasCgemv.restype = None
    cublasCgemv.argtypes = [c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 2470
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCgbmv'):
        cublasCgbmv = _lib.cublasCgbmv
        cublasCgbmv.restype = None
        cublasCgbmv.argtypes = [c_char, c_int, c_int, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2474
for _lib in _libs.values():
    if hasattr(_lib, 'cublasChemv'):
        cublasChemv = _lib.cublasChemv
        cublasChemv.restype = None
        cublasChemv.argtypes = [c_char, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2477
for _lib in _libs.values():
    if hasattr(_lib, 'cublasChbmv'):
        cublasChbmv = _lib.cublasChbmv
        cublasChbmv.restype = None
        cublasChbmv.argtypes = [c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2480
for _lib in _libs.values():
    if hasattr(_lib, 'cublasChpmv'):
        cublasChpmv = _lib.cublasChpmv
        cublasChpmv.restype = None
        cublasChpmv.argtypes = [c_char, c_int, cuComplex, POINTER(cuComplex), POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2537
if hasattr(_libs['cublas'], 'cublasCtrmv'):
    cublasCtrmv = _libs['cublas'].cublasCtrmv
    cublasCtrmv.restype = None
    cublasCtrmv.argtypes = [c_char, c_char, c_char, c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 2540
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCtbmv'):
        cublasCtbmv = _lib.cublasCtbmv
        cublasCtbmv.restype = None
        cublasCtbmv.argtypes = [c_char, c_char, c_char, c_int, c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2543
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCtpmv'):
        cublasCtpmv = _lib.cublasCtpmv
        cublasCtpmv.restype = None
        cublasCtpmv.argtypes = [c_char, c_char, c_char, c_int, POINTER(cuComplex), POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2545
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCtrsv'):
        cublasCtrsv = _lib.cublasCtrsv
        cublasCtrsv.restype = None
        cublasCtrsv.argtypes = [c_char, c_char, c_char, c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2548
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCtbsv'):
        cublasCtbsv = _lib.cublasCtbsv
        cublasCtbsv.restype = None
        cublasCtbsv.argtypes = [c_char, c_char, c_char, c_int, c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2551
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCtpsv'):
        cublasCtpsv = _lib.cublasCtpsv
        cublasCtpsv.restype = None
        cublasCtpsv.argtypes = [c_char, c_char, c_char, c_int, POINTER(cuComplex), POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2598
if hasattr(_libs['cublas'], 'cublasCgeru'):
    cublasCgeru = _libs['cublas'].cublasCgeru
    cublasCgeru.restype = None
    cublasCgeru.argtypes = [c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 2645
if hasattr(_libs['cublas'], 'cublasCgerc'):
    cublasCgerc = _libs['cublas'].cublasCgerc
    cublasCgerc.restype = None
    cublasCgerc.argtypes = [c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 2648
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCher'):
        cublasCher = _lib.cublasCher
        cublasCher.restype = None
        cublasCher.argtypes = [c_char, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2651
for _lib in _libs.values():
    if hasattr(_lib, 'cublasChpr'):
        cublasChpr = _lib.cublasChpr
        cublasChpr.restype = None
        cublasChpr.argtypes = [c_char, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex)]
        break

# /usr/local/cuda/include/cublas.h: 2653
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCher2'):
        cublasCher2 = _lib.cublasCher2
        cublasCher2.restype = None
        cublasCher2.argtypes = [c_char, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 2656
for _lib in _libs.values():
    if hasattr(_lib, 'cublasChpr2'):
        cublasChpr2 = _lib.cublasChpr2
        cublasChpr2.restype = None
        cublasChpr2.argtypes = [c_char, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, POINTER(cuComplex)]
        break

# /usr/local/cuda/include/cublas.h: 2726
if hasattr(_libs['cublas'], 'cublasSgemm'):
    cublasSgemm = _libs['cublas'].cublasSgemm
    cublasSgemm.restype = int
    cublasSgemm.argtypes = [c_char, c_char, c_int, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2808
if hasattr(_libs['cublas'], 'cublasSsymm'):
    cublasSsymm = _libs['cublas'].cublasSsymm
    cublasSsymm.restype = None
    cublasSsymm.argtypes = [c_char, c_char, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2885
if hasattr(_libs['cublas'], 'cublasSsyrk'):
    cublasSsyrk = _libs['cublas'].cublasSsyrk
    cublasSsyrk.restype = None
    cublasSsyrk.argtypes = [c_char, c_char, c_int, c_int, c_float, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 2970
if hasattr(_libs['cublas'], 'cublasSsyr2k'):
    cublasSsyr2k = _libs['cublas'].cublasSsyr2k
    cublasSsyr2k.restype = None
    cublasSsyr2k.argtypes = [c_char, c_char, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int, c_float, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 3043
if hasattr(_libs['cublas'], 'cublasStrmm'):
    cublasStrmm = _libs['cublas'].cublasStrmm
    cublasStrmm.restype = None
    cublasStrmm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 3121
if hasattr(_libs['cublas'], 'cublasStrsm'):
    cublasStrsm = _libs['cublas'].cublasStrsm
    cublasStrsm.restype = None
    cublasStrsm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, c_float, POINTER(c_float), c_int, POINTER(c_float), c_int]

# /usr/local/cuda/include/cublas.h: 3187
if hasattr(_libs['cublas'], 'cublasCgemm'):
    cublasCgemm = _libs['cublas'].cublasCgemm
    cublasCgemm.restype = None
    cublasCgemm.argtypes = [c_char, c_char, c_int, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 3268
if hasattr(_libs['cublas'], 'cublasCsymm'):
    cublasCsymm = _libs['cublas'].cublasCsymm
    cublasCsymm.restype = None
    cublasCsymm.argtypes = [c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 3272
for _lib in _libs.values():
    if hasattr(_lib, 'cublasChemm'):
        cublasChemm = _lib.cublasChemm
        cublasChemm.restype = None
        cublasChemm.argtypes = [c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 3350
if hasattr(_libs['cublas'], 'cublasCsyrk'):
    cublasCsyrk = _libs['cublas'].cublasCsyrk
    cublasCsyrk.restype = None
    cublasCsyrk.argtypes = [c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 3429
if hasattr(_libs['cublas'], 'cublasCherk'):
    cublasCherk = _libs['cublas'].cublasCherk
    cublasCherk.restype = None
    cublasCherk.argtypes = [c_char, c_char, c_int, c_int, c_float, POINTER(cuComplex), c_int, c_float, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 3432
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCsyr2k'):
        cublasCsyr2k = _lib.cublasCsyr2k
        cublasCsyr2k.restype = None
        cublasCsyr2k.argtypes = [c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 3436
for _lib in _libs.values():
    if hasattr(_lib, 'cublasCher2k'):
        cublasCher2k = _lib.cublasCher2k
        cublasCher2k.restype = None
        cublasCher2k.argtypes = [c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int, cuComplex, POINTER(cuComplex), c_int]
        break

# /usr/local/cuda/include/cublas.h: 3511
if hasattr(_libs['cublas'], 'cublasCtrmm'):
    cublasCtrmm = _libs['cublas'].cublasCtrmm
    cublasCtrmm.restype = None
    cublasCtrmm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 3589
if hasattr(_libs['cublas'], 'cublasCtrsm'):
    cublasCtrsm = _libs['cublas'].cublasCtrsm
    cublasCtrsm.restype = None
    cublasCtrsm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, cuComplex, POINTER(cuComplex), c_int, POINTER(cuComplex), c_int]

# /usr/local/cuda/include/cublas.h: 3593
if hasattr(_libs['cublas'], 'cublasXerbla'):
    cublasXerbla = _libs['cublas'].cublasXerbla
    cublasXerbla.restype = None
    cublasXerbla.argtypes = [c_char_p, c_int]

# /usr/local/cuda/include/cublas.h: 3626
if hasattr(_libs['cublas'], 'cublasDasum'):
    cublasDasum = _libs['cublas'].cublasDasum
    cublasDasum.restype = c_double
    cublasDasum.argtypes = [c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 3663
if hasattr(_libs['cublas'], 'cublasDaxpy'):
    cublasDaxpy = _libs['cublas'].cublasDaxpy
    cublasDaxpy.restype = None
    cublasDaxpy.argtypes = [c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 3697
if hasattr(_libs['cublas'], 'cublasDcopy'):
    cublasDcopy = _libs['cublas'].cublasDcopy
    cublasDcopy.restype = None
    cublasDcopy.argtypes = [c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 3732
if hasattr(_libs['cublas'], 'cublasDdot'):
    cublasDdot = _libs['cublas'].cublasDdot
    cublasDdot.restype = c_double
    cublasDdot.argtypes = [c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 3764
if hasattr(_libs['cublas'], 'cublasDnrm2'):
    cublasDnrm2 = _libs['cublas'].cublasDnrm2
    cublasDnrm2.restype = c_double
    cublasDnrm2.argtypes = [c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 3803
if hasattr(_libs['cublas'], 'cublasDrot'):
    cublasDrot = _libs['cublas'].cublasDrot
    cublasDrot.restype = None
    cublasDrot.argtypes = [c_int, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, c_double]

# /usr/local/cuda/include/cublas.h: 3845
if hasattr(_libs['cublas'], 'cublasDrotg'):
    cublasDrotg = _libs['cublas'].cublasDrotg
    cublasDrotg.restype = None
    cublasDrotg.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]

# /usr/local/cuda/include/cublas.h: 3894
if hasattr(_libs['cublas'], 'cublasDrotm'):
    cublasDrotm = _libs['cublas'].cublasDrotm
    cublasDrotm.restype = None
    cublasDrotm.argtypes = [c_int, POINTER(c_double), c_int, POINTER(c_double), c_int, POINTER(c_double)]

# /usr/local/cuda/include/cublas.h: 3937
if hasattr(_libs['cublas'], 'cublasDrotmg'):
    cublasDrotmg = _libs['cublas'].cublasDrotmg
    cublasDrotmg.restype = None
    cublasDrotmg.argtypes = [POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double), POINTER(c_double)]

# /usr/local/cuda/include/cublas.h: 3969
if hasattr(_libs['cublas'], 'cublasDscal'):
    cublasDscal = _libs['cublas'].cublasDscal
    cublasDscal.restype = None
    cublasDscal.argtypes = [c_int, c_double, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4000
if hasattr(_libs['cublas'], 'cublasDswap'):
    cublasDswap = _libs['cublas'].cublasDswap
    cublasDswap.restype = None
    cublasDswap.argtypes = [c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4030
if hasattr(_libs['cublas'], 'cublasIdamax'):
    cublasIdamax = _libs['cublas'].cublasIdamax
    cublasIdamax.restype = c_int
    cublasIdamax.argtypes = [c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4060
if hasattr(_libs['cublas'], 'cublasIdamin'):
    cublasIdamin = _libs['cublas'].cublasIdamin
    cublasIdamin.restype = c_int
    cublasIdamin.argtypes = [c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4124
if hasattr(_libs['cublas'], 'cublasDgemv'):
    cublasDgemv = _libs['cublas'].cublasDgemv
    cublasDgemv.restype = None
    cublasDgemv.argtypes = [c_char, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4173
if hasattr(_libs['cublas'], 'cublasDger'):
    cublasDger = _libs['cublas'].cublasDger
    cublasDger.restype = None
    cublasDger.argtypes = [c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4229
if hasattr(_libs['cublas'], 'cublasDsyr'):
    cublasDsyr = _libs['cublas'].cublasDsyr
    cublasDsyr.restype = None
    cublasDsyr.argtypes = [c_char, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4291
if hasattr(_libs['cublas'], 'cublasDtrsv'):
    cublasDtrsv = _libs['cublas'].cublasDtrsv
    cublasDtrsv.restype = None
    cublasDtrsv.argtypes = [c_char, c_char, c_char, c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4346
if hasattr(_libs['cublas'], 'cublasDtrmv'):
    cublasDtrmv = _libs['cublas'].cublasDtrmv
    cublasDtrmv.restype = None
    cublasDtrmv.argtypes = [c_char, c_char, c_char, c_int, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4416
if hasattr(_libs['cublas'], 'cublasDgemm'):
    cublasDgemm = _libs['cublas'].cublasDgemm
    cublasDgemm.restype = None
    cublasDgemm.argtypes = [c_char, c_char, c_int, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4496
if hasattr(_libs['cublas'], 'cublasDtrsm'):
    cublasDtrsm = _libs['cublas'].cublasDtrsm
    cublasDtrsm.restype = None
    cublasDtrsm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4577
if hasattr(_libs['cublas'], 'cublasZtrsm'):
    cublasZtrsm = _libs['cublas'].cublasZtrsm
    cublasZtrsm.restype = None
    cublasZtrsm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 4652
if hasattr(_libs['cublas'], 'cublasDtrmm'):
    cublasDtrmm = _libs['cublas'].cublasDtrmm
    cublasDtrmm.restype = None
    cublasDtrmm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4735
if hasattr(_libs['cublas'], 'cublasDsymm'):
    cublasDsymm = _libs['cublas'].cublasDsymm
    cublasDsymm.restype = None
    cublasDsymm.argtypes = [c_char, c_char, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4818
if hasattr(_libs['cublas'], 'cublasZsymm'):
    cublasZsymm = _libs['cublas'].cublasZsymm
    cublasZsymm.restype = None
    cublasZsymm.argtypes = [c_char, c_char, c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 4898
if hasattr(_libs['cublas'], 'cublasDsyrk'):
    cublasDsyrk = _libs['cublas'].cublasDsyrk
    cublasDsyrk.restype = None
    cublasDsyrk.argtypes = [c_char, c_char, c_int, c_int, c_double, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 4976
if hasattr(_libs['cublas'], 'cublasZsyrk'):
    cublasZsyrk = _libs['cublas'].cublasZsyrk
    cublasZsyrk.restype = None
    cublasZsyrk.argtypes = [c_char, c_char, c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 5064
if hasattr(_libs['cublas'], 'cublasDsyr2k'):
    cublasDsyr2k = _libs['cublas'].cublasDsyr2k
    cublasDsyr2k.restype = None
    cublasDsyr2k.argtypes = [c_char, c_char, c_int, c_int, c_double, POINTER(c_double), c_int, POINTER(c_double), c_int, c_double, POINTER(c_double), c_int]

# /usr/local/cuda/include/cublas.h: 5130
if hasattr(_libs['cublas'], 'cublasZgemm'):
    cublasZgemm = _libs['cublas'].cublasZgemm
    cublasZgemm.restype = None
    cublasZgemm.argtypes = [c_char, c_char, c_int, c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 5209
if hasattr(_libs['cublas'], 'cublasZtrmm'):
    cublasZtrmm = _libs['cublas'].cublasZtrmm
    cublasZtrmm.restype = None
    cublasZtrmm.argtypes = [c_char, c_char, c_char, c_char, c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 5259
if hasattr(_libs['cublas'], 'cublasZgeru'):
    cublasZgeru = _libs['cublas'].cublasZgeru
    cublasZgeru.restype = None
    cublasZgeru.argtypes = [c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 5309
if hasattr(_libs['cublas'], 'cublasZgerc'):
    cublasZgerc = _libs['cublas'].cublasZgerc
    cublasZgerc.restype = None
    cublasZgerc.argtypes = [c_int, c_int, cuDoubleComplex, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 5390
if hasattr(_libs['cublas'], 'cublasZherk'):
    cublasZherk = _libs['cublas'].cublasZherk
    cublasZherk.restype = None
    cublasZherk.argtypes = [c_char, c_char, c_int, c_int, c_double, POINTER(cuDoubleComplex), c_int, c_double, POINTER(cuDoubleComplex), c_int]

# /usr/local/cuda/include/cublas.h: 97
CUBLAS_STATUS_SUCCESS = 0

# /usr/local/cuda/include/cublas.h: 98
CUBLAS_STATUS_NOT_INITIALIZED = 1

# /usr/local/cuda/include/cublas.h: 99
CUBLAS_STATUS_ALLOC_FAILED = 3

# /usr/local/cuda/include/cublas.h: 100
CUBLAS_STATUS_INVALID_VALUE = 7

# /usr/local/cuda/include/cublas.h: 101
CUBLAS_STATUS_ARCH_MISMATCH = 8

# /usr/local/cuda/include/cublas.h: 102
CUBLAS_STATUS_MAPPING_ERROR = 11

# /usr/local/cuda/include/cublas.h: 103
CUBLAS_STATUS_EXECUTION_FAILED = 13

# /usr/local/cuda/include/cublas.h: 104
CUBLAS_STATUS_INTERNAL_ERROR = 14
