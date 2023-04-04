# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop backends

This file contains main backends of axiprop:
- Numpy
- PyOpenCL
- CuPy
- ArrayFire
- NumPy + MKL
- NumPy + PyFFTW
"""
import numpy as np
from scipy.linalg import pinv as scipy_inv
from scipy.linalg import inv as scipy_inv_sqr

def inv_on_host(self, M, dtype):
    M = scipy_inv(M, check_finite=False)
    M = M.astype(dtype)
    return M

def inv_sqr_on_host(self, M, dtype):
    M = scipy_inv_sqr(M, overwrite_a=True, check_finite=False)
    M = M.astype(dtype)
    return M


AVAILABLE_BACKENDS = {}

backend_strings_ordered = ['CU', 'CL', 'AF','NP_MKL', 'NP_FFTW', 'NP']

################ NumPy ################
class BACKEND_NP():

    name ='NP'
    zeros = np.zeros
    sqrt = np.sqrt
    exp = np.exp
    abs = np.abs
    inv_on_host = inv_on_host
    inv_sqr_on_host = inv_sqr_on_host
    inv = inv_on_host

    def divide_abs_or_set_to_one(self, ar1, ar2):
        condition = (ar2>0)
        result = np.where(condition, np.divide(ar1, ar2, where=condition), 1.0)
        return result

    def to_host(self, arr_in):
        return arr_in

    def to_device(self, arr_in, dtype=None):
        return arr_in

    def make_matmul(self, matrix_in, vec_in, vec_out):
        def matmul(a, b, c):
            c = np.dot(a, b, out=c)
            return c

        return matmul

    def make_fft2(self, vec_in, vec_out):
        def fft2(a, b):
            b = np.fft.fft2(a)
            return b

        def ifft2(a, b):
            b = np.fft.ifft2(a)
            return b
 
        def fftshift(a):
            b = np.fft.fftshift(a)
            return b

        return fft2, ifft2, fftshift

AVAILABLE_BACKENDS['NP'] = BACKEND_NP

################ PyOpenCL ################
try:
    class BACKEND_CL():

        from pyopencl import create_some_context
        from pyopencl import CommandQueue, Program
        import pyopencl.array as arrcl
        import pyopencl.clmath as clmath
        from reikna.cluda import ocl_api
        from reikna.linalg import MatrixMul
        from reikna.fft import FFT, FFTShift

        name = 'CL'
        ctx = create_some_context() #answers=[1,2])
        queue = CommandQueue(ctx)
        api = ocl_api()
        thrd = api.Thread(cqd=queue)

        inv_on_host = inv_on_host
        inv_sqr_on_host = inv_sqr_on_host

        divide_abs_or_set_to_one_prg = Program(ctx,
        """
        __kernel void myknl(
          __global const double *a1,
          __global const double *a2,
          __global double *res
        )
        {
          int gid = get_global_id(0);
          res[gid] = 1.0;
          if (a2[gid]>0.0){
            res[gid] = a1[gid] / a2[gid];
          }
        }
        """).build()
        divide_abs_or_set_to_one_knl = divide_abs_or_set_to_one_prg.myknl

        def to_host(self, arr_in):
            arr_out = arr_in.get()
            return arr_out

        def to_device(self, arr_in, dtype=None):
            arr_out = self.thrd.to_device(np.ascontiguousarray(arr_in))
            return arr_out

        def divide_abs_or_set_to_one(self, ar1, ar2):
            res = self.arrcl.zeros_like(ar2)
            self.divide_abs_or_set_to_one_knl(
                self.queue, ar1.shape, None, ar1.data, ar2.data, res.data
            ).wait()
            return res

        def zeros(self, shape, dtype):
            arr_out = self.arrcl.zeros(self.queue, shape, dtype)
            return arr_out

        def sqrt(self, arr_in):
            arr_out = self.clmath.sqrt(arr_in, self.queue)
            return arr_out

        def exp(self, arr_in):
            arr_out = self.clmath.exp(arr_in, self.queue)
            return arr_out

        def abs(self, arr):
            arr_out = arr.__abs__()
            return arr_out

        def inv(self, M, dtype):
            print("matrix inversion on device is not available [CL]")
            return M

        def make_matmul(self, matrix_in, vec_in, vec_out):
            matmul_reikna = self.MatrixMul(matrix_in, vec_in[:,None],
                out_arr=vec_out[:,None]).compile(self.thrd)

            def matmul(a, b, c):
                matmul_reikna(c, a, b)[0].wait()
                return c

            return matmul

        def make_fft2(self, vec_in, vec_out):
            fft_reikna = self.FFT(vec_in).compile(self.thrd)
            fftshift_reikna = self.FFTShift(vec_in).compile(self.thrd)

            def fft2(a, b):
                fft_reikna(b, a, inverse=0)[0].wait()
                return b

            def ifft2(a, b):
                fft_reikna(b, a, inverse=1)[0].wait()
                return b

            def fftshift(a):
                b = a.copy()
                fftshift_reikna(b, a)[0].wait()
                return b

            return fft2, ifft2, fftshift

    AVAILABLE_BACKENDS['CL'] = BACKEND_CL
except Exception:
    pass

################ CuPy ################
try:
    class BACKEND_CU():

        import cupy as cp

        name = 'CU'
        sqrt = cp.sqrt
        exp = cp.exp
        abs = cp.abs
        inv_on_host = inv_on_host
        inv_sqr_on_host = inv_sqr_on_host

        def divide_abs_or_set_to_one(self, ar1, ar2):
            return ( self.cp.where(ar2>0, ar1/ar2, 1.0) )

        def to_host(self, arr_in):
            arr_out = self.cp.asnumpy(arr_in)
            return arr_out

        def zeros(self, shape, dtype):
            arr_out = self.cp.zeros(shape, dtype)
            return arr_out

        def to_device(self, arr_in, dtype=None):
            arr_out = self.cp.asarray(arr_in)
            return arr_out

        def inv(self, M, dtype):
            M = self.cp.linalg.pinv(M)
            return M

        def make_matmul(self, matrix_in, vec_in, vec_out):
            def matmul(a, b, c):
                c = self.cp.matmul(a, b)
                return c

            return matmul

        def make_fft2(self, vec_in, vec_out):
            def fft2(a, b):
                b = self.cp.fft.fft2(a)
                return b

            def ifft2(a, b):
                b = self.cp.fft.ifft2(a)
                return b

            def fftshift(a):
                b = self.cp.fft.fftshift(a)
                return b

            return fft2, ifft2, fftshift

    AVAILABLE_BACKENDS['CU'] = BACKEND_CU
except Exception:
    pass

################ ArrayFire ################
try:
    class BACKEND_AF():
        import arrayfire as af

        name = 'AF'
        inv_on_host = inv_on_host
        inv_sqr_on_host = inv_sqr_on_host

        def divide_abs_or_set_to_one(self, ar1, ar2):
            return ( self.af.where(a2>0, ar1/ar2, 1.0) )

        def to_host(self, arr_in):
            arr_out = arr_in.to_ndarray()
            return arr_out

        def to_device(self, arr_in, dtype=None):
            if dtype is not None:
                arr_in = arr_in.astype(dtype)
            arr_out = self.af.from_ndarray(arr_in)
            return arr_out

        def zeros(self, shape, dtype):
            arr_out = self.af.from_ndarray(np.zeros(shape, dtype))
            return arr_out

        def sqrt(self, arr_in):
            arr_out = self.af.sqrt(arr_in)
            return arr_out

        def exp(self, arr_in):
            arr_out = self.af.exp(arr_in)
            return arr_out

        def abs(self, arr_in):
            arr_out = self.af.abs(arr_in)
            return arr_out

        def inv(self, M, dtype):
            M = self.af.inverse(M)
            dtype_af = self.af.to_dtype[np.dtype(dtype).char]
            M = M.as_type(dtype_af)
            return M

        def make_matmul(self, matrix_in, vec_in, vec_out):
            def matmul(a, b, c):
                c = self.af.matmul(a, b)
                return c

            return matmul

        def make_fft2(self, vec_in, vec_out):
            def fft2(a, b):
                b = self.af.signal.fft2(a)
                return b

            def ifft2(a, b):
                b = self.af.signal.ifft2(a)
                return b

            def fftshift(a):
                b = self.to_host(a)
                b = np.fft.fftshift(b)
                b = self.bcknd.to_device(b)
                return b

            return fft2, ifft2, fftshift

    AVAILABLE_BACKENDS['AF'] = BACKEND_AF
except Exception:
    pass


################ NumPy+MLK ################
try:
    class BACKEND_NPMKL(BACKEND_NP):

        import mkl_fft

        name ='NP_MKL'
        mklfft2 = mkl_fft.fft2
        mklifft2 = mkl_fft.ifft2

        def make_fft2(self, vec_in, vec_out):
            def fft2(a, b):
                b = self.mklfft2(a)
                return b

            def ifft2(a, b):
                b = self.mklifft2(a)
                return b

            def fftshift(a):
                b = np.fft.fftshift(a)
                return b

            return fft2, ifft2, fftshift

    AVAILABLE_BACKENDS['NP_MKL'] = BACKEND_NPMKL
except Exception:
    pass

################ NumPy+PyFFTW ################
try:
    class BACKEND_NPFFTW(BACKEND_NP):

        import pyfftw

        name ='NP_FFTW'

        def make_fft2(self, vec_in, vec_out):
            threads = 6

            self.fftw2 = self.pyfftw.FFTW( vec_in, vec_out, axes=(-1,0),
                direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), threads=threads)
            self.ifftw2 = self.pyfftw.FFTW( vec_out, vec_in, axes=(-1,0),
                direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), threads=threads,
                normalise_idft=True)

            def fft2(a, b):
                b = self.fftw2(a, b)
                return b

            def ifft2(a, b):
                b = self.ifftw2(a, b)
                return b

            def fftshift(a):
                b = np.fft.fftshift(a)
                return b

            return fft2, ifft2, fftshift

    AVAILABLE_BACKENDS['NP_FFTW'] = BACKEND_NPFFTW
except Exception:
    pass
