# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop backends

This file contains main backends of axiprop:
- PyOpenCL
- CuPy
- ArrayFire
- PropagatorFFT2
"""
import numpy as np
from scipy.linalg import pinv2

BACKENDS = {}

################ NumPy ################
class BACKEND_NP():

    zeros = np.zeros
    sqrt = np.sqrt
    exp = np.exp
    abs = np.abs

    def get(self, arr_in):
        return arr_in

    def send_to_device(self, arr_in):
        return arr_in

    def pinv(self, M, dtype):
        M = pinv2(M)
        M = M.astype(dtype)
        return M

    def make_matmul(self, matrix_in, vec_in, vec_out):
        def matmul(a, b, c):
            c = np.dot(a, b)
            return c

        return matmul

    def make_fft(self, vec_in, vec_out, vec_out2):
        def fft2(a, b):
            b = np.fft.fft2(a, norm="ortho")
            return b

        def ifft2(a, b):
            b = np.fft.ifft2(a, norm="ortho")
            return b

        return fft2, ifft2

BACKENDS['NP'] = BACKEND_NP


################ PyOpenCL ################
try:
    class BACKEND_CL():
        from pyopencl import create_some_context
        from pyopencl import CommandQueue
        import pyopencl.array as arrcl
        import pyopencl.clmath as clmath
        from reikna.cluda import ocl_api
        from reikna.linalg import MatrixMul
        from reikna.fft import FFT

        ctx = create_some_context() #answers=[1,2])
        queue = CommandQueue(ctx)
        api = ocl_api()
        thrd = api.Thread(cqd=queue)

        def get(self, arr_in):
            arr_out = arr_in.get()
            return arr_out

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

        def send_to_device(self, arr_in):
            arr_out = self.thrd.to_device(arr_in)
            return arr_out

        def pinv(self, M, dtype):
            M = pinv2(M, check_finite=False)
            M = self.send_to_device(M)
            return M

        def make_matmul(self, matrix_in, vec_in, vec_out):
            matmul_reikna = self.MatrixMul(matrix_in, vec_in[:,None],
                out_arr=vec_out[:,None]).compile(self.thrd)

            def matmul(a, b, c):
                matmul_reikna(c, a, b)[0].wait()
                return c

            return matmul

        def make_fft(self, vec_in, vec_out, vec_out2):
            fft_reikna = self.FFT(vec_in).compile(self.thrd)

            def fft2(a, b):
                fft_reikna(b, a, inverse=0)[0].wait()
                return b

            def ifft2(a, b):
                fft_reikna(b, a, inverse=1)[0].wait()
                return b

            return fft2, ifft2

    BACKENDS['CL'] = BACKEND_CL
except Exception:
    pass

################ CuPy ################
try:
    class BACKEND_CU():
        import cupy as cp

        sqrt = cp.sqrt
        exp = cp.exp
        abs = cp.abs

        def get(self, arr_in):
            arr_out = self.cp.asnumpy(arr_in)
            return arr_out

        def zeros(self, shape, dtype):
            arr_out = self.cp.zeros(shape, dtype)
            return arr_out

        def send_to_device(self, arr_in):
            arr_out = self.cp.asarray(arr_in)
            return arr_out

        def pinv(self, M, dtype):
            M = self.send_to_device(M)
            M = self.cp.linalg.pinv(M)
            M = M.astype(dtype)
            return M

        def make_matmul(self, matrix_in, vec_in, vec_out):
            def matmul(a, b, c):
                c = self.cp.matmul(a, b)
                return c

            return matmul

        def make_fft(self, vec_in, vec_out, vec_out2):
            def fft2(a, b):
                b = self.cp.fft.fft2(a, norm="ortho")
                return b

            def ifft2(a, b):
                b = self.cp.fft.ifft2(a, norm="ortho")
                return b

            return fft2, ifft2
    BACKENDS['CU'] = BACKEND_CU
except Exception:
    pass

################ ArrayFire ################
try:
    class BACKEND_AF():
        import arrayfire as af

        sqrt = af.sqrt
        exp = af.exp
        abs = af.abs


        def get(self, arr_in):
            arr_out = self.af.asnumpy(arr_in)
            return arr_out

        def zeros(self, shape, dtype):
            arr_out = self.af.from_ndarray(np.zeros(shape, dtype))
            return arr_out

        """
        def sqrt(self, arr_in):
            arr_out = self.af.sqrt(arr_in)
            return arr_out

        def exp(self, arr_in):
            arr_out = self.af.exp(arr_in)
            return arr_out

        def abs(self, arr_in):
            arr_out = self.af.abs(arr_in)
            return arr_out
        """

        def send_to_device(self, arr_in):
            arr_out = self.af.from_ndarray(arr_in)
            return arr_out

        def pinv(self, M, dtype):
            M = self.send_to_device(M)
            M = self.af.inverse(M)
            M = M.as_type(dtype)
            return M

        def make_matmul(self, matrix_in, vec_in, vec_out):
            def matmul(a, b, c):
                c = self.af.matmul(a, b)
                return c

            return matmul

        def make_fft(self, vec_in, vec_out, vec_out2):
            def fft2(a, b):
                b = self.af.signal.fft2(a)
                return b

            def ifft2(a, b):
                b = self.af.signal.ifft2(a)
                return b

            return fft2, ifft2
    BACKENDS['AF'] = BACKEND_AF
except Exception:
    pass


################ NumPy+MLK ################
try:
    class BACKEND_NPMKL(BACKEND_NP):

        import mkl_fft

        mklfft2 = mkl_fft.fft2
        mklifft2 = mkl_fft.ifft2

        def make_fft(self, vec_in, vec_out, vec_out2):
            def fft2(a, b):
                b = self.mklfft2(a)
                return b

            def ifft2(a, b):
                b = self.mklifft2(a)
                return b

            return fft2, ifft2

    BACKENDS['NP_MKL'] = BACKEND_NPMKL
except Exception:
    pass

################ NumPy+PyFFTW ################
try:
    class BACKEND_NPFFTW(BACKEND_NP):

        import pyfftw

        def make_fft(self, vec_in, vec_out, vec_out2):
            threads = 6

            self.fftw2 = self.pyfftw.FFTW( vec_in, vec_out, axes=(-1,0),
                direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), threads=threads)
            self.ifftw2 = self.pyfftw.FFTW( vec_out, vec_out2, axes=(-1,0),
                direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), threads=threads,
                normalise_idft=True)

            def fft2(a, b):
                b = self.fftw2(a, b)
                return b

            def ifft2(a, b):
                b = self.ifftw2(a, b)
                return b

            return fft2, ifft2

    BACKENDS['NP_FFTW'] = BACKEND_NPFFTW
except Exception:
    pass