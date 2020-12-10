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

try:
    class GPU_BACKEND_CL():
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

        def make_fft(self, vec_in, vec_out):
            fft_reikna = self.FFT(vec_in).compile(self.thrd)

            def fft2(a, b):
                fft_reikna(b, a, inverse=0)[0].wait()
                return b

            def ifft2(a, b):
                fft_reikna(b, a, inverse=1)[0].wait()
                return b

            return fft2, ifft2

    BACKENDS['CL'] = GPU_BACKEND_CL
except Exception:
    pass


try:
    class GPU_BACKEND_CU():
        import cupy as cp

        def get(self, arr_in):
            arr_out = self.cp.asnumpy(arr_in)
            return arr_out

        def zeros(self, shape, dtype):
            arr_out = self.cp.zeros(shape, dtype)
            return arr_out

        def sqrt(self, arr_in):
            arr_out = self.cp.sqrt(arr_in)
            return arr_out

        def exp(self, arr_in):
            arr_out = self.cp.exp(arr_in)
            return arr_out

        def abs(self, arr_in):
            arr_out = self.cp.abs(arr_in)
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

        def make_fft(self, vec_in, vec_out):
            def fft2(a, b):
                b = self.cp.fft.fft2(a, norm="ortho")
                return b

            def ifft2(a, b):
                b = self.cp.fft.ifft2(a, norm="ortho")
                return b

            return fft2, ifft2
    BACKENDS['CU'] = GPU_BACKEND_CU
except Exception:
    pass

try:
    class GPU_BACKEND_AF():
        import arrayfire as af

        def get(self, arr_in):
            arr_out = self.af.asnumpy(arr_in)
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

        def make_fft(self, vec_in, vec_out):
            def fft2(a, b):
                b = self.af.signal.fft2(a)
                return b

            def ifft2(a, b):
                b = self.af.signal.ifft2(a)
                return b

            return fft2, ifft2
    BACKENDS['AF'] = GPU_BACKEND_AF
except Exception:
    pass
