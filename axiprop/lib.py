# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop main file

This file contains main propagators of axiprop:
- PropagatorSymmetric
- PropagatorResampling
- PropagatorFFT2
- PropagatorResamplingFresnel
- PropagatorFFT2Fresnel
"""
import numpy as np
from scipy.special import jn
import warnings

from .common import CommonTools
from .steppers import StepperFresnel
from .steppers import StepperNonParaxial


class PropagatorSymmetric(CommonTools, StepperNonParaxial):
    """
    Class for the propagator with the Quasi-Discrete Hankel transform (QDHT)
    described in [M. Guizar-Sicairos, J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)].

    Contains methods to:
    - setup QDHT for TST;
    - perform a forward QDHT;
    - perform a inverse QDHT;

    This propagator uses same matrix for the forward and inverse transforms.
    The inverse transform can be truncated to a smaller radial size (same grid).
    """

    def __init__(self, r_axis, kz_axis, r_axis_new=None,
                 mode=0, dtype=np.complex128,
                 backend=None, verbose=True):
        """
        Construct the propagator.

        Parameters
        ----------
        r_axis: tuple (Rmax, Nr)
          Here:
            Rmax: float (m)
                Radial size of the calculation domain.

            Nr: int
                Number of nodes of the radial grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        r_axis_new: a tuple (Nr_new,) where Nr_new is int (optional)
            New number of nodes of the trancated radial grid. If not defined
            `Nr` will be used.

        mode: integer
            Order of Bessel function used for DHT

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype
        self.mode = mode

        self.init_backend(backend, verbose)
        self.init_kz(kz_axis)
        self.r, self.Rmax, self.Nr = self.init_r_symmetric(r_axis)
        self.init_kr(self.Rmax, self.Nr)

        # Setup a truncated output grid if needed
        if r_axis_new is None:
            self.Nr_new = self.Nr
            self.r_new = self.r
            self.Rmax_new = self.Rmax
        else:
            Nr_new = r_axis_new[0]
            if Nr_new>=self.Nr:
                self.Nr_new = self.Nr
                self.r_new = self.r
                self.Rmax_new = self.Rmax
            else:
                self.Nr_new = Nr_new
                self.r_new = self.r[:Nr_new]
                self.Rmax_new = self.r_new.max() * self.alpha[Nr_new] \
                                / self.alpha[Nr_new-1]

        self.init_TST()

    def init_TST(self):
        """
        Setup QDHT transformation matrix and data buffers.
        """
        Rmax = self.Rmax
        Nr = self.Nr
        Nr_new = self.Nr_new
        dtype = self.dtype
        mode = self.mode
        alpha = self.alpha
        alpha_np1 = self.alpha_np1

        self._j = self.bcknd.to_device( np.abs(jn(mode+1, alpha)) / Rmax )
        denominator = alpha_np1 * np.abs(jn(mode+1, alpha[:,None]) \
                                       * jn(mode+1, alpha[None,:]))

        self.TM = 2 * jn(mode, alpha[:,None] * alpha[None,:] / alpha_np1)\
                     / denominator
        self.TM = self.bcknd.to_device(self.TM, dtype)

        self.shape_trns = (Nr, )
        self.shape_trns_new = (Nr_new, )

        self.u_loc = self.bcknd.zeros(Nr, dtype)
        self.u_ht = self.bcknd.zeros(Nr, dtype)
        self.u_iht = self.bcknd.zeros(Nr_new, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)
        self.iTST_matmul = self.bcknd.make_matmul(self.TM[:Nr_new],
                                           self.u_ht, self.u_iht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_loc /= self._j
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)

    def iTST(self):
        """
        Inverse QDHT transform.
        """
        self.u_iht = self.iTST_matmul(self.TM[:self.Nr_new],
                                      self.u_ht, self.u_iht)
        self.u_iht *= self._j[:self.Nr_new]


class PropagatorResampling(CommonTools, StepperNonParaxial):
    """
    Class for the propagator with the non-symmetric Discrete Hankel transform
    (DHT) and possible different sampling for the input and output radial grids.

    Contains methods to:
    - setup DHT/iDHT transforms for TST;
    - perform a forward DHT;
    - perform a inverse iDHT;

    This propagator creates DHT matrix using numeric inversion of the inverse iDHT.
    This method samples output field on an arbitrary uniform radial grid.
    """

    def __init__(self, r_axis, kz_axis,
                 r_axis_new=None, mode=0,
                 dtype=np.complex128,
                 backend=None, verbose=True):
        """
        Construct the propagator.

        Parameters
        ----------
        r_axis: multiple cases
            tuple (Rmax, Nr)
              Rmax: float (m)
                Radial size of the calculation domain.
              Nr: int
                Number of nodes of the radial grid.

            ndarray (m)
                Radial grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        r_axis_new: multiple cases
            tuple (Rmax_new, Nr_new)
              Rmax_new: float (m)
                New radial size of the calculation domain.
              Nr_new: int
                New number of nodes of the radial grid.

            ndarray (m)
                New radial grid.

            None (default)
                No resampling

        mode: integer
            Order of Bessel function used for DHT

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype
        self.mode = mode

        self.init_backend(backend, verbose)
        self.init_kz(kz_axis)

        if type(r_axis) is tuple:
            self.r, self.Rmax, self.Nr = self.init_r_symmetric(r_axis)
        else:
            self.r, self.Rmax, self.Nr = self.init_r_sampled(r_axis)

        if r_axis_new is None:
            self.r_new, self.Rmax_new, self.Nr_new = self.r, self.Rmax, self.Nr
        elif type(r_axis_new) is tuple:
            self.r_new, self.Rmax_new, self.Nr_new = self.init_r_uniform(r_axis_new)
        else:
            self.r_new, self.Rmax_new, self.Nr_new = self.init_r_sampled(r_axis_new)

        self.init_kr(self.Rmax, self.Nr)
        self.init_TST()

    def init_TST(self):
        """
        Setup DHT transform and data buffers.

        Parameters
        ----------
        Rmax_new: float (m) (optional)
            New radial size for the output calculation domain. If not defined
            `Rmax` will be used.

        Nr_new: int
            New number of nodes of the radial grid. If is `None`, `Nr` will
            be used.
        """
        Nr = self.Nr
        Nr_new = self.Nr_new
        r = self.r
        r_new = self.r_new
        kr = self.kr
        dtype = self.dtype
        mode = self.mode

        self.TM = jn(mode, r[:,None] * kr[None,:])
        if mode == 0:
            self.TM = self.bcknd.inv_sqr_on_host(self.TM, dtype)
        else:
            self.TM = self.bcknd.inv_on_host(self.TM, dtype)

        self.TM = self.bcknd.to_device(self.TM)

        self.invTM = self.bcknd.to_device(\
            jn(mode, r_new[:,None] * kr[None,:]) , dtype)

        self.shape_trns = (Nr, )
        self.shape_trns_new = (Nr_new, )

        self.u_loc = self.bcknd.zeros(Nr, dtype)
        self.u_ht = self.bcknd.zeros(Nr, dtype)
        self.u_iht = self.bcknd.zeros(Nr_new, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)
        self.iTST_matmul = self.bcknd.make_matmul(self.invTM, self.u_ht, self.u_iht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)

    def iTST(self):
        """
        Inverse QDHT transform.
        """
        self.u_iht = self.iTST_matmul(self.invTM, self.u_ht, self.u_iht)


class PropagatorFFT2(CommonTools, StepperNonParaxial):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST.

    Contains methods to:
    - setup TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;
    """

    def __init__(self, x_axis, y_axis, kz_axis,
                 dtype=np.complex128, backend=None,
                 verbose=True):
        """
        Construct the propagator.

        Parameters
        ----------
        x_axis: tuple (Lx, Nx)
          Define the x-axis grid with parameters:
            Lx: float (m)
                Full size of the calculation domain along x-axis.

            Nx: int
                Number of nodes of the x-grid. Better be an odd number,
                in order to make a symmteric grid.

        y_axis: tuple (Ly, Ny)
          Define the y-axis grid with parameters:
            Ly: float (m)
                Full size of the calculation domain along y-axis.

            Ny: int
                Number of nodes of the y-grid.Better be an odd number,
                in order to make a symmteric grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype

        self.init_backend(backend, verbose)
        self.init_kz(kz_axis)
        self.x, self.y, self.r, self.r2 = self.init_xy_uniform(x_axis, y_axis)
        self.init_kxy_uniform(self.x, self.y)
        self.init_TST()

    def init_TST(self):
        """
        Setup data buffers for TST.
        """
        Nx = self.x.size
        Ny = self.y.size
        dtype = self.dtype

        self.shape_trns = (Nx, Ny)
        self.shape_trns_new = (Nx, Ny)

        self.u_loc = self.bcknd.zeros((Nx, Ny), dtype)
        self.u_ht = self.bcknd.zeros((Nx, Ny), dtype)
        self.u_iht = self.bcknd.zeros((Nx, Ny), dtype)

        self.fft2, self.ifft2, fftshift = self.bcknd.make_fft2(self.u_iht, self.u_ht)

    def TST(self):
        """
        Forward FFT transform.
        """
        self.u_ht = self.fft2(self.u_loc, self.u_ht)

    def iTST(self):
        """
        Inverse FFT transform.
        """
        self.u_iht = self.ifft2(self.u_ht, self.u_iht)


class PropagatorResamplingFresnel(CommonTools, StepperFresnel):
    def __init__(self, r_axis, kz_axis,
                 r_axis_new=None, Nkr_new=None,
                 N_pad=4, mode=0, dtype=np.complex128,
                 backend=None, verbose=True):
        """
        The resampling RT propagator.

        Parameters
        ----------
        r_axis: tuple (Rmax, Nr)
          Here:
            Rmax: float (m)
                Radial size of the calculation domain.

            Nr: int
                Number of nodes of the radial grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype
        self.mode = mode
        self.r_axis_new = r_axis_new
        self.gather_on_new_grid = self.gather_on_r_new

        self.init_backend(backend, verbose)
        self.init_kz(kz_axis)

        if type(r_axis) is tuple:
            Rmax, Nr = r_axis
            self.Nr = Nr
            r_axis_ext = ( N_pad * Rmax, N_pad * Nr )
            self.r_ext, self.Rmax_ext, self.Nr_ext = \
                            self.init_r_uniform(r_axis_ext)

            self.r = self.r_ext[:Nr]
            dr_est = (self.r[1:] - self.r[:-1]).mean()
            Rmax = self.r.max()
            self.Rmax = Rmax + 0.5 * dr_est
        else:
            self.r = r_axis.copy()
            Nr = self.r.size
            self.Nr = Nr
            dr_est = (self.r[1:] - self.r[:-1]).mean()
            Rmax = self.r.max()
            self.Rmax = Rmax + 0.5 * dr_est

            self.r_ext = np.r_[ self.r,
                Rmax + dr_est * np.arange( 1, Nr*(N_pad-1))
            ]
            self.Rmax_ext = self.r_ext.max() + 0.5 * dr_est
            self.Nr_ext = self.r_ext.size

        if Nkr_new is None:
            self.Nkr_new = self.Nr_ext
        else:
            self.Nkr_new = Nkr_new
            if Nkr_new > Nr * N_pad:
                warnings.warn(f"Nkr_new>Nr*N_pad={Nr*N_pad} has no effect")

        if r_axis_new is None:
            self.Nr_new = self.Nkr_new
        elif type(r_axis_new) is tuple:
            self.r_new, self.Rmax_new, self.Nr_new = \
                self.init_r_uniform(r_axis_new)
        else:
            self.r_new, self.Rmax_new, self.Nr_new = \
                self.init_r_sampled(r_axis_new)

        self.r2 = self.bcknd.to_device(self.r**2)
        self.init_kr(self.Rmax_ext, self.Nr_ext)
        self.init_TST()

    def init_TST(self):
        """
        Setup DHT transform and data buffers.

        Parameters
        ----------
        """
        mode = self.mode
        dtype = self.dtype
        r_ext = self.r_ext

        Rmax_ext = self.Rmax_ext
        Nr = self.Nr

        Nr_new = self.Nr_new
        Nkr_new = self.Nkr_new
        alpha = self.alpha
        kr = self.kr

        _norm_coef = 2.0 /  ( Rmax_ext * jn(mode+1, alpha[:Nkr_new]) )**2
        self.TM = jn(mode, r_ext[:, None] * kr[None,:Nkr_new]) * _norm_coef[None,:]
        self.TM = self.bcknd.inv_on_host(self.TM, dtype)
        self.TM = self.TM[:,:Nr]

        self.TM = self.bcknd.to_device(self.TM)

        self.shape_trns = (Nr, )
        self.shape_trns_new = (Nr_new, )

        self.u_loc = self.bcknd.zeros(Nr, dtype)
        self.u_ht = self.bcknd.zeros(Nkr_new, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)
        self.u_ht *= 2 * np.pi

    def get_local_grid(self, dz, ikz):
        r_loc = dz * self.kr[:self.Nkr_new] / self.kz[ikz]
        r2_loc = r_loc * r_loc
        return r_loc,  r2_loc

    def check_new_grid(self, dz):
        if self.r_axis_new is None:
            self.r_new =  dz * self.kr[:self.Nr_new] / self.kz.max()

        r_loc_min = dz * self.kr[:self.Nkr_new] / self.kz.max()
        if self.r_new.max()>r_loc_min.max():
            Nkr = int(self.r_new.max() / np.diff(r_loc_min).mean())
            warnings.warn(
                "Extrapolation will be used and may cause noise. "
                + f"In order to avoid this, define Nkr_new>{Nkr+1}.")


class PropagatorFFT2Fresnel(CommonTools, StepperFresnel):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST.

    Contains methods to:
    - setup TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;
    """

    def __init__(self, x_axis, y_axis, kz_axis,
                 Nx_new=None, Ny_new=None,
                 N_pad=2, dtype=np.complex128,
                 backend=None, verbose=True):
        """
        Construct the propagator.

        Parameters
        ----------
        x_axis: tuple (Lx, Nx)
          Define the x-axis grid with parameters:
            Lx: float (m)
                Full size of the calculation domain along x-axis.

            Nx: int
                Number of nodes of the x-grid. Better be an odd number,
                in order to make a symmteric grid.

        y_axis: tuple (Ly, Ny)
          Define the y-axis grid with parameters:
            Ly: float (m)
                Full size of the calculation domain along y-axis.

            Ny: int
                Number of nodes of the y-grid.Better be an odd number,
                in order to make a symmteric grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype

        self.init_backend(backend, verbose)
        self.init_kz(kz_axis)
        self.gather_on_new_grid = self.gather_on_xy_new

        Lx, Nx = x_axis
        Ly, Ny = y_axis

        Lx_ext = N_pad * Lx
        Ly_ext = N_pad * Ly
        Nx_ext = N_pad * Nx
        Ny_ext = N_pad * Ny

        self.Nx = Nx
        self.Ny = Ny
        self.Nx_ext = Nx_ext
        self.Ny_ext = Ny_ext
        self.dV = Lx/Nx * Ly/Ny

        if Nx_new is None:
            self.Nx_new = Nx_ext
        else:
            if Nx_new <= Nx_ext:
                self.Nx_new = Nx_new
            else:
                warnings.warn("Nx_new>Nx*N_pad will be reduced")
                self.Nx_new = Nx_ext

        if Ny_new is None:
            self.Ny_new = Ny_ext
        else:
            if Ny_new <= Ny_ext:
                self.Ny_new = Ny_new
            else:
                warnings.warn("Ny_new>Ny*N_pad will be reduced")
                self.Ny_new = Ny_ext

        self.ix0 = int( np.round( (N_pad - 1) * Nx / 2. ) )
        self.iy0 = int( np.round( (N_pad - 1) * Ny / 2. ) )

        self.x0, self.y0, self.r, self.r2 = self.init_xy_uniform(x_axis, y_axis)
        self.r2 = self.bcknd.to_device(self.r2)

        x, y, r, r2 = self.init_xy_uniform( (Lx_ext, Nx_ext),
                                            (Ly_ext, Ny_ext) )
        self.init_kxy_uniform(x, y, shift=True)
        self.init_TST()

    def init_TST(self):
        """
        Setup data buffers for TST.
        """
        Nx = self.Nx
        Ny = self.Ny
        Nx_ext = self.Nx_ext
        Ny_ext = self.Ny_ext
        Nx_new = self.Nx_new
        Ny_new = self.Ny_new
        dtype = self.dtype

        self.shape_trns_new = (Nx_new, Ny_new)
        self.u_loc = self.bcknd.zeros((Nx, Ny), dtype)

        self.u_iht = self.bcknd.zeros((Nx_ext, Ny_ext), dtype)
        self.u_ht = self.bcknd.zeros((Nx_ext, Ny_ext), dtype)

        self.fft2, ifft2, self.fftshift = self.bcknd.make_fft2(
                                                self.u_iht, self.u_ht)

    def TST(self):
        """
        Forward FFT transform.
        """
        self.u_iht[:] = 0.0
        ix0, iy0 = self.ix0, self.iy0
        Nx, Ny = self.Nx, self.Ny

        for ix_loc in range(Nx):
            ix_glob = ix0 + ix_loc
            self.u_iht[ix_glob, iy0 : iy0 + Ny] = self.u_loc[ix_loc,:]

        self.u_ht = self.fft2(self.u_iht, self.u_ht)
        self.u_ht = self.fftshift(self.u_ht)
        self.u_ht *= self.dV

    def get_local_grid(self, dz, ikz):
        kz_loc = self.kz[ikz]
        x_loc = dz * self.kx / kz_loc
        y_loc = dz * self.ky / kz_loc
        r_loc = (x_loc, y_loc)
        r2_loc = x_loc[:, None]**2 + y_loc[None, :]**2
        return r_loc, r2_loc

    def check_new_grid(self, dz):
        Nx_ext = self.Nx_ext
        Ny_ext = self.Ny_ext
        Nx_new = self.Nx_new
        Ny_new = self.Ny_new

        ix0 = (Nx_ext - Nx_new) // 2
        iy0 = (Ny_ext - Ny_new) // 2
        kz_max = self.kz.max()
        self.x = dz * self.kx[ix0 : ix0 + Nx_new] / kz_max
        self.y = dz * self.ky[iy0 : iy0 + Ny_new] / kz_max
        self.r_new = (self.x, self.y)
