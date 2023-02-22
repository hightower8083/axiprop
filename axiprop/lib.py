# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop main file

This file contains main user classes of axiprop:
- PropagatorSymmetric
- PropagatorResampling
- PropagatorFFT2
"""
import numpy as np
from scipy.special import jn
from scipy.interpolate import interp1d

from .base_classes import PropagatorCommon
from .base_classes import PropagatorFresnel


class PropagatorSymmetric(PropagatorCommon):
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
                self.Nr_new = sel.Nr
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


class PropagatorResampling(PropagatorCommon):
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


class PropagatorFFT2(PropagatorCommon):
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
        self.init_xykxy_fft2(x_axis, y_axis)
        self.init_TST()

    def init_TST(self):
        """
        Setup data buffers for TST.
        """
        Nr = self.Nr
        Nx = self.Nx
        Ny = self.Ny
        self.Nr_new = Nr

        dtype = self.dtype

        self.shape_trns = (Nx, Ny)
        self.shape_trns_new = (Nx, Ny)

        self.u_loc = self.bcknd.zeros((Nx, Ny), dtype)
        self.u_ht = self.bcknd.zeros((Nx, Ny), dtype)
        self.u_iht = self.bcknd.zeros((Nx, Ny), dtype)

        self.fft2, self.ifft2 = self.bcknd.make_fft2(self.u_loc, self.u_ht, self.u_iht)

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


class PropagatorResamplingFresnel(PropagatorFresnel):
    def __init__(self, r_axis, kz_axis,
                 r_axis_new=None, Nkr_new=None,
                 N_pad=4, mode=0, dtype=np.complex128,
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

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype
        self.mode = mode
        self.r_axis_new = r_axis_new

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

            self.r_ext = np.r_[ self.r, Rmax + dr_est * np.arange(1, Nr*(N_pad-1))+1 ]
            self.Rmax_ext = self.r_ext.max() + 0.5 * dr_est
            self.Nr_ext = self.r_ext.size

        if r_axis_new is None:
            self.Nr_new = Nr
        elif type(r_axis_new) is tuple:
            self.r_new, self.Rmax_new, self.Nr_new = \
                self.init_r_uniform(r_axis_new)
        else:
            self.r_new, self.Rmax_new, self.Nr_new = \
                self.init_r_sampled(r_axis_new)

        if Nkr_new is None:
            self.Nkr_new = self.Nr_new
        else:
            self.Nkr_new = Nkr_new

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

        Rmax = self.Rmax
        Rmax_ext = self.Rmax_ext
        Nr = self.Nr
        Nr_ext = self.Nr_ext

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

    def gather_on_r_new( self, u_loc, r_loc, r_new ):
        interp_fu_abs = interp1d(r_loc, np.abs(u_loc),
                                 fill_value='extrapolate',
                                 kind='cubic',
                                 bounds_error=False )
        u_slice_abs = interp_fu_abs(r_new)

        interp_fu_angl = interp1d(r_loc, np.unwrap(np.angle(u_loc)),
                                  fill_value='extrapolate',
                                  kind='cubic',
                                  bounds_error=False )
        u_slice_angl = interp_fu_angl(r_new)
        del interp_fu_abs, interp_fu_angl

        u_slice_new = u_slice_abs * np.exp( 1j * u_slice_angl )
        return u_slice_new


class PropagatorFresnelFFT(PropagatorFFT2, PropagatorFresnel):
    """
    def make_r_new(self, Rmax_new, Nr=None):
        self.Rmax_new = Rmax_new
        self.r_new = self.r
        self.r2_new = self.r_new**2

    def gather_on_r_new( self, u_loc, r_loc ):
        return u_loc
    """
    pass
