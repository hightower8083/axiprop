# Copyright 2023
# Authors: Igor A Andriyash
# License: BSD-3-Clause
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

warnings.simplefilter("always")

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
                 r_axes_types=('bessel', 'uniform'),
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

        r_axes_types: tuple of strings (optional)
            Sampling methods for the axes when generated internally.
            Should be a tuple with two names for `r_axis` and `r_axis_new`
            respectively, that can be either `'bessel'` or `'uniform'`.

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
            if r_axes_types[0]=='bessel':
                self.r, self.Rmax, self.Nr = self.init_r_symmetric(r_axis)
            elif r_axes_types[0]=='uniform':
                self.r, self.Rmax, self.Nr = self.init_r_uniform(r_axis)
            else:
                raise NameError("`r_axes_types` can be either `'uniform'` or `'bessel'`")
        elif type(r_axis) is np.ndarray:
            self.r, self.Rmax, self.Nr = self.init_r_sampled(r_axis)
        else:
            raise TypeError("`r_axis` can be either tuple or 1d-ndarray")

        if r_axis_new is None:
            self.r_new, self.Rmax_new, self.Nr_new = self.r, self.Rmax, self.Nr
        elif type(r_axis_new) is tuple:
            if r_axes_types[1]=='bessel':
                self.r_new, self.Rmax_new, self.Nr_new = self.init_r_symmetric(r_axis_new)
            elif r_axes_types[1]=='uniform':
                self.r_new, self.Rmax_new, self.Nr_new = self.init_r_uniform(r_axis_new)
            else:
                raise NameError("`r_axes_types` can be either `'uniform'` or `'bessel'`")
        elif type(r_axis_new) is np.ndarray:
            self.r_new, self.Rmax_new, self.Nr_new = self.init_r_sampled(r_axis_new)
        else:
            raise TypeError("`r_axis_new` can be either tuple, 1d-ndarray or `None`")

        if self.Rmax_new<=self.Rmax:
            self.r_ext = self.r.copy()
            self.Nr_ext = self.Nr
            self.Rmax_ext = self.Rmax
        else:
            if self.verbose:
                print (
                    'Input r-grid is smaller, than the output one, and will be padded'
                )

            if type(r_axis) is tuple:
                if r_axes_types[0]=='bessel':
                    raise NotImplementedError(
                        "For diffracting cases use `'unifrom'` input sampling." \
                        + "(see `r_axes_types` argument)"
                    )
            else:
                if not self.check_uniform(self.r):
                    raise NotImplementedError(
                        "For diffracting cases `r_axis` must be uniform"
                    )
                dr = self.r[:2].ptp()
                self.r_ext = np.arange(self.r[0], self.Rmax_new, dr)
                self.Nr_ext = self.r_ext.size
                self.Rmax_ext = self.r_ext[-1] + 0.5 * dr

        self.init_kr(self.Rmax_ext, self.Nr_ext)

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
        r = self.r_ext # potentially use extended grid
        r_new = self.r_new
        kr = self.kr
        dtype = self.dtype
        mode = self.mode

        self.TM = jn(mode, r[:,None] * kr[None,:])
        if mode == 0:
            self.TM = self.bcknd.inv_sqr_on_host(self.TM, dtype)
        else:
            self.TM = self.bcknd.inv_on_host(self.TM, dtype)

        self.TM = self.TM[:, :Nr]

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
        if type(x_axis) and type(y_axis) in (list, tuple):
            self.x, self.y, self.r, self.r2 = self.init_xy_uniform(
                                                x_axis, y_axis)
        elif type(x_axis) and type(y_axis) is np.ndarray:
            self.x, self.y, self.r, self.r2 = self.init_xy_sampled(
                                                x_axis, y_axis)

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
    def __init__(self, r_axis, kz_axis, dz,
                 r_axis_new=None, mode=0,
                 dtype=np.complex128,
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

        kz_max = self.kz.max()
        kz_min = self.kz.min()
        kz_mean = self.kz.mean()
        if kz_min<0:
            print('temporal resolution is too high, some data will be lost')

        if type(r_axis) is tuple:
            Rmax, Nr = r_axis
            self.r, self.Rmax, self.Nr = self.init_r_uniform(r_axis)
            dr_est = (self.r[1:] - self.r[:-1]).mean()
        else:
            self.r = r_axis.copy()
            self.Nr = self.r.size
            dr_est = (self.r[1:] - self.r[:-1]).mean()
            self.Rmax = self.r.max() + 0.5 * dr_est

        if type(r_axis_new) is tuple:
            Rmax_new, Nr_new = r_axis_new
            self.r_new, self.Rmax_new, self.Nr_new = self.init_r_uniform(r_axis_new)
            dr_new_est = (self.r_new[1:] - self.r_new[:-1]).mean()
        else:
            self.r_new = r_axis_new.copy()
            self.Nr_new = self.r_new.size
            dr_new_est = (self.r_new[1:] - self.r_new[:-1]).mean()
            self.Rmax_new = self.r_new.max() + 0.5 * dr_new_est

        Rmax_resolved = np.pi * dz / kz_max / dr_est
        if Rmax_resolved < self.Rmax_new:
            print('Requested x-axis is too large')

        self.Nr_ext = int(np.ceil( np.pi * dz / kz_max / dr_est / dr_new_est  ))
        self.r_ext = np.zeros(self.Nr_ext)
        self.r_ext[:self.Nr] = self.r[:]
        self.r_ext[self.Nr:] = self.r[-1] \
            + dr_est * np.arange(1, self.Nr_ext - self.Nr + 1)
        self.Rmax_ext = self.r_ext.max() + 0.5 * dr_est

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
        Nkr_new = Nr_new

        self.Nkr_new = Nkr_new

        alpha = self.alpha
        kr = self.kr

        if mode==0:
            _norm_coef = 2.0 /  (
                Rmax_ext * jn(mode+1, alpha[:Nkr_new]) )**2
        else:
            _norm_coef = np.zeros_like(alpha[:Nkr_new])
            _norm_coef[1:] = 2.0 /  (
                Rmax_ext * jn(mode+1, alpha[1:Nkr_new]) )**2

        self.TM = jn(mode, r_ext[:, None] * kr[None,:Nkr_new]) \
            * _norm_coef[None,:]
        self.TM = self.bcknd.inv_on_host(self.TM, dtype)
        self.TM = self.TM[:,:Nr]
        self.TM *= 2 * np.pi * (-1j)**mode

        self.TM = self.bcknd.to_device(self.TM)

        self.shape_trns = (Nr,)
        self.shape_trns_new = (Nr_new,)

        self.u_loc = self.bcknd.zeros(Nr, dtype)
        self.u_ht = self.bcknd.zeros(Nkr_new, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)

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
            kz_max = self.kz[
                self.kz > dz * self.kr[self.Nkr_new-1] / self.r_new.max()
            ][0]

            lambda_min = 2 * np.pi / kz_max

            warnings.warn(
                "New radius is not fully resolved, so some data for the "
                + f"wavelengths below {lambda_min*1e9:g} nm may be lost. "
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
                 dz, x_axis_new=None, y_axis_new=None,
                 dtype=np.complex128,
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

        kz_max = self.kz.max()
        kz_min = self.kz.min()
        kz_mean = self.kz.mean()

        if type(x_axis) is tuple and type(y_axis):
            self.x0, self.y0, self.r, self.r2 = self.init_xy_uniform(x_axis, y_axis)
        else:
            self.x0, self.y0, self.r, self.r2 = self.init_xy_sampled(x_axis, y_axis)

        self.r2 = self.bcknd.to_device(self.r2)

        Nx = self.x0.size
        Ny = self.y0.size
        Lx = np.ptp( self.x0 )
        Ly = np.ptp( self.y0 )
        dx0 = np.ptp(self.x0[[0,1]])
        dy0 = np.ptp(self.y0[[0,1]])
        x0_max = np.abs(self.x0).max()
        y0_max = np.abs(self.y0).max()

        if x_axis_new is None:
            self.x = 2 * np.pi * dz / kz_mean / dx0 / Nx * (np.arange(Nx) - Nx//2)

        if y_axis_new is None:
            self.y = 2 * np.pi * dz / kz_mean / dy0 / Ny * (np.arange(Ny) - Ny//2)

        if type(x_axis_new) is tuple and type(y_axis_new):
            self.x, self.y, r_new, r2_new = self.init_xy_uniform(x_axis_new, y_axis_new)
        elif type(x_axis_new) is np.ndarray and type(y_axis_new) is np.ndarray:
            self.x, self.y, r_new, r2_new = self.init_xy_sampled(x_axis_new, y_axis_new)

        dx = np.ptp(self.x[[0,1]])
        dy = np.ptp(self.y[[0,1]])
        x_max = np.abs(self.x).max()
        y_max = np.abs(self.y).max()

        if kz_min<0:
            print('temporal resolution is too high, some data will be lost')

        if np.mod(Nx, 2)==0:
            fftfreq_coef_x = 1. - 1.0 / Nx
        else:
            fftfreq_coef_x = 1. - 2.0 / Nx

        if np.mod(Ny, 2)==0:
            fftfreq_coef_y = 1. - 1.0 / Ny
        else:
            fftfreq_coef_y = 1. - 2.0 / Ny

        x_max_resolved = np.pi * fftfreq_coef_x * dz / dx0 / kz_max
        y_max_resolved = np.pi * fftfreq_coef_y * dz / dy0 / kz_max

        if x_max_resolved < x_max:
            print('Requested x-axis is too large')
        if y_max_resolved < y_max:
            print('Requested y-axis is too large')

        Nx_ext = int(np.ceil(2 * np.pi * dz / dx0 / dx / kz_mean)) # to consider `kz_min`
        Ny_ext = int(np.ceil(2 * np.pi * dz / dy0 / dy / kz_mean)) # to consider `kz_min`

        if Nx_ext < Nx:
            Nx_ext = Nx
            self.ix0 = 0
        else:
            self.ix0 = (Nx_ext - Nx) // 2

        if Ny_ext < Ny:
            Ny_ext = Ny
            self.iy0 = 0
        else:
            self.iy0 = (Ny_ext - Ny) // 2

        self.Nx = Nx
        self.Ny = Ny
        self.Nx_ext = Nx_ext
        self.Ny_ext = Ny_ext

        self.dV = dx0 * dy0

        self.Nx_new = self.x.size
        self.Ny_new = self.y.size

        self.x0_ext = np.zeros(Nx_ext)
        ix1 = self.ix0 + Nx
        self.x0_ext[self.ix0 : ix1] = self.x0[:]

        self.x0_ext[ix1:] = self.x0_ext[ix1-1] \
            + dx0 * np.arange(1, Nx_ext - ix1 + 1)

        self.x0_ext[:self.ix0] = self.x0_ext[self.ix0] \
            - dx0 * np.arange(1, self.ix0 + 1)[::-1]

        self.y0_ext = np.zeros(Ny_ext)
        iy1 = self.iy0 + Ny
        self.y0_ext[self.iy0 : iy1] = self.y0[:]

        self.y0_ext[iy1:] = self.y0_ext[iy1-1] \
            + dy0 * np.arange(1, Ny_ext - iy1 + 1)

        self.y0_ext[:self.iy0] = self.y0_ext[self.iy0] \
            - dy0 * np.arange(1, self.iy0 + 1)[::-1]

        self.xmin_ext = self.x0_ext.min()
        self.ymin_ext = self.y0_ext.min()
        self.init_kxy_uniform(self.x0_ext, self.y0_ext, shift=True)
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

        self.xy_init_phase = np.exp(
            -1j * self.kx[:,None] * self.xmin_ext \
            -1j * self.ky[None,:] * self.ymin_ext
        )
        self.xy_init_phase = self.bcknd.to_device(self.xy_init_phase)

    def TST(self):
        """
        Forward FFT transform.
        """
        self.u_iht[:] = 0.0
        ix0, iy0 = self.ix0, self.iy0
        Nx, Ny = self.Nx, self.Ny
        self.u_iht[ix0 : ix0 + Nx, iy0 : iy0 + Ny] = self.u_loc.copy()

        self.u_ht = self.fft2(self.u_iht, self.u_ht)
        self.u_ht = self.fftshift(self.u_ht)
        self.u_ht *= self.dV
        self.u_ht *= self.xy_init_phase

    def get_local_grid(self, dz, ikz):
        kz_loc = self.kz[ikz]
        x_loc = dz * self.kx / kz_loc
        y_loc = dz * self.ky / kz_loc
        r_loc = (x_loc, y_loc)
        r2_loc = x_loc[:, None]**2 + y_loc[None, :]**2
        return r_loc, r2_loc

    def check_new_grid(self, dz):
        self.r_new = (self.x, self.y)
