# Copyright 2023
# Authors: Igor A Andriyash
# License: BSD-3-Clause
"""
Axiprop main file

This file contains older propagators of axiprop:
- PropagatorFFT2Fresnel
"""
import numpy as np
from scipy.special import jn
import warnings

from .common import CommonTools
from .steppers import StepperFresnel
from .steppers import StepperNonParaxial

warnings.simplefilter("always")

class PropagatorResamplingFresnel(CommonTools, StepperFresnel):
    def __init__(self, r_axis, kz_axis,
                 r_axis_new=None, Nkr_new=None,
                 N_pad=1, mode=0, dtype=np.complex128,
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
            Nr_ext_loc = int( np.round( N_pad * Nr  ) )
            r_axis_ext = ( N_pad * Rmax, Nr_ext_loc )
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
            Nr_ext_add = int( np.round(  Nr*(N_pad-1) ) )

            self.r_ext = np.r_[ self.r,
                Rmax + dr_est * np.arange( 1, Nr_ext_add)
            ]
            self.Rmax_ext = self.r_ext.max() + 0.5 * dr_est
            self.Nr_ext = self.r_ext.size

        if Nkr_new is None:
            self.Nkr_new = self.Nr_ext
        else:
            self.Nkr_new = Nkr_new
            if Nkr_new > Nr * N_pad:
                warnings.warn(f"Nkr_new>Nr*N_pad={Nr*int(N_pad)} has no effect")

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
                 Nx_new=None, Ny_new=None,
                 N_pad=1, dtype=np.complex128,
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
        Nx_ext = int( np.round(N_pad * Nx ))
        Ny_ext = int( np.round( N_pad * Ny ))

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

        self.xmin_ext = x.min()
        self.ymin_ext = y.min()
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
