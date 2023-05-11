# Copyright 2023
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop common.py file

This file contains common classed of axiprop:
- CommonTools
- PropagatorExtras
"""
import numpy as np
from scipy.special import jn, jn_zeros
from scipy.interpolate import interp1d, RectBivariateSpline
import os
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import trapezoid

try:
    from unwrap import unwrap as unwrap2d
    unwrap_available = True
except Exception:
    unwrap_available = False

from .backends import AVAILABLE_BACKENDS, backend_strings_ordered


class CommonTools:
    """
    Base class for propagators. Contains methods to:
    - initialize the backend;
    - setup spectral `kz` grid;
    - setup radial `r` and spectral `kr` grids;
    - setup transverse `x`-`y`, and spectral `kx`-`ky` grids;
    - etc

    This class should to be used to derive the actual Propagators
    by adding proper steppers and methods for the Transverse Spectral
    Transforms (TST).
    """

    def init_backend(self, backend, verbose):

        if verbose:
            print('Available backends are: ' \
                + ', '.join(AVAILABLE_BACKENDS.keys()))

        if backend is not None:
            backend_string = backend
        elif 'AXIPROP_BACKEND' in os.environ:
            backend_string = os.environ['AXIPROP_BACKEND']
        else:
            for bknd_str in backend_strings_ordered:
                if bknd_str in AVAILABLE_BACKENDS:
                    backend_string = bknd_str
                    break

        if backend_string not in AVAILABLE_BACKENDS:
            raise Exception(f'Backend {backend_string} is not available')

        self.bcknd = AVAILABLE_BACKENDS[backend_string]()
        if verbose:
            print(f'{self.bcknd.name} is chosen')

    def init_kz(self, kz_axis):
        """
        Setup `kz` spectral grid.

        Parameters
        ----------
        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.
        """
        if type(kz_axis) is tuple:
            k0, Lkz, Nkz = kz_axis
            Nkz_2 = int(np.ceil(Nkz/2))
            half_ax = np.linspace(0, 1., Nkz_2)
            full_ax = np.r_[-half_ax[1:][::-1], half_ax]
            self.Nkz = full_ax.size
            self.kz = k0 + Lkz / 2 * full_ax
        else:
            self.kz = kz_axis.copy()
            self.Nkz = self.kz.size

    def init_kr(self, Rmax, Nr):
        """
        Setup spectral `kr` grid and related data
        """
        mode = self.mode

        if mode !=0:
            alpha = np.r_[0., jn_zeros(mode, Nr)]
        else:
            alpha = jn_zeros(mode, Nr+1)

        self.alpha_np1 = alpha[-1]
        self.alpha = alpha[:-1]
        self.kr = self.alpha / Rmax
        self.kr2 = self.bcknd.to_device( self.kr**2 )

    def init_kxy_uniform(self, x, y, shift=False):
        dx = x[1] - x[0]
        dy = y[1] - y[0]
        Nx = x.size
        Ny = y.size

        kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

        if shift:
            kx = np.fft.fftshift(kx)
            ky = np.fft.fftshift(ky)

        self.kx = kx
        self.ky = ky

        self.kr2 = kx[:,None]**2 + ky[None,:]**2
        self.kr = np.sqrt(self.kr2)
        self.kr2 = self.bcknd.to_device( self.kr2 )

    def init_r_sampled(self, r_axis):
        """
        Setup the radial `r` grid from an array

        Parameters
        ----------
        r_axis: float ndarray (m)
        """
        r = r_axis.copy()
        Nr = r.size
        dr_est = (r[1:] - r[:-1]).mean()
        Rmax = r.max() + dr_est/2
        return r, Rmax, Nr

    def init_r_symmetric(self, r_axis):
        """
        Setup radial `r` grid on jn-roots

        Parameters
        ----------
        r_axis: tuple (Rmax, Nr)
          Here:
            Rmax: float (m)
                Radial size of the calculation domain.

            Nr: int
                Number of nodes of the radial grid.
        """
        Rmax, Nr = r_axis

        alpha = jn_zeros(self.mode, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]
        r = Rmax * alpha / alpha_np1
        return r, Rmax, Nr

    def init_r_uniform(self, r_axis):
        """
        Setup a uniform radial `r` grid

        Parameters
        ----------
        r_axis: tuple (Rmax, Nr)
          Here:
            Rmax: float (m)
                Radial size of the calculation domain.

            Nr: int
                Number of nodes of the radial grid.
        """
        Rmax, Nr = r_axis
        r = np.linspace(0, Rmax, Nr, endpoint=False)
        dr = r[[0,1]].ptp()
        r += 0.5 * dr
        return r, Rmax, Nr

    def init_xy_uniform(self, x_axis, y_axis):
        """
        Setup the transverse `x` and `y` grids

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
        """
        Lx, Nx = x_axis
        Ly, Ny = y_axis

        x = np.linspace(-Lx/2, Lx/2, Nx)
        y = np.linspace(-Ly/2, Ly/2, Ny)
        dx = x[1] - x[0]
        dy = y[1] - y[0]

        if Nx==1 and Ny>1:
            x = np.array( [0.0, ] )
            dx = dy

        if Ny==1 and Nx>1:
            y = np.array( [0.0, ] )
            dy = dx

        r2 = x[:,None]**2 + y[None,:]**2
        r = np.sqrt(r2)
        return x, y, r, r2

    def gather_on_r_new( self, u_loc, r_loc, r_new ):
        interp_fu_abs = interp1d(r_loc, np.abs(u_loc),
                                 fill_value='extrapolate',
                                 kind='cubic',
                                 bounds_error=False )

        interp_fu_angl = interp1d(r_loc, np.unwrap(np.angle(u_loc)),
                                  fill_value='extrapolate',
                                  kind='cubic',
                                  bounds_error=False )

        u_slice_abs = interp_fu_abs(r_new)
        u_slice_angl = interp_fu_angl(r_new)

        u_slice_new = u_slice_abs * np.exp( 1j * u_slice_angl )
        return u_slice_new

    def gather_on_xy_new( self, u_loc, r_loc, r_new ):

        x_loc, y_loc = r_loc
        x_new, y_new = r_new
        if np.alltrue(x_loc==x_new) and np.alltrue(y_loc==y_new):
            return u_loc

        if not unwrap_available:
            raise NotImplementedError("install unwrap")

        interp_fu_abs = RectBivariateSpline(
            x_loc, y_loc, np.abs(u_loc)
        )

        interp_fu_angl = RectBivariateSpline(
            x_loc, y_loc, unwrap2d(np.angle(u_loc))
        )

        u_slice_abs = interp_fu_abs(x_new, y_new)
        u_slice_angl = interp_fu_angl(x_new, y_new)

        u_slice_new = u_slice_abs * np.exp( 1j * u_slice_angl )
        return u_slice_new


class PropagatorExtras:
    """
    Some experimental or obsolete stuff
    """
    def apply_boundary(self, u, nr_boundary=16):
        # apply the boundary "absorbtion"
        absorb_layer_axis = np.r_[0 : np.pi/2 : nr_boundary*1j]
        absorb_layer_shape = np.cos(absorb_layer_axis)**0.5
        absorb_layer_shape[-1] = 0.0
        u[:, -nr_boundary:] *= absorb_layer_shape
        return u

    def get_Ez(self, ux):
        """
        Get a longitudinal field component from the transverse field using the
        Poisson equation in vacuum DIV.E = 0.
        Parameters
        ----------
        ux: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """

        uz = np.zeros_like(ux)
        kx_2d = self.kx[:,None] * np.ones_like(self.ky[None,:])
        kx_2d = self.bcknd.to_device(kx_2d)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(ux[ikz,:])
            self.TST()

            kz_loc = self.bcknd.sqrt(self.bcknd.abs( self.kz[ikz]**2 - \
                                                           self.kr2 ))
            self.u_ht *= - kx_2d / kz_loc
            self.iTST()
            uz[ikz] = self.bcknd.to_host(self.u_iht)

        return uz
