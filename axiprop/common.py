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
        if not unwrap_available:
            raise NotImplementedError("install unwrap")

        x_loc, y_loc = r_loc
        x_new, y_new = r_new

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

    def initiate_stepping(self, u):
        """
        Initiate the stepped propagation mode. This mode allows computation
        of the consequent steps with access to the result on each step.
        In contrast to `step` can operate the `PropagatorResampling` class.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """
        assert u.dtype == self.dtype

        self.stepping_image = self.bcknd.to_device( np.zeros_like(u) )
        self.phase_loc = self.bcknd.to_device( np.zeros_like(u) )
        self.z_propagation = 0.0

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])
            self.TST()

            self.stepping_image[ikz] = self.u_ht.copy()

            phase_loc = self.kz[ikz]**2 - self.kr2
            self.phase_loc[ikz] = self.bcknd.sqrt((phase_loc >= 0.)*phase_loc)

    def stepping(self, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This mode allows computation
        of the consequent steps with access to the result on each step.
        In contrast to `step` can operate the `PropagatorResampling` class.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = np.empty((self.Nkz, *self.shape_trns_new),
                              dtype=self.dtype)

        for ikz in range(self.Nkz):
            self.stepping_image[ikz] *= self.bcknd.exp( \
                1j * dz * self.phase_loc[ikz] )
            self.u_ht = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out

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


class ScalarField:
    def __init__(self, k0, bandwidth, t_range, dt, n_dump=4,
                 dtype_ft=np.complex128, dtype=np.double ):
        self.dtype = dtype
        self.dtype_ft = dtype_ft
        self.t_loc = 0
        self.k0 = k0
        self.omega0 = k0 * c
        self.t_range = t_range
        self.dt = dt
        self.cdt = c * dt

        self.t = np.arange(*t_range, dt)
        if np.mod(self.t.size, 2) == 1:
            self.t = self.t[:-1]
        self.Nt = self.t.size

        self.Nk_freq_full = self.Nt
        self.k_freq_full = 2 * np.pi * np.fft.fftfreq(self.Nt, c*dt)
        self.band_mask = np.abs(self.k_freq_full+k0)<bandwidth

        self.k_freq = np.abs(self.k_freq_full[self.band_mask])
        self.Nk_freq = self.k_freq.size
        self.omega = self.k_freq * c

        self.n_dump = n_dump
        self.dump_mask = np.cos(np.r_[0 : np.pi/2 : n_dump*1j])**0.5
        self.dump_mask[-1] = 0.0

    def make_gaussian_pulse(self, a0, tau, r, R_las, t0=0, phi0=0, n_ord=2, omega0=None):
        self.r_shape = r.shape
        t = self.t
        self.r = r.copy()

        if omega0 is None:
            omega0 = self.omega0

        profile_r = np.exp( -( r/R_las )**n_ord )
        profile_t = np.exp( -(t-t0)**2 / tau**2 ) * np.cos(omega0 * t + phi0)

        profile_r[-self.n_dump:] *= self.dump_mask
        profile_t[-self.n_dump:] *= self.dump_mask
        profile_t[:self.n_dump] *= self.dump_mask[::-1]

        self.E0 = a0 * m_e * c * omega0 / e
        self.Field = self.E0 * profile_t[:, None] * profile_r[None, :]

        self.Field_ft = np.zeros((self.Nk_freq, *self.r_shape), dtype=self.dtype_ft)
        self.time_to_frequency()
        self.Field_ft[-self.n_dump:] *= self.dump_mask[:, None]
        self.Field_ft[:self.n_dump] *= self.dump_mask[::-1][:, None]

    @property
    def Energy(self):
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        Energy = 2 * np.pi * epsilon_0 * trapezoid(
            trapezoid(self.Field**2 * self.r, self.r), c * self.t
        )
        return Energy

    def import_field(self, A, t_loc=None, r=None):
        if t_loc is not None:
            self.t_loc = t_loc

        if r is not None:
            self.r = r
        self.Field = A.copy()
        self.r_shape = A[0].shape
        self.Field_ft = np.zeros((self.Nk_freq, *self.r_shape), dtype=self.dtype_ft)
        self.time_to_frequency()
        self.Field_ft[-self.n_dump:] *= self.dump_mask[:, None]
        self.Field_ft[:self.n_dump] *= self.dump_mask[::-1][:, None]

    def import_field_ft(self, A, t_loc=0, r=None):
        self.t_loc = t_loc
        if r is not None:
            self.r = r
        self.r_shape = A[0].shape
        self.Field_ft = A.copy()
        self.Field_ft[-self.n_dump:] *= self.dump_mask[:, None]
        self.Field_ft[:self.n_dump] *= self.dump_mask[::-1][:, None]
        self.Field = np.zeros((self.Nt, *self.r_shape), dtype=self.dtype)
        self.frequency_to_time()

    def time_to_frequency(self):
        self.Field_ft[:] = np.fft.fft(self.Field, axis=0)[self.band_mask,:]
        self.Field_ft *= np.exp(1j * self.k_freq[:,None] * c * self.t_loc)

    def frequency_to_time(self):
        Field_ft = self.Field_ft * np.exp(-1j * self.k_freq[:,None] * c * self.t_loc)
        Field_ft_full = np.zeros((self.Nk_freq_full, *self.r_shape), dtype=self.dtype_ft)
        Field_ft_full[self.band_mask, :] = Field_ft
        self.Field[:] = 2 * np.fft.ifft(Field_ft_full, axis=0).real