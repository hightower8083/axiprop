# Copyright 2023
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop containers.py file

This file contains container classes for axiprop:
"""
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import trapezoid


def apply_boundary_r(A, dump_mask):
    """
    Attenuate the given field at the transverse boundaries

    Parameters
    ----------
    A: float ndarray
        Some field of the same dimensions as the main
        field of the container
    """
    n_dump = dump_mask.size
    if n_dump==0:
        return A

    if len(A[0].shape)==1:
        A[:,-n_dump:] *= dump_mask[None,:]
    elif len(A[0].shape)==2:
        A[:,:n_dump,:] = dump_mask[::-1][None,:,None]
        A[:,:,:n_dump] = dump_mask[::-1][None,None,:]
        A[:,-n_dump:,:] = dump_mask[None,:,None]
        A[:,:,-n_dump:] = dump_mask[None,None,:]
    return A

def apply_boundary_t(A, dump_mask):
    """
    Attenuate the given field at the transverse boundaries

    Parameters
    ----------
    A: float ndarray
        Some field of the same dimensions as the main
        field of the container
    """
    n_dump = dump_mask.size
    if n_dump==0:
        return A

    if len(A[0].shape)==1:
        A[:n_dump] *= dump_mask[::-1][:,None]
        A[-n_dump:] *= dump_mask[:,None]
    elif len(A[0].shape)==2:
        A[:n_dump] *= dump_mask[::-1][:,None,None]
        A[-n_dump:] *= dump_mask[:,None,None]
    return A


class ScalarField:
    """
    A class to initialize and transform the optical field between temporal
    and frequency domains.
    """
    def __init__(self, k0, bandwidth, t_range, dt, n_dump=0,
                 dtype_ft=np.complex128, dtype=np.double ):
        """
        Initialize the container for the field.

        Parameters
        ----------
        k0: float (m^-1)
          Central wavenumber for the spectral grid

        bandwidth: float (m^-1)
            Width of the spectral grid

        t_range: tuple of floats (t0, t1) (s)
            Range of the initial temporal domain

        dt: float (s)
            Time-step to resolve the temporal domain

        n_dump: int
            Number of cells to be used for attenuating boundaries
        """
        self.dtype = dtype
        self.dtype_ft = dtype_ft
        self.k0 = k0
        self.omega0 = k0 * c
        self.t_range = t_range
        self.dt = dt
        self.cdt = c * dt

        self.t = np.arange(*t_range, dt)
        if np.mod(self.t.size, 2) == 1:
            self.t = self.t[:-1]
        self.Nt = self.t.size
        self.t_loc = self.t[0]

        self.Nk_freq_full = self.Nt // 2 + 1
        self.k_freq_full = 2 * np.pi * np.fft.fftfreq( self.Nt, c*dt )
        self.k_freq_full = self.k_freq_full[:self.Nk_freq_full]

        self.band_mask = ( np.abs(self.k_freq_full - k0) < bandwidth )
        self.k_freq = self.k_freq_full[self.band_mask]
        self.Nk_freq = self.k_freq.size
        self.omega = self.k_freq * c

        self.n_dump = n_dump
        dump_r = np.linspace(0, 1, n_dump)
        self.dump_mask = ( 1 - np.exp(-(2*dump_r)**3) )[::-1]

    def make_gaussian_pulse( self, a0, tau, r, R_las, t0=0,
                             phi0=0, n_ord=2, omega0=None ):
        """
        Initialize the Gaussian pulse

        Parameters
        ----------
        a0: float
          Normalized amplitude of the pulse

        tau: float (s)
            Duration of the pulse (FBPIC definition)

        r: float ndarray (m)
            Radial grid for the container

        t0: float (s)
            Time corresponding the peak field

        phi0: float (rad)
            Carrier envelop phase (CEP)

        n_ord: int
            Order of Gaussian profile of TEM

        omega0: float (s^-1)
            central frequency of the pulse
        """
        self.r_shape = r.shape
        t = self.t
        self.r = r.copy()

        if omega0 is None:
            omega0 = self.omega0

        profile_r = np.exp( -( r/R_las )**n_ord )
        profile_t = np.exp( -(t-t0)**2 / tau**2 ) * np.sin(omega0 * t + phi0)

        if len(profile_r.shape) == 1:
            profile_t = profile_t[:, None]
            self.k_freq_shaped = self.k_freq[:, None]
        elif len(profile_r.shape) == 2:
            profile_t = profile_t[:, None, None]
            self.k_freq_shaped = self.k_freq[:, None, None]

        self.E0 = a0 * m_e * c * omega0 / e
        self.Field = self.E0 * profile_t * profile_r[None, :]

        self.Field = apply_boundary_t(self.Field, self.dump_mask)
        self.Field = apply_boundary_r(self.Field, self.dump_mask)

        self.Field_ft = np.zeros(
            (self.Nk_freq, *self.r_shape), dtype=self.dtype_ft
        )
        self.time_to_frequency()

    @property
    def Energy_ft(self):
        """
        Calculate total energy of the field from the spectral image
        assuming it to correspond to the electric component, and all values
        being defined in SI units.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        Energy = 4 * np.pi * epsilon_0 * c * self.t.ptp() * trapezoid(
            ( np.abs(self.Field_ft)**2 ).sum(0) * self.r, self.r
        )

        return Energy

    @property
    def Energy(self):
        """
        Calculate total energy of the field from the temporal distribution
        assuming it to correspond to the electric component, and all values
        being defined in SI units. This method is typically much slower than
        `Energy_ft` but may have a bit higher precision.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        Energy = 2 * np.pi * epsilon_0 * trapezoid(
            trapezoid(self.Field**2 * self.r, self.r), c * self.t
        )
        return Energy

    def import_field(self, Field, t_loc=None, r=None,
                     transform=True):
        """
        Import the field from the temporal domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the frequency domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if r is not None:
            self.r = r

        if len(Field[0].shape)==1:
            self.k_freq_shaped = self.k_freq[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_shaped = self.k_freq[:, None, None]

        self.Field = Field
        self.r_shape = Field[0].shape
        if transform:
            self.Field = apply_boundary_r(self.Field, self.dump_mask)
            self.time_to_frequency()
            self.Field_ft = apply_boundary_t(self.Field_ft, self.dump_mask)

    def import_field_ft(self, Field, t_loc=None, r=None, transform=True):
        """
        Import the field from the frequency domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container (the frequency domain)

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the temporal domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if r is not None:
            self.r = r

        self.r_shape = Field[0].shape
        self.Field_ft = Field.copy()

        if len(Field[0].shape)==1:
            self.k_freq_shaped = self.k_freq[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_shaped = self.k_freq[:, None, None]

        if transform:
            self.frequency_to_time()

    def time_to_frequency(self):
        """
        Transform the field from temporal to the frequency domain
        """
        if not hasattr(self, 'Field_ft'):
            self.Field_ft = np.zeros((self.Nk_freq, *self.r_shape),
                                     dtype=self.dtype_ft)

        self.Field_ft[:] = np.fft.rfft(self.Field, axis=0, norm='forward')[self.band_mask]
        self.Field_ft = np.conjugate( self.Field_ft )
        self.Field_ft *= np.exp(1j * self.k_freq_shaped * c * self.t_loc)

    def frequency_to_time(self):
        """
        Transform the field from frequency to the temporal domain
        """
        Field_ft = self.Field_ft * np.exp(
            -1j * self.k_freq_shaped * c * self.t_loc
        )

        Field_ft_full = np.zeros(
            (self.Nk_freq_full, *self.r_shape), dtype=self.dtype_ft
        )

        Field_ft_full[self.band_mask] = np.conjugate( Field_ft )

        if not hasattr(self, 'Field'):
            self.Field = np.zeros((self.Nt, *self.r_shape), dtype=self.dtype)

        self.Field[:] = np.fft.irfft(Field_ft_full, axis=0, norm='forward')

    def get_temporal_slice(self, ix=None, iy=None, ir=None):
        """
        Get a slice or linout of the field in the temporal domain
        performing FFT only for the selected points

        Parameters
        ----------
        ix: int or ":" or None
          If integer is used as x-slice coordinate for 3D field.
          If None (default) corresponds to the center of the screen.
          Can also be ":" which selects all, i.e. 2D slice in TX plane.
          In the case ":", `iy` must be either intereger or None.

        iy: int or ":" or None
          If integer is used as y-slice coordinate for 3D field.
          If None (default) corresponds to the center of the screen.
          Can also be ":" which selects all, i.e. 2D slice in TY plane.
          In the case ":", `ix` must be either intereger or None.

        ir: int or None
          If integer is used as r-slice coordinate for RT field.
          If None (default) corresponds to the center of the screen.
        """
        shape_tr = self.Field_ft[0].shape

        if len(shape_tr) == 1:
            if ir is None:
                ir = 0
            Field_ft = self.Field_ft[:, ir].copy()
            Field_ft *= np.exp(-1j * self.k_freq * c * self.t_loc)

        if len(shape_tr) == 2:
            Nx, Ny = shape_tr
            if ix is None and iy is None:
                ix = Nx // 2 - 1
                iy = Ny // 2 - 1
                Field_ft = self.Field_ft[:, ix, iy].copy()
                Field_ft *= np.exp(-1j * self.k_freq * c * self.t_loc)
            elif ix == ":":
                if iy is None:
                    iy = Ny // 2 - 1
                Field_ft = self.Field_ft[:, :, iy].copy()
                Field_ft *= np.exp(-1j * self.k_freq[:, None] * c * self.t_loc)
            elif iy == ":":
                if ix is None:
                    ix = Nx // 2 - 1
                Field_ft = self.Field_ft[:, ix, :].copy()
                Field_ft *= np.exp(-1j * self.k_freq[:, None] * c * self.t_loc)
            else:
                Field_ft = self.Field_ft[:, ix, iy].copy()
                Field_ft *= np.exp(-1j * self.k_freq * c * self.t_loc)

        Field_ft_full = np.zeros((self.Nk_freq_full, *Field_ft[0].shape),
                                 dtype=self.dtype_ft)
        Field_ft_full[self.band_mask] = np.conjugate( Field_ft )
        Field = np.fft.irfft(Field_ft_full, axis=0, norm='forward')
        return Field


class ScalarFieldEnvelope:
    """
    A class to initialize and transform the optical field envelope
    between temporal and frequency domains.
    """
    def __init__(self, k0, t_axis, n_dump=0, dtype=np.complex128 ):
        """
        Initialize the container for the field.

        Parameters
        ----------
        k0: float (m^-1)
          Central wavenumber for the spectral grid

        t_range: tuple of floats (t0, t1) (s)
            Range of the initial temporal domain

        dt: float (s)
            Time-step to resolve the temporal domain

        n_dump: int
            Number of cells to be used for attenuating boundaries
        """
        self.dtype = dtype
        self.k0 = k0
        self.omega0 = k0 * c

        self.t = t_axis.copy()
        self.dt = t_axis[1] - t_axis[0]
        self.cdt = c * self.dt
        self.Nt = self.t.size
        self.t_loc = self.t[0]

        self.k_freq_base = 2 * np.pi * np.fft.fftfreq(self.Nt, c*self.dt)
        self.k_freq = self.k_freq_base + k0

        self.Nk_freq = self.k_freq.size
        self.omega = self.k_freq * c

        self.n_dump = n_dump
        r_dump = np.linspace(0, 1, n_dump)
        self.dump_mask = ( 1 - np.exp(-(2 * r_dump)**3) )[::-1]

    def make_gaussian_pulse(self, a0, tau, r, R_las,
                            t0=0, phi0=0, n_ord=2, omega0=None):
        """
        Initialize the Gaussian pulse

        Parameters
        ----------
        a0: float
          Normalized amplitude of the pulse

        tau: float (s)
            Duration of the pulse (FBPIC definition)

        r: float ndarray (m)
            Radial grid for the container

        t0: float (s)
            Time corresponding the peak field

        phi0: float (rad)
            Carrier envelop phase (CEP)

        n_ord: int
            Order of Gaussian profile of TEM

        omega0: float (s^-1)
            central frequency of the pulse
        """
        self.r_shape = r.shape
        t = self.t
        self.r = r.copy()

        if omega0 is None:
            omega0 = self.omega0

        profile_r = np.exp( -( r/R_las )**n_ord )
        profile_t = np.exp( -(t-t0)**2 / tau**2 ) * np.exp( 1j * phi0 )

        if len(profile_r.shape) == 1:
            profile_t = profile_t[:, None]
            self.k_freq_shaped = self.k_freq[:, None]
            self.k_freq_base_shaped = self.k_freq_base[:, None]
        elif len(profile_r.shape) == 2:
            profile_t = profile_t[:, None, None]
            self.k_freq_shaped = self.k_freq[:, None, None]
            self.k_freq_base_shaped = self.k_freq_base[:, None, None]

        self.E0 = a0 * m_e * c * omega0 / e
        self.Field = (self.E0 * profile_t * profile_r[None,:])\
                        .astype(self.dtype)

        self.Field = apply_boundary_t(self.Field, self.dump_mask)
        self.Field = apply_boundary_r(self.Field, self.dump_mask)

        self.Field_ft = np.zeros(
            (self.Nk_freq, *self.r_shape), dtype=self.dtype
        )
        self.time_to_frequency()

    @property
    def Energy(self):
        """
        Calculate total energy of the field from the temporal distribution
        assuming it to correspond to the electric component, and all values
        being defined in SI units. This method is typically much slower than
        `Energy_ft` but may have a bit higher precision.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        Energy = np.pi * epsilon_0 * trapezoid(
            trapezoid(np.abs(self.Field)**2 * self.r, self.r), c * self.t
        )
        return Energy

    @property
    def Energy_ft(self):
        """
        Calculate total energy of the field from the spectral image
        assuming it to correspond to the electric component, and all values
        being defined in SI units.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        Energy = np.pi * epsilon_0 * c * self.t.ptp() * trapezoid(
            ( np.abs(self.Field_ft)**2 ).sum(0) * self.r, self.r
        )

        return Energy

    def import_field(self, Field, t_loc=None, r=None,
                     transform=True):
        """
        Import the field from the temporal domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the frequency domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if r is not None:
            self.r = r

        if len(Field[0].shape)==1:
            self.k_freq_base_shaped = self.k_freq_base[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_base_shaped = self.k_freq_base[:, None, None]

        self.Field = Field
        self.r_shape = Field[0].shape

        if transform:
            self.Field = apply_boundary_t(self.Field, self.dump_mask)
            self.Field = apply_boundary_r(self.Field, self.dump_mask)
            self.time_to_frequency()

        return self

    def import_field_ft(self, Field, t_loc=None, r=None, transform=True):
        """
        Import the field from the frequency domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container (the frequency domain)

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the temporal domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if r is not None:
            self.r = r

        self.r_shape = Field[0].shape
        self.Field_ft = Field

        if len(Field[0].shape)==1:
            self.k_freq_base_shaped = self.k_freq_base[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_base_shaped = self.k_freq_base[:, None, None]

        if transform:
            self.frequency_to_time()

        return self

    def time_to_frequency(self):
        """
        Transform the field from temporal to the frequency domain
        """
        if not hasattr(self, 'Field_ft'):
            self.Field_ft = np.zeros(
                (self.Nk_freq, *self.r_shape), dtype=self.dtype
            )

        self.Field_ft[:] = np.fft.ifft(self.Field, axis=0, norm="backward")
        self.Field_ft *= np.exp(1j * self.k_freq_base_shaped * c * self.t_loc)

    def frequency_to_time(self):
        """
        Transform the field from frequency to the temporal domain
        """
        Field_ft = self.Field_ft * np.exp(
            -1j * self.k_freq_base_shaped * c * self.t_loc
        )
        if not hasattr(self, 'Field'):
            self.Field = np.zeros((self.Nt, *self.r_shape), dtype=self.dtype)

        self.Field[:] = np.fft.fft(Field_ft, axis=0, norm="backward")

    def get_temporal_slice(self, ix=None, iy=None, ir=None):
        """
        Get a slice or linout of the field in the temporal domain
        performing FFT only for the selected points

        Parameters
        ----------
        ix: int or ":" or None
          If integer is used as x-slice coordinate for 3D field.
          If None (default) corresponds to the center of the screen.
          Can also be ":" which selects all, i.e. 2D slice in TX plane.
          In the case ":", `iy` must be either intereger or None.

        iy: int or ":" or None
          If integer is used as y-slice coordinate for 3D field.
          If None (default) corresponds to the center of the screen.
          Can also be ":" which selects all, i.e. 2D slice in TY plane.
          In the case ":", `ix` must be either intereger or None.

        ir: int or None
          If integer is used as r-slice coordinate for RT field.
          If None (default) corresponds to the center of the screen.
        """
        shape_tr = self.Field_ft[0].shape

        if len(shape_tr) == 1:
            if ir is None:
                ir = 0
            Field_ft = self.Field_ft[:, ir].copy()
            Field_ft *= np.exp(-1j * self.k_freq_base * c * self.t_loc)

        if len(shape_tr) == 2:
            Nx, Ny = shape_tr
            if ix is None and iy is None:
                ix = Nx // 2 - 1
                iy = Ny // 2 - 1
                Field_ft = self.Field_ft[:, ix, iy].copy()
                Field_ft *= np.exp(-1j * self.k_freq_base * c * self.t_loc)
            elif ix == ":":
                if iy is None:
                    iy = Ny // 2 - 1
                Field_ft = self.Field_ft[:, :, iy].copy()
                Field_ft *= np.exp(-1j * self.k_freq_base[:, None] * c * self.t_loc)
            elif iy == ":":
                if ix is None:
                    ix = Nx // 2 - 1
                Field_ft = self.Field_ft[:, ix, :].copy()
                Field_ft *= np.exp(-1j * self.k_freq_base[:, None] * c * self.t_loc)
            else:
                Field_ft = self.Field_ft[:, ix, iy].copy()
                Field_ft *= np.exp(-1j * self.k_freq_base * c * self.t_loc)

        Field = np.fft.fft(Field_ft, axis=0, norm="backward")

        return Field
