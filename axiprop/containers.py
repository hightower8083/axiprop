# Copyright 2023
# Authors: Igor A Andriyash
# License: BSD-3-Clause
"""
Axiprop containers.py file

This file contains container classes for axiprop:
"""
import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import trapezoid
import h5py
from warnings import warn
from types import MethodType

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
        A[:,:n_dump,:] *= dump_mask[::-1][None,:,None]
        A[:,:,:n_dump] *= dump_mask[::-1][None,None,:]
        A[:,-n_dump:,:] *= dump_mask[None,:,None]
        A[:,:,-n_dump:] *= dump_mask[None,None,:]
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


class ScalarFieldEnvelope:
    """
    A class to initialize and transform the optical field envelope
    between temporal and frequency domains.
    """
    def __init__(
        self, k0=None, t_axis=None, n_dump_r=0, n_dump_t=0, dtype=np.complex128, file_name=None
    ):
        """
        Initialize the container for the field.

        Parameters
        ----------
        k0: float (m^-1)
          Central wavenumber for the spectral grid

        t_axis: 1d ndarray (s)
            Temporal grid of the initial temporal domain

        n_dump_r: int
            Number of cells to be used for attenuating transverse boundaries

        n_dump_t: int
            Number of cells to be used for attenuating temporal boundaries
        """
        self.dtype = dtype

        if file_name is not None:
            with h5py.File(file_name, mode='r') as fl:
                for attr in fl.keys():
                    setattr( self, attr, fl[attr][()] )

        elif k0 is not None and t_axis is not None:
            self.k0 = k0
            self.omega0 = k0 * c

            self.t = t_axis.copy()
            self.Nt = self.t.size
            if self.Nt>1:
                self.dt = t_axis[1] - t_axis[0]
                self.cdt = c * self.dt
            else:
                self.dt = np.inf
                self.cdt = np.inf
            self.t_loc = self.t[0]

            self.k_freq_base = 2 * np.pi * np.fft.fftfreq(self.Nt, c*self.dt)
            self.k_freq = self.k_freq_base + k0

            self.Nk_freq = self.k_freq.size
            self.omega = self.k_freq * c

            self.n_dump_r = n_dump_r
            r_dump = np.linspace(0, 1, n_dump_r)
            self.dump_mask_r = ( 1 - np.exp(-(2 * r_dump)**3) )[::-1]

            self.n_dump_t = n_dump_t
            t_dump = np.linspace(0, 1, n_dump_t)
            self.dump_mask_t = ( 1 - np.exp(-(2 * t_dump)**3) )[::-1]
        else:
            print ('Either `k0` and `t_axis` or `file_name` should be provided')


    def make_gaussian_pulse(self, r_axis, tau, w0, a0=None, Energy=None,
                            t0=0.0, phi0=0.0, n_ord=2,
                            omega0=None, transform=True):
        """
        Initialize the Gaussian pulse

        Parameters
        ----------
        r: float ndarray (m)
            Radial grid for the container

        tau: float (s)
            Duration of the pulse (FBPIC definition)

        w0: float (m)
            Waist of the pulse (FBPIC definition)

        a0: float (optional)
          Normalized amplitude of the pulse

        Energy: float (J) (optional)
          Energy of the pulse

        t0: float (s) (optional)
            Time corresponding the peak field

        phi0: float (rad) (optional)
            Carrier envelop phase (CEP)

        n_ord: int (optional)
            Order of Gaussian profile of TEM

        omega0: float (s^-1) (optional)
            central frequency of the pulse

        transform: bool (optional)
            Wether to transform the field to the frequency domain
        """
        t = self.t

        if type(r_axis) is tuple and len(r_axis)==3:
            self.r = r_axis[0].copy()
            self.x = r_axis[1].copy()
            self.y = r_axis[2].copy()
        elif type(r_axis) is np.ndarray and len(r_axis.shape)==1:
            self.r = r_axis.copy()
        else:
            warn("Input `r_axis` must be either 1D ndarray for RZ, or " + \
                 "a tuple `(r, x, y)` with r: 2D ndarray, x/y: 1D ndarrays"
            )

        self.r_shape = self.r.shape

        if Energy is not None and a0 is None:
            a0 = 1.0
        elif Energy is None and a0 is None:
            warn('Either `a0` of `Energy` must be specified.')

        if omega0 is None:
            omega0 = self.omega0

        profile_r = np.exp( -( self.r/w0 )**n_ord )
        profile_t = np.exp( -(t-t0)**2 / tau**2 ) * np.exp( 1j * phi0 )

        if len(profile_r.shape) == 1:
            profile_t = profile_t[:, None]
            self.k_freq_shaped = self.k_freq[:, None]
            self.k_freq_base_shaped = self.k_freq_base[:, None]
        elif len(profile_r.shape) == 2:
            profile_t = profile_t[:, None, None]
            self.k_freq_shaped = self.k_freq[:, None, None]
            self.k_freq_base_shaped = self.k_freq_base[:, None, None]

        E0 = a0 * m_e * c * omega0 / e
        self.Field = ( E0 * profile_t * profile_r[None,:] ) \
                        .astype(self.dtype)

        self.Field = apply_boundary_t(self.Field, self.dump_mask_t)
        self.Field = apply_boundary_r(self.Field, self.dump_mask_r)

        if Energy is not None:
            self.Field *= ( Energy / self.Energy ) ** 0.5

        if transform:
            self.Field_ft = np.zeros(
                (self.Nk_freq, *self.r_shape), dtype=self.dtype
            )
            self.time_to_frequency()

        return self

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

        if len(self.Field[0].shape)==1:
            Energy = np.pi * epsilon_0 * trapezoid(
                trapezoid(np.abs(self.Field)**2 * self.r, self.r), c * self.t
            )
        else:
            dx = np.ptp( self.x[[0,1]] )
            dy = np.ptp( self.y[[0,1]] )
            cdt = c * np.ptp( self.t[[0,1]] )
            Energy = 0.5 * epsilon_0 * \
                np.sum( np.abs(self.Field)**2 ) * dx * dy * cdt

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

        if len(self.Field_ft[0].shape)==1:
            Energy = np.pi * epsilon_0 * c * np.ptp(self.t) * trapezoid(
                ( np.abs(self.Field_ft)**2 ).sum(0) * self.r, self.r
            )
        else:
            dx = np.ptp( self.x[[0,1]] )
            dy = np.ptp( self.y[[0,1]] )

            Energy = 0.5 * epsilon_0 * c * np.ptp(self.t) * \
                ( np.abs(self.Field_ft)**2 ).sum() * dx * dy

        return Energy

    @property
    def w0(self):
        """
        Calculate waist the field from the temporal distribution.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        if len(self.Field[0].shape)==1:
            spot_r = np.abs(self.Field**2).sum(0)
            w0 = 2 * np.sqrt(np.average(self.r**2, weights=spot_r))
        else:
            I_norm = np.abs(self.Field**2)
            spot_x = I_norm.sum(0).sum(-1)
            spot_y = I_norm.sum(0).sum(0)
            w0_x = 2 * np.sqrt(np.average(self.x**2, weights=spot_x) - np.average(self.x, weights=spot_x)**2)
            w0_y = 2 * np.sqrt(np.average(self.y**2, weights=spot_y) - np.average(self.y, weights=spot_y)**2)
            w0 = (w0_x, w0_y)

        return w0

    @property
    def w0_ft(self):
        """
        Calculate waist the field from the temporal distribution.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        if len(self.Field_ft[0].shape)==1:
            spot_r = np.abs(self.Field_ft**2).sum(0)
            w0 = 2 * np.sqrt(np.average(self.r**2, weights=spot_r))
        else:
            I_norm = np.abs(self.Field_ft**2)
            spot_x = I_norm.sum(0).sum(-1)
            spot_y = I_norm.sum(0).sum(0)
            w0_x = 2 * np.sqrt(np.average(self.x**2, weights=spot_x) - np.average(self.x, weights=spot_x)**2)
            w0_y = 2 * np.sqrt(np.average(self.y**2, weights=spot_y) - np.average(self.y, weights=spot_y)**2)
            w0 = (w0_x, w0_y)

        return w0

    @property
    def tau(self):
        fld_onax = np.abs(self.get_temporal_slice())
        tau = 2**.5 * np.sqrt(
            np.average(self.t**2, weights=fld_onax) \
            - np.average(self.t, weights=fld_onax)**2
        )
        return tau

    @property
    def dt_to_center(self):
        fld_onax = np.abs(self.get_temporal_slice())
        dt = np.average(self.t, weights=fld_onax)
        return dt

    @property
    def dt_to_peak(self):
        fld_onax = np.abs(self.get_temporal_slice())
        dt = self.t[fld_onax==fld_onax.max()].mean()
        return dt

    def import_field(self, Field, t_loc=None, r_axis=None,
                     transform=True, make_copy=False):
        """
        Import the field from the temporal domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r_axis: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the frequency domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if r_axis is not None:
            if type(r_axis) is tuple and len(r_axis)==3:
                self.r = r_axis[0].copy()
                self.x = r_axis[1].copy()
                self.y = r_axis[2].copy()
            elif type(r_axis) is np.ndarray and len(r_axis.shape)==1:
                self.r = r_axis.copy()
            else:
                warn("Input `r_axis` must be either 1D ndarray for RZ, or " + \
                     "a tuple `(r, x, y)` with r: 2D ndarray, x/y: 1D ndarrays"
                )

        if len(Field[0].shape)==1:
            self.k_freq_base_shaped = self.k_freq_base[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_base_shaped = self.k_freq_base[:, None, None]

        if make_copy:
            self.Field = Field.copy()
        else:
            self.Field = Field

        self.r_shape = Field[0].shape

        self.Field = apply_boundary_t(self.Field, self.dump_mask_t)
        self.Field = apply_boundary_r(self.Field, self.dump_mask_r)

        if transform:
            self.time_to_frequency()

        return self

    def import_field_ft(self, Field, t_loc=None, r_axis=None,
                        transform=True, clean_boundaries=False,
                        make_copy=False):
        """
        Import the field from the frequency domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container (the frequency domain)

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r_axis: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the temporal domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if r_axis is not None:
            if type(r_axis) is tuple and len(r_axis)==3:
                self.r = r_axis[0].copy()
                self.x = r_axis[1].copy()
                self.y = r_axis[2].copy()
            elif type(r_axis) is np.ndarray and len(r_axis.shape)==1:
                self.r = r_axis.copy()
            else:
                warn("Input `r_axis` must be either 1D ndarray for RZ, or " + \
                     "a tuple `(r, x, y)` with r: 2D ndarray, x/y: 1D ndarrays"
                )

        self.r_shape = Field[0].shape

        if make_copy:
            self.Field_ft = Field.copy()
        else:
            self.Field_ft = Field

        if len(Field[0].shape)==1:
            self.k_freq_base_shaped = self.k_freq_base[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_base_shaped = self.k_freq_base[:, None, None]

        if transform:
            self.frequency_to_time()

        if clean_boundaries:
            self.Field = apply_boundary_t(self.Field, self.dump_mask_t)
            self.Field = apply_boundary_r(self.Field, self.dump_mask_r)
            self.time_to_frequency()

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

    def save_to_file(self, file_name='axiprop_container.h5'):
        attr_select = np.asarray(self.__dir__() )

        attr_select = attr_select[[
            '__' not in attr for attr in attr_select
        ]]

        attr_exclude = [
            'dtype', 'Energy', 'Energy_ft', 'w0', 'w0_ft',
            'tau', 'dt_to_center', 'dt_to_peak'
        ]

        attr_select = attr_select[[
            attr not in attr_exclude for attr in attr_select
        ]]

        attr_select = attr_select[[
            getattr(self, attr).__class__ is not MethodType for attr in attr_select
        ]]

        with h5py.File(file_name, mode='w') as fl:
            for attr in attr_select:
                fl[attr] = getattr(self, attr)


class ScalarField(ScalarFieldEnvelope):
    """
    A class to initialize and transform the optical field between temporal
    and frequency domains.
    """
    def __init__(self, k0, t_axis, bandwidth, n_dump=0,
                 dtype_ft=np.complex128, dtype=np.double ):
        """
        Initialize the container for the field.

        Parameters
        ----------
        k0: float (m^-1)
          Central wavenumber for the spectral grid

        t_axis: 1d ndarray (s)
            Temporal grid of the initial temporal domain

        bandwidth: float (m^-1)
            Width of the spectral grid

        n_dump: int
            Number of cells to be used for attenuating boundaries
        """
        self.dtype = dtype
        self.dtype_ft = dtype_ft
        self.k0 = k0
        self.omega0 = k0 * c

        self.t = t_axis.copy()
        self.dt = np.ptp( t_axis[[0,1]] )
        self.cdt = c * self.dt

        self.Nt = self.t.size
        self.t_loc = self.t[0]

        self.k_freq_full = 2 * np.pi * np.fft.rfftfreq( self.Nt, c*self.dt )
        self.Nk_freq_full = self.k_freq_full.size
        self.k_freq_full = self.k_freq_full[:self.Nk_freq_full]

        if type(bandwidth) in [tuple, list]:
            lambda_min, lambda_max = bandwidth
            k_max = 2 * np.pi / lambda_min
            k_min = 2 * np.pi / lambda_max
            self.band_mask = (self.k_freq_full>k_min) * (self.k_freq_full<k_max)
        else:
            self.band_mask = ( np.abs(self.k_freq_full - k0) < bandwidth )

        self.k_freq = self.k_freq_full[self.band_mask]
        self.Nk_freq = self.k_freq.size
        self.omega = self.k_freq * c

        self.n_dump = n_dump
        dump_r = np.linspace(0, 1, n_dump)
        self.dump_mask = ( 1 - np.exp(-(2*dump_r)**3) )[::-1]

    def make_gaussian_pulse( self, r_axis, tau, w0, a0=None,
                            Energy=None, t0=0.0, phi0=0.0, n_ord=2,
                            omega0=None, transform=True):
        """
        Initialize the Gaussian pulse

        Parameters
        ----------
        r_axis: float ndarray (m)
            Radial grid for the container

        tau: float (s)
            Duration of the pulse (FBPIC definition)

        w0: float (m)
            Waist of the pulse (FBPIC definition)

        a0: float
          Normalized amplitude of the pulse

        Energy: float (J)
          Energy of the pulse in Joules

        t0: float (s)
            Time corresponding the peak field

        phi0: float (rad)
            Carrier envelop phase (CEP)

        n_ord: int
            Order of Gaussian profile of TEM

        omega0: float (s^-1)
            central frequency of the pulse
        """
        t = self.t

        if type(r_axis) is tuple and len(r_axis)==3:
            self.r = r_axis[0].copy()
            self.x = r_axis[1].copy()
            self.y = r_axis[2].copy()
        elif type(r_axis) is np.ndarray and len(r_axis.shape)==1:
            self.r = r_axis.copy()
        else:
            warn("Input `r_axis` must be either 1D ndarray for RZ, or " + \
                 "a tuple `(r, x, y)` with r: 2D ndarray, x/y: 1D ndarrays"
            )

        self.r_shape = self.r.shape

        if omega0 is None:
            omega0 = self.omega0

        if Energy is not None and a0 is None:
            a0 = 1.0
        elif Energy is None and a0 is None:
            warn('Either `a0` of `Energy` must be specified.')

        profile_r = np.exp( -( self.r / w0 )**n_ord )
        profile_t = np.exp( -(t-t0)**2 / tau**2 ) * np.sin(omega0 * t + phi0)

        if len(profile_r.shape) == 1:
            profile_t = profile_t[:, None]
            self.k_freq_shaped = self.k_freq[:, None]
        elif len(profile_r.shape) == 2:
            profile_t = profile_t[:, None, None]
            self.k_freq_shaped = self.k_freq[:, None, None]

        E0 = a0 * m_e * c * omega0 / e
        self.Field = ( E0 * profile_t * profile_r[None,:] ) \
                        .astype(self.dtype)

        self.Field = apply_boundary_t(self.Field, self.dump_mask)
        self.Field = apply_boundary_r(self.Field, self.dump_mask)

        if Energy is not None:
            self.Field *= ( Energy / self.Energy ) ** 0.5

        if transform:
            self.Field_ft = np.zeros(
                (self.Nk_freq, *self.r_shape), dtype=self.dtype_ft
            )
            self.time_to_frequency()

        return self

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

        if len(self.Field[0].shape)==1:
            Energy = 2 * np.pi * epsilon_0 * trapezoid(
                trapezoid(np.abs(self.Field)**2 * self.r, self.r), c * self.t
            )
        else:
            dx = np.ptp( self.x[[0,1]] )
            dy = np.ptp( self.y[[0,1]] )
            cdt = c * np.ptp( self.t[[0,1]] )
            Energy = epsilon_0 * \
                np.sum( np.abs(self.Field)**2 ) * dx * dy * cdt


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

        if len(self.Field_ft[0].shape)==1:
            Energy = 4 * np.pi * epsilon_0 * c * np.ptp(self.t) * trapezoid(
                ( np.abs(self.Field_ft)**2 ).sum(0) * self.r, self.r
            )
        else:
            dx = np.ptp( self.x[[0,1]] )
            dy = np.ptp( self.y[[0,1]] )

            Energy = 2 * epsilon_0 * c * np.ptp(self.t) * \
                ( np.abs(self.Field_ft)**2 ).sum() * dx * dy


        return Energy

    @property
    def w0(self):
        """
        Calculate waist the field from the temporal distribution.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        if len(self.Field[0].shape)==1:
            spot_r = (np.abs(self.Field)**2).sum(0)
            w0 = 2 * np.sqrt(np.average(self.r**2, weights=spot_r))
        else:
            I_norm = np.abs(self.Field)**2
            spot_x = I_norm.sum(0).sum(-1)
            spot_y = I_norm.sum(0).sum(0)
            w0_x = 2 * np.sqrt(np.average(self.x**2, weights=spot_x) \
                                - np.average(self.x, weights=spot_x)**2)
            w0_y = 2 * np.sqrt(np.average(self.y**2, weights=spot_y) \
                                - np.average(self.y, weights=spot_y)**2)
            w0 = (w0_x, w0_y)

        return w0

    @property
    def w0_ft(self):
        """
        Calculate waist the field from the temporal distribution.
        """
        if not hasattr(self, 'r'):
            print('provide r-axis')
            return None

        if len(self.Field_ft[0].shape)==1:
            spot_r = (np.abs(self.Field_ft)**2).sum(0)
            w0 = 2 * np.sqrt(np.average(self.r**2, weights=spot_r))
        else:
            I_norm = np.abs(self.Field_ft)**2
            spot_x = I_norm.sum(0).sum(-1)
            spot_y = I_norm.sum(0).sum(0)
            w0_x = 2 * np.sqrt(np.average(self.x**2, weights=spot_x) \
                                - np.average(self.x, weights=spot_x)**2)
            w0_y = 2 * np.sqrt(np.average(self.y**2, weights=spot_y) \
                                - np.average(self.y, weights=spot_y)**2)
            w0 = (w0_x, w0_y)

        return w0

    @property
    def tau(self):
        fld_onax = np.abs(self.get_temporal_slice())
        tau = 2**.5 * np.sqrt(
            np.average(self.t**2, weights=fld_onax) \
            - np.average(self.t, weights=fld_onax)**2
        )
        return tau

    @property
    def dt_to_center(self):
        fld_onax = np.abs(self.get_temporal_slice())
        dt = np.average(self.t, weights=fld_onax)
        return dt

    @property
    def dt_to_peak(self):
        fld_onax = np.abs(self.get_temporal_slice())
        dt = self.t[fld_onax==fld_onax.max()].mean()
        return dt

    def import_field(self, Field, t_loc=None, r_axis=None,
                     transform=True, make_copy=False):
        """
        Import the field from the temporal domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r_axis: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the frequency domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if type(r_axis) is tuple and len(r_axis)==3:
            self.r = r_axis[0].copy()
            self.x = r_axis[1].copy()
            self.y = r_axis[2].copy()
        elif type(r_axis) is np.ndarray and len(r_axis.shape)==1:
            self.r = r_axis.copy()
        else:
            warn("Input `r_axis` must be either 1D ndarray for RZ, or " + \
                 "a tuple `(r, x, y)` with r: 2D ndarray, x/y: 1D ndarrays"
            )

        if len(Field[0].shape)==1:
            self.k_freq_shaped = self.k_freq[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_shaped = self.k_freq[:, None, None]

        if make_copy:
            self.Field = Field.copy()
        else:
            self.Field = Field

        self.r_shape = Field[0].shape

        if transform:
            self.Field = apply_boundary_t(self.Field, self.dump_mask)
            self.Field = apply_boundary_r(self.Field, self.dump_mask)
            self.time_to_frequency()

        return self

    def import_field_ft(self, Field, t_loc=None, r_axis=None,
                        transform=True, clean_boundaries=False,
                        make_copy=False):
        """
        Import the field from the frequency domain

        Parameters
        ----------
        Field: float ndarray
          The field  to be imported of the same dimensions
          as the main field of the container (the frequency domain)

        t_loc: float (s)
            Local time for the field to be considered in frequency space

        r_axis: float ndarray (m)
            Radial grid for the container

        transform: bool
            Wether to transform the field to the temporal domain
        """
        if t_loc is not None:
            self.t_loc = t_loc

        if type(r_axis) is tuple and len(r_axis)==3:
            self.r = r_axis[0].copy()
            self.x = r_axis[1].copy()
            self.y = r_axis[2].copy()
        elif type(r_axis) is np.ndarray and len(r_axis.shape)==1:
            self.r = r_axis.copy()
        else:
            warn("Input `r_axis` must be either 1D ndarray for RZ, or " + \
                 "a tuple `(r, x, y)` with r: 2D ndarray, x/y: 1D ndarrays"
            )

        self.r_shape = Field[0].shape

        if make_copy:
            self.Field_ft = Field.copy()
        else:
            self.Field_ft = Field

        if len(Field[0].shape)==1:
            self.k_freq_shaped = self.k_freq[:, None]
        elif len(Field[0].shape)==2:
            self.k_freq_shaped = self.k_freq[:, None, None]

        if transform:
            self.frequency_to_time()
            if clean_boundaries:
                self.Field = apply_boundary_t(self.Field, self.dump_mask)
                self.Field = apply_boundary_r(self.Field, self.dump_mask)
                self.time_to_frequency()

        return self

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
