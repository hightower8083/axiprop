# Copyright 2023
# Authors: Igor A Andriyash
# License: BSD-3-Clause
"""
Axiprop utils file

This file contains utility methods for Axiprop tool
"""
import numpy as np
from scipy.constants import c
from scipy.interpolate import Akima1DInterpolator
from scipy.interpolate import RegularGridInterpolator

from axiprop.containers import ScalarFieldEnvelope

try:
    from skimage.restoration import unwrap_phase as unwrap2d
    unwrap_available = True
except Exception:
    unwrap_available = False

if not unwrap_available:
    try:
        from unwrap import unwrap as unwrap2d
        unwrap_available = True
    except Exception:
        unwrap_available = False


# try import numba and make dummy methods if cannot
try:
    from numba import njit, prange
    njit = njit(parallel=True)
except Exception:
    prange = range
    def njit(func):
        def func_wrp(*args, **kw_args):
            print(f"Install Numba to get `{func.__name__}` " + \
                   "function greatly accelerated")
            return func(*args, **kw_args)
        return func_wrp

def refine1d(A, refine_ord, kind='linear'):
    if A.dtype not in (np.double, np.complex128):
        print("Data type must be `np.double` or `np.complex128`")
        return None
    if len(A.shape) != 1:
        print("Data must be 1D array")
        return None

    refine_ord = int(refine_ord)
    Nx = A.size
    x = np.arange(Nx, dtype=np.double)
    x_new = np.linspace(x.min(), x.max(), refine_ord * (Nx-1) + 1 )

    if kind == 'linear':
        if A.dtype == np.double:
            A_new = np.interp(x_new, x, A)
        elif A.dtype == np.complex128:
            slice_abs = np.interp( x_new, x, np.abs(A) )
            slice_angl = np.interp( x_new, x, np.unwrap(np.angle(A)) )
            A_new = slice_abs * np.exp(1j * slice_angl)
    elif kind == 'cubic':
        if A.dtype == np.double:
            A_new = Akima1DInterpolator(x, A)(x_new)
        elif A.dtype == np.complex128:
            slice_abs = Akima1DInterpolator(x, np.abs(A))(x_new)
            slice_angl = Akima1DInterpolator(x, np.unwrap(np.angle(A)))(x_new)
            A_new = slice_abs * np.exp(1j * slice_angl)

    return A_new

def refine1d_TR(A, refine_ord, kind='linear'):
    Nt, Nr = A.shape
    Nt_new = refine_ord * (Nt-1) + 1
    A_new = np.zeros((Nt_new, Nr), dtype=A.dtype)

    for ir in range(Nr):
        A_new[:, ir] = refine1d(A[:, ir], refine_ord=refine_ord, kind=kind)
    return A_new

def init_fresnel_rt( dz, r_axis, kz_axis, r_axis_new, **prop_args):
    prop_args['kz_axis'] = kz_axis
    k_max = kz_axis.max()

    if type(r_axis_new) in (tuple, list):
        assert ( len(r_axis_new) == 2 )
        R_2, Nr_2 = r_axis_new
    elif type(r_axis_new) is np.ndarray:
        assert ( len(r_axis_new.shape) == 1 )
        R_2 = r_axis_new.max()
        Nr_2 = r_axis_new.size

    prop_args['r_axis_new'] = r_axis_new

    if type(r_axis) in (tuple, list):
        assert ( len(r_axis) == 2 )
        R_1, Nr_1 = r_axis
        if Nr_1 is not None:
            prop_args['r_axis'] = (R_1, Nr_1)
    elif type(r_axis) is np.ndarray:
        assert ( len(r_axis.shape) == 1 )
        R_1 = r_axis.max()
        Nr_1 = r_axis.size
        prop_args['r_axis'] = r_axis

    if Nr_1 is None:
        Nr_1 = int( np.ceil( R_1 * R_2 * k_max / (np.pi * dz) ) )
        prop_args['r_axis'] = (R_1, Nr_1)
        N_pad = Nr_2 / Nr_1
        if N_pad < 1 :
            N_pad = 1
    else:
        dr_2 = R_2 / Nr_2
        N_pad = np.pi * dz / (dr_2 * R_1 * k_max)
        if N_pad < 1 :
            N_pad = 1
        R_2_eff = np.pi * dz * Nr_1 / R_1 / k_max
        if R_2 <= R_2_eff:
            prop_args['Nkr_new'] = int(np.ceil( N_pad * Nr_1 * R_2 / R_2_eff ))
        else:
            print('warning: higher `r_axis` resolution is needed')

    prop_args['N_pad'] = N_pad

    return prop_args

def unwrap1d(angl_in, period=2*np.pi, n_span=4, n_order=1):
    """
    from scipy.special import binom
    FD_shapes = [ (-1)**(n_order-np.arange(n_order+1)) \
                              * binom(n_order, np.arange(n_order+1)) \
                          for n_order in range(10)]
    FD_shape = FD_shapes[n_order][::-1]
    """
    angl = angl_in.copy()
    period_span = period * np.arange(-n_span, n_span+1)

    FD_shapes = [
        np.array([1.]),
        np.array([ 1., -1.]),
        np.array([ 1., -2., 1.]),
        np.array([ 1., -3., 3., -1.]),
        np.array([ 1., -4.,  6., -4., 1.]),
        np.array([ 1., -5., 10., -10., 5., -1.])
        ]

    FD_shape = FD_shapes[n_order]
    angle_values = np.zeros_like(FD_shape)

    for i_angl in range(1, angl.size):

        for i_order in range(n_order+1):
            if i_angl>i_order:
                angle_values[i_order] = angl[i_angl-i_order]
            else:
                angle_values[i_order] = angl[i_order-1]

        index_minimum_div = np.abs( (FD_shape * angle_values).sum() + \
                                    period_span ).argmin()

        angl[i_angl:] += period_span[index_minimum_div]

    return angl

def laser_from_fu(fu, kz, r, normalize=False):
    """
    Generate array with spectra-radial field
    distribution with the pre-defined function
    """

    fu = njit(fu)

    a0 = fu( ( kz * np.ones((*kz.shape, *r.shape)).T ).T,
             r[None,:] * np.ones((*kz.shape, *r.shape)) )

    if normalize:
        a0 /= (np.abs(a0)**2).sum(0).max()**0.5

    return a0

def mirror_parabolic(f0, kz, r):
    """
    Generate array with spectra-radial phase
    representing the on-axis Parabolic Mirror
    """
    s_ax = r**2 / 4 / f0

    val = np.exp( -2j * s_ax[None,:] * \
                 ( kz * np.ones((*kz.shape, *r.shape)).T ).T)
    return val

def import_from_lasy(laser):
    r"""
    Extract field from `lasy` object and import it to a `container` object or list
    """
    time_axis_indx = -1
    omega0 = laser.profile.omega0
    t_axis = laser.grid.axes[time_axis_indx]

    if laser.dim == "rt":
        Container = []
        for i_m in range( laser.grid.azimuthal_modes.size ):
            Container.append(
                ScalarFieldEnvelope(omega0 / c, t_axis).import_field(
                    np.transpose(laser.grid.field[i_m]).copy()
                )
            )
    else:
        Container = ScalarFieldEnvelope(omega0 / c, t_axis).import_field(
            np.moveaxis(laser.grid.field, 0, -1).copy()
        )

    return Container

def export_to_lasy(Container, polarization=(1,0), dimensions='rt'):
    r"""
    Export field from the `container` to the `lasy` object
    """
    try:
        from lasy.laser import Laser
        from lasy.profiles import CombinedLongitudinalTransverseProfile
        from lasy.profiles.longitudinal.longitudinal_profile import LongitudinalProfile
        from lasy.profiles.transverse.transverse_profile import TransverseProfile
    except Exception:
        print ( "Error: `lasy` is not installed" )
        return None

    wavelength = 2 * np.pi * c / Container.omega0
    empty_longitudinal_profile = LongitudinalProfile(wavelength=wavelength)
    empty_transverse_profile = TransverseProfile()
    empty_profile = CombinedLongitudinalTransverseProfile(
        wavelength=wavelength,
        pol=polarization,
        laser_energy=0.0,
        long_profile=empty_longitudinal_profile,
        trans_profile=empty_transverse_profile,
    )

    if dimensions == 'rt':
        lo = ( Container.r.min(), Container.t.min() )
        hi = ( Container.r.max(), Container.t.max() )
        num_points = ( Container.r.size, Container.t.size )
    elif dimensions == 'xyt':
        lo = ( Container.x.min(), Container.y.min(), Container.t.min() )
        hi = ( Container.x.min(), Container.y.min(), Container.t.max() )
        num_points = ( Container.x.size, Container.y.size, Container.t.size )

    if not hasattr(Container, 'Field'):
        Container.frequency_to_time()

    laser = Laser(dimensions, lo, hi, num_points, empty_profile)
    laser.grid.set_temporal_field(np.moveaxis(Container.Field, 0, -1)[None,...])

    return laser


def txy_to_mtr(laser_txy, Nm, dr=None, Nm_ext=64 ):

    x = laser_txy.x.copy()
    y = laser_txy.y.copy()

    rmax = np.max([
        np.abs(x).max(), np.abs(y).max()
    ])

    if dr is None:
        dxy = np.min([
            np.ptp(x[[0,1]]),
            np.ptp(y[[0,1]])
        ])
    else:
        dxy = dr

    r = np.arange(0, rmax, dxy)
    th = np.linspace(0, 2*np.pi, Nm_ext, endpoint=False)

    Nr = r.size
    Nt = laser_txy.t.size

    xx_pg = r[None, :] * np.cos( th[:, None] )
    yy_pg = r[None, :] * np.sin( th[:, None] )

    m_axis_ext =  (Nm_ext * np.fft.fftfreq(Nm_ext)).astype(np.int64)
    m_axis = (Nm * np.fft.fftfreq(Nm)).astype(np.int64)

    laser_mtr = []

    for im, m in enumerate(m_axis):
        laser_mtr.append(
            ScalarFieldEnvelope(laser_txy.k0, t_axis=laser_txy.t) \
                .import_field(
                    np.zeros( (Nt, Nr), dtype=laser_txy.dtype),
                    t_loc=laser_txy.t_loc,
                    r_axis=r
                )
        )

    for it in range(Nt):
        E_slice = laser_txy.Field[it]
        E_slice_abs = np.abs(E_slice)
        E_slice_angl = unwrap2d(np.angle(E_slice))

        E_abs_pg = RegularGridInterpolator((x,y), E_slice_abs, bounds_error=False, fill_value=0.0, method='cubic')((xx_pg, yy_pg))
        E_angl_pg = RegularGridInterpolator((x,y), E_slice_angl, bounds_error=False, fill_value=0.0, method='cubic')((xx_pg, yy_pg))

        E_slice_pg = E_abs_pg * np.exp(1j * E_angl_pg)
        E_slice_pg = np.fft.ifft(E_slice_pg, axis=0)

        for im, m in enumerate(m_axis):
            im_ext = np.argwhere(m_axis_ext==m).flatten()[0]
            laser_mtr[im].Field[it] = E_slice_pg[im_ext]

    return laser_mtr, m_axis


def mtr_to_txy(laser_mtr, m_axis, x, y ):
    k0 = laser_mtr[0].k0
    t_axis = laser_mtr[0].t
    r = laser_mtr[0].r
    t_loc = laser_mtr[0].t_loc
    dtype = laser_mtr[0].dtype

    Nm = m_axis.size
    Nx = x.size
    Ny = y.size
    Nt = t_axis.size

    r_proj = np.sqrt(x[:, None]**2 + y[None, :]**2)
    th_proj = np.arctan2(y[None, :], x[:, None])

    laser_txy = ScalarFieldEnvelope(
        laser_mtr[0].k0, t_axis=laser_mtr[0].t
    ).import_field(
        np.zeros((Nt, Nx, Ny), dtype=dtype),
        t_loc=t_loc,
        r_axis=(r, x, y),
        )

    for im, m in enumerate(m_axis):
        field_tr = laser_mtr[im].Field

        field_tr_abs = np.abs(field_tr)
        field_tr_angl = np.unwrap( np.angle(field_tr) )

        for it in range(laser_txy.t.size):
            field_txy_abs = Akima1DInterpolator(r, field_tr_abs[it])(r_proj)
            field_txy_angl = Akima1DInterpolator(r, field_tr_angl[it])(r_proj)
            phase_loc = field_txy_angl - m * th_proj
            laser_txy.Field[it] += field_txy_abs * np.exp(1j * phase_loc)

    return laser_txy
