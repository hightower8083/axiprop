# Copyright 2023
# Authors: Igor A Andriyash
# License: BSD-3-Clause
"""
Axiprop utils file

This file contains utility methods for Axiprop tool
"""
import numpy as np
from scipy.constants import c
from scipy.interpolate import interp1d

from axiprop.containers import ScalarFieldEnvelope

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

def refine1d(A, refine_ord):
    refine_ord = int(refine_ord)
    x = np.arange(A.size, dtype=np.double)
    x_new = np.linspace(x.min(), x.max(), x.size*refine_ord)

    if A.dtype == np.double:
        interp_fu = interp1d(x, A, assume_sorted=True)
        A_new = interp_fu(x_new)
    elif A.dtype == np.complex128:
        interp_fu_abs = interp1d(x, np.abs(A), assume_sorted=True)
        slice_abs = interp_fu_abs(x_new)

        interp_fu_angl = interp1d(x, np.unwrap(np.angle(A)), assume_sorted=True)
        slice_angl = interp_fu_angl(x_new)

        A_new = slice_abs * np.exp(1j * slice_angl)
    else:
        print("Data type must be `np.double` or `np.complex128`")
        return None

    return A_new

def refine1d_TR(A, refine_ord):
    refine_ord = int(refine_ord)

    t = np.arange(A.shape[0], dtype=np.double)
    t_new = np.linspace(t.min(), t.max(), t.size*refine_ord)

    A_new = np.zeros((t_new.size, A.shape[1]), dtype=A.dtype)

    for ir in range(A.shape[1]):
        interp_fu_abs = interp1d(t, np.abs(A[:, ir]), assume_sorted=True)
        slice_abs = interp_fu_abs(t_new)

        interp_fu_angl = interp1d( t, np.unwrap(np.angle(A[:, ir])),
                                   assume_sorted=True )
        slice_angl = interp_fu_angl(t_new)

        A_new[:, ir] = slice_abs * np.exp(1j * slice_angl)

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

@njit
def get_temporal_1d(u, u_t, t, kz):
    """
    Resonstruct temporal field distribution
    """
    Nt = t.size

    for it in prange(Nt):
        FFT_factor = np.exp(-1j * kz * c * t[it])
        u_t[it] = np.real(u * FFT_factor).sum()

    return u_t

@njit
def get_temporal_radial(u, u_t, t, kz):
    """
    Resonstruct temporal-radial field distribution
    """
    Nr = u.shape[1]
    Nt = t.size

    assert u_t.shape[-1] == Nr
    assert u_t.shape[0] == Nt

    for it in prange(Nt):
        FFT_factor = np.exp(-1j * kz * c * t[it])
        for ir in range(Nr):
            u_t[it, ir] += np.real(u[:, ir] * FFT_factor).sum()
    return u_t

@njit
def get_temporal_slice2d(u, u_t, t, kz):
    """
    Resonstruct temporal-radial field distribution
    """
    _, Nx, Ny = u.shape
    Nt = t.size

    assert u_t.shape[-1] == Nx

    for it in prange(Nt):
        FFT_factor = np.exp(-1j * kz * c * t[it])
        for ix in range(Nx):
            u_t[it, ix] += np.real(u[:, ix, Ny//2-1] * FFT_factor).sum()

    return u_t

@njit
def get_temporal_3d(u, t, kz):
    """
    Resonstruct temporal-radial field distribution
    """
    _, Nx, Ny = u.shape
    Nt = t.size

    u_t = np.empty((Nt, Nx, Ny))

    for it in prange(Nt):
        FFT_factor = np.exp(-1j * kz * c * t[it])
        for ix in range(Nx):
            for iy in range(Ny):
                u_t[it, ix, iy] = np.real(u[:, ix, iy] * FFT_factor).sum()

    return u_t

def export_from_lasy(laser):
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
            np.transpose(laser.grid.field).copy()
        )

    return Container

#### FBPIC profile
@njit
def get_E_r(t, u, kz):
    #  print(t, u, kz)
    FFT_factor = (np.exp(-1j * kz * c * t) * np.ones_like(u).T).T
    u_r = np.real(u * FFT_factor).sum(0) # / FFT_factor.size
    return u_r


class LaserProfile( object ):

    def __init__( self, propagation_direction, gpu_capable=False ):
        assert propagation_direction in [-1, 1]
        self.propag_direction = float(propagation_direction)
        self.gpu_capable = gpu_capable

class AxipropLaser( LaserProfile ):

    def __init__( self, E0, u, kz, r, time_offset=0.0, z0=0.0,
                  theta_pol=0., lambda0=0.8e-6 ):

        LaserProfile.__init__(self, propagation_direction=1, gpu_capable=False)

        self.u = u
        self.kz = kz
        self.r = r
        self.time_offset = time_offset
        self.z0 = z0

        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)

    def E_field( self, x, y, z, t ):

        Ex = np.zeros_like(x)
        Ey = np.zeros_like(x)
        for iz in range(x.shape[0]):
            x_p, y_p, z_p = x[iz], y[iz], z[iz]
            z_p0 = z_p[0,0]
            u_r = get_E_r( -z_p0/c + self.z0/c + self.time_offset,
                            self.u, self.kz)
            fu = interp1d(self.r, u_r,  kind='cubic',
                      fill_value=0.0, bounds_error=False )

            r_p = np.sqrt(x_p*x_p + y_p*y_p)

            prof_p = fu(r_p)

            Ex[iz] = self.E0x * prof_p
            Ey[iz] = self.E0y * prof_p


        return( Ex.real, Ey.real )


class AxipropLaserAntenna( LaserProfile ):

    def __init__( self, E0, u, kz, r, time_offset=0.0, z0=0.0,
                  theta_pol=0., lambda0=0.8e-6 ):

        LaserProfile.__init__(self, propagation_direction=1, gpu_capable=False)

        self.u = u
        self.kz = kz
        self.r = r
        self.time_offset = time_offset
        self.z0 = z0

        self.E0x = E0 * np.cos(theta_pol)
        self.E0y = E0 * np.sin(theta_pol)

    def E_field( self, x, y, z, t ):
        if type(z) == np.ndarray:
            z_a = z[0]
        else:
            z_a = z

        if type(t) == np.ndarray:
            t_loc = t[0]
        else:
            t_loc = t

        u_r = get_E_r( t_loc - z_a/c + self.z0/c + self.time_offset,
                        self.u, self.kz)

        r_p = np.sqrt(x*x + y*y)
        fu = interp1d(self.r, u_r,  kind='cubic',
                      fill_value=0.0, bounds_error=False )
        profile = fu(r_p)
        Ex = self.E0x * profile
        Ey = self.E0y * profile
        return( Ex.real, Ey.real )

######## WARPX [WIP]

"""
The following methods are taken from the WarpX examples
https://github.com/ECP-WarpX/WarpX/tree/development/Examples/Modules/laser_injection_from_file
and all rights belong to WarpX development group Copyright (c) 2018
"""

def write_file_unf(fname, x, y, t, E):
    """ For a given filename fname, space coordinates x and y, time coordinate t
    and field E, write a WarpX-compatible input binary file containing the
    profile of the laser pulse. This function should be used in the case
    of a uniform spatio-temporal mesh
    """

    with open(fname, 'wb') as file:
        flag_unif = 1
        file.write(flag_unif.to_bytes(1, byteorder='little'))
        file.write((len(t)).to_bytes(4, byteorder='little', signed=False))
        file.write((len(x)).to_bytes(4, byteorder='little', signed=False))
        file.write((len(y)).to_bytes(4, byteorder='little', signed=False))
        file.write(t[0].tobytes())
        file.write(t[-1].tobytes())
        file.write(x[0].tobytes())
        file.write(x[-1].tobytes())
        if len(y) == 1 :
            file.write(y[0].tobytes())
        else :
            file.write(y[0].tobytes())
            file.write(y[-1].tobytes())
        file.write(E.tobytes())


def write_file(fname, x, y, t, E):
    """ For a given filename fname, space coordinates x and y, time coordinate t
    and field E, write a WarpX-compatible input binary file containing the
    profile of the laser pulse
    """

    with open(fname, 'wb') as file:
        flag_unif = 0
        file.write(flag_unif.to_bytes(1, byteorder='little'))
        file.write((len(t)).to_bytes(4, byteorder='little', signed=False))
        file.write((len(x)).to_bytes(4, byteorder='little', signed=False))
        file.write((len(y)).to_bytes(4, byteorder='little', signed=False))
        file.write(t.tobytes())
        file.write(x.tobytes())
        file.write(y.tobytes())
        file.write(E.tobytes())
