# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop utils file

This file contains utility methods for Axiprop tool
"""
import numpy as np
from numba import njit, prange
from scipy.constants import c

# try import numba and make dummy methods if it is not
try:
    from numba import njit, prange
    njit = njit(parallel=True, fastmath=True)
except Exception:
    prange = range
    def njit(func):
        def func_wrp(*args, **kw_args):
            print(f"Install Numba to get `{func.__name__}` " + \
                   "function greatly accelerated")
            return func(*args, **kw_args)
        return func_wrp

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
    s_ax = r**2/4/f0

    val = np.exp(-2j * s_ax[None,:] * \
                 ( kz * np.ones((*kz.shape, *r.shape)).T ).T)
    return val

@njit
def get_temporal_1d(u, u_t, t, kz, Nr_loc):
    """
    Resonstruct temporal-radial field distribution
    """
    Nkz, Nr = u.shape
    Nt = t.size

    for it in prange(Nt):
        FFT_factor = np.exp(-1j * kz * c * t[it])
        for ir in range(Nr_loc):
            u_t[it] += np.real(u[:,ir] * FFT_factor).sum()

    return u_t
