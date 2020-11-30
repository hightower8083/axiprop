# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop utils file

This file contains utility methods for Axiprop tool
"""
import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange

try:
    from numba import njit, prange
    njit = njit(parallel=True, fastmath=True)
except Exception:
    prange = range
    def njit(func):
        print("This function is greatly accelerated if Numba is installed")
        return func

def laser_from_fu(fu, kz, r, normalize=False):
    """
    Generate array with spectra-radial field
    distribution with the pre-defined function
    """
    a0 = fu( kz[:,None], r[None,:] )

    if normalize:
        a0 /= (np.abs(a0)**2).sum(0).max()**0.5

    return a0

def mirror_parabolic(f0, kz, r):
    """
    Generate array with spectra-radial phase
    representing the on-axis Parabolic Mirror
    """
    s_ax = r**2/4/f0
    return np.exp(-2j * s_ax[None,:] * kz[:,None])

@njit
def get_temporal_onaxis(time_ax, freq, A_freqR, A_temp):
    """
    Resonstruct temporal-radial field distribution
    """
    A_temp[:] = 0.0
    Nw_loc = A_freqR.shape[0]
    Nr_loc = A_freqR.shape[1]
    Nt_loc = time_ax.size

    for it in prange(Nt_loc):
        propag = np.exp(-1j*freq*time_ax[it])
        for ir in range(Nr_loc):
            A_temp[it] += np.real(A_freqR[:,ir] * propag).sum()

    return A_temp
