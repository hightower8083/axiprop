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

def laser_from_fu(fu, r, kz, normalize=False):
    """
    Generate array with spectra-radial field
    distribution with the pre-defined function
    """
    a0 = fu( r[None,:], kz[:,None] )

    if normalize:
        a0 /= (np.abs(a0)**2).sum(0).max()**0.5

    return a0

def mirror_parabolic(f0, r, kz):
    """
    Generate array with spectra-radial phase
    representing the on-axis Parabolic Mirror
    """
    s_ax = r**2/4/f0
    return np.exp(-2j * s_ax[None,:] * kz[:,None])

def mirror_axiparabola(f0, d0, r, kz):
    """
    Generate array with spectra-radial phase representing
    the on-axis Axiparabola with analytic expression (see
    Eq. (4) in [Smartsev et al Opt. Lett. 44, 3414 (2019)])
    """
    s_ax = r**2/4/f0 - d0/(8*f0**2*Rmax**2)*r**4 \
         + d0*(Rmax**2+8*f0*d0)/(96*f0**4*Rmax**4)*r**6

    return np.exp(-2j * s_ax[None,:] * kz[:,None])

def mirror_axiparabola2(f0, d0, r, kz):
    """
    Generate array with spectra-radial phase representing
    the on-axis Axiparabola solving sag-equation numerically
    (see Eq. (2) in [Smartsev et al Opt. Lett. 44, 3414 (2019)])
    """
    sag_equation = lambda r, s : (s - (f0 + d0 * np.sqrt(r/Rmax)) +
            np.sqrt(r**2 + ((f0 + d0 * np.sqrt(r/Rmax) - s)**2))/r)

    s_ax = solve_ivp( sag_equation,
                      (r[0], r[-1]),
                      [r[0]/(4*f0),],
                      t_eval=r
                    ).y.flatten()

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
