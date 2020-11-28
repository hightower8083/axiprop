import numpy as np
from scipy.integrate import solve_ivp
from numba import njit, prange


def mirror_analytic(f0, d0, r, Rmax, wvnum, freq_symm,
                            a_hole=None, tau_ret=None):

    s_ax = r**2/4/f0 - d0/(8*f0**2*Rmax**2)*r**4 \
        + d0*(Rmax**2+8*f0*d0)/(96*f0**4*Rmax**4)*r**6

    phase_on_mirror = -2 * s_ax[None,:] * wvnum[:,None]
    phase_on_mirror = np.exp(1j*phase_on_mirror)
    return phase_on_mirror

def mirror_numeric(f0, d0, r, Rmax, wvnum, freq_symm,
                         a_hole=False, tau_ret=None):

    sag_equation = lambda r, s : (s - (f0 + d0 * np.sqrt(r/Rmax)) +
            np.sqrt(r**2 + ((f0 + d0 * np.sqrt(r/Rmax) - s)**2))/r)

    s_ax = solve_ivp( sag_equation,
                      (r[0], r[-1]),
                      [r[0]/(4*f0),],
                      t_eval=r
                    ).y.flatten()

    phase_on_mirror = -2 * s_ax[None,:] * wvnum[:,None]
    phase_on_mirror = np.exp(1j*phase_on_mirror)

    return phase_on_mirror

@njit(parallel=True, fastmath=True)
def get_temporal_onaxis(time_ax, freq, A_freqR, A_temp):

    A_temp[:] = 0.0
    Nw_loc = A_freqR.shape[0]
    Nr_loc = A_freqR.shape[1]
    Nt_loc = time_ax.size

    for it in prange(Nt_loc):
        propag = np.exp(-1j*freq*time_ax[it])
        for ir in range(Nr_loc):
            A_temp[it] += np.real(A_freqR[:,ir] * propag).sum()

    return A_temp
