import numpy as np
from scipy.constants import c, e, m_e, epsilon_0
from scipy.integrate import solve_ivp
from scipy.special import j0, j1, jn_zeros
from scipy.linalg import pinv2
from numba import njit, prange

def init_radial_axes(Nr, Rmax, method='j_zeros'):
    if method=='j_zeros':
        alphas = jn_zeros(0, Nr+1)
        alpha_np1 = alphas[-1]
        alphas = alphas[:-1]
        r_ax = Rmax * alphas/alpha_np1
        
    elif method=='uniform':
        alphas = jn_zeros(0, Nr)
        r_ax = Rmax * np.r_[0:1:Nr*1j]

    kr_ax = alphas/Rmax

    return r_ax, kr_ax

def init_freq_axis(lam0, freq_width, Nfreq):
    k_las = 2 * np.pi / lam0
    freq_las = k_las * c
    freq_symm = 4 * freq_width * np.linspace(-1, 1, Nfreq)

    freq = freq_symm + freq_las
    wvnum = freq / c
    wvlgth = 2*np.pi / wvnum
    
    return freq, wvnum, wvlgth, freq_symm

def get_mirror_phase_approx(f0, d0, r, Rmax, wvnum, freq_symm, a_hole=None, tau_ret=None):

    s_ax = r**2/4/f0 - d0/(8*f0**2*Rmax**2)*r**4 \
        + d0*(Rmax**2+8*f0*d0)/(96*f0**4*Rmax**4)*r**6
    
    phase_on_mirror = -2 * s_ax[None,:] * wvnum[:,None]
    phase_on_mirror = np.exp(1j*phase_on_mirror)

    return phase_on_mirror


def get_mirror_phase_num(f0, d0, r, Rmax, wvnum, freq_symm, a_hole=False, tau_ret=None):

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

def dht_init(r_start, r_end, k_r):
    if r_start.size<r_end.size:
        print('The output Nr should not exceed the input one (memory size)')
        return None, None
    
    Nr_start = r_start.size
    alphas = jn_zeros(0, Nr_start)
    jp1 = np.abs(j1(alphas))

    Rmax_start = r_start.max()
    invTM = j0( r_start[:,None] * k_r[None, :] ) \
        / ( jp1[None, :]**2 * Rmax_start * np.pi)

    TM = pinv2(invTM,check_finite=False)

    Rmax_end = r_end.max()    
    invTM = j0( r_end[:,None] * k_r[None, :] ) \
        / ( jp1[None, :]**2 * Rmax_end * np.pi)
    
    return TM, invTM

def dht_propagate_single(u, wvnum, dz_prop, TM, invTM, k_r):
    Nr_end = invTM.shape[0]
    Nz = u.shape[0]
    u_loc = np.zeros(u.shape[1], dtype=u.dtype)
    u_ht = np.zeros_like(u_loc)
    u_iht = np.zeros(Nr_end, dtype=u.dtype)
    
    for ikz in range(Nz):
        u_loc[:] = u[ikz,:]
        u_ht = np.dot(TM.astype(u_loc.dtype), u_loc, out=u_ht)
        u_ht *= np.exp( 1j * dz_prop * np.sqrt(wvnum[ikz]**2 - k_r**2) )
        u_iht = np.dot(invTM.astype(u_ht.dtype), u_ht, out=u_iht)
        u[ikz, :Nr_end] = u_iht
        
    u = u[:, :Nr_end]

    return u

def dht_propagate_multi(u, wvnum, dz_steps, TM, invTM, k_r):
    Nsteps = len(dz_steps)
    Nr_loc = invTM.shape[0]
    Nr_in = u.shape[1]
    Nz_loc = u.shape[0]
    
    u_multi = np.empty((Nz_loc, Nr_loc, Nsteps), dtype=u.dtype)
    u_loc = np.zeros(Nr_in, dtype=u.dtype)
    u_ht = np.zeros(Nr_in, dtype=u.dtype)
    u_iht = np.zeros(Nr_loc, dtype=u.dtype)
    
    for ikz in range(Nz_loc):
        u_loc[:] = u[ikz,:]
        u_ht = np.dot(TM.astype(u_loc.dtype), u_loc, out=u_ht)
        
        k_loc = np.sqrt(wvnum[ikz]**2-k_r**2)
        
        for i_step in range(Nsteps):
            u_ht = u_ht * np.exp( 1j * dz_steps[i_step] *k_loc )
            u_iht = np.dot(invTM.astype(u_ht.dtype), u_ht, out=u_iht)
            u_multi[ikz, :, i_step] = u_iht

    return u_multi

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
