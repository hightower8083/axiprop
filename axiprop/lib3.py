import numpy as np
from scipy.constants import c
from scipy.integrate import solve_ivp
from scipy.special import j0, j1, jn_zeros
from numba import njit, prange


def init_freq_axis(lam0, freq_width, Nfreq):

    k_las = 2 * np.pi / lam0
    freq_las = k_las * c
    freq_symm = 4 * freq_width * np.linspace(-1, 1, Nfreq)
    freq = freq_symm + freq_las
    wvnum = freq / c
    wvlgth = 2*np.pi / wvnum
    return freq, wvnum, wvlgth, freq_symm

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


class Propagator:
    def __init__(self, Nr, Rmax):

        alpha = jn_zeros(0, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        self.Nr = Nr
        self.Rmax = Rmax
        self.r = Rmax * alpha/alpha_np1
        self.kr = alpha/Rmax
        self._j = np.abs(j1(alpha))/Rmax

        denominator = alpha_np1 * np.abs(j1(alpha[:,None]) * j1(alpha[None,:]))
        self.TM = 2 * j0(alpha[:,None]* alpha[None,:]/alpha_np1) / denominator

    def step(self, u, dz, wvnum, Nr_new=None):

        Nz = u.shape[0]
        assert (u.shape[1]==self.Nr)
        udtype = u.dtype
        if Nr_new is None:
            Nr_new = self.Nr

        u_loc = np.zeros(self.Nr, dtype=udtype)
        u_ht = np.zeros(u_loc, dtype=udtype)
        u_iht = np.zeros(Nr_new, dtype=udtype)

        for ikz in range(Nz):
            u_loc[:] = u[ikz,:]

            u_ht = np.dot(self.TM.astype(udtype),
                          u_loc/self._j.astype(udtype),
                          out=u_ht)
            u_ht *= np.exp( 1j * dz * np.sqrt(wvnum[ikz]**2 - self.kr**2) )
            u_iht = np.dot(self.TM[:Nr_new].astype(udtype), u_ht, out=u_iht)
            u[ikz, :Nr_new] = u_iht * self._j[:Nr_new].astype(udtype)

        u = u[:, :Nr_new]

        return u

    def steps(u, wvnum, dz=[], Nr_new=None):

        Nsteps = len(dz)
        Nz = u.shape[0]
        assert (u.shape[1]==self.Nr)
        udtype = u.dtype
        if Nr_new is None:
            Nr_new = self.Nr

        u_loc = np.zeros(self.Nr, dtype=udtype)
        u_ht = np.zeros(self.Nr, dtype=udtype)
        u_iht = np.zeros(Nr_new, dtype=udtype)
        u_steps = np.empty((Nz, Nr_new, Nsteps), dtype=udtype)

        for ikz in range(Nz_loc):
            u_loc[:] = u[ikz,:]
            u_ht = np.dot(self.TM.astype(udtype),
                          u_loc/self._j.astype(udtype),
                          out=u_ht)

            k_loc = np.sqrt( wvnum[ikz]**2 - self.kr**2 )

            for i_step in range(Nsteps):
                u_ht = u_ht * np.exp( 1j * dz[i_step] *k_loc )
                u_iht = np.dot(self.TM[:Nr_new].astype(udtype), u_ht, out=u_iht)
                u_steps[ikz, :, i_step] = u_iht * self._j.astype(udtype)[:Nr_new]

        return u_steps
