import numpy as np
from scipy.constants import c
from scipy.special import j0, j1, jn_zeros


class Propagator:

    def __init__(self, Rmax, omega_width, Nr, Nkz, lambda0):

        self.Rmax = Rmax
        self.Nr = Nr
        self.Nkz = Nkz

        alpha = jn_zeros(0, Nr+1)
        self.alpha_np1 = alpha[-1]
        self.alpha = alpha[:-1]

        self.r = Rmax * self.alpha / self.alpha_np1
        self.kr = self.alpha/Rmax

        self.k0 = 2 * np.pi / lambda0
        self.omega0 = self.k0 * c
        self.omega_symm = 4 * omega_width * np.linspace(-1., 1., Nkz)
        self.omega = self.omega_symm + self.omega0
        self.kz = self.omega / c
        self.wvlgth = 2*np.pi / self.kz
        self.dtype = np.complex
        self.create_DHT()

    def create_DHT(self):

        self._j = np.abs(j1(self.alpha))/self.Rmax
        denominator = self.alpha_np1 * np.abs(j1(self.alpha[:,None])) \
                     * np.abs(j1(self.alpha[None,:]))
        self.TM = 2 * j0(self.alpha[:,None]*self.alpha[None,:]/self.alpha_np1) \
                   / denominator

    def DHT(self, u_in, u_out):

        u_out = np.dot(self.TM.astype(self.dtype), u_in/self._j, out=u_out)
        return u_out

    def iDHT(self, u_in, u_out):

        Nr_new = u_out.size
        u_out = np.dot(self.TM[:Nr_new].astype(self.dtype), u_in, out=u_out)
        u_out *= self._j[:Nr_new]
        return u_out

    def step(self, u, dz, Nr_new=None):

        Nz = u.shape[0]
        if Nr_new is None:
            Nr_new = self.Nr
        self.r_new = self.r[:Nr_new]
        assert (u.shape[1]==self.Nr)

        self.dtype = u.dtype
        self._j = self._j.astype(self.dtype)

        u_loc = np.zeros(self.Nr, dtype=self.dtype)
        u_ht = np.zeros(self.Nr, dtype=self.dtype)
        u_iht = np.zeros(Nr_new, dtype=self.dtype)

        for ikz in range(Nz):
            u_loc[:] = u[ikz,:]
            u_ht = self.DHT(u_loc, u_ht)
            u_ht *= np.exp( 1j * dz * np.sqrt(self.kz[ikz]**2 - self.kr**2) )
            u_iht = self.iDHT(u_ht, u_iht)
            u[ikz, :Nr_new] = u_iht

        u = u[:, :Nr_new]

        return u

    def steps(u, wvnum, dz=[], Nr_new=None):

        Nsteps = len(dz)
        Nz = u.shape[0]
        assert (u.shape[1]==self.Nr)
        self.dtype = u.dtype
        if Nr_new is None:
            Nr_new = self.Nr
        self.r_new = self.r[:Nr_new]

        u_loc = np.zeros(self.Nr, dtype=self.dtype)
        u_ht = np.zeros(self.Nr, dtype=self.dtype)
        u_iht = np.zeros(Nr_new, dtype=self.dtype)
        u_steps = np.empty((Nz, Nr_new, Nsteps), dtype=self.dtype)

        for ikz in range(Nz_loc):
            u_loc[:] = u[ikz,:]
            u_ht = self.DHT(u_loc, u_ht)
            k_loc = np.sqrt( wvnum[ikz]**2 - self.kr**2 )

            for i_step in range(Nsteps):
                u_ht = u_ht * np.exp( 1j * dz[i_step] *k_loc )
                u_iht = self.iDHT(u_ht, u_iht)
                u_steps[ikz, :, i_step] = u_iht

        return u_steps
