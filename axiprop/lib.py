import numpy as np
from scipy.constants import c
from scipy.special import j0, j1, jn_zeros
from scipy.linalg import pinv2

class PropagatorCommon:

    def init_kz(self, omega_width, Nkz, lambda0):

        self.k0 = 2 * np.pi / lambda0
        self.Nkz = Nkz
        self.omega0 = self.k0 * c
        self.omega_symm = 4 * omega_width * np.linspace(-1., 1., Nkz)
        self.omega = self.omega_symm + self.omega0
        self.kz = self.omega / c
        self.wvlgth = 2*np.pi / self.kz
        self.dtype = np.complex

    def step(self, u, dz):

        assert u.shape[0] == self.Nkz
        assert u.shape[1] == self.Nr
        assert self.dtype == u.dtype

        for ikz in range(self.Nkz):
            self.u_loc[:] = u[ikz,:]
            self.u_ht = self.DHT(self.u_loc, self.u_ht)
            self.u_ht *= np.exp(1j * dz * np.sqrt(self.kz[ikz]**2 - self.kr**2))
            self.u_iht = self.iDHT(self.u_ht, self.u_iht)
            u[ikz, :self.Nr_new] = self.u_iht

        u = u[:, :self.Nr_new]

        return u

    def steps(self, u, dz, verbose=True):

        Nsteps = len(dz)
        if Nsteps==0:
            return None

        assert u.shape[0] == self.Nkz
        assert u.shape[1] == self.Nr
        assert self.dtype == u.dtype

        u_steps = np.empty((self.Nkz, self.Nr_new, Nsteps), dtype=self.dtype)

        if verbose:
            print('Propagating:')

        for ikz in range(self.Nkz):
            self.u_loc[:] = u[ikz,:]
            self.u_ht = self.DHT(self.u_loc, self.u_ht)
            ik_loc = 1j * np.sqrt(self.kz[ikz]**2 - self.kr**2)
            for i_step in range(Nsteps):
                self.u_ht *= np.exp( dz[i_step] * ik_loc )
                self.u_iht = self.iDHT(self.u_ht, self.u_iht)
                u_steps[ikz, :, i_step] = self.u_iht

                if verbose:
                    print(f"Done step {i_step} of {Nsteps} "+ \
                          f"for wavelength {ikz+1} of {self.Nkz}",
                      end='\r', flush=True)

        return u_steps

class PropagatorSymmetric(PropagatorCommon):

    def __init__(self, Rmax, omega_width, Nr, Nkz, lambda0,
                 Nr_new=None, dtype=np.complex):

        self.init_kz(omega_width, Nkz, lambda0)
        self.init_rkr_and_DHT(Rmax, Nr, Nr_new, dtype)

    def init_rkr_and_DHT(self, Rmax, Nr, Nr_new, dtype):
        self.Rmax = Rmax
        self.Nr = Nr
        self.dtype = dtype

        alpha = jn_zeros(0, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        self.r = Rmax * alpha / alpha_np1
        self.kr = alpha/Rmax

        self._j = (np.abs(j1(alpha)) / Rmax).astype(dtype)
        denominator = alpha_np1 * np.abs(j1(alpha[:,None]) * j1(alpha[None,:]))
        self.TM = 2 * j0(alpha[:,None]*alpha[None,:]/alpha_np1) / denominator

        self.Nr_new = Nr_new
        if self.Nr_new is None:
            self.Nr_new = Nr

        self.r_new = self.r[:self.Nr_new]
        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr_new, dtype=dtype)

    def DHT(self, u_in, u_out):

        u_out = np.dot(self.TM.astype(self.dtype), u_in/self._j, out=u_out)
        return u_out

    def iDHT(self, u_in, u_out):

        u_out = np.dot(self.TM[:self.Nr_new].astype(self.dtype),
                       u_in, out=u_out)
        u_out *= self._j[:self.Nr_new]
        return u_out

class PropagatorResampling(PropagatorCommon):

    def __init__(self, Rmax, omega_width, Nr, Nkz, lambda0,
                 Rmax_new=None, Nr_new=None, dtype=np.complex):

        self.init_kz(omega_width, Nkz, lambda0)
        self.init_rkr_and_DHT(Rmax, Nr, Rmax_new, Nr_new, dtype)

    def init_rkr_and_DHT(self, Rmax, Nr, Rmax_new, Nr_new, dtype):
        self.Rmax = Rmax
        self.Nr = Nr
        self.dtype = dtype

        alpha = jn_zeros(0, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        self.r = Rmax * alpha / alpha_np1
        self.kr = alpha/Rmax

        self.Rmax_new = Rmax_new
        if self.Rmax_new is None:
            self.Rmax_new = Rmax

        self.Nr_new = Nr_new
        if self.Nr_new is None:
            self.Nr_new = Nr
        self.r_new = np.linspace(0, self.Rmax_new, self.Nr_new)

        invTM = j0(self.r[:,None] * self.kr[None,:])
        self.TM = pinv2(invTM, check_finite=False)
        self.invTM = j0(self.r_new[:,None] * self.kr[None,:])

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr_new, dtype=dtype)

    def DHT(self, u_in, u_out):

        u_out = np.dot(self.TM.astype(self.dtype), u_in, out=u_out)
        return u_out

    def iDHT(self, u_in, u_out):

        u_out = np.dot(self.invTM.astype(self.dtype), u_in, out=u_out)
        return u_out