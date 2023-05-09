import numpy as np
from tqdm.auto import tqdm
from scipy.constants import e, m_e, c, pi, epsilon_0, fine_structure, mu_0
from mendeleev import element as table_element
from scipy.special import gamma as gamma_func
from numba import njit, prange

from .lib import PropagatorResampling
from .lib import PropagatorSymmetric

#njit = njit(parallel=True)
mc2_to_eV =  m_e*c**2/e


class StepperNonParaxialPlasma:
    """
    Class of steppers for the non-paraxial propagators with account
    for the plasma response. Contains methods to:
    - initiate the stepped propagation mode a single-step calculation;
    - perform a propagation step with account for the plasma refraction;
    - perform a propagation step with account for the arbitrary plasma current;

    This class should to be used to derive the actual Propagators
    by adding proper methods for the Transverse Spectral Transforms (TST).
    """

    def initiate_stepping(self, u):
        """
        Initiate the stepped propagation mode. This mode allows 
        computation of the consequent steps with access to the result on 
        each step.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """
        assert u.dtype == self.dtype

        self.stepping_image = self.bcknd.zeros( u.shape, self.dtype )
        self.z_propagation = 0.0

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:].copy())
            self.TST()
            self.stepping_image[ikz] = self.u_ht.copy()

    def stepping_step_kp2(self, kp2, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode with account 
        for the plasma refraction. This mode allows computation of the 
        consequent steps with access to the result on each step.

        Parameters
        ----------
        kp2: float (m^-2)
            Square of plasma wavenumber

        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = np.empty((self.Nkz, *self.shape_trns_new),
                              dtype=self.dtype)

        for ikz in range(self.Nkz):

            phase_term = self.kz[ikz]**2 - self.kr2 - kp2
            phase_term_mask = (phase_term>0.0)
            phase_term = self.bcknd.sqrt(
                self.bcknd.abs(phase_term) * phase_term_mask
            )
            self.stepping_image[ikz] *= self.bcknd.exp(1j * dz * phase_term)

            self.u_ht[:] = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out

    def stepping_step_withJ(self, kp2, J, dz, u_out=None, conservative=True):
        """
        Perform a step in the stepped propagation mode with account 
        for the arbitrary plasma current. This mode allows computation 
        of the consequent steps with access to the result on each step. 
        In contrast to `step` can operate the `PropagatorResampling` 
        class.

        Parameters
        ----------
        kp2: float (m^-2)
            Square of plasma wavenumber

        J: 2darray of complex (A)
            Plasma current in frequency space.

        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = np.empty((self.Nkz, *self.shape_trns_new),
                              dtype=self.dtype)

        for ikz in range(self.Nkz):

            phase_term = self.kz[ikz]**2 - self.kr2 - kp2
            phase_term_mask = (phase_term>0.0)
            phase_term = self.bcknd.sqrt(self.bcknd.abs(phase_term))
            k_ll_inv = phase_term_mask / phase_term
            phase_term *= phase_term_mask

            self.u_loc = self.bcknd.to_device(J[ikz,:].copy())
            self.TST_stepping()

            E_0 = self.stepping_image[ikz].copy()
            E_0_abs = self.bcknd.abs(E_0)

            Upsilon = -1j * self.kz[ikz] * c * mu_0 * self.u_ht - kp2 * E_0
            E_new = E_0 - 0.5j * k_ll_inv * dz * Upsilon

            if conservative:
                E_new_abs = self.bcknd.abs(E_new)
                E_new *= self.bcknd.divide_abs_or_set_to_one(
                    E_0_abs, E_new_abs)

            self.stepping_image[ikz] = E_new \
                * self.bcknd.exp(1j * dz * phase_term ) \

            self.u_ht[:] = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out


class PropagatorResamplingPlasma(
    PropagatorResampling, StepperNonParaxialPlasma):
    """
    A propagator with account for plasma response,
    based on `PropagatorResampling`
    """

    def init_TST_resampled(self):
        """
        Setup DHT transform
        """
        Nr = self.Nr
        Nr_new = self.Nr_new
        r = self.r
        r_new = self.r_new
        kr = self.kr
        dtype = self.dtype
        mode = self.mode

        invTM = jn(mode, r_new[:,None] * kr[None,:])
        self.TM_resampled = self.bcknd.inv_on_host(invTM, dtype)
        self.TM_resampled = self.bcknd.to_device(self.TM_resampled)
        self.TST_resampled_matmul = self.bcknd.make_matmul(
            self.TM_resampled, self.u_iht, self.u_ht)

    def TST_stepping(self):
        """
        Forward QDHT transform.
        """
        if not hasattr(self, 'TST_resampled_matmul'):
            self.init_TST_resampled()

        self.u_ht = self.TST_resampled_matmul(
            self.TM_resampled, self.u_loc, self.u_ht)


class PropagatorSymmetricPlasma(
    PropagatorSymmetric, StepperNonParaxialPlasma):
    """
    A propagator with account for plasma response,
    based on `PropagatorSymmetric`
    """
    TST_stepping = PropagatorSymmetric.TST


def make_ADK(my_element):
    """
    Prepare the useful data for ADK probabplity calculation for a given
    element. This part is mostly a copy-pased from FBPIC code
    [https://github.com/fbpic/fbpic]
    """
    r_e = e**2 / m_e / c**2 / 4 / pi /epsilon_0
    omega_a = fine_structure**3 * c / r_e
    Ea = m_e*c**2/e * fine_structure**4/r_e
    UH = table_element('H').ionenergies[1]

    Uion = np.array(list(table_element(my_element).ionenergies.values()))
    Z_states = np.arange( Uion.size ) + 1

    n_eff = Z_states * np.sqrt( UH/Uion )
    l_eff = n_eff[0] - 1

    C2 = 2**(2*n_eff) / (n_eff * gamma_func(n_eff+l_eff+1) * \
        gamma_func(n_eff-l_eff))

    adk_power = - (2*n_eff - 1)
    adk_prefactor = omega_a * C2 * ( Uion/(2*UH) ) \
                * ( 2 * (Uion/UH)**(3./2) * Ea )**(2*n_eff - 1)
    adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea
    return (adk_power, adk_prefactor, adk_exp_prefactor), Uion

@njit
def get_ADK_probability(E_fld, dt, adk_power, \
                        adk_prefactor, adk_exp_prefactor):
    """
    Calculate the probability of ADK ionization for each ion state from
    over the time `dt` in a static field `E_fld`
    """
    Propab = np.zeros(adk_power.size)
    if E_fld != 0.0:
        W_rate = adk_prefactor * E_fld**adk_power * \
            np.exp( adk_exp_prefactor/E_fld )
        Propab += 1. - np.exp( - W_rate * dt )
    return Propab

@njit #(parallel=True)
def get_plasma_OFI(E_laser, A_laser, dt, n_gas,
                   pack_ADK, Uion, Z_init=0, Zmax=-1,
                   ionization_current=False
                   ):

    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape

    J_plasma = np.zeros_like(E_laser)
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)

    for ir in range(Nr):

        n_gas_loc = n_gas[ir]

        E_loc = E_laser[:,ir].copy()
        U_loc = -e / (m_e * c) * A_laser[:,ir].copy()

        Gamma_loc = np.sqrt( 1 + U_loc * U_loc )
        Wth_loc = (Gamma_loc - 1) * mc2_to_eV
        beta_loc = U_loc / Gamma_loc

        Jt = np.zeros_like(E_loc)

        ion_fracts_loc = np.zeros(num_ions)
        ion_fracts_loc[Z_init] = 1.

        electron_temp_dens = 0
        all_new_events = 0

        for it in range(Nt):

            if np.abs(ion_fracts_loc[Zmax]-1)<1e-8:
                break

            probs = get_ADK_probability( np.abs( E_loc[it] ), dt, *pack_ADK )

            for ion_state in range(num_ions-1):
                new_events = ion_fracts_loc[ion_state] * probs[ion_state]
                if new_events>0:
                    ion_fracts_loc[ion_state+1] += new_events
                    ion_fracts_loc[ion_state] -= new_events

                    all_new_events += new_events
                    electron_temp_dens += Wth_loc[it] * new_events

                    if ionization_current:
                        J_ion = n_gas_loc * new_events * Uion[ion_state] \
                            * e / E_loc[it] / dt
                        Jt[it] += J_ion

                    Jt[it:] += -e * new_events * n_gas_loc * c * (beta_loc[it:] - beta_loc[it])

        if Z_init>0:
            Jt += -e * Z_init * n_gas_loc * c * beta_loc

        if all_new_events>0:
            T_e[ir] = electron_temp_dens/all_new_events
        else:
            T_e[ir] = 0.0

        n_e[ir] =  all_new_events * n_gas_loc
        J_plasma[:, ir] = Jt.copy()

    return J_plasma, n_e, T_e

def get_plasma_ionized(E_laser, A_laser, dt, n_pe):
    U_loc = e / (m_e * c) * A_laser.copy()
    Gamma_loc = np.sqrt( 1.0 + U_loc * U_loc )
    J_plasma = e * n_pe[None, :] * c * U_loc / Gamma_loc
    return J_plasma, n_pe, np.zeros_like(n_pe)
