import numpy as np
from tqdm.auto import tqdm
from scipy.constants import e, m_e, c, pi, epsilon_0, fine_structure
from mendeleev import element as table_element
from scipy.special import gamma as gamma_func
from numba import njit, prange

#njit = njit(parallel=True)

mc2_to_eV =  m_e*c**2/e

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
def get_plasma_OFI(E_laser, A_laser, dt, n_gas, pack_ADK, Uion, Zmax=-1):

    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape

    J_plasma = np.zeros_like(E_laser)
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)

    for ir in range(Nr):

        n_gas_loc = n_gas[ir]

        E_loc = E_laser[:,ir].copy()
        U_loc = e / (m_e * c) * A_laser[:,ir].copy()

        Gamma_loc = np.sqrt( 1 + U_loc * U_loc )
        Wth_loc = (Gamma_loc - 1) * mc2_to_eV
        beta_loc = U_loc / Gamma_loc

        Jt = np.zeros_like(E_loc)

        ion_fracts_loc = np.zeros(num_ions)
        ion_fracts_loc[0] = 1.

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

                    J_ion = n_gas_loc * new_events * Uion[ion_state] * e / E_loc[it] / dt
                    Jt[it] += J_ion
                    Jt[it:] += -n_gas_loc * new_events * e * c * (beta_loc[it:] - beta_loc[it])

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
    #beta_loc = U_loc / Gamma_loc
    #J_plasma = e * n_pe[None,:] * c * beta_loc
    J_plasma = e * n_pe[None,:] * c * e / (m_e * c) * A_laser.copy() #/ Gamma_loc
    print(Gamma_loc.max())
    return J_plasma, n_pe, np.zeros_like(n_pe)