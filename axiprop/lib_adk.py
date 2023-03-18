import numpy as np

from scipy.constants import fine_structure, e, m_e, c, epsilon_0, pi
from mendeleev import element as table_element
from scipy.special import gamma as gamma_func
from numba import njit

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

@njit
def get_ADK_heating(Et, At, dt, adk_power, adk_prefactor, adk_exp_prefactor):

        num_ions = adk_power.size + 1

        ion_fracts = np.zeros(num_ions)
        ion_fracts[0] = 1.

        electron_temp_dens = 0
        all_new_events = 0

        for it in range(Et.size-1):
            E_loc = np.abs(Et[it])
            A_loc = np.abs(At[it])

            probs = np.zeros(num_ions-1)
            if E_loc>0:
                W_rate = adk_prefactor * np.power(E_loc, adk_power) * \
                    np.exp( adk_exp_prefactor/E_loc )
                probs[:] = 1. - np.exp( - W_rate * dt )

            for ion_state in range(num_ions-1):
                new_events = ion_fracts[ion_state]*probs[ion_state]
                ion_fracts[ion_state+1] += new_events
                ion_fracts[ion_state] -= new_events

                if new_events>0:
                    all_new_events += new_events
                    electron_temp_dens += 0.5 * A_loc**2 * mc2_to_eV * new_events

        if all_new_events>0:
            Te = electron_temp_dens/all_new_events
        else:
            Te = 0.0

        return Te, ion_fracts

