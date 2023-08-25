import numpy as np
from numba import njit, prange
from scipy.constants import e, m_e, c, pi, mu_0
from scipy.constants import epsilon_0, fine_structure

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

@njit(parallel=True)
def get_plasma_ADK( E_laser, A_laser, dt, n_gas, pack_ADK, Uion,
                    Z_init, Zmax, ionization_current ):

    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape

    J_plasma = np.zeros_like(E_laser)
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)

    for ir in prange(Nr):

        n_gas_loc = n_gas[ir]

        E_ir_t = E_laser[:,ir].copy()
        A_ir_t = A_laser[:,ir].copy()

        J_ir_t = np.zeros_like(E_ir_t)

        P_loc = e * A_ir_t
        v_loc = P_loc / m_e / \
            np.sqrt( 1 + 0.5 * np.abs(P_loc)**2 / (m_e*c)**2 )

        if Z_init>0:
            J_ir_t += -e * Z_init * n_gas_loc * v_loc

        ion_fracts_loc = np.zeros(num_ions)
        ion_fracts_loc[Z_init] = 1.
        all_new_events = 0

        for it in range(Nt):

            if np.abs(ion_fracts_loc[Zmax]-1)<1e-8:
                break

            probs = get_ADK_probability( np.abs( E_ir_t[it] ), dt, *pack_ADK )

            v_follow = P_loc[it:] / m_e / \
                np.sqrt( 1 + 0.5 * np.abs(P_loc[it:])**2 / (m_e*c)**2 )

            for ion_state in range(num_ions-1):
                new_events = ion_fracts_loc[ion_state] * probs[ion_state]
                if new_events>0:
                    ion_fracts_loc[ion_state+1] += new_events
                    ion_fracts_loc[ion_state] -= new_events
                    all_new_events += new_events

                    if ionization_current:
                        J_ion = n_gas_loc * new_events \
                            * Uion[ion_state] / E_ir_t[it] / dt
                        J_ir_t[it] += J_ion

                    J_ir_t[it:] += -e * new_events * n_gas_loc * v_follow

        n_e[ir] =  all_new_events * n_gas_loc
        J_plasma[:, ir] = J_ir_t.copy()

    return J_plasma, n_e, T_e

@njit(parallel=True)
def get_plasma_ADK_OFI(
    E_laser, A_laser, t_axis, omega0, n_gas, pack_ADK, Uion,
    Z_init=0, Zmax=-1, ionization_current=True ):

    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape

    ion_fracts_loc = np.zeros(num_ions)
    Xi = np.zeros( (Nr, num_ions) )
    J_plasma = np.zeros_like(E_laser)
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)

    dt = t_axis[1] - t_axis[0]
    phase_env = np.exp(-1j * omega0 * t_axis)
    phase_env_inv = np.exp(1j * omega0 * t_axis)

    for ir in prange(Nr):

        n_gas_ir = n_gas[ir]

        E_ir_t = E_laser[:,ir].copy()
        A_ir_t = A_laser[:,ir].copy()

        E_ir_t_re = np.real( E_ir_t * phase_env )
        A_ir_t_re = np.real( A_ir_t * phase_env )

        P_ir_env = np.abs( e * A_ir_t )
        P_ir_re = e * A_ir_t_re
        v_ir_re = P_ir_re / m_e / \
            np.sqrt( 1 + 0.5 * P_ir_env**2 / (m_e*c)**2 )

        J_ir_t_re = np.zeros_like(E_ir_t_re)

        if Z_init>0:
            J_ir_t_re += -e * Z_init * n_gas_ir * v_ir_re

        ion_fracts_loc[:] = 0.0
        ion_fracts_loc[Z_init] = 1.

        electron_temp_dens = 0
        all_new_events = 0

        for it in range(Nt):

            if np.abs(ion_fracts_loc[Zmax]-1)<1e-8:
                break

            probs = get_ADK_probability(
                np.abs( E_ir_t_re[it] ), dt, *pack_ADK )

            P0 = -P_ir_re[it]

            W_ir = m_e * c**2 * (
                np.sqrt( 1 + P0**2 / (m_e*c)**2 ) - 1.0
            )

            P_follow_re = P_ir_re[it:] + P0
            P_follow_env = P_ir_env[it:]
            v_follow = P_follow_re / m_e / \
                np.sqrt( 1 + 0.5 * P_follow_env**2 / (m_e*c)**2 )

            for ion_state in range(num_ions-1):
                new_events = ion_fracts_loc[ion_state] * probs[ion_state]
                if new_events>0:
                    ion_fracts_loc[ion_state+1] += new_events
                    ion_fracts_loc[ion_state] -= new_events

                    all_new_events += new_events
                    electron_temp_dens += W_ir * new_events

                    if ionization_current:
                        J_ir_t_re[it] += n_gas_ir * new_events \
                            * Uion[ion_state] * e / E_ir_t_re[it] / dt

                    J_ir_t_re[it:] += -e * new_events * n_gas_ir * v_follow

        if all_new_events>0:
            T_e[ir] = electron_temp_dens/all_new_events
        else:
            T_e[ir] = 0.0

        n_e[ir] =  all_new_events * n_gas_ir
        Xi[ir, :] += ion_fracts_loc

        J_ir_t = 2 * phase_env_inv * J_ir_t_re
        J_plasma[:, ir] = J_ir_t.copy()

    return J_plasma, n_e, T_e, Xi
