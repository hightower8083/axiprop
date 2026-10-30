import numpy as np
from numba import jit, prange
from scipy.constants import e, m_e, c

def wake_kernel_integrate(kernel_t, k_p, xi_ax):
    delta_n = np.zeros_like(kernel_t)
    dxi = xi_ax[1] - xi_ax[0]
    for it in range(1, xi_ax.size):
        xi = xi_ax[it]
        delta_n[it] = dxi * np.sum(
            kernel_t[:it] * np.sin( k_p * (xi - xi_ax[:it, None]) ),
            axis=0
        )
    return delta_n

def laplacian_fd(A, xi_ax, r_ax):
    r_inv = 1.0 / r_ax
    dxi_inv = 1.0 / ( xi_ax[1] - xi_ax[0] )
    dxi2_inv = dxi_inv * dxi_inv
    dr_inv = 1.0 / ( r_ax[1] - r_ax[0] )
    dr2_inv = dr_inv * dr_inv

    nabla2_A = np.zeros_like(A)

    nabla2_A[1:-1, 1:-1] = \
        dxi2_inv * (A[2:, 1:-1] - 2* A[1:-1, 1:-1] + A[:-2, 1:-1]) \
      + dr2_inv * (A[1:-1, 2:] - 2* A[1:-1, 1:-1] + A[1:-1, :-2,]) \
      + r_inv[None, 1:-1] * 0.5 * dr_inv * (A[1:-1, 2:] - A[1:-1, :-2] )

    nabla2_A[:,0] = nabla2_A[:,1]

    return nabla2_A


@jit(nopython=True, cache=True)
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

@jit(nopython=True, cache=True, parallel=True)
def get_plasma_ADK( E_laser, A_laser, dt, n_gas, pack_ADK, Uion,
                    Z_init, Zmax, ionization_current ):
    """Cycle-averaged ADK plasma response for a complex envelope."""
    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape
    z_stop = num_ions - 1 if Zmax < 0 else Zmax

    J_plasma = np.zeros_like(E_laser)
    n_e = np.zeros(Nr)
    IonizationEnergy = np.zeros(Nr)

    for ir in prange(Nr):
        n_gas_loc = n_gas[ir]
        E_ir_t = E_laser[:, ir].copy()
        A_ir_t = A_laser[:, ir].copy()
        P_loc = e * A_ir_t
        v_loc = P_loc / m_e / np.sqrt(
            1.0 + 0.5 * np.abs(P_loc)**2 / (m_e*c)**2
        )

        ion_fracts_loc = np.zeros(num_ions)
        ion_fracts_loc[Z_init] = 1.0
        n_e_slice = np.full(Nt, Z_init * n_gas_loc)
        J_ion_t = np.zeros_like(E_ir_t)

        for it in range(Nt):
            if ion_fracts_loc[z_stop] >= 1.0 - 1.0e-16:
                break
            probs = get_ADK_probability(np.abs(E_ir_t[it]), dt, *pack_ADK)
            for ion_state in range(z_stop):
                fraction = ion_fracts_loc[ion_state]
                if fraction < 1.0e-16:
                    continue
                new_events = fraction * probs[ion_state]
                if new_events <= 0.0:
                    continue
                ion_fracts_loc[ion_state + 1] += new_events
                ion_fracts_loc[ion_state] -= new_events
                if ionization_current and np.abs(E_ir_t[it]) > 0.0:
                    J_ion_t[it] += (n_gas_loc * new_events * Uion[ion_state]
                                    / E_ir_t[it] / dt)
                n_e_slice[it:] += new_events * n_gas_loc

        J_ir_t = -e * n_e_slice * v_loc + J_ion_t
        n_e[ir] = n_e_slice[-1]
        J_plasma[:, ir] = J_ir_t
        IonizationEnergy[ir] = 0.5 * np.real(
            np.sum(J_ir_t * np.conj(E_ir_t))
        ) * dt

    return J_plasma, n_e, IonizationEnergy


@jit(nopython=True, cache=True, parallel=True)
def get_OFI_heating(
    E_laser, P_laser, t_axis, omega0,
    n_gas, pack_ADK,
    Z_init=0, Zmax=-1 ):

    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape

    Xi = np.zeros( (Nr, num_ions) )
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)

    dt = t_axis[1] - t_axis[0]
    phase_env = np.exp(-1j * omega0 * t_axis)

    for ir in prange(Nr):

        ion_fracts_loc = np.zeros(num_ions)
        n_gas_ir = n_gas[ir]

        E_ir_t = E_laser[:,ir].copy()
        P_ir_t = P_laser[:,ir].copy()

        E_ir_t_re = np.real( E_ir_t * phase_env )
        P_ir_t_re = np.real( P_ir_t * phase_env )

        ion_fracts_loc[:] = 0.0
        ion_fracts_loc[Z_init] = 1.

        electron_temp_dens = 0.0
        all_new_events = 0.0

        for it in range(Nt):

#            if np.abs(ion_fracts_loc[Zmax]-1)<1e-8:
#                break

            probs = get_ADK_probability(
                np.abs( E_ir_t_re[it] ), dt, *pack_ADK )

            W_ir = m_e * c**2 * (
                np.sqrt( 1 + P_ir_t_re[it]**2 / (m_e*c)**2 ) - 1.0
            )

            for ion_state in range(num_ions-1):
                new_events = ion_fracts_loc[ion_state] * probs[ion_state]
                if new_events>0:
                    ion_fracts_loc[ion_state+1] += new_events
                    ion_fracts_loc[ion_state] -= new_events

                    all_new_events += new_events
                    # ideal electrons gas is assumed so `E = 3/2 k_B T`
                    electron_temp_dens += 2. / 3. * W_ir * new_events

        if all_new_events>0:
            T_e[ir] = electron_temp_dens / all_new_events
        else:
            T_e[ir] = 0.0

        n_e[ir] = all_new_events * n_gas_ir
        Xi[ir, :] += ion_fracts_loc

    return n_e, T_e, Xi

@jit(nopython=True, cache=True, parallel=True)
def get_plasma_ADK_ref( E_laser, A_laser, t_axis, omega0,
                        n_gas, pack_ADK, Uion,
                        Z_init, Zmax, ionization_current ):
    """Carrier-resolved ADK plasma response on a real optical field."""
    num_ions = pack_ADK[0].size + 1
    Nt, Nr = E_laser.shape
    dt = t_axis[1] - t_axis[0]
    phase_env = np.exp(-1j * omega0 * t_axis)
    z_stop = num_ions - 1 if Zmax < 0 else Zmax

    J_loc_t_re = np.zeros((Nt, Nr), dtype=np.double)
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)
    Xi = np.zeros((Nr, num_ions))

    for ir in prange(Nr):
        n_gas_loc = n_gas[ir]
        E_ir_t_re = np.real(E_laser[:, ir] * phase_env)
        A_ir_t_re = np.real(A_laser[:, ir] * phase_env)
        P_ir_t_re = e * A_ir_t_re
        v_ir_t_re = P_ir_t_re / m_e / np.sqrt(
            1.0 + P_ir_t_re**2 / (m_e*c)**2
        )

        ion_fracts_loc = np.zeros(num_ions)
        ion_fracts_loc[Z_init] = 1.0
        n_e_slice = np.full(Nt, Z_init * n_gas_loc)
        J_ion_t_re = np.zeros(Nt)
        all_new_events = 0.0
        electron_temp_dens = 0.0

        for it in range(Nt):
            if ion_fracts_loc[z_stop] >= 1.0 - 1.0e-16:
                break
            probs = get_ADK_probability(np.abs(E_ir_t_re[it]), dt, *pack_ADK)
            W_ir = m_e * c**2 * (
                np.sqrt(1.0 + P_ir_t_re[it]**2 / (m_e*c)**2) - 1.0
            )
            for ion_state in range(z_stop):
                fraction = ion_fracts_loc[ion_state]
                if fraction < 1.0e-16:
                    continue
                new_events = fraction * probs[ion_state]
                if new_events <= 0.0:
                    continue
                ion_fracts_loc[ion_state + 1] += new_events
                ion_fracts_loc[ion_state] -= new_events
                all_new_events += new_events
                electron_temp_dens += 2.0 / 3.0 * W_ir * new_events
                if ionization_current and np.abs(E_ir_t_re[it]) > 0.0:
                    J_ion_t_re[it] += (
                        n_gas_loc * new_events * Uion[ion_state]
                        / E_ir_t_re[it] / dt
                    )
                n_e_slice[it:] += new_events * n_gas_loc

        J_ir_t_re = -e * n_e_slice * v_ir_t_re + J_ion_t_re
        n_e[ir] = n_e_slice[-1]
        J_loc_t_re[:, ir] = J_ir_t_re
        if all_new_events > 0.0:
            T_e[ir] = electron_temp_dens / all_new_events
        Xi[ir, :] = ion_fracts_loc

    return J_loc_t_re, n_e, T_e, Xi

# --- Finite-frequency PPT ionization ---------------------------------------
# Append this block to axiprop/simulation/inline_methods.py.  That module
# already imports numpy, numba.jit/prange and scipy.constants e, m_e, c.
from scipy.special import dawsn as _scipy_dawsn

_DAWSON_STEP = 1.0e-3
_DAWSON_XMAX = 40.0
_DAWSON_TABLE = _scipy_dawsn(
    np.arange(int(_DAWSON_XMAX / _DAWSON_STEP) + 1) * _DAWSON_STEP
)


@jit(nopython=True, cache=True)
def _dawson_numba(x):
    """Dawson integral D(x), x >= 0, using a host-generated lookup table."""
    if x <= 0.0:
        return 0.0
    if x < _DAWSON_XMAX:
        q = x / _DAWSON_STEP
        i = int(q)
        f = q - i
        return _DAWSON_TABLE[i] * (1.0 - f) + _DAWSON_TABLE[i + 1] * f

    # Asymptotic expansion, used only in the far tail.
    inv_x2 = 1.0 / (x * x)
    return 0.5 / x * (1.0 + 0.5*inv_x2 + 0.75*inv_x2**2
                      + 1.875*inv_x2**3 + 6.5625*inv_x2**4
                      + 29.53125*inv_x2**5)


@jit(nopython=True, cache=True)
def _ppt_g(gamma):
    """Finite-frequency correction g(gamma) in the PPT tunnelling exponent."""
    if gamma < 1.0e-3:
        g2 = gamma * gamma
        return (1.0 - g2 / 10.0 + 9.0 * g2**2 / 280.0
                - 5.0 * g2**3 / 336.0 + 35.0 * g2**4 / 4224.0)
    return (1.5 / gamma) * (
        (1.0 + 0.5 / gamma**2) * np.arcsinh(gamma)
        - np.sqrt(1.0 + gamma**2) / (2.0 * gamma)
    )


@jit(nopython=True, cache=True)
def _ppt_alpha(gamma):
    """PPT photon-channel spacing parameter alpha(gamma)."""
    if gamma < 1.0e-2:
        g2 = gamma * gamma
        return gamma**3 * (
            2.0 / 3.0 - 3.0 * g2 / 5.0 + 15.0 * g2**2 / 28.0
            - 35.0 * g2**3 / 72.0 + 315.0 * g2**4 / 704.0
        )
    return 2.0 * (np.arcsinh(gamma)
                  - gamma / np.sqrt(1.0 + gamma**2))


@jit(nopython=True, cache=True)
def _ppt_a0(gamma, nu):
    """PPT A_m factor for m=0.

    For gamma < 0.4, use the continuum approximation to the discrete photon
    sum.  At the switch the difference from the explicit sum is below 0.25%
    (and decreases as gamma decreases).  Above the switch, sum over integer
    photon channels explicitly, beginning at ceil(nu).
    """
    if gamma <= 1.0e-8:
        return 1.0

    alpha = _ppt_alpha(gamma)
    beta = 2.0 * gamma / np.sqrt(1.0 + gamma**2)

    if gamma < 0.4:
        # Integral approximation to sum exp[-alpha q] D(sqrt(beta q)).
        return (2.0 / np.sqrt(3.0)) * gamma**2 / (1.0 + gamma**2) \
            * np.sqrt(beta) / (np.sqrt(alpha) * (alpha + beta))

    delta = np.ceil(nu) - nu
    total = 0.0
    q = delta
    while alpha * q < 36.0:
        total += np.exp(-alpha * q) * _dawson_numba(np.sqrt(beta * q))
        q += 1.0

    return (4.0 / np.sqrt(3.0 * np.pi)) * gamma**2 \
        / (1.0 + gamma**2) * total


@jit(nopython=True, cache=True)
def get_PPT_rate(E_fld, omega_au, Ip_au, n_eff, C2, Ea, omega_a):
    """Cycle-averaged single-electron PPT rate in s^-1.

    E_fld and Ea are SI electric fields; omega_au, Ip_au, n_eff and C2
    are atomic-unit parameters.  This is the linearly-polarized, m=0,
    effective-hydrogenic PPT rate used by both envelope and refined models.
    """
    if E_fld <= 0.0:
        return 0.0

    F = E_fld / Ea
    kappa = np.sqrt(2.0 * Ip_au)
    gamma = omega_au * kappa / F
    E0 = (2.0 * Ip_au)**1.5

    exponent = -(2.0 * E0 / (3.0 * F)) * _ppt_g(gamma)
    if exponent < -745.0:
        return 0.0

    nu = 0.0
    if gamma >= 0.4:
        nu = (Ip_au / omega_au) * (1.0 + 1.0 / (2.0 * gamma**2))
    A0 = _ppt_a0(gamma, nu)
    if A0 <= 0.0:
        return 0.0

    power = 2.0 * n_eff - 1.5
    log_rate_au = (
        np.log(C2 * Ip_au * np.sqrt(6.0 / np.pi))
        + power * np.log(2.0 * E0 / F)
        + 0.75 * np.log1p(gamma**2)
        + np.log(A0)
        + exponent
    )
    if log_rate_au < -745.0:
        return 0.0
    if log_rate_au > 690.0:
        return 1.0e300
    return omega_a * np.exp(log_rate_au)


@jit(nopython=True, cache=True)
def get_PPT_probability(E_fld, dt, omega_au, Ip_au, n_eff, C2, Ea, omega_a):
    """Cycle-averaged single-electron PPT probability over dt."""
    rate = get_PPT_rate(E_fld, omega_au, Ip_au, n_eff, C2, Ea, omega_a)
    x = rate * dt
    if x > 50.0:
        return 1.0
    return -np.expm1(-x)


@jit(nopython=True, cache=True)
def _log_i0e(x):
    """log(exp(-x) I0(x)) for x >= 0, using a stable approximation."""
    if x < 3.75:
        y = (x / 3.75)**2
        i0 = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (
            1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
        return -x + np.log(i0)
    y = 3.75 / x
    scaled_i0 = 0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (
        -0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (
        0.02635537 + y * (-0.01647633 + y * 0.00392377)))))))
    return -0.5 * np.log(x) + np.log(scaled_i0)


@jit(nopython=True, cache=True)
def _i1_i0_ratio(x):
    """I1(x)/I0(x) for x >= 0, stable over small and large arguments."""
    if x <= 0.0:
        return 0.0
    if x < 3.75:
        y = (x / 3.75)**2
        i0 = 1.0 + y * (3.5156229 + y * (3.0899424 + y * (
            1.2067492 + y * (0.2659732 + y * (0.0360768 + y * 0.0045813)))))
        i1 = x * (0.5 + y * (0.87890594 + y * (0.51498869 + y * (
            0.15084934 + y * (0.02658733 + y * (0.00301532 + y * 0.00032411))))))
        return i1 / i0
    y = 3.75 / x
    p0 = 0.39894228 + y * (0.01328592 + y * (0.00225319 + y * (
        -0.00157565 + y * (0.00916281 + y * (-0.02057706 + y * (
        0.02635537 + y * (-0.01647633 + y * 0.00392377)))))))
    p1 = 0.39894228 + y * (-0.03988024 + y * (-0.00362018 + y * (
        0.00163801 + y * (-0.01031555 + y * (0.02282967 + y * (
        -0.02895312 + y * (0.01787654 - y * 0.00420059)))))))
    return p1 / p0


@jit(nopython=True, cache=True)
def _ppt_yi_phase_factor(E_env, E_real, omega_au, Ip_au, Ea):
    """PPT-normalized Yudin-Ivanov sub-cycle modulation factor.

    The original PPT rate is cycle averaged and has no instantaneous phase
    dependence.  Yudin-Ivanov supplies the nonadiabatic sub-cycle shape.  We
    normalize that shape by its exact cycle average so that the mean of this
    factor over a carrier cycle is one and the cycle-averaged rate remains the
    PPT rate used by PlasmaIonizationPPT.
    """
    if E_env <= 0.0:
        return 0.0
    F = E_env / Ea
    kappa = np.sqrt(2.0 * Ip_au)
    gamma = omega_au * kappa / F

    # In Yudin-Ivanov notation sigma_1 = 2(asinh(gamma)-gamma/sqrt(1+gamma^2)),
    # which is the same function as _ppt_alpha(gamma).
    sigma1 = _ppt_alpha(gamma)
    Q = 0.5 * F * F / omega_au**3 * sigma1
    if Q < 0.0:
        Q = 0.0

    cos_phase = E_real / E_env
    if cos_phase > 1.0:
        cos_phase = 1.0
    elif cos_phase < -1.0:
        cos_phase = -1.0
    sin2_phase = 1.0 - cos_phase * cos_phase

    x = 0.5 * Q
    log_mean = _log_i0e(x)  # log(<exp(-Q sin^2(phi))>_phi)
    log_factor = -Q * sin2_phase - log_mean
    if log_factor < -745.0:
        return 0.0
    if log_factor > 100.0:
        return np.exp(100.0)
    return np.exp(log_factor)


@jit(nopython=True, cache=True)
def _ppt_yi_cos2_average(E_env, omega_au, Ip_au, Ea):
    """<cos^2(phi)> weighted by the YI sub-cycle modulation."""
    if E_env <= 0.0:
        return 0.5
    F = E_env / Ea
    gamma = omega_au * np.sqrt(2.0 * Ip_au) / F
    Q = 0.5 * F * F / omega_au**3 * _ppt_alpha(gamma)
    ratio = _i1_i0_ratio(0.5 * Q)
    return 0.5 * (1.0 + ratio)


@jit(nopython=True, cache=True, parallel=True)
def get_plasma_PPT(E_laser, A_laser, dt, n_gas, pack_PPT, Uion,
                  omega_au, Ea, omega_a, Z_init, Zmax,
                  ionization_current):
    """Envelope-based PPT plasma current and electron density.

    pack_PPT = (Ip_au, n_eff, C2), one value per ionization stage.
    The output follows get_plasma_ADK:
      (complex envelope current, final electron density, absorbed work density).
    """
    num_ions = pack_PPT[0].size + 1
    Nt, Nr = E_laser.shape
    J_plasma = np.zeros_like(E_laser)
    n_e = np.zeros(Nr)
    absorbed_work = np.zeros(Nr)
    z_stop = num_ions - 1 if Zmax < 0 else Zmax

    for ir in prange(Nr):
        n_gas_loc = n_gas[ir]
        E_ir_t = E_laser[:, ir]
        A_ir_t = A_laser[:, ir]
        P_loc = e * A_ir_t
        v_loc = P_loc / m_e / np.sqrt(
            1.0 + 0.5 * np.abs(P_loc)**2 / (m_e*c)**2
        )

        ion_fracts = np.zeros(num_ions)
        ion_fracts[Z_init] = 1.0
        n_e_slice = np.full(Nt, Z_init * n_gas_loc)
        J_ion_t = np.zeros_like(E_ir_t)

        for it in range(Nt):
            if ion_fracts[z_stop] >= 1.0 - 1.0e-16:
                break

            E_abs = np.abs(E_ir_t[it])
            if E_abs == 0.0:
                continue

            # Ascending order permits sequential multiple ionization steps
            # within one time sample when a single step has a large probability.
            for ion_state in range(z_stop):
                fraction = ion_fracts[ion_state]
                if fraction < 1.0e-16:
                    continue

                prob = get_PPT_probability(
                    E_abs, dt, omega_au,
                    pack_PPT[0][ion_state],
                    pack_PPT[1][ion_state],
                    pack_PPT[2][ion_state], Ea, omega_a
                )
                new_events = fraction * prob
                if new_events <= 0.0:
                    continue

                ion_fracts[ion_state] -= new_events
                ion_fracts[ion_state + 1] += new_events
                n_e_slice[it:] += new_events * n_gas_loc

                if ionization_current:
                    # Cycle-averaged energy debit: J_ion * E = n_ionized * Ip/dt.
                    J_ion_t[it] += (n_gas_loc * new_events * Uion[ion_state]
                                    / E_ir_t[it] / dt)

        J_ir_t = -e * n_e_slice * v_loc + J_ion_t
        n_e[ir] = n_e_slice[-1]
        J_plasma[:, ir] = J_ir_t
        absorbed_work[ir] = 0.5 * np.real(
            np.sum(J_ir_t * np.conj(E_ir_t))
        ) * dt

    return J_plasma, n_e, absorbed_work

@jit(nopython=True, cache=True, parallel=True)
def get_plasma_PPT_ref(E_laser, A_laser, t_axis, omega0, n_gas, pack_PPT,
                      Uion, omega_au, Ea, omega_a, Z_init, Zmax,
                      ionization_current):
    """Carrier-resolved PPT/Yudin-Ivanov plasma response on a real field.

    E_laser and A_laser are complex envelopes.  The carrier-resolved real
    fields are reconstructed with exp(-i*omega0*t).  The PPT cycle-averaged
    rate is modulated within each optical cycle by the Yudin-Ivanov
    nonadiabatic phase factor, normalized so its cycle average is exactly the
    PPT rate.  Returns (real current, final electron density, T_e, ion fractions).
    """
    num_ions = pack_PPT[0].size + 1
    Nt, Nr = E_laser.shape
    dt = t_axis[1] - t_axis[0]
    phase_env = np.exp(-1j * omega0 * t_axis)
    z_stop = num_ions - 1 if Zmax < 0 else Zmax

    J_loc_t_re = np.zeros((Nt, Nr), dtype=np.float64)
    n_e = np.zeros(Nr)
    T_e = np.zeros(Nr)
    Xi = np.zeros((Nr, num_ions))

    for ir in prange(Nr):
        n_gas_loc = n_gas[ir]
        E_env_t = E_laser[:, ir]
        A_env_t = A_laser[:, ir]
        E_real_t = np.real(E_env_t * phase_env)
        A_real_t = np.real(A_env_t * phase_env)

        P_real_t = e * A_real_t
        v_real_t = P_real_t / m_e / np.sqrt(
            1.0 + (P_real_t / (m_e * c))**2
        )

        ion_fracts = np.zeros(num_ions)
        ion_fracts[Z_init] = 1.0
        n_e_slice = np.full(Nt, Z_init * n_gas_loc)
        J_ion_t = np.zeros(Nt, dtype=np.float64)
        all_new_events = 0.0
        electron_temp_dens = 0.0

        for it in range(Nt):
            if ion_fracts[z_stop] >= 1.0 - 1.0e-16:
                break

            E_env_abs = np.abs(E_env_t[it])
            E_real = E_real_t[it]
            if E_env_abs <= 0.0:
                continue

            P = P_real_t[it]
            W_quiver = m_e * c**2 * (
                np.sqrt(1.0 + (P / (m_e*c))**2) - 1.0
            )

            # Each charge state has its own PPT envelope rate and YI phase
            # dependence through its own ionization potential.
            for ion_state in range(z_stop):
                fraction = ion_fracts[ion_state]
                if fraction < 1.0e-16:
                    continue

                Ip_au = pack_PPT[0][ion_state]
                W_cycle = get_PPT_rate(
                    E_env_abs, omega_au, Ip_au,
                    pack_PPT[1][ion_state], pack_PPT[2][ion_state],
                    Ea, omega_a
                )
                phase_factor = _ppt_yi_phase_factor(
                    E_env_abs, E_real, omega_au, Ip_au, Ea
                )
                W_inst = W_cycle * phase_factor
                x = W_inst * dt
                if x > 50.0:
                    prob = 1.0
                else:
                    prob = -np.expm1(-x)

                new_events = fraction * prob
                if new_events <= 0.0:
                    continue

                ion_fracts[ion_state] -= new_events
                ion_fracts[ion_state + 1] += new_events
                all_new_events += new_events
                electron_temp_dens += (2.0 / 3.0) * W_quiver * new_events
                n_e_slice[it:] += new_events * n_gas_loc

                # Deposit ionization energy at the actual carrier phase.
                # The phase-resolved rate suppresses events close to E=0;
                # retain a small guard against roundoff at field zero-crossings.
                if ionization_current:
                    # Smooth, phase-resolved ionization current.  The
                    # normalization by <cos^2(phi)>_YI makes its cycle-averaged
                    # work equal to the ionization energy while avoiding 1/E
                    # singularities at carrier zero-crossings.
                    cos2_avg = _ppt_yi_cos2_average(
                        E_env_abs, omega_au, Ip_au, Ea
                    )
                    J_ion_t[it] += (
                        n_gas_loc * new_events * Uion[ion_state]
                        * E_real / (E_env_abs**2 * cos2_avg * dt)
                    )

        J_real_t = -e * n_e_slice * v_real_t + J_ion_t
        n_e[ir] = n_e_slice[-1]
        J_loc_t_re[:, ir] = J_real_t
        if all_new_events > 0.0:
            T_e[ir] = electron_temp_dens / all_new_events
        Xi[ir, :] = ion_fracts

    return J_loc_t_re, n_e, T_e, Xi
