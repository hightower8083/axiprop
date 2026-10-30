import numpy as np

from scipy.constants import e, m_e, c, pi, epsilon_0
from scipy.constants import mu_0, fine_structure
from scipy.special import gamma as gamma_func
from scipy.signal import hilbert
from mendeleev import element as table_element

from ..containers import ScalarFieldEnvelope
from ..utils import refine1d, refine1d_TR
from .inline_methods import get_plasma_ADK
from .inline_methods import get_plasma_PPT
from .inline_methods import get_plasma_ADK_ref
from .inline_methods import get_plasma_PPT_ref
from .inline_methods import get_OFI_heating
from .inline_methods import wake_kernel_integrate
from .inline_methods import laplacian_fd

r_e = e**2 / m_e / c**2 / 4 / pi /epsilon_0
mc_m2 = 1. / ( m_e * c )**2
e_mc = e / m_e / c

omega_a = fine_structure**3 * c / r_e
Ea = m_e * c**2 / e * fine_structure**4 / r_e
UH = table_element('H').ionenergies[1]


class PlasmaSimple:

    def __init__(self, n_pe, dens_func, sim, **kw_args):
        self.n_pe = n_pe
        self.dens_func = dens_func
        self.sim = sim
        self.coef_RHS = -0.5j * e**2 * mu_0 / m_e * sim.prop.k_z_inv

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        n_pe_z = self.n_pe * self.dens_func( sim.z_loc + dz, prop.r_new )[0]

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_simple(E_ts, dz)
        else:
            E_loc = E_ts

        Jp_ts = n_pe_z * E_loc * self.coef_RHS

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        return Jp_ts


class PlasmaSimpleNonuniform(PlasmaSimple):

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        n_pe = self.n_pe * self.dens_func(
            sim.z_loc + dz, prop.r_new )[None,:]

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_and_iTST(E_ts, dz)
        else:
            E_loc = prop.perform_iTST(E_ts)

        if sim.pulse_centering:
            sim._pulse_center(E_loc)

        n_pe = sim.prop.bcknd.to_device( n_pe * np.ones(E_loc.shape) )
        Jp_ts = sim.prop.perform_TST( E_loc * n_pe )
        Jp_ts *= self.coef_RHS

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        return Jp_ts


class PlasmaRelativistic:

    def __init__(self, n_pe, dens_func, sim, wake=False, **kw_args):
        self.n_pe = n_pe
        self.dens_func = dens_func
        self.sim = sim
        self.wake = wake

        omega = sim.prop.bcknd.to_host(sim.prop.omega)
        k_z_inv = sim.prop.bcknd.to_host(sim.prop.k_z_inv)
        self.coef_RHS = -0.5 * mu_0 * omega * k_z_inv
        self.coef_RHS = sim.prop.bcknd.to_device(self.coef_RHS)

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        omega = sim.prop.kz[:, None] * c
        n_pe = self.n_pe * self.dens_func( sim.z_loc + dz, prop.r_new )[None,:]

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_and_iTST_transfer(E_ts, dz)
        else:
            E_loc = prop.perform_iTST_transfer(E_ts)

        if sim.pulse_centering:
            sim._pulse_center(E_loc)

        P_loc = -1j * e * E_loc * prop.omega_inv

        P_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        P_loc_obj.t += sim.dt_shift
        P_loc_obj.t_loc += sim.dt_shift
        P_loc_t = P_loc_obj.import_field_ft(P_loc).Field

        Jp_loc_t = - e * n_pe * P_loc_t / m_e \
            / np.sqrt( 1.0 + 0.5 * np.abs(P_loc_t)**2 * mc_m2 )

        if self.wake:

            A_loc = -1j * prop.omega_inv * E_loc
            A_loc *= (omega>0.0)
            A_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
            A_loc_obj.t += sim.dt_shift
            A_loc_obj.t_loc += sim.dt_shift
            A_loc_t = A_loc_obj.import_field_ft(A_loc).Field

            xi_ax = c * sim.EnvArgs[1]
            r_ax = prop.r_new.copy()
            k_p = np.sqrt( 4 * np.pi * r_e * self.n_pe )

            # cycle-averaged square of normalized vector potential
            A2_t = 0.5 * np.abs(A_loc_t)**2
            kernel_t = 0.5 / k_p * laplacian_fd(A2_t, xi_ax, r_ax)

            # do the integral
            delta_n = wake_kernel_integrate(kernel_t, k_p, xi_ax)

            delta_n *= (delta_n >= -1.0)
            Jp_loc_t *= 1.0 + delta_n

        Jp_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        Jp_obj.t += sim.dt_shift
        Jp_obj.t_loc += sim.dt_shift
        Jp_ft = Jp_obj.import_field(Jp_loc_t).Field_ft

        Jp_ts = prop.perform_transfer_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        Jp_ts *= self.coef_RHS

        return Jp_ts

    def get_density(self, E_ft ):
        sim = self.sim
        prop = self.sim.prop
        n_pe = self.n_pe
        xi_ax = c * ScalarFieldEnvelope(*sim.EnvArgs).t
        r_ax = prop.r_new
        k_p = np.sqrt( 4 * np.pi * r_e * n_pe )

        # get normalized vector potential
        A_ft = -1j * prop.omega_inv * e_mc * E_ft
        A_ft *= (prop.kz[:, None]>0.0)

        # bring from frequency to time domain
        A_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        A_obj.t += sim.dt_shift
        A_obj.t_loc += sim.dt_shift
        A_t = A_obj.import_field_ft(A_ft).Field

        # cycle-averaged square of normalized vector potential
        A2_t = 0.5 * np.abs(A_t)**2
        kernel_t = 0.5 / k_p * laplacian_fd(A2_t, xi_ax, r_ax)

        # do the integral
        delta_n = wake_kernel_integrate(kernel_t, k_p, xi_ax)

        n_t = n_pe * ( 1.0 + delta_n )

        return n_t


class PlasmaIonization(PlasmaRelativistic):

    def __init__( self, n_gas, dens_func, sim, my_element,
                  Z_init=0, Zmax=-1, ionization_current=True,
                  ionization_mode='AC', Nr_max=None, wake=False,**kw_args):

        super().__init__(n_gas, dens_func, sim)
        self.n_gas = n_gas

        self.Z_init = Z_init
        self.Zmax = Zmax
        self.Nr_max = Nr_max
        self.wake = wake
        self.dt = sim.t_axis[1] - sim.t_axis[0]
        self.make_ADK(my_element, ionization_mode)
        zmax_eff = self.Uion.size if self.Zmax < 0 else self.Zmax
        if not (0 <= self.Z_init <= zmax_eff <= self.Uion.size):
            raise ValueError(
                f'Expected 0 <= Z_init <= Zmax <= {self.Uion.size}; '
                f'got Z_init={self.Z_init}, Zmax={self.Zmax}'
            )
        self.ionization_current = ionization_current

        omega_shift = sim.prop.kz[:, None] * c - sim.omega0
        self.lowpass_filt = np.cos(
            0.5 * np.pi * omega_shift / omega_shift.max()
        )

    def make_ADK(self, my_element, ionization_mode):
        """
        Prepare the useful data for ADK probabplity calculation for a given
        element. This part is mostly a copy-pased from FBPIC code
        [https://github.com/fbpic/fbpic] but corrected for the enveloped
        field following approach of SMILEI ionization
        [https://smileipic.github.io/Smilei/Understand/ionization.html]
        """

        Uion = np.array(list(table_element(my_element).ionenergies.values()))
        Z_states = np.arange( Uion.size ) + 1

        n_eff = Z_states * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1

        C2 = 2**(2*n_eff) / (n_eff * gamma_func(n_eff+l_eff+1) * \
            gamma_func(n_eff-l_eff))

        if ionization_mode == 'AC':
            self.adk_power = - (2*n_eff - 1.5)
            self.adk_prefactor = (6 / np.pi)**0.5 * omega_a * C2 * \
                ( Uion/(2*UH) ) * ( 2 * (Uion/UH)**1.5 * Ea )**(2*n_eff - 1.5)
        elif ionization_mode == 'DC':
            self.adk_power = - (2*n_eff - 1)
            self.adk_prefactor = omega_a * C2 * ( Uion/(2*UH) ) \
                * ( 2 * (Uion/UH)**1.5 * Ea )**(2*n_eff - 1)
        else:
            print("`ionization_mode` must be `AC` or `DC`")

        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea
        self.Uion = e * Uion # convert to Joules

    def _get_plasma_response(self, E_loc_t, A_loc_t, n_gas):
        """Return the envelope plasma response (overridden by rate models)."""
        return get_plasma_ADK(
            E_loc_t, A_loc_t, self.dt, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current
        )

    def get_RHS(self, E_ts, dz=0.0):
        sim = self.sim
        prop = self.sim.prop
        omega = sim.prop.kz[:, None] * c

        n_gas = self.n_gas * self.dens_func( sim.z_loc + dz, prop.r_new )

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_and_iTST_transfer(E_ts, dz)
        else:
            E_loc = prop.perform_iTST_transfer(E_ts)

        if sim.pulse_centering:
            sim._pulse_center(E_loc)

        A_loc = -1j * prop.omega_inv * E_loc
        A_loc *= (omega>0.0)

        Jp_ft = np.zeros_like(E_loc)
        Nr = A_loc.shape[-1]

        if self.Nr_max is None:
            Nr_max = Nr
        elif self.Nr_max>Nr:
            Nr_max = Nr
        else:
            Nr_max = self.Nr_max

        E_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        E_loc_obj.t += sim.dt_shift
        E_loc_obj.t_loc += sim.dt_shift
        E_loc_t = E_loc_obj.import_field_ft(E_loc[:,:Nr_max]).Field

        A_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        A_loc_obj.t += sim.dt_shift
        A_loc_obj.t_loc += sim.dt_shift
        A_loc_t = A_loc_obj.import_field_ft(A_loc[:,:Nr_max]).Field

        Jp_loc_t, self.n_e, self.T_e = self._get_plasma_response(
            E_loc_t, A_loc_t, n_gas
        )

        if self.wake:
            n_pe = self.n_e.max()

            if n_pe>0:
                xi_ax = c * sim.EnvArgs[1]
                r_ax = prop.r_new[:Nr_max]
                k_p = np.sqrt( 4 * np.pi * r_e * n_pe )

                # cycle-averaged square of normalized vector potential
                A2_t = 0.5 * np.abs(A_loc_t)**2
                kernel_t = 0.5 / k_p * laplacian_fd(A2_t, xi_ax, r_ax)

                # do the integral
                delta_n = wake_kernel_integrate(kernel_t, k_p, xi_ax)

                delta_n *= (delta_n >= -1.0)
                Jp_loc_t *= 1.0 + delta_n

        Jp_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        Jp_obj.t += sim.dt_shift
        Jp_obj.t_loc += sim.dt_shift
        Jp_ft[:,:Nr_max] = Jp_obj.import_field(Jp_loc_t).Field_ft

        Jp_ft *= self.lowpass_filt

        Jp_ts = prop.perform_transfer_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        Jp_ts *= self.coef_RHS

        return Jp_ts




class PlasmaIonizationPPT(PlasmaIonization):
    """Cycle-averaged finite-frequency PPT ionization for an optical envelope.

    Uses abs(envelope E) and the carrier frequency sim.omega0. It follows the
    effective-hydrogenic, single-active-electron m=0 approximation used by the
    existing Axiprop ADK model. This cycle-averaged rate must not be applied to
    the reconstructed carrier-resolved real field.
    """

    def __init__(self, n_gas, dens_func, sim, my_element,
                 Z_init=0, Zmax=-1, ionization_current=True,
                 Nr_max=None, wake=False, **kw_args):
        super().__init__(
            n_gas, dens_func, sim, my_element,
            Z_init=Z_init, Zmax=Zmax,
            ionization_current=ionization_current,
            ionization_mode='AC', Nr_max=Nr_max, wake=wake, **kw_args
        )

    def make_ADK(self, my_element, ionization_mode='AC'):
        """Prepare PPT atomic parameters (name kept for parent compatibility)."""
        Uion = np.asarray(list(table_element(my_element).ionenergies.values()),
                          dtype=float)
        if Uion.size == 0 or np.any(~np.isfinite(Uion)) or np.any(Uion <= 0.0):
            raise ValueError(f'No valid ionization energies found for {my_element!r}')

        Z_states = np.arange(Uion.size, dtype=float) + 1.0
        Ip_au = Uion / (2.0 * UH)
        n_eff = Z_states * np.sqrt(UH / Uion)
        l_eff = n_eff[0] - 1.0
        C2 = 2.0**(2.0*n_eff) / (
            n_eff * gamma_func(n_eff + l_eff + 1.0)
            * gamma_func(n_eff - l_eff)
        )
        if self.sim.omega0 <= 0.0:
            raise ValueError('PPT ionization requires sim.omega0 > 0')

        self.ppt_pack = (Ip_au, n_eff, C2)
        self.omega_au = self.sim.omega0 / omega_a
        self.Uion = e * Uion

    def _get_plasma_response(self, E_loc_t, A_loc_t, n_gas):
        return get_plasma_PPT(
            E_loc_t, A_loc_t, self.dt, n_gas,
            self.ppt_pack, self.Uion, self.omega_au, Ea, omega_a,
            self.Z_init, self.Zmax, self.ionization_current
        )



class PlasmaIonizationRefine(PlasmaIonization):

    def __init__( self, n_gas, dens_func, sim, my_element,
                  Z_init=0, Zmax=-1, ionization_current=True,
                  Nr_max=None, refine_ord=8, **kw_args):

        super().__init__(n_gas, dens_func, sim,
            my_element=my_element, Z_init=Z_init, Zmax=Zmax,
            ionization_current=ionization_current,
            ionization_mode='DC', Nr_max=Nr_max)

        self.refine_ord = refine_ord

    def _get_refined_plasma_response(self, E_loc_t, A_loc_t, t_axis, n_gas):
        return get_plasma_ADK_ref(
            E_loc_t, A_loc_t, t_axis, self.sim.omega0, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current
        )

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        omega = sim.prop.kz[:, None] * c

        n_gas = self.n_gas * self.dens_func( sim.z_loc + dz, prop.r_new )

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_and_iTST_transfer(E_ts, dz)
        else:
            E_loc = prop.perform_iTST_transfer(E_ts)

        if sim.pulse_centering:
            sim._pulse_center(E_loc)

        A_loc = -1j * prop.omega_inv * E_loc
        A_loc *= (omega>0.0)

        Jp_ft = np.zeros_like(E_loc)
        Nr = A_loc.shape[-1]

        if self.Nr_max is None:
            Nr_max = Nr
        elif self.Nr_max>Nr:
            Nr_max = Nr
        else:
            Nr_max = self.Nr_max

        E_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        E_loc_obj.t += sim.dt_shift
        E_loc_obj.t_loc += sim.dt_shift
        E_loc_t = E_loc_obj.import_field_ft(E_loc[:,:Nr_max]).Field

        A_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        A_loc_obj.t += sim.dt_shift
        A_loc_obj.t_loc += sim.dt_shift
        A_loc_t = A_loc_obj.import_field_ft(A_loc[:,:Nr_max]).Field

        if self.refine_ord>1:
            E_loc_t = refine1d_TR(E_loc_t, self.refine_ord)
            A_loc_t = refine1d_TR(A_loc_t, self.refine_ord)
            t_axis  = refine1d(sim.t_axis, self.refine_ord)
        else:
            t_axis = sim.t_axis.copy()

        Jp_loc_t_re, self.n_e, self.T_e, self.Xi = self._get_refined_plasma_response(
            E_loc_t, A_loc_t, t_axis, n_gas
        )

        Jp_loc_t = np.exp(1j * sim.omega0 * sim.t_axis)[:, None] \
            * np.conj(
                hilbert(Jp_loc_t_re, axis=0)[::self.refine_ord]
              )

        Jp_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        Jp_obj.t += sim.dt_shift
        Jp_obj.t_loc += sim.dt_shift
        Jp_ft[:,:Nr_max] = Jp_obj.import_field(Jp_loc_t).Field_ft

        Jp_ft *= self.lowpass_filt

        Jp_ts = prop.perform_transfer_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        Jp_ts *= self.coef_RHS

        return Jp_ts




class PlasmaIonizationRefinePPT(PlasmaIonizationRefine):
    """Carrier-resolved PPT ionization using a PPT-normalized Yudin-Ivanov rate.

    Original PPT is cycle averaged and has no sub-cycle instantaneous rate.
    This class uses its cycle-averaged PPT rate together with the Yudin-Ivanov
    nonadiabatic phase modulation.  The phase factor is normalized to unit
    cycle average, so the mean ionization rate is the same as in
    PlasmaIonizationPPT.  It is intended for the refined real-field pathway.
    """

    def make_ADK(self, my_element, ionization_mode='DC'):
        Uion = np.asarray(list(table_element(my_element).ionenergies.values()),
                          dtype=float)
        if Uion.size == 0 or np.any(~np.isfinite(Uion)) or np.any(Uion <= 0.0):
            raise ValueError(f'No valid ionization energies found for {my_element!r}')

        Z_states = np.arange(Uion.size, dtype=float) + 1.0
        Ip_au = Uion / (2.0 * UH)
        n_eff = Z_states * np.sqrt(UH / Uion)
        l_eff = n_eff[0] - 1.0
        C2 = 2.0**(2.0*n_eff) / (
            n_eff * gamma_func(n_eff + l_eff + 1.0)
            * gamma_func(n_eff - l_eff)
        )
        if self.sim.omega0 <= 0.0:
            raise ValueError('PPT ionization requires sim.omega0 > 0')

        self.ppt_pack = (Ip_au, n_eff, C2)
        self.omega_au = self.sim.omega0 / omega_a
        self.Uion = e * Uion

    def _get_refined_plasma_response(self, E_loc_t, A_loc_t, t_axis, n_gas):
        return get_plasma_PPT_ref(
            E_loc_t, A_loc_t, t_axis, self.sim.omega0, n_gas,
            self.ppt_pack, self.Uion, self.omega_au, Ea, omega_a,
            self.Z_init, self.Zmax, self.ionization_current
        )



class OFI_heating:

    def __init__( self, n_gas, dens_func, sim, my_element,
                  Z_init=0, Zmax=-1, refine_ord=16):

        self.follow_process = True

        self.n_gas = n_gas
        self.dens_func = dens_func
        self.sim = sim
        self.refine_ord = refine_ord

        self.Z_init = Z_init
        self.Zmax = Zmax
        self.dt = sim.t_axis[1] - sim.t_axis[0]
        self.make_ADK_OFI(my_element)

    def get_RHS(self, *args, **kw_args ):
        return 0.0

    def make_ADK_OFI(self, my_element):

        Uion = np.array(list(table_element(my_element).ionenergies.values()))
        Z_states = np.arange( Uion.size ) + 1

        n_eff = Z_states * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1

        C2 = 2**(2*n_eff) / (n_eff * gamma_func(n_eff+l_eff+1) * \
            gamma_func(n_eff-l_eff))

        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = omega_a * C2 * ( Uion/(2*UH) ) \
            * ( 2 * (Uion/UH)**1.5 * Ea )**(2*n_eff - 1)

        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea
        self.Uion = e * Uion # convert to Joules

    def get_data(self, E_obj):

        sim = self.sim
        prop = self.sim.prop
        n_gas_loc = self.n_gas * self.dens_func( sim.z_loc, prop.r_new )

        E_loc_t =  E_obj.Field.copy()

        P_loc = -1j * e * E_obj.Field_ft * prop.omega_inv

        P_loc_t = ScalarFieldEnvelope(*sim.EnvArgs)
        P_loc_t.t += sim.dt_shift
        P_loc_t.t_loc += sim.dt_shift
        P_loc_t = P_loc_t.import_field_ft(P_loc).Field

        if self.refine_ord>1:
            E_loc_t = refine1d_TR(E_loc_t, self.refine_ord)
            P_loc_t = refine1d_TR(P_loc_t, self.refine_ord)
            t_axis  = refine1d(sim.t_axis, self.refine_ord)
        else:
            t_axis = sim.t_axis.copy()

        self.n_e, self.T_e, self.Xi = get_OFI_heating(
            E_loc_t, P_loc_t, t_axis, sim.omega0, n_gas_loc,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Z_init, self.Zmax)
