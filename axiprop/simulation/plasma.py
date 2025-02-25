import numpy as np

from scipy.constants import e, m_e, c, pi, epsilon_0
from scipy.constants import mu_0, fine_structure
from scipy.special import gamma as gamma_func
from scipy.signal import hilbert
from mendeleev import element as table_element

from ..containers import ScalarFieldEnvelope
from ..utils import refine1d, refine1d_TR
from .ionization_inline import get_plasma_ADK
from .ionization_inline import get_plasma_ADK_ref
from .ionization_inline import get_OFI_heating


r_e = e**2 / m_e / c**2 / 4 / pi /epsilon_0
mc_m2 = 1. / ( m_e * c )**2
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

        n_pe = sim.prop.bcknd.to_device( n_pe * np.ones(E_loc.shape) )
        Jp_ts = sim.prop.perform_TST( E_loc * n_pe )
        Jp_ts *= self.coef_RHS

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        return Jp_ts


class PlasmaRelativistic:

    def __init__(self, n_pe, dens_func, sim, **kw_args):
        self.n_pe = n_pe
        self.dens_func = dens_func
        self.sim = sim
        self.coef_RHS = -0.5 * mu_0 * sim.prop.omega * sim.prop.k_z_inv

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        n_pe = self.n_pe * self.dens_func( sim.z_loc + dz, prop.r_new )[None,:]

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_and_iTST_transfer(E_ts, dz)
        else:
            E_loc = prop.perform_iTST_transfer(E_ts)

        P_loc = -1j * e * E_loc * prop.omega_inv

        P_loc_obj = ScalarFieldEnvelope(*sim.EnvArgs)
        P_loc_obj.t += sim.dt_shift
        P_loc_obj.t_loc += sim.dt_shift
        P_loc_t = P_loc_obj.import_field_ft(P_loc).Field

        Jp_loc_t = - e * n_pe * P_loc_t / m_e \
            / np.sqrt( 1.0 + 0.5 * np.abs(P_loc_t)**2 * mc_m2 )

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


class PlasmaIonization(PlasmaRelativistic):

    def __init__( self, n_gas, dens_func, sim, my_element,
                  Z_init=0, Zmax=-1, ionization_current=True,
                  ionization_mode='AC', Nr_max=None, **kw_args):

        super().__init__(n_gas, dens_func, sim)
        self.n_gas = n_gas

        self.Z_init = Z_init
        self.Zmax = Zmax
        self.Nr_max = Nr_max
        self.dt = sim.t_axis[1] - sim.t_axis[0]
        self.make_ADK(my_element, ionization_mode)
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

        Jp_loc_t, self.n_e, self.T_e = get_plasma_ADK(
            E_loc_t, A_loc_t, self.dt, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current
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


class PlasmaIonizationRefine(PlasmaIonization):

    def __init__( self, n_gas, dens_func, sim, my_element,
                  Z_init=0, Zmax=-1, ionization_current=True,
                  Nr_max=None, refine_ord=8, **kw_args):

        super().__init__(n_gas, dens_func, sim,
            my_element=my_element, Z_init=Z_init, Zmax=Zmax,
            ionization_current=ionization_current,
            ionization_mode='DC', Nr_max=Nr_max)

        self.refine_ord = refine_ord

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

        Jp_loc_t_re, self.n_e, self.T_e, self.Xi = get_plasma_ADK_ref(
            E_loc_t, A_loc_t, t_axis, sim.omega0, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current
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
