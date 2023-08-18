import numpy as np

from scipy.constants import e, m_e, c, pi, epsilon_0
from scipy.constants import mu_0, fine_structure
from scipy.special import gamma as gamma_func
from mendeleev import element as table_element

from ..containers import ScalarFieldEnvelope
from ..utils import refine1d, refine1d_TR
from .ionization_inline import get_plasma_ADK
from .ionization_inline import get_plasma_ADK_OFI

r_e = e**2 / m_e / c**2 / 4 / pi /epsilon_0

mc_m2 = 1. / ( m_e * c )**2

omega_a = fine_structure**3 * c / r_e
Ea = m_e * c**2 / e * fine_structure**4 / r_e
UH = table_element('H').ionenergies[1]


class PlasmaSimple:

    def __init__(self, n_pe, dens_func, sim ):
        self.n_pe = n_pe
        self.dens_func = dens_func
        self.sim = sim
        self.coef_RHS = -0.5j * e**2 * mu_0 / m_e * sim.prop.k_z_inv

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        n_pe_z = self.n_pe * self.dens_func( sim.z_0 + dz, prop.r_new )[0]

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
            sim.z_0 + dz, prop.r_new )[None,:]

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

    def __init__(self, n_pe, dens_func, sim):
        self.n_pe = n_pe
        self.dens_func = dens_func
        self.sim = sim
        self.coef_RHS = -0.5 * mu_0 * sim.prop.omega * sim.prop.k_z_inv

    def get_RHS(self, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        n_pe = self.n_pe * self.dens_func( sim.z_0 + dz, prop.r_new )[None,:]

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = prop.step_and_iTST_transfer(E_ts, dz)
        else:
            E_loc = prop.perform_iTST_transfer(E_ts)

        P_loc = -1j * e * E_loc * prop.omega_inv

        P_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            P_loc).Field

        Jp_loc_t = - e * n_pe * P_loc_t / m_e \
            / np.sqrt( 1.0 + 0.5 * np.abs(P_loc_t)**2 * mc_m2 )

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            Jp_loc_t ).Field_ft

        Jp_ts = prop.perform_transfer_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = prop.step_simple(Jp_ts, -dz)

        Jp_ts *= self.coef_RHS

        return Jp_ts

class PlasmaIonization(PlasmaRelativistic):

    def __init__( self, n_gas, dens_func, sim, my_element,
                  Z_init=0, Zmax=-1, ionization_current=True):

        super().__init__(n_gas, dens_func, sim)

        self.Z_init = Z_init
        self.Zmax = Zmax
        self.make_ADK(my_element)
        self.ionization_current = ionization_current

    def make_ADK(self, my_element):
        """
        Prepare the useful data for ADK probabplity calculation for a given
        element. This part is mostly a copy-pased from FBPIC
        [https://github.com/fbpic/fbpic]
        """

        Uion = np.array(list(table_element(my_element).ionenergies.values()))
        Z_states = np.arange( Uion.size ) + 1

        n_eff = Z_states * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1

        C2 = 2**(2*n_eff) / (n_eff * gamma_func(n_eff+l_eff+1) * \
            gamma_func(n_eff-l_eff))

        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = omega_a * C2 * ( Uion/(2*UH) ) \
                    * ( 2 * (Uion/UH)**(3./2) * Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea
        self.Uion = Uion

    def get_RHS(self, sim, E_ts, dz=0.0 ):
        sim = self.sim
        prop = self.sim.prop
        omega = sim.prop.kz[:, None] * c

        n_gas = self.n_gas * self.dens_func( sim.z_0 + dz, prop.r_new )

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = sim.prop.prop.step_and_iTST_transfer(E_ts, dz)
        else:
            E_loc = prop.perform_iTST_transfer(E_ts)

        A_loc = -1j * prop.omega_inv * E_loc
        A_loc *= (omega>0.0)

        E_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            E_loc).Field
        A_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            A_loc).Field

        dt = sim.t_axis[1] - sim.t_axis[0]

        Jp_loc_t, self.n_e, self.T_e = get_plasma_ADK(
            E_loc_t, A_loc_t, dt, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current)

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            Jp_loc_t ).Field_ft

        Jp_ts = prop.perform_transfer_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = sim.prop.step_simple(Jp_ts, -dz)

        Jp_ts *= self.coef_RHS

        return Jp_ts


class PlasmaIonizationOFI:

    def __init__( self, n_gas, dens_func, my_element, refine_ord=1,
                  Z_init=0, Zmax=-1, ionization_current=True):

        self.n_gas = n_gas
        self.dens_func = dens_func
        self.make_ADK(my_element)
        self.refine_ord = refine_ord
        self.Z_init = Z_init
        self.Zmax = Zmax
        self.ionization_current = ionization_current

    def make_ADK(self, my_element):
        """
        Prepare the useful data for ADK probabplity calculation for a given
        element. This part is mostly a copy-pased from FBPIC code
        [https://github.com/fbpic/fbpic]
        """

        Uion = np.array(list(table_element(my_element).ionenergies.values()))
        Z_states = np.arange( Uion.size ) + 1

        n_eff = Z_states * np.sqrt( UH/Uion )
        l_eff = n_eff[0] - 1

        C2 = 2**(2*n_eff) / (n_eff * gamma_func(n_eff+l_eff+1) * \
            gamma_func(n_eff-l_eff))

        self.adk_power = - (2*n_eff - 1)
        self.adk_prefactor = omega_a * C2 * ( Uion/(2*UH) ) \
                    * ( 2 * (Uion/UH)**(3./2) * Ea )**(2*n_eff - 1)
        self.adk_exp_prefactor = -2./3 * ( Uion/UH )**(3./2) * Ea
        self.Uion = Uion

    def coef_RHS(self, sim, kp_base=0.0):
        k_z2 = sim.prop.kz[:, None]**2 - sim.prop.kr[None,:]**2 - kp_base**2
        cond = (k_z2>0.0)
        k_z_inv = np.divide(
            1., np.sqrt(k_z2, where=cond), where=cond)
        k_z_inv *= cond

        omega = sim.prop.kz[:, None] * c
        coef_RHS = sim.prop.bcknd.to_device(-0.5 * mu_0 * omega * k_z_inv)

        return coef_RHS

    def get_RHS(self, sim, E_ts, dz=0.0 ):

        r_axis = sim.prop.r_new
        omega = sim.prop.kz[:, None] * c

        n_gas = self.n_gas * self.dens_func( sim.z_0 + dz, r_axis )
        self.kp_z0 = 0.0

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = sim.prop.step_and_iTST(E_ts, dz, kp=self.kp_z0)
        else:
            E_loc = sim.prop.perform_iTST(E_ts)

        A_loc = -1j * np.divide(1, omega, where=(omega>0.0)) * E_loc
        A_loc *= (omega>0.0)

        E_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            E_loc).Field
        A_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            A_loc).Field

        dt = sim.t_axis[1] - sim.t_axis[0]

        Jp_loc_t, self.n_e, self.T_e = get_plasma_ADK_OFI(
            E_loc_t, A_loc_t, sim.t_axis, sim.omega0, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.refine_ord,
            self.ionization_current)

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            Jp_loc_t ).Field_ft

        Jp_ts = sim.prop.perform_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = sim.prop.step_simple(Jp_ts, -dz, kp=self.kp_z0)

        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

        return Jp_ts

    def get_plasma(self, sim, E_loc, z0 ):

        r_axis = sim.prop.r_new
        n_gas = self.n_gas * self.dens_func( z0, r_axis )
        omega = sim.prop.kz[:, None] * c

        A_loc = -1j * np.divide(1, omega, where=(omega>0.0)) * E_loc
        A_loc *= (omega>0.0)

        E_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            E_loc).Field
        A_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            A_loc).Field

        if self.refine_ord>1:
            E_loc_t = refine1d_TR(E_loc_t, self.refine_ord)
            A_loc_t = refine1d_TR(A_loc_t, self.refine_ord)
            t_axis = refine1d(sim.t_axis, self.refine_ord)
        else:
            t_axis = sim.t_axis.copy()

        Jp_loc_t, n_e, T_e, Xi = get_plasma_ADK_OFI(
            E_loc_t, A_loc_t, t_axis, sim.omega0, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current)

        return Jp_loc_t, n_e, T_e, Xi
