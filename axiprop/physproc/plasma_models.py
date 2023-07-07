import numpy as np

from scipy.constants import e, m_e, c, pi, epsilon_0, mu_0, fine_structure
from scipy.special import gamma as gamma_func
from scipy.interpolate import interp1d

from mendeleev import element as table_element

from ..containers import ScalarFieldEnvelope
from .ionization_inline import get_plasma_ADK
from .ionization_inline import get_plasma_ADK_OFI

r_e = e**2 / m_e / c**2 / 4 / pi /epsilon_0
omega_a = fine_structure**3 * c / r_e
Ea = m_e * c**2 / e * fine_structure**4 / r_e
UH = table_element('H').ionenergies[1]

def refine1d(A, refine_ord):
    refine_ord = int(refine_ord)
    x = np.arange(A.size, dtype=np.double)
    x_new = np.linspace(x.min(), x.max(), x.size*refine_ord)

    if A.dtype == np.double:
        interp_fu = interp1d(x, A, assume_sorted=True)
        A_new = interp_fu(x_new)
    elif A.dtype == np.complex128:
        interp_fu_abs = interp1d(x, np.abs(A), assume_sorted=True)
        slice_abs = interp_fu_abs(x_new)

        interp_fu_angl = interp1d(x, np.unwrap(np.angle(A)), assume_sorted=True)
        slice_angl = interp_fu_angl(x_new)

        A_new = slice_abs * np.exp(1j * slice_angl)
    else:
        print("Data type must be `np.double` or `np.complex128`")
        return None

    return A_new

def refine1d_TR(A, refine_ord):
    refine_ord = int(refine_ord)

    t = np.arange(A.shape[0], dtype=np.double)
    t_new = np.linspace(t.min(), t.max(), t.size*refine_ord)

    A_new = np.zeros((t_new.size, A.shape[1]), dtype=A.dtype)

    for ir in range(A.shape[1]):
        interp_fu_abs = interp1d(t, np.abs(A[:, ir]), assume_sorted=True)
        slice_abs = interp_fu_abs(t_new)

        interp_fu_angl = interp1d( t, np.unwrap(np.angle(A[:, ir])),
                                   assume_sorted=True )
        slice_angl = interp_fu_angl(t_new)

        A_new[:, ir] = slice_abs * np.exp(1j * slice_angl)

    return A_new


class PlasmaSimpleNonuniform:

    def __init__(self, n_pe, dens_func, variative=True):
        self.n_pe = n_pe
        self.dens_func = dens_func
        self.variative = variative

    def coef_RHS(self, sim, kp):
        k_z2 = sim.prop.kz[:, None]**2 - sim.prop.kr[None,:]**2 - kp**2
        cond = (k_z2>0.0)
        k_z_inv = np.divide(
            1., np.sqrt(k_z2, where=cond), where=cond)
        k_z_inv *= cond
        coef_RHS = sim.prop.bcknd.to_device(-0.5j * e**2 * mu_0 / m_e * k_z_inv)

        return coef_RHS

    def get_RHS(self, sim, E_ts, dz=0.0 ):

        r_axis = sim.prop.r_new
        n_p0_z0 = self.n_pe * self.dens_func( sim.z_0, r_axis )[0]
        n_pe = self.n_pe * self.dens_func( sim.z_0 + dz, r_axis )[None,:]

        if self.variative:
            n_pe -= n_p0_z0
            self.kp_z0 = ( n_p0_z0 * e**2 / m_e / epsilon_0 )**0.5 / c
        else:
            self.kp_z0 = 0.0

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = sim.prop.step_and_iTST(E_ts, dz, kp=self.kp_z0)
        else:
            E_loc = sim.prop.perform_iTST(E_ts)

        E_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            E_loc).Field

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            E_loc_t * n_pe ).Field_ft

        Jp_ts = sim.prop.perform_TST( Jp_ft )
        Jp_ts *= self.coef_RHS(sim, self.kp_z0)

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = sim.prop.step_simple(Jp_ts, -dz, kp=self.kp_z0)

        return Jp_ts


class PlasmaRelativistic:

    def __init__(self, n_pe, dens_func):
        self.n_pe = n_pe
        self.dens_func = dens_func

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

        n_pe = self.n_pe * self.dens_func( sim.z_0 + dz, r_axis )[None,:]
        self.kp_z0 = 0.0

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = sim.prop.step_and_iTST(E_ts, dz, kp=self.kp_z0)
        else:
            E_loc = sim.prop.perform_iTST(E_ts)

        P_loc = -1j * e * E_loc * np.divide(1, omega, where=(omega>0.0))
        P_loc *= (omega>0.0)

        P_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            P_loc).Field

        Jp_loc_t = - e * n_pe * P_loc_t / m_e \
            / np.sqrt( 1 + 0.5 * np.abs(P_loc_t)**2 / (m_e*c)**2 )

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            Jp_loc_t ).Field_ft

        Jp_ts = sim.prop.perform_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = sim.prop.step_simple(Jp_ts, -dz, kp=self.kp_z0)

        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

        return Jp_ts


class PlasmaIonization:

    def __init__( self, n_gas, dens_func, my_element,
                  Z_init=0, Zmax=-1, ionization_current=True):

        self.n_gas = n_gas
        self.dens_func = dens_func
        self.make_ADK(my_element)
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

        Jp_loc_t, self.n_e, self.T_e = get_plasma_ADK(
            E_loc_t, A_loc_t, dt, n_gas,
            (self.adk_power, self.adk_prefactor, self.adk_exp_prefactor),
            self.Uion, self.Z_init, self.Zmax, self.ionization_current)

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            Jp_loc_t ).Field_ft

        Jp_ts = sim.prop.perform_TST( Jp_ft )

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = sim.prop.step_simple(Jp_ts, -dz, kp=self.kp_z0)

        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

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