import numpy as np
from scipy.constants import e, m_e, c, pi, epsilon_0, mu_0
from ..containers import ScalarFieldEnvelope

class PlasmaSimpleNonuniform:

    def __init__(self, n_pe, dens_func):
        self.n_pe = n_pe
        self.dens_func = dens_func

    def coef_RHS(self, sim, kp_base=0.0):
        k_z2 = sim.prop.kz[:, None]**2 - sim.prop.kr[None,:]**2 - kp_base**2
        cond = (k_z2>0.0)
        k_z_inv = np.divide(
            1., np.sqrt(k_z2, where=cond), where=cond)
        k_z_inv *= cond
        coef_RHS = sim.prop.bcknd.to_device(-0.5j * e**2 * mu_0 / m_e * k_z_inv)

        return coef_RHS

    def get_RHS(self, sim, E_ts, dz=0.0 ):

        r_axis = sim.prop.r_new
        n_p0_z0 = self.n_pe * self.dens_func( sim.z_0, r_axis )[0]
        self.kp_z0 = ( n_p0_z0 * e**2 / m_e / epsilon_0 )**0.5 / c
        n_pe = self.n_pe * self.dens_func( sim.z_0 + dz, r_axis )[None,:]
        n_pe -= n_p0_z0

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
        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

        if dz != 0.0:
            sim.t_axis -= dz / c
            Jp_ts = sim.prop.step_simple(Jp_ts, -dz, kp=self.kp_z0)

        return Jp_ts

    def get_RHS0(self, sim, E_ts ):

        r_axis = sim.prop.r_new
        n_p0_z0 = self.n_pe * self.dens_func( sim.z_0, r_axis )[0]
        self.kp_z0 = ( n_p0_z0 * e**2 / m_e / epsilon_0 )**0.5 / c

        E_loc = sim.prop.perform_iTST(E_ts)

        n_pe = self.n_pe * self.dens_func( sim.z_0, r_axis )[None,:]
        n_pe -= n_p0_z0

        Jp_ft = E_loc * n_pe
        Jp_ts = sim.prop.perform_TST( Jp_ft )
        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

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
        coef_RHS = sim.prop.bcknd.to_device(-0.5j * e**2 * mu_0 / m_e * k_z_inv)

        return coef_RHS

    def get_RHS(self, sim, E_ts, dz=0.0, dz2=0.0 ):

        r_axis = sim.prop.r_new
        n_p0_z0 = self.n_pe * self.dens_func( sim.z_0, r_axis )[0]
        self.kp_z0 = ( n_p0_z0 * e**2 / m_e / epsilon_0 )**0.5 / c

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = sim.prop.step_and_iTST(E_ts, dz, kp=self.kp_z0)
        else:
            E_loc = sim.prop.perform_iTST(E_ts)

        E_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            E_loc).Field

        #n_pe = self.n_pe * self.dens_func( sim.z_0 + dz, r_axis )[None,:]
        n_pe = self.n_pe * self.dens_func( sim.z_0, r_axis )[None,:]
        n_pe -= n_p0_z0

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            E_loc_t * n_pe ).Field_ft

        Jp_ts = sim.prop.perform_TST( Jp_ft )
        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

        if dz2 != 0.0:
            sim.t_axis += dz2 / c
            Jp_ts = sim.prop.step_simple(Jp_ts, dz2, kp=self.kp_z0)

        return Jp_ts

    def get_RHS0(self, sim, E_ts ):

        r_axis = sim.prop.r_new
        n_p0_z0 = self.n_pe * self.dens_func( sim.z_0, r_axis )[0]
        self.kp_z0 = ( n_p0_z0 * e**2 / m_e / epsilon_0 )**0.5 / c

        E_loc = sim.prop.perform_iTST(E_ts)

        n_pe = self.n_pe * self.dens_func( sim.z_0, r_axis )[None,:]
        n_pe -= n_p0_z0

        Jp_ft = E_loc * n_pe
        Jp_ts = sim.prop.perform_TST( Jp_ft )
        Jp_ts *= self.coef_RHS(sim, kp_base=self.kp_z0)

        return Jp_ts
