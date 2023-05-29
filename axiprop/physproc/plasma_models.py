import numpy as np
from scipy.constants import e, m_e, c, pi, epsilon_0, mu_0
from ..containers import ScalarFieldEnvelope

class SimplePlasma:

    def __init__(self, n_pe, dens_func):
        self.n_pe = n_pe
        self.dens_func = dens_func

    def get_RHS(self, sim, E_ts, dz=0.0, dz2=0.0 ):

        r_axis = sim.prop.r_new
        n_p0_z0 = self.n_pe * self.dens_func( sim.z_0, r_axis )[0]
        self.kp_z0 = ( n_p0_z0 * e**2 / m_e / epsilon_0 )**0.5 / c

        if dz != 0.0:
            sim.t_axis += dz / c
            E_loc = sim.prop.step_and_iTST(E_ts, dz, kp=self.kp_z0)
            E_loc *= np.exp(1j * sim.k0 * dz)
        else:
            E_loc = sim.prop.perform_iTST(E_ts)

        E_loc_t =  ScalarFieldEnvelope(*sim.EnvArgs).import_field_ft(
            E_loc).Field

        n_pe = self.n_pe * self.dens_func( sim.z_0 + dz, r_axis )[None,:]

        Jp_ft = ScalarFieldEnvelope(*sim.EnvArgs).import_field(
            E_loc_t * (n_pe-n_p0_z0) ).Field_ft

        Jp_ts = sim.prop.perform_TST( Jp_ft )
        Jp_ts *= sim.coef_RHS(kp_base=self.kp_z0)

        if dz2 != 0.0:
            sim.t_axis += dz2 / c
            Jp_ts = sim.prop.step_simple(Jp_ts, dz2, kp=self.kp_z0)
            Jp_ts *= np.exp(1j * sim.k0 * dz2)

        return Jp_ts
