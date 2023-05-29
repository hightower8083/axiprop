import numpy as np
from scipy.constants import e, m_e, c, pi, epsilon_0, mu_0
from ..containers import ScalarFieldEnvelope


class Simulation:
    def __init__(self, prop, t_axis, k0, z_0, n_dump=0):

        self.prop = prop
        self.t_axis = t_axis.copy()
        self.k0 = k0
        self.EnvArgs = (self.k0, self.t_axis, n_dump)
        self.z_0 = z_0

    def adjust_dz( self, dz, err, dz_min=1e-6, err_max=1e-4, err_min=0.0,
                   damp_rate=0.7, growth_rate=1.01):
        if err>err_max:
            dz *= damp_rate

        if err<err_min:
            dz *= 1 / damp_rate

        dz *= growth_rate

        if dz<dz_min:
            dz = dz_min

        return dz

    def opt_dz( self, dz, err, dz_min=2e-6, err_max=1e-3, growth_rate=1.01):

        ErrFact = err_max / err
        if ErrFact<1:
            dz *= 0.9 * ErrFact**0.5
        else:
            dz *= growth_rate

        if dz<dz_min:
            dz = dz_min

        return dz

    def coef_RHS(self, kp_base=0.0):
        k_z2 = self.prop.kz[:, None]**2 - self.prop.kr[None,:]**2 - kp_base**2
        cond = (k_z2>0.0)
        k_z_inv = np.divide(
            1., np.sqrt(k_z2, where=cond), where=cond)
        k_z_inv *= cond
        coef_RHS = self.prop.bcknd.to_device(-0.5j * e**2 * mu_0 / m_e * k_z_inv)

        return coef_RHS

    def step(self, En_ts, dz, physprocs=[], method='RK4'):

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS(self, En_ts, 0.0, 0.0)
            kp_z0 = physproc.kp_z0

        if method == 'Ralston2':
            C_k2 = 2./3
        else:
            C_k2 = 1./2

        k2 = 0.0
        En_pre_ts = En_ts + C_k2 * dz * k1
        for physproc in physprocs:
            k2 += physproc.get_RHS(
                self, En_pre_ts, C_k2 * dz, -C_k2 * dz)

        if method == 'RK4':
            k3 = 0.0
            En_pre_ts = En_ts + 0.5 * dz * k2
            for physproc in physprocs:
                k3 += physproc.get_RHS(
                    self, En_pre_ts, 0.5 * dz, -0.5 * dz)

            En_pre_ts = En_ts + dz * k3

            k4 = 0.0
            for physproc in physprocs:
                k4 += physproc.get_RHS(
                    self, En_pre_ts, dz, -dz)

            k_tot = ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6
            k_lower = k2
        elif method == 'LF2':
            k_tot = k2
            k_lower = k1
        elif method == 'Ralston2':
            k_tot = 0.25 * k1 + 0.75 * k2
            k_lower = k1

        En_ts += dz * k_tot

        En_ts = self.prop.step_simple(En_ts, dz, kp=kp_z0)
        En_ts *= np.exp(1j * self.k0 * dz)

        self.t_axis += dz / c
        self.z_0 += dz

        val_intgral = 0.5 * (
            self.prop.bcknd.abs(k_tot).sum() \
            + self.prop.bcknd.abs(k_lower).sum()
            ).get()

        err_abs = 0.5 * self.prop.bcknd.abs(k_tot-k_lower).sum().get()

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        return En_ts, err

    def stepRK4(self, En_ts, dz, physprocs=[]):

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS(self, En_ts, 0.0, 0.0)
            kp = physproc.kp

        En_pre_ts = En_ts + 0.5 * dz * k1

        k2 = 0.0
        for physproc in physprocs:
            k2 += physproc.get_RHS(
                self, En_pre_ts, 0.5 * dz, -0.5 * dz)

        En_pre_ts = En_ts + 0.5 * dz * k2

        k3 = 0.0
        for physproc in physprocs:
            k3 += physproc.get_RHS(
                self, En_pre_ts, 0.5 * dz, -0.5 * dz)

        En_pre_ts = En_ts + dz * k3

        k4 = 0.0
        for physproc in physprocs:
            k4 += physproc.get_RHS(
                self, En_pre_ts, dz, -dz)

        k_tot = ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6

        En_ts += dz * k_tot

        En_ts = self.prop.step_simple(En_ts, dz, kp=kp)
        En_ts *= np.exp(1j * self.k0 * dz)

        self.t_axis += dz / c
        self.z_0 += dz

        val_intgral = 0.5 * (
            self.prop.bcknd.abs(k_tot).sum().get() \
            + self.prop.bcknd.abs(k2).sum().get()
            )

        err_abs = self.prop.bcknd.abs(k_tot-k2).sum().get()

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        return En_ts, err

    def stepLF2(self, En_ts, dz, physprocs=[]):

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS(self, En_ts, 0.0, 0.0)
            kp = physproc.kp

        En_pre_ts = En_ts + 0.5 * dz * k1

        k2 = 0.0
        for physproc in physprocs:
            k2 += physproc.get_RHS(
                self, En_pre_ts, 0.5 * dz, -0.5 * dz)

        En_ts += dz * k2

        En_ts = self.prop.step_simple(En_ts, dz, kp=kp)
        En_ts *= np.exp(1j * self.k0 * dz)

        self.t_axis += dz / c
        self.z_0 += dz

        val_intgral = 0.5 * (
            self.prop.bcknd.abs(k1).sum().get() \
            + self.prop.bcknd.abs(k2).sum().get()
            )

        err_abs = self.prop.bcknd.abs(k1-k2).sum().get()

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        return En_ts, err

    def stepRalston(self, En_ts, dz, physprocs=[]):

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS(self, En_ts, 0.0, 0.0)
            kp = physproc.kp

        En_pre_ts = En_ts + 2./3 * dz * k1

        k2 = 0.0
        for physproc in physprocs:
            k2 += physproc.get_RHS(
                self, En_pre_ts, 2./3 * dz, -2./3 * dz)

        k_tot = 0.25 * k1 + 0.75 * k2
        En_ts += dz * k_tot

        En_ts = self.prop.step_simple(En_ts, dz, kp=kp)
        En_ts *= np.exp(1j * self.k0 * dz)

        self.t_axis += dz / c
        self.z_0 += dz

        val_intgral = 0.5 * (
            self.prop.bcknd.abs(k1).sum().get() \
            + self.prop.bcknd.abs(k_tot).sum().get()
            )

        err_abs = self.prop.bcknd.abs(k1-k_tot).sum().get()

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        return En_ts, err

    def stepLF(self, En_ts, dz, physprocs=[]):

        Jn_ts = 0.0
        for physproc in physprocs:
            Jn_ts += physproc.get_RHS(self, En_ts, 0.0, 0.0)
            kp = physproc.kp

        En_half_ts = En_ts + 0.5 * dz * Jn_ts

        Jn_half_ts = 0.0
        for physproc in physprocs:
            Jn_half_ts += physproc.get_RHS(
                self, En_half_ts, 0.5 * dz, -0.5 * dz)

        En_ts += dz * Jn_half_ts
        En_ts = self.prop.step_simple(En_ts, dz, kp=kp)
        En_ts *= np.exp(1j * self.k0 * dz)

        self.t_axis += dz / c
        self.z_0 += dz

        val_intgral = 0.5 * ( self.prop.bcknd.abs(Jn_ts).sum().get() \
            + self.prop.bcknd.abs(Jn_half_ts).sum().get() )
        err_abs = self.prop.bcknd.abs(Jn_ts - Jn_half_ts).sum().get()
        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        return En_ts, err
