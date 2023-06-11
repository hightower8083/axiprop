import numpy as np
from scipy.constants import e, m_e, c, pi, epsilon_0, mu_0
from ..containers import ScalarFieldEnvelope


class Simulation:
    def __init__(self, prop, t_axis, k0, z_0, n_dump=0,
                 max_wavelength=4e-6):

        self.prop = prop
        self.t_axis = t_axis.copy()
        self.k0 = k0
        self.omega0 = k0 * c
        self.EnvArgs = (self.k0, self.t_axis, n_dump)
        self.z_0 = z_0

        kz_real_min = 2 * np.pi /  max_wavelength
        kz2_real = self.prop.kz[:,None]**2 - self.prop.kr[None,:]**2
        kz2_real[kz2_real<0.0] = 0.0

        if max_wavelength is not None:
            DC_filter = 1.0 - np.exp(-0.5*(kz2_real/kz_real_min**2)**2)
            self.DC_filter = self.prop.bcknd.to_device(DC_filter)
        else:
            self.DC_filter = None

    def step(self, En_ts, dz, physprocs=[], method='RK4'):
        kp_z0 = 0.0

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS( self, En_ts )
            kp_z0 = physproc.kp_z0

        if method in ['Ralston', 'MP', 'RK4']:
            if method=='Ralston':
                C_k2 = 2./3
            else:
                C_k2 = 1./2

            k2 = 0.0
            En_pre_ts = En_ts + C_k2 * dz * k1
            for physproc in physprocs:
                k2 += physproc.get_RHS(
                    self, En_pre_ts, C_k2 * dz)

        if method=='Euler':
            k_tot = k1
            k_lower = k1
        elif method=='MP':
            k_tot = k2
            k_lower = k1
        elif method=='Ralston':
            k_tot = 0.25 * k1 + 0.75 * k2
            k_lower = k1
        elif method == 'RK4':
            k3 = 0.0
            En_pre_ts = En_ts + 0.5 * dz * k2
            for physproc in physprocs:
                k3 += physproc.get_RHS(
                    self, En_pre_ts, 0.5 * dz)

            En_pre_ts = En_ts + dz * k3

            k4 = 0.0
            for physproc in physprocs:
                k4 += physproc.get_RHS(
                    self, En_pre_ts, dz)

            k_tot = ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6
            k_lower = k2

        En_ts += dz * k_tot
        En_ts = self.prop.step_simple(En_ts, dz, kp=kp_z0)
        if self.DC_filter is not None:
            En_ts *= self.DC_filter

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

        if err_max<err:
            ErrFact = err_max / err
            dz *= 0.9 * ErrFact**0.5
        else:
            dz *= growth_rate

        if dz<dz_min:
            dz = dz_min

        return dz
