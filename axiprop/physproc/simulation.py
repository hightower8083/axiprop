import numpy as np
from scipy.constants import e, m_e, c, pi, epsilon_0, mu_0
from tqdm.auto import tqdm
from ..containers import ScalarFieldEnvelope


class Simulation:
    def __init__(self, prop, t_axis, k0, z_0, n_dump=0,
                 max_wavelength=4e-6):

        self.prop = prop

        self.k0 = k0
        self.omega0 = k0 * c
        self.t_axis = t_axis.copy()
        self.EnvArgs = (self.k0, self.t_axis, n_dump)
        self.z_0 = z_0
        self.z_loc = z_0

        k_z2 = prop.kz[:, None]**2 - prop.kr[None,:]**2
        cond = ( k_z2 > 0.0 )
        prop.k_z = np.sqrt( k_z2, where=cond )
        prop.k_z_inv = np.divide( 1., prop.k_z, where=cond )

        k_z2 *= cond
        prop.k_z *= cond
        prop.k_z_inv *= cond

        if max_wavelength is not None:
            k_z_min = 2 * np.pi /  max_wavelength
            DC_filter = 1.0 - np.exp( -0.5 * ( k_z2 / k_z_min**2 )**2 )
            DC_filter *= cond
            self.DC_filter = prop.bcknd.to_device(DC_filter)
        else:
            self.DC_filter = None

        prop.omega = prop.kz[:, None] * c
        cond = ( prop.omega > 0.0 )
        prop.omega_inv = np.divide(1, prop.omega, where=cond)
        prop.omega *= cond
        prop.omega_inv *= cond

        prop.k_z = prop.bcknd.to_device(prop.k_z)
        prop.k_z_inv = prop.bcknd.to_device(prop.k_z_inv)
        prop.omega = prop.bcknd.to_device(prop.omega)
        # prop.omega_inv = prop.bcknd.to_device(prop.omega_inv) # add later

    def step(self, En_ts, dz, physprocs=[], method='RK4'):

        bcknd = self.prop.bcknd

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS( En_ts )

        if method in ['Ralston', 'MP', 'RK4']:
            if method=='Ralston':
                C_k2 = 2./3
            else:
                C_k2 = 1./2

            k2 = 0.0
            En_pre_ts = En_ts + C_k2 * dz * k1
            for physproc in physprocs:
                k2 += physproc.get_RHS( En_pre_ts, C_k2 * dz )

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
                k3 += physproc.get_RHS( En_pre_ts, 0.5 * dz )

            En_pre_ts = En_ts + dz * k3

            k4 = 0.0
            for physproc in physprocs:
                k4 += physproc.get_RHS( En_pre_ts, dz )

            k_tot = ( k1 + 2 * k2 + 2 * k3 + k4 ) / 6
            k_lower = k2

        En_ts += dz * k_tot
        En_ts = self.prop.step_simple(En_ts, dz)
        if self.DC_filter is not None:
            En_ts *= self.DC_filter

        self.t_axis += dz / c
        self.z_loc += dz

        val_intgral = 0.5 * bcknd.sum(
            bcknd.abs(k_tot) + bcknd.abs(k_lower)
        )

        err_abs = 0.5 * bcknd.sum( bcknd.abs(k_tot-k_lower) )

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        return En_ts, err

    def opt_dz( self, dz, err, dz_min=2e-6, err_max=1e-2,
                growth_rate=None):

        if growth_rate is None:
            if err>0:
                ErrFact = err_max / err
                dz *= 0.98 * ErrFact**0.5
        else:
            if err_max<err:
                ErrFact = err_max / err
                dz *= 0.9 * ErrFact**0.5
            else:
                dz *= growth_rate

        if dz<dz_min:
            dz = dz_min

        return dz

    def pbar_init(self, Lz):
        tqdm_bar_format = '{l_bar}{bar}| {elapsed}<{remaining} [{rate_fmt}{postfix}]'
        self.pbar = tqdm(total=100, bar_format=tqdm_bar_format)
        self.Lz = Lz

    def pbar_update(self, dz):
        # update progress bar
        self.pbar.update(dz/self.Lz * 100)
        print("".join( 79 * [' '] ), end='\r', flush=True)
        print(f'z = {(self.z_loc-self.z_0)*1e3:.3f} mm; dz = {dz*1e6:.3f} um',
              end='\r', flush=True)
