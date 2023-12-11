import numpy as np
from scipy.constants import c
from tqdm.auto import tqdm
from ..containers import ScalarFieldEnvelope, apply_boundary_r
from ..utils import refine1d

class Simulation:
    def __init__(self, prop, t_axis, k0, z_0,
                 diag_fields=('E_t_env', 'Energy'),
                 n_dump_current=0,
                 n_dump_field=0,
                 open_boundaries_r=False,
                 physprocs = [],
                 max_wavelength=4e-6,
                 refine_ord=1,
                 pulse_centering=False,
                 verbose=True):

        self.prop = prop
        self.physprocs = physprocs

        self.k0 = k0
        self.omega0 = k0 * c
        self.t_axis = t_axis.copy()
        self.EnvArgs = (self.k0, self.t_axis, n_dump_current)
        self.z_0 = z_0
        self.z_loc = z_0
        self.dt_shift = 0.0
        self.open_boundaries_r = open_boundaries_r
        self.pulse_centering = pulse_centering

        k_z2 = prop.kz[:, None]**2 - prop.kr[None,:]**2
        cond = ( k_z2 > 0.0 )

        if not hasattr(prop, 'k_z') :
            prop.k_z = np.sqrt( k_z2, where=cond )
            prop.k_z_inv = np.divide( 1., prop.k_z, where=cond )

            prop.k_z = np.nan_to_num(prop.k_z)
            prop.k_z_inv = np.nan_to_num(prop.k_z_inv)

            prop.k_z *= cond
            prop.k_z_inv *= cond

            prop.k_z = prop.bcknd.to_device(prop.k_z)
            prop.k_z_inv = prop.bcknd.to_device(prop.k_z_inv)

        k_z2 *= cond

        if max_wavelength is not None:
            k_z_min = 2 * np.pi /  max_wavelength
            DC_filter = 1.0 - np.exp( -0.5 * ( k_z2 / k_z_min**2 )**2 )
            DC_filter *= cond
            self.DC_filter = prop.bcknd.to_device(DC_filter)
        else:
            self.DC_filter = None

        if open_boundaries_r:
            self.dump_mask = np.zeros(n_dump_field)
            r_dump = np.linspace(0, 1, n_dump_field//2)
            self.dump_mask[:n_dump_field//2] = \
                ( 1 - np.exp(-(2 * r_dump)**6) )[::-1]
            self.dump_mask = prop.bcknd.to_device( self.dump_mask )

        if not hasattr(prop, 'omega') :
            prop.omega = prop.kz[:, None] * c
            cond = ( prop.omega > 0.0 )
            prop.omega_inv = np.divide(1, prop.omega, where=cond)
            prop.omega_inv = np.nan_to_num(prop.omega_inv)

            prop.omega *= cond
            prop.omega_inv *= cond

            prop.omega = prop.bcknd.to_device(prop.omega)
            # prop.omega_inv = prop.bcknd.to_device(prop.omega_inv) # add later

        self.diags = {}
        diags_default = ('z_axis',)
        if self.pulse_centering:
            diags_default += ('dt_shift',)

        for diag_str in tuple(diag_fields) + diags_default:
            self.diags[diag_str] = []
        self.refine_ord = refine_ord

        self.verbose = verbose

    def run(self, E0, Lz, dz0, N_diags,
            method='RK4', adjust_dz=True, dz_min=2e-7, err_max=1e-2,
            growth_rate=None):

        physprocs = self.physprocs
        for diag_str in ['n_e', 'T_e', 'Xi']:
            for i_physproc, physproc in enumerate(physprocs):
                self.diags[ diag_str + str(i_physproc) ] = []

        # field in Fourier-Bessel space (updated by simulation)
        En_ts = self.prop.perform_transfer_TST(E0, stepping=False)
        dz = dz0

        # lists to be filled numerical error data
        self.z_axis_err = []
        self.errors = []

        z_diag = np.linspace(self.z_0, self.z_0 + Lz, N_diags)
        do_diag_next = False

        # simulation loop
        if self.verbose:
            self._pbar_init(Lz)

        while (self.z_loc < self.z_0 + Lz) :
            # record diagnostics data
            if do_diag_next:
                self._record_diags(En_ts, physprocs)
                do_diag_next = False

            # simulation step
            En_ts, err = self._step(
                En_ts, dz,
                physprocs,
                method
            )

            # record error data
            self.errors.append(err)
            self.z_axis_err.append(self.z_loc)

            # adjust dz to the error
            if adjust_dz and err>0:
                dz = self._opt_dz( dz, err, dz_min, err_max, growth_rate )
            else:
                dz = dz0

            # adjust dz to diags
            if self.z_loc <= z_diag[-1] - dz:
                dz, do_diag_next = self._match_dz_to_diags( dz, z_diag)

            if self.verbose:
               self._pbar_update(dz)

        print('End of simulation')

    def _step(self, En_ts, dz, physprocs, method):

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


        val_intgral = 0.5 * bcknd.sum(
            bcknd.abs(k_tot) + bcknd.abs(k_lower)
        )

        err_abs = 0.5 * bcknd.sum( bcknd.abs(k_tot-k_lower) )

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        # field advance
        En_ts += dz * k_tot

        if self.DC_filter is not None:
            En_ts *= self.DC_filter

        En_ts = self.prop.step_simple(En_ts, dz)

        if self.open_boundaries_r:
            En_r_space = self.prop.perform_iTST(En_ts)
            En_r_space = apply_boundary_r(En_r_space, self.dump_mask)
            En_ts = self.prop.perform_TST(En_r_space)

        self.t_axis += dz / c
        self.z_loc += dz
        # field advance: end

        return En_ts, err

    def _pulse_center(self, E_ft):
        field = ScalarFieldEnvelope(*self.EnvArgs)
        field.t += self.dt_shift
        field.t_loc += self.dt_shift
        field = field.import_field_ft(E_ft, transform=False)
        self.dt_shift = field.dt_to_center - self.z_loc/c

    def _record_diags(self, E_fb, physprocs):
        self.diags['z_axis'].append(self.z_loc)
        E_ft = self.prop.perform_iTST_transfer(E_fb.copy())

        if self.pulse_centering:
            self._pulse_center(E_ft)
            self.diags['dt_shift'].append(self.dt_shift)

        E_obj = ScalarFieldEnvelope(*self.EnvArgs)
        E_obj.t += self.dt_shift
        E_obj.t_loc += self.dt_shift
        E_obj = E_obj.import_field_ft( E_ft, r_axis=self.prop.r_new )

        for i_physproc, physproc in enumerate(physprocs):
            i_physproc_str = str(i_physproc)

            if hasattr(physproc, 'follow_process'):
                physproc.get_data(E_obj)

            if hasattr(physproc, 'n_e'):
                self.diags['n_e'+i_physproc_str].append(physproc.n_e)

            if hasattr(physproc, 'T_e'):
                self.diags['T_e'+i_physproc_str].append(physproc.T_e)

            if hasattr(physproc, 'Xi'):
                self.diags['Xi'+i_physproc_str].append(physproc.Xi)

        if 'E_ft' in self.diags.keys():
            self.diags['E_ft'].append(E_obj.Field_ft)

        if 'E_ft_onax' in self.diags.keys():
            self.diags['E_ft'].append(E_obj.Field_ft[:, 0])

        if 'E_t_env' in self.diags.keys():
            self.diags['E_t_env'].append(E_obj.Field)

        if 'E_t_env_onax' in self.diags.keys():
            self.diags['E_t_env_onax'].append(E_obj.Field[:, 0])

        if 'E_t_onax' in self.diags.keys():
            t_axis_refine = refine1d(self.t_axis, self.refine_ord).real
            phs = np.exp(-1j * t_axis_refine * c * self.k0)
            self.diags['E_t_onax'].append(
                np.real(refine1d(E_obj.Field[:,0], self.refine_ord) * phs )
            )

        if 'Energy_ft' in self.diags.keys():
            self.diags['Energy_ft'].append(E_obj.Energy_ft)

        if 'Energy' in self.diags.keys():
            self.diags['Energy'].append(E_obj.Energy)

    def _opt_dz( self, dz, err, dz_min=2e-6, err_max=1e-2,
                growth_rate=None):

        if growth_rate is None:
            if err>0:
                ErrFact = err_max / err
                dz *= 0.95 * ErrFact**0.5
        else:
            if err_max<err:
                ErrFact = err_max / err
                dz *= 0.95 * ErrFact**0.5
            else:
                dz *= growth_rate

        if dz<dz_min:
            dz = dz_min

        return dz

    def _match_dz_to_diags( self, dz, z_diags):
        dz_to_next_diag = z_diags[z_diags>self.z_loc][0] - self.z_loc

        if dz_to_next_diag < dz:
            dz = dz_to_next_diag
            do_diag_next = True
        else:
            do_diag_next = False

        return dz, do_diag_next

    def _pbar_init(self, Lz):
        tqdm_bar_format = '{l_bar}{bar}| {elapsed}<{remaining} [{rate_fmt}{postfix}]'
        self.pbar = tqdm(total=100, bar_format=tqdm_bar_format)
        self.Lz = Lz

    def _pbar_update(self, dz):
        # update progress bar
        self.pbar.update(dz/self.Lz * 100)
        print("".join( 79 * [' '] ), end='\r', flush=True)
        print(f'distance left = {(self.z_0+self.Lz-self.z_loc)*1e3:.3f} mm; '+
              f'dz = {dz*1e6:.3f} um', end='\r', flush=True)

    def diags_to_numpy(self):
        for diag_str in self.diags.keys():
            self.diags[diag_str] = np.asarray(self.diags[diag_str])

        self.errors = np.asarray(self.errors)
        self.z_axis_err = np.asarray(self.z_axis_err)
