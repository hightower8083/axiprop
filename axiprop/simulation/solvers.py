import numpy as np
from scipy.constants import c
from tqdm.auto import tqdm
import os

from ..containers import apply_boundary_r
from .diags import Diagnostics


class SolverBase(Diagnostics):
    def __init__(self,
                 prop, t_axis, k0, z_0,
                 physprocs=[],
                 diag_fields=('all',),
                 adjust_dz=True,
                 err_max=3e-3,
                 iterations_max=16,
                 dz_min=2e-7,
                 growth_rate=None,
                 max_wavelength=4e-6,
                 pulse_centering=False,
                 open_boundaries_r=False,
                 n_dump_current=0,
                 n_dump_field=0,
                 refine_ord=1,
                 verbose=True,
    ):

        self.prop = prop
        self.t_axis = t_axis.copy()
        self.k0 = k0
        self.z_0 = z_0

        self.physprocs = physprocs
        self.adjust_dz = adjust_dz
        self.dz_min = dz_min
        self.err_max = err_max
        self.iterations_max = iterations_max
        self.dz_min = dz_min
        self.growth_rate = growth_rate

        self.open_boundaries_r = open_boundaries_r
        self.pulse_centering = pulse_centering

        self.refine_ord = refine_ord
        self.verbose = verbose

        self.diags = {}
        diags_default = ('z_axis',)

        if self.pulse_centering:
            diags_default += ('dt_shift',)

        for diag_str in tuple(diag_fields) + diags_default:
            self.diags[diag_str] = []

        self.omega0 = k0 * c

        self.EnvArgs = (self.k0, self.t_axis, n_dump_current)
        self.z_loc = z_0
        self.dt_shift = 0.0

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
            r_dump = np.linspace(0, 1, n_dump_field)
            self.dump_mask[:r_dump.size] = np.exp(-(2 * r_dump)**4)

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

    def run(self, E0, Lz, dz0, N_diags, write_dir='diags'):

        physprocs = self.physprocs
        dz = dz0
        z_diag = np.linspace(self.z_0, self.z_0 + Lz, N_diags)

        # lists to be filled numerical error data
        self.z_axis_err = []
        self.errors = []
        self.iterations = []

        # create diag folder (if needed)
        if 'all' in self.diags.keys() and write_dir not in os.listdir('./'):
            os.mkdir(write_dir)

        # initalize plasma OFI diags (if needed)
        for diag_str in ['n_e', 'T_e', 'Xi']:
            for i_physproc, physproc in enumerate(physprocs):
                self.diags[ diag_str + str(i_physproc) ] = []

        # field in Fourier-Bessel space (updated by simulation)
        En_ts = self.prop.perform_transfer_TST(E0, stepping=False)

        # simulation loop
        if self.verbose:
            self._pbar_init(Lz)

        i_diag = 0
        do_diag_next = False

        while (self.z_loc <= self.z_0 + Lz) and self.is_finite(En_ts):
            # simulation step
            En_ts, err, iterations = self._step(
                En_ts, dz,
                physprocs
            )

            # field filtering
            if self.DC_filter is not None:
                En_ts *= self.DC_filter

            if self.open_boundaries_r:
                En_r_space = self.prop.perform_iTST(En_ts)
                En_r_space = apply_boundary_r(En_r_space, self.dump_mask)
                En_ts = self.prop.perform_TST(En_r_space)

            # simulation time advance
            self.t_axis += dz / c
            self.z_loc += dz

            # record error data
            self.errors.append(err)
            self.iterations.append(iterations)
            self.z_axis_err.append(self.z_loc)

            if self.adjust_dz:
                dz = self._opt_dz( dz, dz0, err, iterations )

            # adjust dz to diags
            if self.z_loc <= z_diag[-1] - dz:
                dz, do_diag_next = self._match_dz_to_diags( dz, z_diag)
            else:
                dz = z_diag[-1] - self.z_loc
                do_diag_next = True

            if self.verbose:
               self._pbar_update(dz,iterations)

            # record diagnostics data
            if do_diag_next:
                self._record_diags(En_ts, physprocs, i_diag, write_dir)
                i_diag += 1
                do_diag_next = False
                if self.z_loc == self.z_0 + Lz:
                    self.diags_to_file()
                    print ('End of simulation')
                    return

    def _pbar_init(self, Lz):
        tqdm_bar_format = '{l_bar}{bar}| {elapsed}<{remaining} [{rate_fmt}{postfix}]'
        self.pbar = tqdm(total=100, bar_format=tqdm_bar_format)
        self.Lz = Lz

    def _pbar_update(self, dz, iterations):
        # update progress bar
        self.pbar.update(dz/self.Lz * 100)
        print("".join( 79 * [' '] ), end='\r', flush=True)
        print(f'distance left = {(self.z_0+self.Lz-self.z_loc)*1e3:.3f} mm; '+
              f'dz = {dz*1e6:.3f} um; iterations {iterations:d}', end='\r', flush=True)

    def is_finite(self, En_ts):
        infinite_count = np.isnan(En_ts).sum()
        if infinite_count>0:
            print ('NaN elements detected. Try to restart simulation.')
            return False
        else:
            return True

class SolverExplicitBase(SolverBase):

    def _step(self, En_ts, dz, physprocs):
        pass

    def _opt_dz( self, dz, dz0, err, iterations=None ):
        if err == 0:
            return dz0

        if self.growth_rate is not None:
            if err<self.err_max:
                dz *= self.growth_rate
            else:
                ErrFact = self.err_max / err
                dz *= 0.95 * ErrFact**0.5
        else:
            ErrFact = self.err_max / err
            dz *= 0.95 * ErrFact**0.5

        if dz<self.dz_min:
            dz = self.dz_min

        return dz


class SolverImplicitBase(SolverBase):
    def _step(self, En_ts, dz, physprocs):
        pass

    def _opt_dz(self, dz, dz0, err, iterations):
        return dz0



class SolverExplicitMulti(SolverExplicitBase):

    def _step(self, En_ts, dz, physprocs):

        method = self.method
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
        En_ts_next = En_ts + dz * k_tot
        En_ts_next = self.prop.step_simple(En_ts_next, dz)

        return En_ts_next, err, 0

    def _opt_dz( self, dz, dz0, err, iterations=None ):
        if err == 0:
            return dz0

        if self.growth_rate is not None:
            if err<self.err_max:
                dz *= self.growth_rate
            else:
                ErrFact = self.err_max / err
                dz *= 0.95 * ErrFact**0.5
        else:
            ErrFact = self.err_max / err
            dz *= 0.95 * ErrFact**0.5

        if dz<self.dz_min:
            dz = self.dz_min

        return dz


class SolverEuler(SolverExplicitBase):

    def _step(self, En_ts, dz, physprocs):
        k_tot = 0.0
        for physproc in physprocs:
            k_tot += physproc.get_RHS( En_ts )

        # field advance
        En_ts_next = En_ts + dz * k_tot
        En_ts_next = self.prop.step_simple(En_ts_next, dz)

        return En_ts_next, 0.0, 0


class SolverMP(SolverExplicitBase):

    def _step(self, En_ts, dz, physprocs):

        bcknd = self.prop.bcknd

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS( En_ts )

        k2 = 0.0
        En_pre_ts = En_ts + 0.5 * dz * k1
        for physproc in physprocs:
            k2 += physproc.get_RHS( En_pre_ts, 0.5 * dz )

        k_tot = k2
        k_lower = k1

        val_intgral = 0.5 * bcknd.sum(
            bcknd.abs(k_tot) + bcknd.abs(k_lower)
        )

        err_abs = 0.5 * bcknd.sum( bcknd.abs(k_tot-k_lower) )

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        # field advance
        En_ts_next = En_ts + dz * k_tot
        En_ts_next = self.prop.step_simple(En_ts_next, dz)

        return En_ts_next, err, 0


class SolverRK4(SolverExplicitBase):

    def _step(self, En_ts, dz, physprocs):

        bcknd = self.prop.bcknd

        k1 = 0.0
        for physproc in physprocs:
            k1 += physproc.get_RHS( En_ts )

        En_pre_ts = En_ts + 0.5 * dz * k1

        k2 = 0.0
        for physproc in physprocs:
            k2 += physproc.get_RHS( En_pre_ts, 0.5 * dz )

        En_pre_ts = En_ts + 0.5 * dz * k2

        k3 = 0.0
        for physproc in physprocs:
            k3 += physproc.get_RHS( En_pre_ts, 0.5 * dz )

        En_pre_ts = En_ts + dz * k3

        k4 = 0.0
        for physproc in physprocs:
            k4 += physproc.get_RHS( En_pre_ts, dz )

        k_tot = ( k1 + 2*k2 + 2*k3 + k4 ) / 6
        k_lower = k2 #2*k2 - k1

        val_intgral = 0.5 * bcknd.sum(
            bcknd.abs(k_tot) + bcknd.abs(k_lower)
        )

        err_abs = 0.5 * bcknd.sum( bcknd.abs(k_tot-k_lower) )

        if val_intgral>0:
            err = err_abs / val_intgral
        else:
            err = 0.0

        En_ts_next = En_ts + dz * k_tot
        En_ts_next = self.prop.step_simple(En_ts_next, dz)

        return En_ts_next, err, 0


class SolverAM1(SolverImplicitBase):

    def _step(self, En_ts, dz, physprocs):

        bcknd = self.prop.bcknd

        En_ts_prev = En_ts.copy()
        err = 1.0
        iterations = 0

        while err>self.err_max and iterations<=self.iterations_max:
            self.k0 = 0.0
            for physproc in physprocs:
                self.k0 += physproc.get_RHS( En_ts_prev, dz )

            En_ts_next = En_ts + dz * self.k0

            val_intgral = 0.5 * bcknd.sum(
                bcknd.abs(En_ts_next) + bcknd.abs(En_ts_prev)
            )

            err_abs = 0.5 * bcknd.sum( bcknd.abs(En_ts_next-En_ts_prev) )

            if val_intgral>0:
                err = err_abs / val_intgral
            else:
                err = 0.0

            En_ts_prev = En_ts_next.copy()
            iterations += 1

        En_ts_next = self.prop.step_simple(En_ts_next, dz)

        return En_ts_next, err, iterations


class SolverAM2(SolverImplicitBase):

    def _step(self, En_ts, dz, physprocs, propagate_coeff=False):

        bcknd = self.prop.bcknd

        En_ts_prev = En_ts.copy()
        err = 1.0
        iterations = 0

        self.k0 = 0.0
        for physproc in physprocs:
            self.k0 += physproc.get_RHS( En_ts_prev )

        En_ts_prev = En_ts + dz * self.k0

        while err>self.err_max and iterations<=self.iterations_max:
            self.k1 = 0.0
            for physproc in physprocs:
                self.k1 += physproc.get_RHS( En_ts_prev, dz )

            En_ts_next = En_ts + 0.5 * dz * ( self.k0 + self.k1 )

            val_intgral = 0.5 * bcknd.sum(
                bcknd.abs(En_ts_next) + bcknd.abs(En_ts_prev)
            )

            err_abs = 0.5 * bcknd.sum( bcknd.abs(En_ts_next-En_ts_prev) )

            if val_intgral>0:
                err = err_abs / val_intgral
            else:
                err = 0.0

            En_ts_prev = En_ts_next.copy()
            iterations += 1

        En_ts_next = self.prop.step_simple(En_ts_next, dz)

        if propagate_coeff:
            self.k0 = self.prop.step_simple(self.k0, dz)
            self.k1 = self.prop.step_simple(self.k1, dz)

        return En_ts_next, err, iterations


class SolverAM3(SolverImplicitBase):

    _step0 = SolverAM2._step

    def _step(self, En_ts, dz, physprocs):

        bcknd = self.prop.bcknd

        if not hasattr(self, 'k1'):
            return self._step0(En_ts, dz, physprocs, propagate_coeff=True)

        En_ts_prev = En_ts.copy()
        err = 1.0
        iterations = 0

        while err>self.err_max and iterations<=self.iterations_max:

            self.k2 = 0.0
            for physproc in physprocs:
                self.k2 += physproc.get_RHS( En_ts_prev, dz )

            En_ts_next = En_ts + dz * ( 5./12. * self.k2  + 8./12. * self.k1 - 1./12. * self.k0)

            val_intgral = 0.5 * bcknd.sum(
                bcknd.abs(En_ts_next) + bcknd.abs(En_ts_prev)
            )

            err_abs = 0.5 * bcknd.sum( bcknd.abs(En_ts_next-En_ts_prev) )

            if val_intgral>0:
                err = err_abs / val_intgral
            else:
                err = 0.0

            En_ts_prev = En_ts_next.copy()
            iterations += 1

        En_ts_next = self.prop.step_simple(En_ts_next, dz)
        self.k0 = self.prop.step_simple(self.k1, dz)
        self.k1 = self.prop.step_simple(self.k2, dz)

        return En_ts_next, err, iterations
