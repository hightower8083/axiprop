import numpy as np
from scipy.special import j0, j1, jn, jn_zeros
from scipy.interpolate import interp1d
import os

from .lib import PropagatorCommon
from .lib import PropagatorFFT2
from .lib import backend_strings_ordered
from .lib import AVAILABLE_BACKENDS, backend_strings_ordered
from .utils import unwrap1d

from .lib import tqdm_available
if tqdm_available:
    from .lib import tqdm
    from .lib import bar_format

class PropagatorFresnel(PropagatorCommon):

    def step(self, u, dz, overwrite=False, show_progress=False):
        """
        Propagate wave `u` over the distance `dz`.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        dz: float (m)
            Distance over which wave should be propagated.

        Returns
        -------
        u: 2darray of complex or double
            Overwritten array with the propagated field.
        """
        assert u.dtype == self.dtype

        if not overwrite:
            u_step = np.empty((self.Nkz, *self.shape_trns_new),
                              dtype=u.dtype)
        else:
            u_step = u

        r2 = self.bcknd.to_device(self.r**2)

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz, bar_format=bar_format)

        if self.r_axis_new is None:
            self.r_new =  dz * self.kr[:self.Nr] / self.kz[self.kz.size//2]

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])
            self.u_loc *= self.bcknd.exp(0.5j * self.kz[ikz] / dz * r2)
            self.TST()

            u_slice_loc = self.bcknd.to_host(self.u_ht)

            r_loc = dz * self.kr[:self.Nr_new] / self.kz[ikz]
            phase_loc = self.kz[ikz] * dz * (1 + 0.5 * (r_loc*r_loc) / (dz*dz) )
            coef_loc = self.kz[ikz] / (1j * 2 * np.pi * dz)
            u_slice_loc *= coef_loc * np.exp( 1j * phase_loc )

            u_slice_loc = self.gather_on_r_new(u_slice_loc, r_loc, self.r_new)

            u_step[ikz] = u_slice_loc

            if tqdm_available and show_progress:
                pbar.update(1)

        if tqdm_available and show_progress:
            pbar.close()

        return u_step

    def steps(self, u, z_axis, show_progress=True):
        """
        Propagate wave `u` over the multiple steps.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        z_axis: array of floats (m)
            Axis over which wave should be propagated.

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the propagated field.
        """
        assert u.dtype == self.dtype
        Nsteps = len(z_axis)
        if Nsteps==0:
            return None

        u_steps = np.empty( (Nsteps, self.Nkz, *self.shape_trns_new),
                         dtype=u.dtype)

        if tqdm_available and show_progress:
            pbar = tqdm(total=Nsteps, bar_format=bar_format)

        for i_step, z_dest in enumerate(z_axis):
            u_steps[i_step] = self.step(u, z_dest)
            if tqdm_available and show_progress:
                pbar.update(1)

        if tqdm_available and show_progress:
            pbar.close()

        return u_steps


class PropagatorFresnelFFT(PropagatorFFT2, PropagatorFresnel):
    pass

"""
    def make_r_new(self, Rmax_new, Nr=None):
        self.Rmax_new = Rmax_new
        self.r_new = self.r
        self.r2_new = self.r_new**2

    def gather_on_r_new( self, u_loc, r_loc ):
        return u_loc
"""

class PropagatorFresnelHT(PropagatorFresnel):
    def __init__(self, r_axis, kz_axis,
                 r_axis_new=None, N_pad=4, mode=0,
                 dtype=np.complex128, backend=None):
        """
        Construct the propagator.

        Parameters
        ----------
        r_axis: tuple (Rmax, Nr)
          Here:
            Rmax: float (m)
                Radial size of the calculation domain.

            Nr: int
                Number of nodes of the radial grid.

        kz_axis: a tuple (k0, Lkz, Nkz) or a 1D numpy.array
            When tuple is given the axis is created using:

              k0: float (1/m)
                Central wavenumber of the spectral domain.

              Lkz: float (1/m)
                Total spectral width in units of wavenumbers.

              Nkz: int
                Number of spectral modes (wavenumbers) to resolve the temporal
                profile of the wave.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        backend: string
            Backend to be used. See axiprop.backends.AVAILABLE_BACKENDS for the
            list of available options.
        """
        self.dtype = dtype
        self.mode = mode
        self.r_axis_new = r_axis_new

        self.init_backend(backend)
        self.init_kz(kz_axis)

        if type(r_axis) is tuple:
            Rmax, Nr = r_axis
            self.Nr = Nr
            r_axis_ext = ( N_pad * Rmax, N_pad * Nr )
            self.r_ext, self.Rmax_ext, self.Nr_ext = \
                            self.init_r_uniform(r_axis_ext)

            self.r = self.r_ext[:Nr]
            dr_est = (self.r[1:] - self.r[:-1]).mean()
            Rmax = self.r.max()
            self.Rmax = Rmax + 0.5 * dr_est
        else:
            self.r = r_axis.copy()
            Nr = self.r.size
            self.Nr = Nr
            dr_est = (self.r[1:] - self.r[:-1]).mean()
            Rmax = self.r.max()
            self.Rmax = Rmax + 0.5 * dr_est

            self.r_ext = np.r_[ self.r, Rmax + dr_est * np.arange(1, Nr*(N_pad-1))+1 ]
            self.Rmax_ext = self.r_ext.max() + 0.5 * dr_est
            self.Nr_ext = self.r_ext.size

        if r_axis_new is None:
            self.Nr_new = Nr
        elif type(r_axis_new) is tuple:
            self.r_new, self.Rmax_new, self.Nr_new = \
                self.init_r_uniform(self.r_axis_new)
        else:
            self.r_new, self.Rmax_new, self.Nr_new = \
                self.init_r_sampled(self.r_axis_new)

        self.init_kr(self.Rmax_ext, self.Nr_ext)
        self.init_TST()

    def init_TST(self):
        """
        Setup DHT transform and data buffers.

        Parameters
        ----------
        """
        mode = self.mode
        dtype = self.dtype

        r_ext = self.r_ext

        Rmax = self.Rmax
        Rmax_ext = self.Rmax_ext
        Nr = self.Nr
        Nr_ext = self.Nr_ext

        Nr_new = self.Nr_new

        alpha = self.alpha
        kr = self.kr

        _norm_coef = 2.0 /  ( Rmax_ext * jn(mode+1, alpha[:Nr_new]) )**2
        self.TM = jn(mode, r_ext[:, None] * kr[None,:Nr_new]) * _norm_coef[None,:]
        self.TM = self.bcknd.inv_on_host(self.TM, dtype)
        self.TM = self.TM[:,:Nr]

#        _norm_coef = 2.0 /  ( Rmax * jn(mode+1, alpha) )**2
#        self.TM = jn(mode, r[:, None] * kr[None,:]) * _norm_coef[None,:]
#        self.TM = self.bcknd.inv_sqr_on_host(self.TM, dtype)
#        self.TM = self.TM[:Nr_new, :Nr]

        self.TM = self.bcknd.to_device(self.TM)

        self.shape_trns = (Nr, )
        self.shape_trns_new = (Nr_new, )

        self.u_loc = self.bcknd.zeros(Nr, dtype)
        self.u_ht = self.bcknd.zeros(Nr_new, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)
        self.u_ht *= 2 * np.pi

    def gather_on_r_new( self, u_loc, r_loc, r_new ):
        interp_fu_abs = interp1d(r_loc, np.abs(u_loc),
                             fill_value='extrapolate',
                             kind='linear',
                             bounds_error=False )
        u_slice_abs = interp_fu_abs(r_new)

        interp_fu_angl = interp1d(r_loc, unwrap1d(np.angle(u_loc)),
                             fill_value='extrapolate',
                             kind='linear',
                             bounds_error=False )
        u_slice_angl = interp_fu_angl(r_new)
        del interp_fu_abs, interp_fu_angl

        u_slice_new = u_slice_abs * np.exp( 1j * u_slice_angl )
        return u_slice_new