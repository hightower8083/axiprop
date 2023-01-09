import numpy as np
from scipy.special import j0, j1, jn_zeros
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

#        Rmax_new = dz * self.kr[:self.Nr_new].max() / self.kz.max()
#        self.make_r_new( Rmax_new )
        Rmax_new = dz * self.kr[:self.Nr_new].min() / self.kz[self.kz.size//2]
        self.make_r_new(  Rmax_new=None, r_ax=dz * self.kr[:self.Nr_new] / self.kz.max() )
        r2 = self.bcknd.to_device(self.r**2)

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz, bar_format=bar_format)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])
            self.u_loc *= self.bcknd.exp( 0.5j * self.kz[ikz] / dz \
                                            * r2 )
            self.TST()

            u_slice_loc = self.bcknd.to_host(self.u_ht)

            """
            r_loc = dz * self.kr[:self.Nr_new] / self.kz[ikz]
            phase_loc = self.kz[ikz] * dz * (1 + 0.5 * (r_loc*r_loc) / (dz*dz) )
            coef_loc = self.kz[ikz] / (1j * 2 * np.pi * dz)
            u_slice_loc *= coef_loc * np.exp( 1j * phase_loc )
            """

            u_step[ikz] = u_slice_loc #self.gather_on_r_new( u_slice_loc, r_loc )

            if tqdm_available and show_progress:
                pbar.update(1)

        if tqdm_available and show_progress:
            pbar.close()

        return u_step


class PropagatorFresnelFFT(PropagatorFFT2, PropagatorFresnel):

    def make_r_new(self, Rmax_new, Nr=None):
        self.Rmax_new = Rmax_new
        self.r_new = self.r
        self.r2_new = self.r_new**2

    def gather_on_r_new( self, u_loc, r_loc ):
        return u_loc


class PropagatorFresnelHT(PropagatorFresnel):
    def __init__(self, r_axis, kz_axis,
                 Nr_new=None, N_pad=4, mode=0,
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

        self.init_backend(backend)
        self.init_kz(kz_axis)
        self.init_rkr_jroot_padded(r_axis, N_pad, mode)

        if Nr_new is None:
            self.Nr_new = self.Nr
        else:
            self.Nr_new = Nr_new

        self.init_TST(mode)

    def init_rkr_jroot_padded(self, r_axis, N_pad, mode):
        self.Rmax, self.Nr = r_axis

        self.Rmax_ext = self.Rmax * N_pad
        self.Nr_ext = self.Nr * N_pad

        self.r_ext = np.linspace(0, self.Rmax_ext, self.Nr_ext)
        dr = self.r_ext[[0,1]].ptp()
        self.r_ext += 0.5 * dr
        self.r = self.r_ext[:self.Nr]
        self.Rmax = self.r.max()

        alpha = jn_zeros(mode, self.Nr_ext+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        self.kr = alpha/self.Rmax_ext
        self.alpha = alpha

    def make_r_new(self, Rmax_new=None, r_ax=None):
        if r_ax is None:
            self.Rmax_new = Rmax_new
            alpha = jn_zeros(0, self.Nr_new + 1)
            alpha_np1 = alpha[-1]
            alpha = alpha[:-1]

            self.r_new = alpha / alpha_np1 * Rmax_new
        else:
            self.Rmax_new = Rmax_new
            self.r_new = r_ax.copy()

        #self.r_new = np.linspace(0, Rmax_new, Nr)
        #dr_new = self.r_new[[0,1]].ptp()
        #self.r_new += 0.5 * dr_new

        self.r2_new = self.r_new**2

    def gather_on_r_new( self, u_loc, r_loc ):
        return u_loc

    def gather_on_r_new0( self, u_loc, r_loc ):
        interp_fu_abs = interp1d(r_loc, np.abs(u_loc),
                             fill_value='extrapolate',
                             kind='cubic',
                             bounds_error=False )
        u_slice_abs = interp_fu_abs(self.r_new)

        interp_fu_angl = interp1d(r_loc, unwrap1d(np.angle(u_loc)),
                             fill_value='extrapolate',
                             kind='cubic',
                             bounds_error=False )
        u_slice_angl = interp_fu_angl(self.r_new)
        del interp_fu_abs, interp_fu_angl
        u_slice_new = u_slice_abs * np.exp( 1j * u_slice_angl )
        return u_slice_new

    def init_TST(self, mode):
        """
        Setup DHT transform and data buffers.

        Parameters
        ----------
        """
        Rmax = self.Rmax
        Nr = self.Nr
        dtype = self.dtype

        kr = self.kr
        r = self.r_ext

        jn_fu =  [j0,j1][mode]
        jnp1_fu =  [j0,j1][mode+1]

        _norm_coef = 2.0 /  ( Rmax * jnp1_fu(self.alpha[:self.Nr_new]) )**2
        self.TM = jn_fu(r[:, None] * kr[None,:self.Nr_new]) * _norm_coef[None,:]
        self.TM = self.bcknd.inv_on_host(self.TM, dtype)
        self.TM = self.TM[:,:self.Nr]

#        _norm_coef = 2.0 /  ( Rmax * jnp1_fu(self.alpha) )**2
#        self.TM = jn_fu(r[:, None] * kr[None,:]) * _norm_coef[None,:]
#        self.TM = self.bcknd.inv_sqr_on_host(self.TM, dtype)
#        self.TM = self.TM[:self.Nr_new, :self.Nr]

        self.TM = self.bcknd.to_device(self.TM)

        self.shape_trns = (self.Nr, )
        self.shape_trns_new = (self.Nr_new, )

        self.u_loc = self.bcknd.zeros(self.Nr, dtype)
        self.u_ht = self.bcknd.zeros(self.Nr_new, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)