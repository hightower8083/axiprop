import numpy as np
from scipy.special import j0, j1, jn_zeros
from scipy.interpolate import interp1d
import os

from .lib import PropagatorCommon
from .lib import PropagatorFFT2
from .lib import backend_strings_ordered
from .lib import AVAILABLE_BACKENDS

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

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz, bar_format=bar_format)

        r2_loc = self.bcknd.to_device(self.r**2)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])

            self.u_loc *= self.bcknd.exp( 0.5j * self.kz[ikz] / dz * r2_loc )
            self.TST()

            phase_loc = self.kz[ikz] * dz * (1 + 0.5 * r2_loc / dz**2)
            coef_loc = self.kz[ikz] / (1j * 2 * np.pi * dz)
            self.u_ht *= self.bcknd.exp( 1j * phase_loc )
            self.u_ht *= coef_loc

            u_step[ikz] = self.bcknd.to_host(self.u_ht)
            if tqdm_available and show_progress:
                pbar.update(1)

        kr = self.bcknd.to_host(self.kr)

        Nr = kr.size

        alpha = jn_zeros(0, Nr+1) # mode 0 only
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]
        self.Rmax_new = dz * kr.max() / self.kz.min()
        self.r_new = self.Rmax_new * alpha / alpha_np1

#        self.r_new = dz * kr / self.kz[self.Nkz//2]
#        self.Rmax_new = self.r_new.max()

        # self.Rmax_new = dz * kr.max() / self.kz.min()
        # self.r_new = np.linspace(0, self.Rmax_new, self.Nr)

        u_slice_abs = np.zeros(self.Nr, dtype=np.double)
        u_slice_angl = np.zeros(self.Nr, dtype=np.double)

        for ikz in range(self.Nkz):
            r_loc = dz * kr / self.kz[ikz]

            interp_fu = interp1d(r_loc, np.abs(u_step[ikz]),
                                 fill_value='extrapolate',
                                 kind='quadratic',
                                 bounds_error=False )
            u_slice_abs = interp_fu(self.r_new)

            interp_fu = interp1d(r_loc, np.unwrap(np.angle(u_step[ikz])),
                                 fill_value='extrapolate',
                                 kind='quadratic',
                                 bounds_error=False )
            u_slice_angl = interp_fu(self.r_new)

            u_step[ikz] = u_slice_abs * np.exp( 1j * u_slice_angl )

        if tqdm_available and show_progress:
            pbar.close()

        return u_step

class PropagatorFresnelFFT(PropagatorFFT2, PropagatorFresnel):
    pass


class PropagatorFresnelHT(PropagatorFresnel):
    def __init__(self, r_axis, kz_axis,
                 N_pad=4, mode=0,
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
        self.init_TST(mode)

    def init_rkr_jroot_padded(self, r_axis, N_pad, mode):
        self.Rmax, self.Nr = r_axis

        self.Rmax_ext = self.Rmax * N_pad
        self.Nr_ext = self.Nr * N_pad

        alpha = jn_zeros(mode, self.Nr_ext+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

#        self.r_ext = np.linspace(0, self.Rmax_ext, self.Nr_ext)
        self.r_ext = self.Rmax_ext * alpha / alpha_np1
        self.kr_ext = self.bcknd.to_device(alpha/self.Rmax_ext)
        self.alpha_ext = alpha

        self.r = self.r_ext[:self.Nr]
        self.Rmax = self.r.max()
        self.kr = self.kr_ext[:self.Nr]
        self.alpha = self.alpha_ext[:self.Nr]

    def init_TST(self, mode):
        """
        Setup DHT transform and data buffers.

        Parameters
        ----------
        """
        Rmax = self.Rmax
        Nr = self.Nr
        dtype = self.dtype

        kr = self.bcknd.to_host(self.kr)
        r = self.bcknd.to_host(self.r)

        #kr = self.bcknd.to_host(self.kr_ext)
        #r = self.bcknd.to_host(self.r_ext)

        jn_fu =  [j0,j1][mode]
        jnp1_fu =  [j0,j1][mode+1]

        self.TM = jn_fu(r[:,None] * kr[None,:])
        self.TM = self.bcknd.inv_on_host(self.TM, dtype)
        #self.TM = self.TM[:self.Nr, :self.Nr]
        #self.TM = self.TM[:self.Nr, :]
        self.TM = self.bcknd.to_device(self.TM)

        self._TST_norm_coef = 0.5 * ( Rmax * jnp1_fu(self.alpha)) **2
        self._TST_norm_coef = self.bcknd.to_device( self._TST_norm_coef )

        self.shape_trns = (self.Nr, )
        self.shape_trns_new = (self.Nr, )

        self.u_loc = self.bcknd.zeros(self.Nr, dtype)
        self.u_ht = self.bcknd.zeros(self.Nr, dtype)

        self.TST_matmul = self.bcknd.make_matmul(self.TM, self.u_loc, self.u_ht)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_ht = self.TST_matmul(self.TM, self.u_loc, self.u_ht)
        self.u_ht *= self._TST_norm_coef