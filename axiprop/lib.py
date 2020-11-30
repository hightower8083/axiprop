# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop main file

This file contains main classes of axiprop:
- PropagatorCommon
- PropagatorSymmetric
- PropagatorResampling
"""
import numpy as np
from scipy.constants import c
from scipy.special import j0, j1, jn_zeros
from scipy.linalg import pinv2


class PropagatorCommon:
    """
    Base class for propagators. Contains methods to:
    - setup spectral `kz` grid;
    - perform a single-step calculation;
    - perform a multi-step calculation;

    This class should to be used to derive the actual Propagators
    by adding proper methods to setup radial and `kr` grids, and
    DHT /iDHT transforms.
    """

    def init_kz(self, Lkz, Nkz, k0):
        """
        Setup `kz` spectral grid.

        Parameters
        ----------
        Lkz: float (1/m)
            Total spectral width in units of wavenumbers.

        Nkz: int
            Number of spectral modes (wavenumbers) to resolve the temporal
            profile of the wave.

        k0: float (1/m)
            Central wavenumber of the spectral domain.
        """
        self.Nkz = Nkz
        self.kz = k0 + Lkz / 2 * np.linspace(-1., 1., Nkz)
        self.dtype = np.complex

    def init_rkr_jroot_both(self, Rmax, Nr, dtype):
        """
        Setup radial `r` and spectral `kr` grids, and fix data type.

        Parameters
        ----------
        Rmax: float (m)
            Radial size of the calculation domain.

        Nr: int
            Number of nodes of the radial grid.

        dtype: type
            Data type to be used.
        """
        self.Rmax = Rmax
        self.Nr = Nr
        self.dtype = dtype

        alpha = jn_zeros(0, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        self.r = Rmax * alpha / alpha_np1
        self.kr = alpha/Rmax

    def step(self, u, dz):
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
        assert u.shape[0] == self.Nkz
        assert u.shape[1] == self.Nr
        assert self.dtype == u.dtype

        for ikz in range(self.Nkz):
            self.u_loc[:] = u[ikz,:]
            self.u_ht = self.DHT(self.u_loc, self.u_ht)
            self.u_ht *= np.exp(1j * dz * np.sqrt(self.kz[ikz]**2 - self.kr**2))
            self.u_iht = self.iDHT(self.u_ht, self.u_iht)
            u[ikz, :self.Nr_new] = self.u_iht

        u = u[:, :self.Nr_new]

        return u

    def steps(self, u, dz, verbose=True):
        """
        Propagate wave `u` over the multiple steps.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        dz: array of floats (m)
            Steps over which wave should be propagated.

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the propagated field.
        """
        Nsteps = len(dz)
        if Nsteps==0:
            return None

        assert u.shape[0] == self.Nkz
        assert u.shape[1] == self.Nr
        assert self.dtype == u.dtype

        u_steps = np.empty((Nsteps, self.Nkz, self.Nr_new), dtype=self.dtype)

        if verbose:
            print('Propagating:')

        for ikz in range(self.Nkz):
            self.u_loc[:] = u[ikz,:]
            self.u_ht = self.DHT(self.u_loc, self.u_ht)
            ik_loc = 1j * np.sqrt(self.kz[ikz]**2 - self.kr**2)
            for i_step in range(Nsteps):
                self.u_ht *= np.exp( dz[i_step] * ik_loc )
                self.u_iht = self.iDHT(self.u_ht, self.u_iht)
                u_steps[i_step, ikz, :] = self.u_iht

                if verbose:
                    print(f"Done step {i_step} of {Nsteps} "+ \
                          f"for wavelength {ikz+1} of {self.Nkz}",
                      end='\r', flush=True)

        return u_steps

class PropagatorSymmetric(PropagatorCommon):
    """
    Class for the propagator described in [M. Guizar-Sicairos,
    J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)].

    Contains methods to:
    - setup DHT transformation;
    - perform a forward DHT transform;
    - perform a inverse DHT transform;

    This propagator uses same DHT matrix for forward and inverse transforms.
    Inverse transform can be truncated to a smaller radial size (same grid).
    """

    def __init__(self, Rmax, Lkz, Nr, Nkz, k0,
                 Nr_new=None, dtype=np.complex):
        """
        Construct the propagator.

        Parameters
        ----------
        Rmax: float (m)
            Radial size of the calculation domain.

        Lkz: float (1/m)
            Total spectral width in units of wavenumbers.

        Nr: int
            Number of nodes of the radial grid.

        Nkz: int
            Number of spectral modes (wavenumbers) to resolve the temporal
            profile of the wave.

        k0: float (1/m)
            Central wavenumber of the spectral domain.

        Nr_new: int (optional)
            New number of nodes of the trancated radial grid. If not defined
            `Nr` will be used.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.
        """
        self.init_kz(Lkz, Nkz, k0)
        self.init_rkr_jroot_both(Rmax, Nr, dtype)
        self.init_DHT(Nr_new)

    def init_DHT(self, Nr_new):
        """
        Setup DHT transformation matrix and data buffers.

        Parameters
        ----------
        Nr_new: int
            New number of nodes of the trancated radial grid. If is `None`,
            `Nr` will be used.
        """
        Rmax = self.Rmax
        Nr = self.Nr
        dtype = self.dtype

        self.Nr_new = Nr_new
        if self.Nr_new is None:
            self.Nr_new = Nr
        self.r_new = self.r[:self.Nr_new]

        alpha = jn_zeros(0, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        self._j = (np.abs(j1(alpha)) / Rmax).astype(dtype)
        denominator = alpha_np1 * np.abs(j1(alpha[:,None]) * j1(alpha[None,:]))
        self.TM = 2 * j0(alpha[:,None]*alpha[None,:]/alpha_np1) / denominator

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr_new, dtype=dtype)

    def DHT(self, u_in, u_out):
        """
        Forward DHT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-radial field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-spectral field.
        """
        u_in = u_in/self._j
        u_out = np.dot(self.TM.astype(self.dtype), u_in, out=u_out)
        return u_out

    def iDHT(self, u_in, u_out):
        """
        Inverse DHT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-spectral field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-radial field.
        """
        u_out = np.dot(self.TM[:self.Nr_new].astype(self.dtype),
                       u_in, out=u_out)
        u_out *= self._j[:self.Nr_new]
        return u_out

class PropagatorResampling(PropagatorCommon):
    """
    Class for the propagator with possible different sampling for
    input and output radial grids.

    Contains methods to:
    - setup DHT/iDHT transformations;
    - perform a forward DHT transform;
    - perform a inverse DHT transform;

    This propagator creates inverse iDHT matrix using numeric inversion
    of DHT. This method samples output field on an arbitrary uniform
    radial grid.
    """

    def __init__(self, Rmax, Lkz, Nr, Nkz, k0,
                 Rmax_new=None, Nr_new=None, dtype=np.complex):
        """
        Construct the propagator.

        Parameters
        ----------
        Rmax: float (m)
            Radial size of the calculation domain.

        Lkz: float (1/m)
            Total spectral width in units of wavenumbers.

        Nr: int
            Number of nodes of the radial grid.

        Nkz: int
            Number of spectral modes (wavenumbers) to resolve the temporal
            profile of the wave.

        k0: float (1/m)
            Central wavenumber of the spectral domain.

        Rmax_new: float (m) (optional)
            New radial size for the output calculation domain. If not defined
            `Rmax` will be used.

        Nr_new: int (optional)
            New number of nodes of the radial grid. If not defined `Nr`
            will be used.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.
        """
        self.init_kz(Lkz, Nkz, k0)
        self.init_rkr_jroot_both(Rmax, Nr, dtype)
        self.init_DHT(Rmax_new, Nr_new)

    def init_DHT(self, Rmax_new, Nr_new):
        """
        Setup DHT transformation and data buffers.

        Parameters
        ----------
        Rmax_new: float (m) (optional)
            New radial size for the output calculation domain. If not defined
            `Rmax` will be used.

        Nr_new: int
            New number of nodes of the radial grid. If is `None`, `Nr` will
            be used.
        """
        Rmax = self.Rmax
        Nr = self.Nr
        dtype = self.dtype

        self.Rmax_new = Rmax_new
        if self.Rmax_new is None:
            self.Rmax_new = Rmax

        self.Nr_new = Nr_new
        if self.Nr_new is None:
            self.Nr_new = Nr
        self.r_new = np.linspace(0, self.Rmax_new, self.Nr_new)

        alpha = jn_zeros(0, Nr+1)
        alpha_np1 = alpha[-1]
        alpha = alpha[:-1]

        invTM = j0(self.r[:,None] * self.kr[None,:])
        self.TM = pinv2(invTM, check_finite=False)
        self.invTM = j0(self.r_new[:,None] * self.kr[None,:])

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr_new, dtype=dtype)

    def DHT(self, u_in, u_out):
        """
        Forward DHT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-radial field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-spectral field.
        """
        u_out = np.dot(self.TM.astype(self.dtype), u_in, out=u_out)
        return u_out

    def iDHT(self, u_in, u_out):
        """
        Inverse DHT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-spectral field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-radial field.
        """
        u_out = np.dot(self.invTM.astype(self.dtype), u_in, out=u_out)
        return u_out