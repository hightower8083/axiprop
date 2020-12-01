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

try:
    import pyfftw
    have_pyfftw = True
except Exception:
    have_pyfftw = False

class PropagatorCommon:
    """
    Base class for propagators. Contains methods to:
    - setup spectral `kz` grid;
    - setup radial `r` and spectral `kr` grids;
    - setup transverse `x`-`y`, and spectral `kx`-`ky` grids;
    - perform a single-step calculation;
    - perform a multi-step calculation;

    This class should to be used to derive the actual Propagators
    by adding proper methods for the Transverse Spectral Transforms (TST).
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
        Nkz_2 = int(np.ceil(Nkz/2))
        half_ax = np.linspace(0, 1., Nkz_2)
        full_ax = np.r_[-half_ax[1:][::-1], half_ax]

        self.kz = k0 + Lkz / 2 * full_ax
        self.Nkz = full_ax.size
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

    def init_xykxy_fft2(self, Lx, Ly, Nx, Ny, dtype):
        """
        Setup the transverse `x` and `y` and corresponding spectral
        `kx` and `ky` grids, and fix data type.

        Parameters
        ----------
        Lx: float (m)
            Full size of the calculation domain along x-axis.

        Ly: float (m)
            Full size of the calculation domain along y-axis.

        Nx: int
            Number of nodes of the x-grid.

        Ny: int
            Number of nodes of the y-grid.

        dtype: type
            Data type to be used.
        """
        self.dtype = dtype

        self.Lx = Lx
        self.Ly = Ly
        self.Nx = Nx
        self.Ny = Ny

        self.x = np.linspace(-Lx/2, Lx/2, Nx)
        self.y = np.linspace(-Ly/2, Ly/2, Ny)
        dx = self.x[1] - self.x[0]
        dy = self.y[1] - self.y[0]

        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

        self.r = np.sqrt(self.x[:,None]**2 + self.y[None,:]**2 ).flatten()
        self.kr = np.sqrt(self.kx[:,None]**2 + self.ky[None,:]**2).flatten()
        self.Nr = self.r.size

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
            self.u_ht = self.TST(self.u_loc, self.u_ht)
            self.u_ht *= np.exp(1j * dz * np.sqrt(self.kz[ikz]**2 - self.kr**2))
            self.u_iht = self.iTST(self.u_ht, self.u_iht)
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
            self.u_ht = self.TST(self.u_loc, self.u_ht)
            ik_loc = 1j * np.sqrt(self.kz[ikz]**2 - self.kr**2)
            for i_step in range(Nsteps):
                self.u_ht *= np.exp( dz[i_step] * ik_loc )
                self.u_iht = self.iTST(self.u_ht, self.u_iht)
                u_steps[i_step, ikz, :] = self.u_iht

                if verbose:
                    print(f"Done step {i_step} of {Nsteps} "+ \
                          f"for wavelength {ikz+1} of {self.Nkz}",
                      end='\r', flush=True)

        return u_steps

class PropagatorSymmetric(PropagatorCommon):
    """
    Class for the propagator with the Quasi-Discrete Hankel transform (QDHT)
    described in [M. Guizar-Sicairos, J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)].

    Contains methods to:
    - setup QDHT for TST;
    - perform a forward QDHT;
    - perform a inverse QDHT;

    This propagator uses same matrix for the forward and inverse transforms.
    The inverse transform can be truncated to a smaller radial size (same grid).
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
        self.init_TST(Nr_new)

    def init_TST(self, Nr_new):
        """
        Setup QDHT transformation matrix and data buffers.

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

    def TST(self, u_in, u_out):
        """
        Forward QDHT transform.

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

    def iTST(self, u_in, u_out):
        """
        Inverse QDHT transform.

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
    Class for the propagator with the non-symmetric Discrete Hankel transform
    (DHT) and possible different sampling for the input and output radial grids.

    Contains methods to:
    - setup DHT/iDHT transforms for TST;
    - perform a forward DHT;
    - perform a inverse iDHT;

    This propagator creates DHT matrix using numeric inversion of the inverse iDHT.
    This method samples output field on an arbitrary uniform radial grid.
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
        self.init_TST(Rmax_new, Nr_new)

    def init_TST(self, Rmax_new, Nr_new):
        """
        Setup DHT transform and data buffers.

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

    def TST(self, u_in, u_out):
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

    def iTST(self, u_in, u_out):
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

class PropagatorFFT2(PropagatorCommon):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST.

    Contains methods to:
    - setup TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;

    This class uses serial Numpy `fft` library.
    """

    def __init__(self, Lx, Ly, Lkz, Nx, Ny, Nkz, k0,
                 Rmax_new=None, Nr_new=None, dtype=np.complex):
        """
        Construct the propagator.

        Parameters
        ----------
        Lx: float (m)
            Full size of the calculation domain along x-axis.

        Ly: float (m)
            Full size of the calculation domain along y-axis.

        Lkz: float (1/m)
            Total spectral width in units of wavenumbers.

        Nx: int
            Number of nodes of the x-grid.

        Ny: int
            Number of nodes of the y-grid.

        Nkz: int
            Number of spectral modes (wavenumbers) to resolve the temporal
            profile of the wave.

        k0: float (1/m)
            Central wavenumber of the spectral domain.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.
        """
        self.init_kz(Lkz, Nkz, k0)
        self.init_xykxy_fft2(Lx, Ly, Nx, Ny, dtype)
        self.init_TST()

    def init_TST(self):
        """
        Setup data buffers for TST.
        """
        Nr = self.Nr
        Nx = self.Nx
        Ny = self.Ny
        self.Nr_new = Nr

        dtype = self.dtype

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr, dtype=dtype)

    def TST(self, u_in, u_out):
        """
        Forward FFT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-radial field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-spectral field.
        """
        u_in = u_in.reshape(self.Nx, self.Ny)
        u_out[:] = np.fft.fft2(u_in, norm='ortho').flatten()
        return u_out

    def iTST(self, u_in, u_out):
        """
        Inverse FFT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-spectral field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-radial field.
        """
        u_in = u_in.reshape(self.Nx, self.Ny)
        u_out[:] = np.fft.ifft2(u_in, norm='ortho').flatten()
        return u_out

class PropagatorFFTW(PropagatorCommon):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST. This class uses Numpy `fft` library.

    Contains methods to:
    - setup FFTW object and TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;

    This class uses FFTW library via `pyfftw` wrapper, and can be used with
    multiple processors. This method is typically 2 times faster than
    PropagatorFFT2.
    """

    def __init__(self, Lx, Ly, Lkz, Nx, Ny, Nkz, k0, Rmax_new=None,
                 Nr_new=None, dtype=np.complex, threads=4):
        """
        Construct the propagator.

        Parameters
        ----------
        Lx: float (m)
            Full size of the calculation domain along x-axis.

        Ly: float (m)
            Full size of the calculation domain along y-axis.

        Lkz: float (1/m)
            Total spectral width in units of wavenumbers.

        Nx: int
            Number of nodes of the x-grid.

        Ny: int
            Number of nodes of the y-grid.

        Nkz: int
            Number of spectral modes (wavenumbers) to resolve the temporal
            profile of the wave.

        k0: float (1/m)
            Central wavenumber of the spectral domain.

        dtype: type (optional)
            Data type to be used. Default is np.complex128.

        threads: int
            Number of threads to use for computations. Default is 4.
        """
        if not have_pyfftw:
            print("This method requires `pyfftw`. " + \
                "Install `pyfftw`, or use PropagatorFFT2 (slower numpy fft)")
            return

        self.init_kz(Lkz, Nkz, k0)
        self.init_xykxy_fft2(Lx, Ly, Nx, Ny, dtype)
        self.init_TST(threads)

    def init_TST(self, threads):
        """
        Setup data buffers for DHT transformation.

        Parameters
        ----------
        threads: int
            Number of threads to use for computations.
        """
        Nr = self.Nr
        Nx = self.Nx
        Ny = self.Ny
        self.Nr_new = Nr

        dtype = self.dtype

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr, dtype=dtype)

        self.in_buff = pyfftw.empty_aligned( (Nx, Ny), dtype=np.complex128 )
        self.out_buff = pyfftw.empty_aligned( (Nx, Ny), dtype=np.complex128 )
        self.fft = pyfftw.FFTW( self.in_buff, self.out_buff, axes=(-1,0),
            direction='FFTW_FORWARD', threads=threads)
        self.ifft = pyfftw.FFTW( self.in_buff, self.out_buff, axes=(-1,0),
            direction='FFTW_BACKWARD', threads=threads, normalise_idft=True)

    def TST(self, u_in, u_out):
        """
        Forward FFT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-radial field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-spectral field.
        """
        self.in_buff[:] = u_in.reshape(self.Nx, self.Ny)
        self.fft()
        u_out[:] = self.out_buff.flatten()
        return u_out

    def iTST(self, u_in, u_out):
        """
        Inverse FFT transform.

        Parameters
        ----------
        u_in: 2darray of complex
            Array with the spectral-spectral field.

        u_out: 2darray of complex (is also Returned)
            Array with the spectral-radial field.
        """
        self.in_buff[:] = u_in.reshape(self.Nx, self.Ny)
        self.ifft()
        u_out[:] = self.out_buff.flatten()
        return u_out