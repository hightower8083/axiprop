# Copyright 2020
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop main file

This file contains main classes of axiprop:
- PropagatorCommon
- PropagatorSymmetric
- PropagatorResampling
- PropagatorFFT2
- PropagatorFFTW
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

try:
    import mkl_fft
    fft2 = mkl_fft._numpy_fft.fft2
    ifft2 = mkl_fft._numpy_fft.ifft2
except Exception:
    fft2 = np.fft.fft2
    ifft2 = np.fft.ifft2

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
            Number of nodes of the x-grid. Better be an odd number,
            in order to make a symmteric grid.

        Ny: int
            Number of nodes of the y-grid.Better be an odd number,
            in order to make a symmteric grid.

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

        if Nx==1 and Ny>1:
            self.x = np.array( [0.0, ] )
            dx = dy

        if Ny==1 and Nx>1:
            self.y = np.array( [0.0, ] )
            dy = dx

        self.kx = 2 * np.pi * np.fft.fftfreq(Nx, dx)
        self.ky = 2 * np.pi * np.fft.fftfreq(Ny, dy)

        self.r = np.sqrt(self.x[:,None]**2 + self.y[None,:]**2 )
        self.kr = np.sqrt(self.kx[:,None]**2 + self.ky[None,:]**2)
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
        assert u.dtype == self.dtype
        u_out = np.empty((self.Nkz, *self.shape_trns_new),
                         dtype=self.dtype)

        for ikz in range(self.Nkz):
            self.u_loc[:] = u[ikz,:]
            self.TST()
            self.u_ht *= np.exp(1j*dz * np.sqrt(self.kz[ikz]**2 - self.kr**2))
            self.iTST()
            u_out[ikz, :] = self.u_iht

        return u_out

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
        assert u.dtype == self.dtype
        Nsteps = len(dz)
        if Nsteps==0:
            return None

        u_steps = np.empty((Nsteps, self.Nkz, *self.shape_trns_new),
                           dtype=self.dtype)

        if verbose:
            print('Propagating the wave:')

        for ikz in range(self.Nkz):
            self.u_loc[:] = u[ikz,:]
            self.TST()
            ik_loc = 1j * np.sqrt(self.kz[ikz]**2 - self.kr**2)
            for i_step in range(Nsteps):
                self.u_ht *= np.exp( dz[i_step] * ik_loc )
                self.iTST()
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

        self.shape_trns_new = (self.Nr_new,)

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr_new, dtype=dtype)

    def TST(self):
        """
        Forward QDHT transform.
        """
        self.u_loc = self.u_loc/self._j
        self.u_ht = np.dot(self.TM.astype(self.dtype), self.u_loc,
                           out=self.u_ht)

    def iTST(self):
        """
        Inverse QDHT transform.
        """
        self.u_iht = np.dot(self.TM[:self.Nr_new].astype(self.dtype),
                       self.u_ht, out=self.u_iht)
        self.u_iht *= self._j[:self.Nr_new]

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

        self.shape_trns_new = (self.Nr_new,)

        self.u_loc = np.zeros(self.Nr, dtype=dtype)
        self.u_ht = np.zeros(self.Nr, dtype=dtype)
        self.u_iht = np.zeros(self.Nr_new, dtype=dtype)

    def TST(self):
        """
        Forward DHT transform.
        """
        self.u_ht = np.dot(self.TM.astype(self.dtype), self.u_loc,
                           out=self.u_ht)

    def iTST(self):
        """
        Inverse DHT transform.
        """
        self.u_iht = np.dot(self.invTM.astype(self.dtype), self.u_ht,
                       out=self.u_iht)

class PropagatorFFT2(PropagatorCommon):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST.

    Contains methods to:
    - setup TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;

    This class uses either Intel's `mkl_fft` or if unavailable the serial Numpy `fft`.
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

        self.shape_trns_new = (Nx, Ny)

        self.u_loc = np.zeros((Nx, Ny), dtype=dtype)
        self.u_ht = np.zeros((Nx, Ny), dtype=dtype)
        self.u_iht = np.zeros((Nx, Ny), dtype=dtype)

    def TST(self):
        """
        Forward FFT transform.
        """
        self.u_ht[:] = fft2(self.u_loc, norm='ortho')

    def iTST(self):
        """
        Inverse FFT transform.
        """
        self.u_iht[:] = ifft2(self.u_ht, norm='ortho')

class PropagatorFFTW(PropagatorCommon):
    """
    Class for the propagator with two-dimensional Fast Fourier transform (FFT2)
    for TST.

    Contains methods to:
    - setup FFTW object and TST data buffers;
    - perform a forward FFT;
    - perform a inverse FFT;

    This class uses FFTW library via `pyfftw` wrapper, and can be used with
    multiple processors. This method can be ~2 times faster than serial
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

        self.shape_trns_new = (Nx, Ny)

        self.u_loc = pyfftw.empty_aligned((Nx, Ny), dtype=dtype)
        self.u_ht = pyfftw.empty_aligned((Nx, Ny), dtype=dtype)
        self.u_iht = pyfftw.empty_aligned((Nx, Ny), dtype=dtype)

        self.fft = pyfftw.FFTW( self.u_loc, self.u_ht, axes=(-1,0),
            direction='FFTW_FORWARD', flags=('FFTW_MEASURE', ), threads=threads)
        self.ifft = pyfftw.FFTW( self.u_ht, self.u_iht, axes=(-1,0),
            direction='FFTW_BACKWARD', flags=('FFTW_MEASURE', ), threads=threads,
            normalise_idft=True)

    def TST(self):
        """
        Forward FFT transform.
        """
        self.fft()

    def iTST(self):
        """
        Inverse FFT transform.
        """
        self.ifft()
