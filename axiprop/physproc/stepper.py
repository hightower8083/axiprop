import numpy as np
from tqdm.auto import tqdm
from scipy.constants import e, m_e, c, pi, epsilon_0, fine_structure, mu_0
from mendeleev import element as table_element
from scipy.special import gamma as gamma_func

from ..lib import PropagatorResampling

mc2_to_eV =  m_e*c**2/e


class StepperNonParaxialPlasma:
    """
    Class of steppers for the non-paraxial propagators with account
    for the plasma response. Contains methods to:
    - initiate the stepped propagation mode a single-step calculation;
    - perform a propagation step with account for the plasma refraction;
    - perform a propagation step with account for the arbitrary plasma current;

    This class should to be used to derive the actual Propagators
    by adding proper methods for the Transverse Spectral Transforms (TST).
    """

    def perform_iTST(self, image, u_out=None):
        """
        Perform a step in the stepped propagation mode with account 
        for the plasma refraction. This mode allows computation of the 
        consequent steps with access to the result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        kp2: float (m^-2) (optional)
            Square of plasma wavenumber

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = np.zeros( image.shape, dtype=self.dtype )

        for ikz in range(self.Nkz):

            self.u_ht[:] = image[ikz]
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        return u_out

    def perform_TST(self, u):
        """
        Initiate the stepped propagation mode. This mode allows 
        computation of the consequent steps with access to the result on 
        each step.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """
        assert u.dtype == self.dtype

        image = self.bcknd.zeros( u.shape, self.dtype )

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:].copy())
            self.TST()
            image[ikz] = self.u_ht.copy()

        return image

    def step_simple(self, image, dz, kp=0.0, u_out=None):
        """
        Perform a step in the stepped propagation mode with account 
        for the plasma refraction. This mode allows computation of the 
        consequent steps with access to the result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        kp: float (m^-1) (optional)
            Plasma wavenumber (uniform)

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = self.bcknd.zeros( image.shape, self.dtype )

        for ikz in range(self.Nkz):

            phase_term = self.kz[ikz]**2 - self.kr2 - kp**2
            phase_term_mask = (phase_term>0.0)
            phase_term = self.bcknd.sqrt(
                self.bcknd.abs(phase_term) * phase_term_mask
            )

            u_out[ikz] = image[ikz] * self.bcknd.exp(1j * dz * phase_term)

        return u_out

    def step_and_iTST(self, image, dz, kp=0.0, u_out=None):
        """
        Perform a step in the stepped propagation mode with account 
        for the plasma refraction. This mode allows computation of the 
        consequent steps with access to the result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        kp: float (m^-1) (optional)
            Plasma wavenumber (uniform)

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = np.empty((self.Nkz, *self.shape_trns_new),
                              dtype=self.dtype)

        for ikz in range(self.Nkz):

            phase_term = self.kz[ikz]**2 - self.kr2 - kp**2
            phase_term_mask = (phase_term>0.0)
            phase_term = self.bcknd.sqrt(
                self.bcknd.abs(phase_term) * phase_term_mask
            )

            self.u_ht[:] = image[ikz] * self.bcknd.exp(1j * dz * phase_term)
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        return u_out

class PropagatorResamplingStepping(
    PropagatorResampling, StepperNonParaxialPlasma):
    """
    A propagator with account for plasma response,
    based on `PropagatorResampling`
    """

    def init_TST_resampled(self):
        """
        Setup DHT transform
        """
        Nr = self.Nr
        Nr_new = self.Nr_new
        r = self.r
        r_new = self.r_new
        kr = self.kr
        dtype = self.dtype
        mode = self.mode

        invTM = jn(mode, r_new[:,None] * kr[None,:])
        self.TM_resampled = self.bcknd.inv_on_host(invTM, dtype)
        self.TM_resampled = self.bcknd.to_device(self.TM_resampled)
        self.TST_resampled_matmul = self.bcknd.make_matmul(
            self.TM_resampled, self.u_iht, self.u_ht)

    def TST_stepping(self):
        """
        Forward QDHT transform.
        """
        if not hasattr(self, 'TST_resampled_matmul'):
            self.init_TST_resampled()

        self.u_ht = self.TST_resampled_matmul(
            self.TM_resampled, self.u_loc, self.u_ht)
