import numpy as np
from scipy.special import jn

from ..lib import PropagatorResampling
from ..lib import PropagatorSymmetric
from .steppers import StepperNonParaxialPlasma


class PropagatorResamplingStepping(
    PropagatorResampling, StepperNonParaxialPlasma):
    """
    A propagator with account for plasma response,
    based on `PropagatorResampling`
    """

    def init_TST_stepping(self):
        """
        Setup DHT transform
        """
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
            self.init_TST_stepping()

        self.u_ht = self.TST_resampled_matmul(
            self.TM_resampled, self.u_iht, self.u_ht)


class PropagatorSymmetricStepping(
    PropagatorSymmetric, StepperNonParaxialPlasma):
    """
    A propagator with account for plasma response,
    based on `PropagatorSymmetric`
    """

    def init_TST_stepping(self):
        Rmax = self.Rmax
        Nr = self.Nr
        Nr_new = self.Nr_new
        dtype = self.dtype
        mode = self.mode
        alpha = self.alpha
        alpha_np1 = self.alpha_np1

        self._j_stepping = self.bcknd.to_device(
            np.abs(jn(mode+1, alpha)) / Rmax
        )[:Nr_new]

        denominator = alpha_np1 * np.abs(jn(mode+1, alpha[:,None]) \
                                       * jn(mode+1, alpha[None,:]))

        TM = 2 * jn(mode, alpha[:,None] * alpha[None,:] / alpha_np1)\
                     / denominator

        self.TM_stepping = self.bcknd.to_device(TM[:,:Nr_new], dtype)

        self.TST_stepping_matmul = self.bcknd.make_matmul(
            self.TM_stepping, self.u_iht, self.u_ht
        )

    def TST_stepping(self):
        """
        """
        if not hasattr(self, 'TST_stepping_matmul'):
            self.init_TST_stepping()

        self.u_iht /= self._j_stepping

        self.u_ht = self.TST_stepping_matmul(
            self.TM_stepping, self.u_iht, self.u_ht)

