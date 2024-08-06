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

    def init_TST_resampled(self):
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
            self.init_TST_resampled()

        self.u_ht = self.TST_resampled_matmul(
            self.TM_resampled, self.u_loc, self.u_ht)

class PropagatorSymmetricStepping(
    PropagatorSymmetric, StepperNonParaxialPlasma):
    """
    A propagator with account for plasma response,
    based on `PropagatorSymmetric`
    """
    TST_stepping = PropagatorSymmetric.TST
