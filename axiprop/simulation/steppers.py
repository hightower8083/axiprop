import numpy as np


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
        Parameters
        ----------
        image: 2darray of complex or double

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            out_shape = (self.Nkz, *self.shape_trns_new)
            u_out = self.bcknd.zeros( out_shape, self.dtype )

        for ikz in range(self.Nkz):
            self.u_ht[:] = image[ikz]
            self.iTST()
            u_out[ikz] = self.u_iht.copy()

        return u_out

    def perform_TST(self, u, u_out=None, stepping=True):
        """
        Initiate the stepped propagation mode. This mode allows
        computation of the consequent steps with access to the result on
        each step.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """
        if u_out is None:
            out_shape = (self.Nkz, *self.shape_trns)
            u_out = self.bcknd.zeros( out_shape, self.dtype )

        if stepping:
            for ikz in range(self.Nkz):
                self.u_iht = u[ikz,:].copy()
                self.TST_stepping()
                u_out[ikz] = self.u_ht.copy()
        else:
            for ikz in range(self.Nkz):
                self.u_loc = u[ikz,:].copy()
                self.TST()
                u_out[ikz] = self.u_ht.copy()

        return u_out

    def step_simple(self, image, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This mode
        allows computation of the consequent steps with access to the
        result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = self.bcknd.zeros( image.shape, self.dtype )

        for ikz in range(self.Nkz):
            if self.kz[ikz] <= 0:
                continue
            phase_term = self.k_z[ikz]
            u_out[ikz] = image[ikz] * self.bcknd.exp(1j * dz * phase_term)

        return u_out

    def step_and_iTST(self, image, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This
        mode allows computation of the consequent steps with
        access to the result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            out_shape = (self.Nkz, *self.shape_trns_new)
            u_out = self.bcknd.zeros( out_shape, self.dtype )

        for ikz in range(self.Nkz):
            if self.kz[ikz] <= 0:
                continue

            phase_term = self.k_z[ikz]
            self.u_ht[:] = image[ikz] * self.bcknd.exp(1j * dz * phase_term)
            self.iTST()
            u_out[ikz] = self.u_iht.copy()

        return u_out

    def perform_iTST_transfer(self, image, u_out=None):
        """
        Perform a step in the stepped propagation mode with account
        for the plasma refraction. This mode allows computation of the
        consequent steps with access to the result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            out_shape = (self.Nkz, *self.shape_trns_new)
            u_out = np.zeros( out_shape, dtype=self.dtype )

        for ikz in range(self.Nkz):
            self.u_ht[:] = image[ikz]
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        return u_out

    def perform_transfer_TST(self, u, image=None, stepping=True):
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

        if image is None:
            out_shape = (self.Nkz, *self.shape_trns)
            image = self.bcknd.zeros( out_shape, self.dtype )

        if stepping:
            for ikz in range(self.Nkz):
                self.u_iht = self.bcknd.to_device(u[ikz,:].copy())
                self.TST_stepping()
                image[ikz] = self.u_ht.copy()
        else:
            for ikz in range(self.Nkz):
                self.u_loc = self.bcknd.to_device(u[ikz,:].copy())
                self.TST()
                image[ikz] = self.u_ht.copy()

        return image

    def step_and_iTST_transfer(self, image, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This
        mode allows computation of the consequent steps with
        access to the result on each step.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            out_shape = (self.Nkz, *self.shape_trns_new)
            u_out = np.empty(out_shape, dtype=self.dtype)

        for ikz in range(self.Nkz):
            if self.kz[ikz] <= 0:
                continue

            phase_term = self.k_z[ikz]
            self.u_ht[:] = image[ikz] * self.bcknd.exp(1j * dz * phase_term)
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        return u_out
