# Copyright 2023
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop extras file

This file contains extras classes of axiprop:
- PropagatorExtras
"""
import numpy as np


class PropagatorExtras:
    def apply_boundary(self, u, nr_boundary=16):
        # apply the boundary "absorbtion"
        absorb_layer_axis = np.r_[0 : np.pi/2 : nr_boundary*1j]
        absorb_layer_shape = np.cos(absorb_layer_axis)**0.5
        absorb_layer_shape[-1] = 0.0
        u[:, -nr_boundary:] *= absorb_layer_shape
        return u

    def initiate_stepping(self, u):
        """
        Initiate the stepped propagation mode. This mode allows computation
        of the consequent steps with access to the result on each step.
        In contrast to `step` can operate the `PropagatorResampling` class.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """
        assert u.dtype == self.dtype

        self.stepping_image = self.bcknd.to_device( np.zeros_like(u) )
        self.phase_loc = self.bcknd.to_device( np.zeros_like(u) )
        self.z_propagation = 0.0

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])
            self.TST()

            self.stepping_image[ikz] = self.u_ht.copy()

            phase_loc = self.kz[ikz]**2 - self.kr2
            self.phase_loc[ikz] = self.bcknd.sqrt((phase_loc >= 0.)*phase_loc)

    def stepping(self, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This mode allows computation
        of the consequent steps with access to the result on each step.
        In contrast to `step` can operate the `PropagatorResampling` class.

        Parameters
        ----------
        dz: float (m)
            Step over which wave should be propagated.

        u_out: 2darray of complex or double (optional)
            Array to which data should be written.
            If not provided will be allocated.
        """
        if u_out is None:
            u_out = np.empty((self.Nkz, *self.shape_trns_new),
                              dtype=self.dtype)

        for ikz in range(self.Nkz):
            self.stepping_image[ikz] *= self.bcknd.exp( \
                1j * dz * self.phase_loc[ikz] )
            self.u_ht = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out

    def get_Ez(self, ux):
        """
        Get a longitudinal field component from the transverse field using the
        Poisson equation in vacuum DIV.E = 0.
        Parameters
        ----------
        ux: 2darray of complex or double
            Spectral-radial distribution of the field to be propagated.
        """

        uz = np.zeros_like(ux)
        kx_2d = self.kx[:,None] * np.ones_like(self.ky[None,:])
        kx_2d = self.bcknd.to_device(kx_2d)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(ux[ikz,:])
            self.TST()

            kz_loc = self.bcknd.sqrt(self.bcknd.abs( self.kz[ikz]**2 - \
                                                           self.kr2 ))
            self.u_ht *= - kx_2d / kz_loc
            self.iTST()
            uz[ikz] = self.bcknd.to_host(self.u_iht)

        return uz