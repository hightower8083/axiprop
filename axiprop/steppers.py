# Copyright 2023
# Authors: Igor Andriyash
# License: GNU GPL v3
"""
Axiprop base_classes file

This file contains stepper classes of axiprop:
- StepperNonParaxial
- StepperFresnel
"""
import numpy as np
from scipy.constants import mu_0, c

try:
    from tqdm.auto import tqdm
    tqdm_available = True
    bar_format='{l_bar}{bar}| {elapsed}<{remaining} [{rate_fmt}{postfix}]'
except Exception:
    tqdm_available = False


class StepperNonParaxial:
    """
    Class of steppers for non-paraxial propagators. Contains methods to:
    - perform a single-step calculation;
    - perform a multi-step calculation;

    This class should to be used to derive the actual Propagators
    by adding proper methods for the Transverse Spectral Transforms (TST).
    """
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

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])
            self.TST()

            phase_loc = self.kz[ikz]**2 - self.kr2
            phase_loc = self.bcknd.sqrt( (phase_loc>=0.)*phase_loc )
            self.u_ht *= self.bcknd.exp( 1j * dz * phase_loc )

            self.iTST()
            u_step[ikz] = self.bcknd.to_host(self.u_iht)
            if tqdm_available and show_progress:
                pbar.update(1)

        if tqdm_available and show_progress:
            pbar.close()

        return u_step

    def steps(self, u, dz=None, z_axis=None, show_progress=True):
        """
        Propagate wave `u` over the multiple steps.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        dz: array of floats (m)
            Steps over which wave should be propagated.

        z_axis: array of floats (m) (optional)
            Axis over which wave should be propagated. Overrides dz.

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the propagated field.
        """
        assert u.dtype == self.dtype
        if z_axis is not None:
            dz = np.r_[z_axis[0], np.diff(z_axis)]

        Nsteps = len(dz)
        if Nsteps==0:
            return None

        u_steps = np.empty( (Nsteps, self.Nkz, *self.shape_trns_new),
                         dtype=u.dtype)

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz*Nsteps, bar_format=bar_format)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz])
            self.TST()
            k_loc = self.bcknd.sqrt(self.bcknd.abs( self.kz[ikz]**2 - \
                                                     self.kr2 ))
            for i_step in range(Nsteps):
                self.u_ht *= self.bcknd.exp(1j * dz[i_step] * k_loc )
                self.iTST()
                u_steps[i_step, ikz, :] = self.bcknd.to_host(self.u_iht)

                if tqdm_available and show_progress:
                    pbar.update(1)
                elif show_progress and not tqdm_available:
                    print(f"Done step {i_step} of {Nsteps} "+ \
                          f"for wavelength {ikz+1} of {self.Nkz}",
                          end='\r', flush=True)

        if tqdm_available and show_progress:
            pbar.close()

        return u_steps

class StepperFresnel:
    """
    Class of steppers for paraxial propagators. Contains methods to:
    - perform a single-step calculation;
    - perform a multi-step calculation;

    This class should to be used to derive the actual Propagators
    by adding proper methods for the Transverse Spectral Transforms (TST).
    """
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

        self.check_new_grid(dz)

        for ikz in range(self.Nkz):
            self.u_loc[:] = self.bcknd.to_device(u[ikz,:])
            self.u_loc *= self.bcknd.exp(0.5j * self.kz[ikz] / dz * self.r2)
            self.TST()
            u_slice_loc = self.bcknd.to_host(self.u_ht)

            r_loc,  r2_loc = self.get_local_grid(dz, ikz)
            phase_loc = self.kz[ikz] * dz * (1 + 0.5 * r2_loc / (dz*dz) )
            coef_loc = self.kz[ikz] / (1j * 2 * np.pi * dz)
            u_slice_loc *= coef_loc * np.exp( 1j * phase_loc )
            u_slice_loc = self.gather_on_new_grid(u_slice_loc, r_loc, self.r_new)
            u_step[ikz] = u_slice_loc.copy()

            if tqdm_available and show_progress:
                pbar.update(1)

        if tqdm_available and show_progress:
            pbar.close()

        return u_step

    def steps(self, u, z_axis, show_progress=True):
        """
        Propagate wave `u` over the multiple steps.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        z_axis: array of floats (m)
            Axis over which wave should be propagated.

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the propagated field.
        """
        assert u.dtype == self.dtype
        Nsteps = len(z_axis)
        if Nsteps==0:
            return None

        u_steps = np.empty( (Nsteps, self.Nkz, *self.shape_trns_new),
                         dtype=u.dtype)

        if tqdm_available and show_progress:
            pbar = tqdm(total=Nsteps, bar_format=bar_format)

        for i_step, z_dest in enumerate(z_axis):
            u_steps[i_step] = self.step(u, z_dest)
            if tqdm_available and show_progress:
                pbar.update(1)

        if tqdm_available and show_progress:
            pbar.close()

        return u_steps



class StepperNonParaxialPlasma:
    """
    Class of steppers for non-paraxial propagators. Contains methods to:
    - XXXXXX perform a single-step calculation;
    - XXXXXX perform a multi-step calculation;

    This class should to be used to derive the actual Propagators
    by adding proper methods for the Transverse Spectral Transforms (TST).
    """

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
        self.z_propagation = 0.0

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz,:])
            self.TST()
            self.stepping_image[ikz] = self.u_ht.copy()

    def stepping_simple_sclr(self, k_p, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This mode
        allows computation of the consequent steps with access to the
        result on each step. In contrast to `step` can operate the
        `PropagatorResampling` class.

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

            phase_term = self.kz[ikz]**2 - self.kr2 - k_p**2
            phase_term = self.bcknd.sqrt(phase_term * (phase_term>0.0))

            self.stepping_image[ikz] *= self.bcknd.exp(1j * dz * phase_term )

            self.u_ht[:] = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out

    def stepping_simple(self, kp2_E, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This mode
        allows computation of the consequent steps with access to the
        result on each step. In contrast to `step` can operate the
        `PropagatorResampling` class.

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

            E_0 = self.stepping_image[ikz]
            E_0_abs = self.bcknd.to_host(self.bcknd.abs(E_0))
            E_0_abs_max = E_0_abs.max()
            E_0_mask = self.bcknd.to_device(
                np.double( E_0_abs > 1.0e-10 * E_0_abs.max() )
                )

            self.u_loc = self.bcknd.to_device(kp2_E[ikz,:])
            self.TST()

            k2_p = self.u_ht.copy()  / E_0
            #* self.bcknd.where(
            #    E_0_mask, 1.0 / E_0, 0.0)

            phase_term = self.kz[ikz]**2 - self.kr2 - k2_p
            phase_term = self.bcknd.sqrt(phase_term * (phase_term>0.0))

            self.stepping_image[ikz] *= self.bcknd.exp(1j * dz * phase_term )

            self.u_ht[:] = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out

    def stepping(self, J_plasma, dz, u_out=None):
        """
        Perform a step in the stepped propagation mode. This mode
        allows computation of the consequent steps with access to the
        result on each step. In contrast to `step` can operate the
        `PropagatorResampling` class.

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

            E_0 = self.stepping_image[ikz]
            E_0_abs = self.bcknd.to_host(self.bcknd.abs(E_0))
            E_0_abs_max = E_0_abs.max()
            E_0_mask = self.bcknd.to_device(
                np.double( E_0_abs > 1.0e-10 * E_0_abs.max() )
                )

            self.u_loc = self.bcknd.to_device(J_plasma[ikz,:])
            self.TST()

            J_plasma_ht = self.u_ht.copy() * (1j * self.kz[ikz] * c * mu_0) \
                * self.bcknd.where(E_0_mask, 1.0 / E_0, 0.0)

            phase_term = self.kz[ikz]**2 - self.kr2 - J_plasma_ht
            phase_mask = self.bcknd.to_device(
                np.double(np.real( self.bcknd.to_host(phase_term) ) > 0.0) )

            phase_term = self.bcknd.sqrt(phase_term)
            phase_term *= phase_mask

            self.stepping_image[ikz] *= self.bcknd.exp(1j * dz * phase_term )

            self.u_ht[:] = self.stepping_image[ikz].copy()
            self.iTST()
            u_out[ikz] = self.bcknd.to_host(self.u_iht)

        self.z_propagation += dz
        return u_out
