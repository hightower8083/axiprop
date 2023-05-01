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
from scipy.constants import c

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
            self.u_loc = self.bcknd.to_device(u[ikz,:].copy())
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
            self.u_loc = self.bcknd.to_device(u[ikz].copy())
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

    def t2z(self, u, z_axis=None, z0=0.0, t0=0.0, show_progress=True):
        """
        Reconstruct wave `u` in the spatial domain.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to propagate.

        z_axis: array of floats (m)
            Axis over which wave should be reconstructed.

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the reconstructed field.
        """
        assert u.dtype == self.dtype

        Nsteps = len(z_axis)
        if Nsteps==0:
            return None

        u_steps = np.zeros( (Nsteps, *self.shape_trns_new),
                            dtype=u.dtype)

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz*Nsteps, bar_format=bar_format)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz].copy())
            self.TST()
            k_loc = self.bcknd.sqrt(self.bcknd.abs( self.kz[ikz]**2 - \
                                                     self.kr2 ))

            u_ht0 = self.u_ht.copy() * np.exp( 1j * self.kz[ikz] * c * t0 )
            for i_step in range(Nsteps):
                self.u_ht[:] = u_ht0 * \
                    self.bcknd.exp( 1j * (z_axis[i_step]-z0) * k_loc )
                self.iTST()
                u_steps[i_step, :] += self.bcknd.to_host(self.u_iht )

                if tqdm_available and show_progress:
                    pbar.update(1)
                elif show_progress and not tqdm_available:
                    print(f"Done step {i_step} of {Nsteps} "+ \
                          f"for wavelength {ikz+1} of {self.Nkz}",
                          end='\r', flush=True)

        if tqdm_available and show_progress:
            pbar.close()

        return u_steps

    def z2t(self, u, t_axis, z0=0.0, t0=0.0, show_progress=True):
        """
        Reconstruct wave `u` in the temporal domain.

        Parameters
        ----------
        u: 2darray of complex or double
            Spectral-radial distribution of the field to convert.

        t_axis: array of floats (s)
            Axis over which wave should be reconstructed.

        Returns
        -------
        u: 3darray of complex or double
            Array with the steps of the propagated field.
        """
        assert u.dtype == self.dtype

        Nsteps = len(t_axis)
        if Nsteps==0:
            return None

        u_steps = np.zeros( (Nsteps, *self.shape_trns_new),
                         dtype=u.dtype)

        if tqdm_available and show_progress:
            pbar = tqdm(total=self.Nkz*Nsteps, bar_format=bar_format)

        for ikz in range(self.Nkz):
            self.u_loc = self.bcknd.to_device(u[ikz].copy())
            self.TST()
            k_loc = self.bcknd.sqrt(self.bcknd.abs( self.kz[ikz]**2 + \
                                                     self.kr2 ))

            u_ht0 = self.u_ht.copy() * np.exp( 1j * self.kz[ikz] * z0 )

            for i_step in range(Nsteps):
                self.u_ht[:] = u_ht0 * self.bcknd.exp(
                    -1j * k_loc * c * ( t_axis[i_step] - t0 )
                )
                self.iTST()
                u_steps[i_step, :] += self.bcknd.to_host(self.u_iht)

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
            self.u_loc[:] = self.bcknd.to_device(u[ikz,:].copy())
            self.u_loc *= self.bcknd.exp(0.5j * self.kz[ikz] / dz * self.r2)
            self.TST()
            u_slice_loc = self.bcknd.to_host(self.u_ht)

            r_loc,  r2_loc = self.get_local_grid(dz, ikz)
            phase_loc = self.kz[ikz] * dz * (1 + 0.5 * r2_loc / (dz*dz) )
            coef_loc = self.kz[ikz] / (1j * 2 * np.pi * dz)
            u_slice_loc *= coef_loc * np.exp( 1j * phase_loc )
            u_slice_loc = self.gather_on_new_grid(
                u_slice_loc, r_loc, self.r_new)
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
