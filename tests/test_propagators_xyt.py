import numpy as np
from axiprop.containers import ScalarFieldEnvelope as ContainerEnv
from axiprop.lib import PropagatorFFT2Fresnel, PropagatorFFT2
from axiprop.utils import mirror_parabolic

from scipy.constants import c

LaserEnergy = 1.0
R_las = 10e-3
tau = 30e-15
lambda0 = 0.8e-6

k0 = 2 * np.pi / lambda0
f_N = 70
f0 = 2 * R_las * f_N
w0 = 2 / np.pi * lambda0 * f_N

def container_env():
    Nt = 16
    t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )
    return ContainerEnv(k0, t_axis)

def propagator_fft2_fresnel(container):
    Nxy = 256
    Lxy = 5 * R_las
    return PropagatorFFT2Fresnel(
        x_axis=(Lxy, Nxy),
        y_axis=(Lxy, Nxy),
        Nx_new=Nxy,
        Ny_new=Nxy,
        kz_axis=container().k_freq,
        N_pad=2,
        )

def propagator_fft2(container):
    Nxy = 1024 * 3
    Lxy = 5 * R_las
    return PropagatorFFT2(
        x_axis=(Lxy, Nxy),
        y_axis=(Lxy, Nxy),
        kz_axis=container().k_freq,
        )

def gaussian_xyt(container, r_axis):
    LaserObject = container.make_gaussian_pulse(
            r_axis, tau, R_las, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject


def check_energy(LaserObject):
    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=1e-7, atol=0)
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=1e-2, atol=0)

def check_tau(LaserObject):
    assert np.allclose(LaserObject.tau, tau, rtol=3e-4, atol=0)

def check_waist_xyt(LaserObject):
    w0_est = LaserObject.w0
    w0_est_ft = LaserObject.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-2, atol=0)
    assert np.allclose(w0_est[1], w0, rtol=1e-2, atol=0)
    assert np.allclose(w0_est_ft[0], w0, rtol=1e-2, atol=0)
    assert np.allclose(w0_est_ft[1], w0, rtol=1e-2, atol=0)


def test_propagate():

    for container in [container_env, ]:
        for propagator_method in [
            propagator_fft2_fresnel, propagator_fft2,
            ]:
            prop = propagator_method(container)
            if hasattr(prop, 'x0'):
                LaserObject = gaussian_xyt(container(), (prop.r, prop.x0, prop.y0))
            else:
                LaserObject = gaussian_xyt(container(), (prop.r, prop.x, prop.y))

            E0 = LaserObject.Field_ft.copy()
            E0 *= mirror_parabolic(f0, prop.kz, prop.r )
            E1 = prop.step(E0, f0)

            r_axis_new = (
                np.sqrt(prop.x[:, None]**2 + prop.y[None,:]**2), prop.x, prop.y
            )

            container_new = container()
            container_new.t += f0/c
            container_new.t_loc += f0/c
            LaserObject = container_new.import_field_ft(
                E1, r_axis=r_axis_new
            )

            check_tau(LaserObject)
            check_waist_xyt(LaserObject)
            check_tau(LaserObject)
