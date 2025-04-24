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
f_N = 300
f0 = 2 * R_las * f_N
w0 = 2 / np.pi * lambda0 * f_N


def container_env():
    Nt = 32
    t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )
    return ContainerEnv(k0, t_axis)

def propagator_fft2_fresnel(container, dz):
    Nx = 128
    Ny = 129
    Lxy = 5 * R_las

    Nx_new = 32
    Ny_new = 33
    Lxy_new = 5 * w0
    return PropagatorFFT2Fresnel(
        x_axis=(Lxy, Nx),
        y_axis=(Lxy, Ny),
        dz=dz,
        x_axis_new=(Lxy_new, Nx_new),
        y_axis_new=(Lxy_new, Ny_new),
        kz_axis=container().k_freq,
        )

def propagator_fft2(container, dz):
    Nx = 1024
    Ny = 1028
    Lxy = 5 * R_las
    return PropagatorFFT2(
        x_axis=(Lxy, Nx),
        y_axis=(Lxy, Ny),
        kz_axis=container().k_freq,
        )

def gaussian_xyt(container, r_axis):
    LaserObject = container.make_gaussian_pulse(
            r_axis, tau, R_las, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject

def check_energy(LaserObject):
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=5e-2, atol=0)
    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=5e-2, atol=0)

def check_tau(LaserObject):
    assert np.allclose(LaserObject.tau, tau, rtol=1e-3, atol=0)

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
                propagator_fft2_fresnel,
                propagator_fft2,
            ]:
            prop = propagator_method(container, f0)
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
            check_energy(LaserObject)
