import numpy as np
from axiprop.containers import ScalarField as ContainerFull
from axiprop.containers import ScalarFieldEnvelope as ContainerEnv

from axiprop.lib import PropagatorResampling
from axiprop.lib import PropagatorSymmetric
from axiprop.lib import PropagatorResamplingFresnel

from axiprop.utils import mirror_parabolic

from scipy.constants import c

LaserEnergy = 1.0
R_las = 10e-3
tau = 30e-15
lambda0 = 0.8e-6

k0 = 2 * np.pi / lambda0
f_N = 200
f0 = 2 * R_las * f_N
w0 = 2 / np.pi * lambda0 * f_N

R0 = 3.5 * R_las
R1 = 3.5 * w0

def propagator_resample(container):
    Nr0 = 1024 * 4
    Nr1 = 256
    return PropagatorResampling(
        r_axis=(R0, Nr0), kz_axis=container().k_freq,
        r_axis_new=(R1, Nr1) )


def propagator_symmetric(container):
    Nr0 = 512 * 8
    Nr1 = int( R1 / R0 * Nr0 )
    return PropagatorSymmetric(
        r_axis=(R0, Nr0), kz_axis=container().k_freq,
        r_axis_new=(Nr1,) )

def propagator_resample_fresnel(container):
    Nr0 = 512
    Nr1 = 128
    return PropagatorResamplingFresnel(
        r_axis=(R0, Nr0), kz_axis=container().k_freq,
        dz=f0, r_axis_new=(R1, Nr1)
    )

def container_env():
    Nt = 128
    t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )
    return ContainerEnv(k0, t_axis)

def container_full():
    Nt = 768
    t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )
    return ContainerFull(k0, t_axis, 8/tau)

def gaussian_rt(container, r_axis):
    LaserObject = container.make_gaussian_pulse(
            r_axis, tau, R_las, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject

def check_energy(LaserObject):
    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=5e-2, atol=0)
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=5e-2, atol=0)

def check_tau(LaserObject):
    assert np.allclose(LaserObject.tau, tau, rtol=3e-4, atol=0)

def check_waist_rt(LaserObject):
    assert np.allclose(LaserObject.w0, w0, rtol=5e-2, atol=0)
    assert np.allclose(LaserObject.w0_ft, w0, rtol=5e-2, atol=0)

def check_waist_xyt(LaserObject):
    w0_est = LaserObject.w0
    w0_est_ft = LaserObject.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-7, atol=0)
    assert np.allclose(w0_est[1], w0, rtol=1e-7, atol=0)
    assert np.allclose(w0_est_ft[0], w0, rtol=1e-7, atol=0)
    assert np.allclose(w0_est_ft[1], w0, rtol=1e-7, atol=0)


def test_propagate():

    for container in [
            container_env,
            container_full
        ]:
        for propagator_method in [
                propagator_resample_fresnel,
                propagator_resample,
                propagator_symmetric
            ]:
            prop = propagator_method(container)
            LaserObject = gaussian_rt(container(), prop.r)

            E0 = LaserObject.Field_ft.copy()
            E0 *= mirror_parabolic(f0, prop.kz, prop.r )
            E1 = prop.step(E0, f0)

            container_new = container()
            container_new.t += f0/c
            container_new.t_loc += f0/c
            LaserObject = container_new.import_field_ft(
                E1, r_axis=prop.r_new
            )

            check_tau(LaserObject)
            check_waist_rt(LaserObject)
            check_energy(LaserObject)
