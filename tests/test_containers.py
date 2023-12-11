import numpy as np
from axiprop.containers import ScalarField as ContainerFull
from axiprop.containers import ScalarFieldEnvelope as ContainerEnv

LaserEnergy = 1.0
w0 = 10.e-6
tau = 30e-15
lambda0 = 0.8e-6
k0 = 2 * np.pi / lambda0

def container_env():
    Nt = 128
    t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )
    return ContainerEnv(k0, t_axis)

def container_full():
    Nt = 768
    t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )
    return ContainerFull(k0, t_axis, 8/tau)

def gaussian_rt(container):
    Nr = 512
    r_axis = np.linspace( 0.0, 3.5*w0, Nr )

    LaserObject = container.make_gaussian_pulse(
            r_axis, tau, w0, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject

def gaussian_xyt(container):
    Nx = 256
    x = np.linspace( -3.5*w0, 3.5*w0, Nx )
    y = np.linspace( -3.5*w0, 3.5*w0, Nx )
    r = np.sqrt( x[:,None]**2 + y[None,:]**2 )

    LaserObject = container.make_gaussian_pulse(
            (r, x, y), tau, w0, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject

def check_energy(LaserObject):
    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=1e-7, atol=0)
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=1e-2, atol=0)

def check_tau(LaserObject):
    assert np.allclose(LaserObject.tau, tau, rtol=2e-4, atol=0)

def check_waist_rt(LaserObject):
    assert np.allclose(LaserObject.w0, w0, rtol=3e-3, atol=0)
    assert np.allclose(LaserObject.w0_ft, w0, rtol=3e-3, atol=0)

def check_waist_xyt(LaserObject):
    w0_est = LaserObject.w0
    w0_est_ft = LaserObject.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-7, atol=0)
    assert np.allclose(w0_est[1], w0, rtol=1e-7, atol=0)
    assert np.allclose(w0_est_ft[0], w0, rtol=1e-7, atol=0)
    assert np.allclose(w0_est_ft[1], w0, rtol=1e-7, atol=0)

def check_imports(LaserObject, r_axis, check_waist, container):
    LaserObject = container.import_field(
        LaserObject.Field, r_axis=r_axis, make_copy=True
    )
    check_energy(LaserObject)
    check_waist(LaserObject)
    check_tau(LaserObject)

    LaserObject = container.import_field_ft(
        LaserObject.Field_ft, r_axis=r_axis, make_copy=True
    )
    check_energy(LaserObject)
    check_waist(LaserObject)
    check_tau(LaserObject)

def test_all():
    for container in (container_env, container_full):

        laser_rt = gaussian_rt(container())
        laser_xyt = gaussian_xyt(container())

        lasers = [ laser_rt, laser_xyt]
        check_waist_methods = [check_waist_rt, check_waist_xyt]
        r_axes = [laser_rt.r, (laser_xyt.r, laser_xyt.x, laser_xyt.y)]
        geometries = zip(lasers, check_waist_methods, r_axes)

        for LaserObject, check_waist, r_axis in geometries:
            check_energy(LaserObject)
            check_waist(LaserObject)
            check_tau(LaserObject)
            check_imports(LaserObject, r_axis, check_waist, container())
