import numpy as np
from axiprop.containers import ScalarFieldEnvelope

LaserEnergy = 1.0
w0 = 10.e-6
tau = 30e-15
lambda0 = 0.8e-6

Nt = 128
k0 = 2 * np.pi / lambda0
t_axis = np.linspace( -3.5 * tau, 3.5 * tau, Nt )

def gaussian_rt():
    Nr = 512
    r_axis = np.linspace( 0.0, 3.5*w0, Nr )

    container = ScalarFieldEnvelope(k0, t_axis)
    LaserObject = container.make_gaussian_pulse(
            r_axis, tau, w0, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject

def gaussian_xyt():
    Nx = 256
    x = np.linspace( -3.5*w0, 3.5*w0, Nx )
    y = np.linspace( -3.5*w0, 3.5*w0, Nx )
    r = np.sqrt( x[:,None]**2 + y[None,:]**2 )

    container = ScalarFieldEnvelope(k0, t_axis)
    LaserObject = container.make_gaussian_pulse(
            (r, x, y), tau, w0, Energy=LaserEnergy, n_ord=2
    )
    return LaserObject

def check_energy(LaserObject):
    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=1e-2)

def check_waist_rt(LaserObject):
    assert np.allclose(LaserObject.w0, w0, rtol=3e-3)
    assert np.allclose(LaserObject.w0_ft, w0, rtol=3e-3)

def check_waist_xyt(LaserObject):
    w0_est = LaserObject.w0
    w0_est_ft = LaserObject.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)
    assert np.allclose(w0_est_ft[0], w0, rtol=1e-7)
    assert np.allclose(w0_est_ft[1], w0, rtol=1e-7)

def check_imports(LaserObject, r_axis, check_waist):
    container = ScalarFieldEnvelope(k0, t_axis)
    LaserObject = container.import_field(
        LaserObject.Field, r_axis=r_axis, make_copy=True
    )
    check_energy(LaserObject)
    check_waist(LaserObject)

    container = ScalarFieldEnvelope(k0, t_axis)
    LaserObject = container.import_field_ft(
        LaserObject.Field_ft, r_axis=r_axis, make_copy=True
    )
    check_energy(LaserObject)
    check_waist(LaserObject)

def test_all():
    laser_rt = gaussian_rt()
    laser_xyt = gaussian_xyt()

    lasers = [ laser_rt, laser_xyt]
    check_waist_methods = [check_waist_rt, check_waist_xyt]
    r_axes = [laser_rt.r, (laser_xyt.r, laser_xyt.x, laser_xyt.y)]
    geometries = zip(lasers, check_waist_methods, r_axes)

    for LaserObject, check_waist, r_axis in geometries:
        check_energy(LaserObject)
        check_waist(LaserObject)
        check_imports(LaserObject, r_axis, check_waist)
