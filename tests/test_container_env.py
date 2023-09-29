# -*- coding: utf-8 -*-

import pytest
import numpy as np
from scipy.constants import c
from copy import deepcopy

from axiprop.containers import ScalarFieldEnvelope

LaserEnergy = 1.0e-3
w0 = 10.e-3
tau = 30e-15
lambda0 = 0.8e-6

Nt = 128
k0 = 2 * np.pi / lambda0
t_axis = np.linspace( -3 * tau, 4 * tau, Nt )

@pytest.fixture(scope="function")
def container():
    scl_obj = ScalarFieldEnvelope(k0, t_axis)
    return scl_obj


def test_rz(container):
    Nr = 512
    r_axis = np.linspace( 0.0, 3.5*w0, Nr )

    container_sup = deepcopy(container)

    LaserObject = container.make_gaussian_pulse(
            r_axis, tau, w0, Energy=LaserEnergy, n_ord=2
    )

    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=1e-2)
    assert np.allclose(LaserObject.w0, w0, rtol=3e-3)
    assert np.allclose(LaserObject.w0_ft, w0, rtol=3e-3)

    LaserObject_import = container_sup.import_field(
        LaserObject.Field, r_axis=r_axis
    )

    assert np.allclose(LaserObject_import.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject_import.Energy_ft, LaserEnergy, rtol=1e-2)
    assert np.allclose(LaserObject_import.w0, w0, rtol=3e-3)
    assert np.allclose(LaserObject_import.w0_ft, w0, rtol=3e-3)

    LaserObject_import = container_sup.import_field_ft(
        LaserObject.Field_ft, r_axis=r_axis
    )

    assert np.allclose(LaserObject_import.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject_import.Energy_ft, LaserEnergy, rtol=1e-2)
    assert np.allclose(LaserObject_import.w0, w0, rtol=3e-3)
    assert np.allclose(LaserObject_import.w0_ft, w0, rtol=3e-3)


def test_3d(container):
    Nx = 256
    xaxis = np.linspace( -3.5*w0, 3.5*w0, Nx )
    yaxis = np.linspace( -3.5*w0, 3.5*w0, Nx )

    r_axis = np.sqrt( xaxis[:,None]**2 + yaxis[None,:]**2 )

    container_sup = deepcopy(container)

    LaserObject = container.make_gaussian_pulse(
            (r_axis, xaxis, yaxis), tau, w0, Energy=LaserEnergy, n_ord=2
    )

    assert np.allclose(LaserObject.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject.Energy_ft, LaserEnergy, rtol=1e-2)

    w0_est = LaserObject.w0
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)

    w0_est = LaserObject.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)

    LaserObject_import = container_sup.import_field(
        LaserObject.Field, r_axis=(r_axis, xaxis, yaxis)
    )

    assert np.allclose(LaserObject_import.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject_import.Energy_ft, LaserEnergy, rtol=1e-2)

    w0_est = LaserObject_import.w0
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)

    w0_est = LaserObject_import.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)

    LaserObject_import = container_sup.import_field_ft(
        LaserObject.Field_ft, r_axis=(r_axis, xaxis, yaxis)
    )

    assert np.allclose(LaserObject_import.Energy, LaserEnergy, rtol=1e-7)
    assert np.allclose(LaserObject_import.Energy_ft, LaserEnergy, rtol=1e-2)

    w0_est = LaserObject_import.w0
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)

    w0_est = LaserObject_import.w0_ft
    assert np.allclose(w0_est[0], w0, rtol=1e-7)
    assert np.allclose(w0_est[1], w0, rtol=1e-7)
