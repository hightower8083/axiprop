# Axiprop
A simple tool to compute optical propagation, based on the discrete Hankel and 
Fourier transforms.

## Contents

This library contains methods and convenience tools to model propagation of the 3D optical
field. 

### Basic propagation methods include:
- `PropagatorSymmetric`: cylindical axisymmetric propagator with the symmetric DHT proposed in 
[[M. Guizar-Sicairos, J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)](https://doi.org/10.1364/JOSAA.21.000053)]
- `PropagatorResampling`: cylindical axisymmetric propagator with a more generic DHT which allows for arbitrary
sampling of radial axis [[K. Oubrerie, I.A. Andriyash et al, J. Opt. 24, 045503 (2022)](https://doi.org/10.1088/2040-8986/ac57d2)]
- `PropagatorResamplingFresnel`: cylindical axisymmetric propagator based on the Fresnel approximation as given by `Eq. (4-17)` of [JW Goodman _Introduction to Fourier Optics_] and is suited for translations between far and near fields.
- `PropagatorFFT2`: fully 3D FFT-based propagator
- `PropagatorFFT2Fresnel`: ully 3D FFT-based propagator based on the Fresnel approximation 

### Propagation with plasma models:
- `Simulation` class includes a few algorithms (`'Euler'` `'Ralston'`, `'MP'`, `'RK4'`), method to provide adjustive steps, diagnostics
- Plasma models: `PlasmaSimple` (linear current, $n_{pe}(z)$ ), `PlasmaSimpleNonuniform` (linear current, $n_{pe}(z, r)$), `PlasmaRelativistic` (non-linear current, $n_{pe}(z, r)$), `PlasmaIonization` (non-linear current and ionization, $n_{g}(z, r)$)

### Convenience tools include:
- Container classes to create and handle time-frequency transformations for the **carrier-frequency-resolved** and **enveloped** fields.
- Methods to convert fields between temporal and spatial representations

Computations can be done using a number of backends:
- [NumPy](https://numpy.org) (**CPU**) optionally enhanced via [mkl_fft](https://github.com/IntelPython/mkl_fft) or
[pyfftw](https://github.com/pyFFTW/pyFFTW)
- [CuPy](https://cupy.dev) for **GPU** calculations via Nvidia CUDA to be discontinuedI
- [PyOpenCL](https://documen.tician.de/pyopencl) for **GPU** calculations/OpenCL
- [ArrayFire](https://arrayfire.com) for **GPU** calculations via CUDA/OpenCL (to be discontinued)

## Documentation

Documentation is organized in a few sections:

 - [Equations of field propagation](./docs/main.md)


## Installation
### From PyPI
Major versions are released at [PyPI](https://pypi.org) and can be installed with `pip`:
```bash
pip install axiprop
```

### From source
You can build and install `axiprop` by cloning the source 
```bash
git clone https://github.com/hightower8083/axiprop.git
cd axiprop
python setup.py install
```
or directly from Github  with `pip`:
```bash
pip install git+https://github.com/hightower8083/axiprop.git
```

### Additional dependencies
Note that, while base backend `NP` requires only NumPy and SciPy, other 
backends have specific dependencies:
- `NP_MKL`:  [mkl_fft](https://github.com/IntelPython/mkl_fft)
- `NP_FFTW`: [pyfftw](https://github.com/pyFFTW/pyFFTW)
- `CU`: [cupy](https://cupy.dev)
- `CL`: [pyopencl](https://documen.tician.de/pyopencl) and [reikna](https://github.com/fjarri/reikna)
- `AF`:  [ArrayFire](https://arrayfire.com) and [arrayfire-python](https://github.com/arrayfire/arrayfire-python)

Optional enhancements of utilities are achieved if [Numba](https://numba.pydata.org) is installed.
