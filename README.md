# Axiprop
A simple tool to compute optical propagation, based on the discrete Hankel and 
Fourier transforms

### Contents

This library contains methods and convenience tools to model propagation of the 3D optical
field. Computations can be done using a number of backends:
- [NumPy](https://numpy.org) (**CPU**) optionally enhanced via [mkl_fft](https://github.com/IntelPython/mkl_fft) or
[pyfftw](https://github.com/pyFFTW/pyFFTW)
- [CuPy](https://cupy.dev) for **GPU** calculations via Nvidia CUDA API
- [ArrayFire](https://arrayfire.com) for **GPU** calculations via CUDA or OpenCL APIs
- [PyOpenCL](https://documen.tician.de/pyopencl) for **GPU** calculations via OpenCL API

Currently methods include:
- `PropagatorSymmetric`: cylindical axisymmetric propagator with the symmetric DHT proposed in 
[[M. Guizar-Sicairos, J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)](https://doi.org/10.1364/JOSAA.21.000053)]
- `PropagatorResampling`: cylindical axisymmetric propagator with a more generic DHT which allows for arbitrary
sampling of radial axis [[K. Oubrerie, I.A. Andriyash et al, J. Opt. 24, 045503 (2022)](https://doi.org/10.1088/2040-8986/ac57d2)]
- `PropagatorFFT2`: fully 3D FFT-based propagator

### Usage

Consider a laser,
```python 
k0 = 2 * np.pi / 0.8e-6            # 800 nm wavelength
tau = 35e-15/ (2*np.log(2))**0.5   # 35 fs FWHM duration
R_las = 10e-3                      # 10 mm radius

def fu_laser(kz, r):
    """
    Gaussian spot with the Gaussian temporal profile
    """
    profile_r = np.exp( -(r/R_las)**2 )
    profile_kz = np.exp( -( (kz-k0) * c * tau / 2 )**2 )
    return profile_r * profile_kz
```

and some focusing optics,
```python
f_N = 40                      # f-number f/40 
f0 = 2 * R_las * f_N          # focal length
```

Define the propagator,
```python
prop = PropagatorSymmetric((Rmax, Nr), (k0, L_kz, Nkz), Nr_end)
```
and setup the laser reflected from the focusing mirror
```python
A0 = laser_from_fu( fu_laser, prop.kz, prop.r )
A0 = A0 * mirror_parabolic( f0, prop.kz, prop.r )
```

Use `AXIPROP` to compute the field after propagation of `dz` distance 
(e.g. `dz=f0` for field at focus):
```python
A0 = prop.step(A0, f0)
```
or evaluate it at `Nsteps` along some `Distance` around the focus,
```python
dz =  Distance / Nsteps
zsteps = Nsteps * [dz,]
zsteps[0] = f0 - Distance/2
A_multi = prop.steps(A0, zsteps)
```

Plot the results using your favorite tools 

![example_image](https://github.com/hightower8083/axiprop/blob/main/examples/example_figure.png)

For more info checkout the example notebooks for [radial](https://github.com/hightower8083/axiprop/blob/main/examples/example.ipynb) and [cartesian](https://github.com/hightower8083/axiprop/blob/main/examples/test2d.ipynb) cases, and also look for methods documentation.

### Installation

Install `axiprop` by cloning the source 
```bash
git clone https://github.com/hightower8083/axiprop.git
cd axiprop
python setup.py install
```
or directly via PiPy
```bash
pip install git+https://github.com/hightower8083/axiprop.git
```

#### Additional requirements

Note that, while base backend `NP` requires only NumPy and SciPy, other 
backends have specific dependencies:
- `NP_MKL`:  [mkl_fft](https://github.com/IntelPython/mkl_fft)
- `NP_FFTW`: [pyfftw](https://github.com/pyFFTW/pyFFTW)
- `CU`: [cupy](https://cupy.dev)
- `CL`: [pyopencl](https://documen.tician.de/pyopencl) and [reikna](https://github.com/fjarri/reikna)
- `AF`:  [ArrayFire](https://arrayfire.com) and [arrayfire-python](https://github.com/arrayfire/arrayfire-python)

Optional enhancements of utilities are achieved if [Numba](https://numba.pydata.org) is installed.
