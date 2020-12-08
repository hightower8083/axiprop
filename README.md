# Axiprop
Simple tool to compute optical propagation, based on the discrete Hankel and 
Fourier transforms

### Contents

This library contains methods and convenience tools to model propagation of the 3D optical
field. Computation part is implemented on
- **CPU** using the NumPy, optionally enhanced via [mkl_fft](https://github.com/IntelPython/mkl_fft) or [pyfftw](https://github.com/pyFFTW/pyFFTW)
- **GPU** using [CuPy](https://cupy.dev) or [ArrayFire](https://github.com/arrayfire/arrayfire-python)

Currently methods include:
- `PropagatorSymmetric`: cylindical axisymmetric propagator with the symmetric DHT proposed in 
[[M. Guizar-Sicairos, J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)](https://doi.org/10.1364/JOSAA.21.000053)]
- `PropagatorResampling`: cylindical axisymmetric propagator with a more generic DHT which allows for arbitrary
sampling of radial axis
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
prop = PropagatorSymmetric(Rmax, L_kz, Nr, Nkz, k0,  Nr_end)
```
and setup the laser reflected from the focusing mirror
```python
A0 = laser_from_fu( fu_laser, prop.kz, prop.r, normalize=True )
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

![example_image](https://github.com/hightower8083/axiprop/blob/main/examples/example_figure.jpg)

For more info checkout the 
[example notebook](https://github.com/hightower8083/axiprop/blob/main/examples/example.ipynb)
, and inline documentation of the methods.

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

Note that enhancement libraries, 
[numba](https://numba.pydata.org), 
[pyfftw](https://github.com/pyFFTW/pyFFTW), 
[mkl_fft](https://github.com/IntelPython/mkl_fft), 
[ArrayFire](https://arrayfire.com)+
[arrayfire-python](https://github.com/arrayfire/arrayfire-python) and 
[CuPy](https://cupy.dev) are not *required* nor included, and should be 
installated separately.

To use *GPU* methods use `lib_cu` and `lib_af` for CuPy and ArrayFire 
implementations.