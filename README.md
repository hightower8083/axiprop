# Axiprop
Simple tool to compute optical propagation, based on the discrete 
Hankel transform.

### Contents

This library contains methods to model propagation of axisymmetric 3D optical
wave. Methods include:
- `PropagatorSymmetric`: scheme with symmetric DHT proposed in  [[M. Guizar-Sicairos, 
J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004)](https://doi.org/10.1364/JOSAA.21.000053)]
- `PropagatorResampling`: a more generic scheme which allows using an arbitrary
sampling of radial axis

### Installation

Can be installed by cloning the source 
```bash
git clone https://github.com/hightower8083/axiprop.git
cd CheatSheet
python setup.py install
```
or via PiPy
```bash
pip install git+https://github.com/hightower8083/axiprop.git
```

### Contributions

Feel free to propose your favorite formulas or fixes
