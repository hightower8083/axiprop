# Axiprop
Simple tool to compute optical propagation tool based on discrete 
Hankel transform.

This library contains methods to model propagation of axisymmetric 3D optical
wave. Methods include:
- `PropagatorSymmetric`: scheme with symmetric DHT proposed in  [(M. Guizar-Sicairos, 
J.C. Gutiérrez-Vega, JOSAA 21, 53 (2004))[https://doi.org/10.1364/JOSAA.21.000053]]
- `PropagatorResampling`: a more generic scheme which allows using an arbitrary
sampling of radial axis
