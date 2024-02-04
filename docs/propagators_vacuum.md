## Propagation

Propagation calculations in `axiprop` are realized with spectral transformations in time and transverse spatial domains.

### Non-paraxial propagator

#### 3D

#### RZ

### Fresnel propagator

For a mode $k=2\pi / \lambda$

$$ E(x, y, k, z) = \cfrac{ \exp\left(i k z \left(1 + \frac{x^2+y^2}{2 z^2}\right) \right) }{i \lambda z} \int  \int_{-\infty}^{\infty} 
dx' dy' \left[E(x', y', k, z=0) \exp\left(\frac{i k}{2 z} \left(x'^2+y'^2\right) \right) \right] \exp\left(-\frac{i k}{z} 
\left(xx'+yy'\right)\right) $$

## [Return to README](https://github.com/hightower8083/axiprop/blob/new-docs/README.md#documentation)