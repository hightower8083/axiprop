## Basic model

For optical propagation we need to describe evolution of electromagnetic field from the state defined at some time and space region to the state that it has after some time and in a different space region. This computation follows from the Maxwell equations for electric and magnetic fields:

$$ \nabla \times \mathbf{E} = - \partial_t \mathbf{B} $$

$$ \nabla \times \mathbf{B} = \frac{1}{c^2} \partial_t \mathbf{E} + \mu_0 \mathbf{J}$$

In most cases that concern laser pulse propagation, it is enough to describe electric components of the field, for which we can find the equation by combinng the above ones:

$$ \nabla \times \nabla \times \mathbf{E} = \nabla (\nabla \mathbf{E}) - \nabla^2 \mathbf{E} = - \partial_t \left( \frac{1}{c^2} 
\partial_t \mathbf{E} + \mu_0 \mathbf{J}\right) = - \frac{1}{c^2} \partial_t^2 \mathbf{E} - \mu_0  \partial_t\mathbf{J}$$

The term $\nabla (\nabla \mathbf{E})$ here describes contributions of charge density perturbations (plasma waves) which 
we may neglect, and for the acccount of fast media resopnse we can use:

$$ \nabla^2 \mathbf{E} = \frac{1}{c^2}  \partial_t^2 \mathbf{E} + \mu_0  \partial_t\mathbf{J} $$

In the following we will always consider a scalar version of this equation meaning the component along the laser field polarisation.

#### Vacuum case

In a case when field propagates in vacuum, there is no current $J = 0$ and equation simplifies to 

$$ \nabla^2 E = \frac{1}{c^2}  \partial_t^2 E $$

### Plasma dispersion

Simplest plasma response can be described with help of non-relativistic motion equation:

$$ \mu_0 \partial_t J = -e \mu_0 n_{pe} \partial_t v = \omega_{pe}^2 / c^2 E $$

where $\omega_{pe} = \sqrt{\frac{e^2 n_{pe} }{\epsilon_0  m_e}}$ is a plasma frequency.

The resulting field equation is:

$$ \nabla^2 E = \frac{1}{c^2}  \partial_t^2 E + \frac{ \omega_{pe}^2 }{ c^2 } E  $$

and its spatio-temporal Fourier transform leads to the basic dispersion relation

$$ \omega^2 = k^2 c^2 + \omega_{pe}^2 $$

### Field representation

#### Temporal representation

We assumed the field propagation along z-axis is  and the temporal representation. The later means that the full field $E(t,x,y,z)$ is considered at a fixed $z=z_0$, and is measured on a finite $(x,y)$-plane over a time interval $t\in[t_{min},t_{max}]$, so the input as $E_0 = E(t,x,y,z=z_0)$. 

The field equation can be written as following:

$$ \partial_z^2 E =  \frac{1}{c^2}  \partial_t^2 E - \nabla_{\perp}^2 E + \mu_0  \partial_t J   $$

#### Geometries

Two main geometries are usually considered
- cartesian $(x, y, t)$, where transverse Laplace operator is $\nabla_\perp^2 = \partial_x^2 + \partial_y^2$
- cylindrical $(r, \theta, t)$, where $\nabla_{\perp}^2 = r^{-1} \partial_r (r \partial_r) +  r^{-2} \partial_\theta^2$. 

#### Enelope

If one considers the field around its central frequency $\omega_0$ as:

$$ E(t) = \mathrm{Re}[\hat{E}(t) \exp(- i \omega_0 t) ], $$

in many practical cases the complex function $\hat{E}(t)$ can be assumed to be much slower than the actual $E(t)$. This presentation called envelope is often preferrable for analysis, and is also used in `Lasy`

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


# [Return](https://github.com/hightower8083/axiprop/blob/new-docs/README.md)
