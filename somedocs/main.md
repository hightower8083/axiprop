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

#### Linear plasma dispersion

Simplest nonrelativistic plasma response can be described with help of motion equation:

$$ \mu_0 \partial_t J = -e \mu_0 n_{pe} \partial_t v = \omega_{pe}^2 / c^2 E $$

where $\omega_{pe} = \sqrt{\frac{e^2 n_{pe} }{\epsilon_0  m_e}}$ is a plasma frequency.

The resulting field equation is:

$$ \nabla^2 E = \frac{1}{c^2}  \partial_t^2 E + \frac{ \omega_{pe}^2 }{ c^2 } E  $$

and its spatio-temporal Fourier transform leads to the basic dispersion relation

$$ \omega^2 = k^2 c^2 + \omega_{pe}^2 $$


## In-code representation

In `axiprop` the field propagation along z-axis is assumed and the temporal representation is considered. The later means that full field $E(t,x,y,z)$ is considered at a fixed $z=z_0$, and is measured on a finite $(x,y)$-plane over a time interval $t\in[t_{min},t_{max}]$, so the input as $E_0 = E(t,x,y,z=z_0)$. 

The field equation can be written as following:

$$ \partial_z^2 E =  \frac{1}{c^2}  \partial_t^2 E - \nabla_{\perp}^2 E + \mu_0  \partial_t J   $$

Two main geometries are consider the 3D cartesian $(x, y, t)$ and 2D RZ $(r, t)$. In 3D the transverse Laplace operator is $\nabla_\perp^2 = \partial_x^2 + \partial_y^2$ while in RZ it is $\nabla_{\perp}^2 = r^{-1} \partial_r (r \partial_r)$ 

## Propagation


The RHS of wave equation can be linearized in Fourier space, and for  the temporal representation we can define it as:

$$ E = \sum_{\omega,k_x,k_y} \; \hat{E}_{\omega,k_x,k_y}  \exp\big[ -i (\omega t - k_x x - k_y y)\big]  $$

For propagation in positive direction along $z$ axis, the solution to the wave equation gives:

$$ \hat{E}(z) =  \hat{E}_0 \;  \exp\left[ i (z-z_0) \; \sqrt{\omega^2/c^2 - k_x^2 - k_y^2} \;\right]$$

Go to RZ geometry and consider only polarisation component:

___
$$ \partial_z^2 E = \frac{1}{c^2} \; \partial_t^2 E - \nabla_\perp^2 E + \mu_0  \partial_t J$$
___
