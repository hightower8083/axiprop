## Basic model

For optical propagation we calculate the electromagnetic field that evolves from the state defined at some time and space, to another state that it takes after some time, and typically in a different location. Formally, for this me need to consider the vectorial Maxwell equations for electric and magnetic fields,

$$ \nabla \times \mathbf{E} = - \partial_t \mathbf{B}, $$

$$ \nabla \times \mathbf{B} = \frac{1}{c^2} \partial_t \mathbf{E} + \mu_0 \mathbf{J},$$

For the partical case of the optical field it is typically enough to describe the electric field, for which the equation can be written by combining the above equations:

$$ \nabla \times \nabla \times \mathbf{E} = \nabla (\nabla \mathbf{E}) - \nabla^2 \mathbf{E} = - \partial_t \left( \frac{1}{c^2} 
\partial_t \mathbf{E} + \mu_0 \mathbf{J}\right) = - \frac{1}{c^2} \partial_t^2 \mathbf{E} - \mu_0  \partial_t\mathbf{J}$$

The term $\nabla (\nabla \mathbf{E})$ here describes contributions of charge density perturbations (plasma waves) which 
we may neglect, and to account for the fast media resopnse we can use:

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


___________________________
## [Return to README](https://github.com/hightower8083/axiprop/blob/new-docs/README.md#documentation)
