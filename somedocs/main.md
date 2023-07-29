For optical propagation we operate the field equations:

$$ \nabla \times \mathbf{E} = - \partial_t \mathbf{B} $$

$$ \nabla \times \mathbf{B} = \frac{1}{c^2} \partial_t \mathbf{E} + \mu_0 \mathbf{J}$$

We can describe optical field with it's electric components, for which an equation can be written as:

$$ \nabla \times \nabla \times \mathbf{E} = \nabla (\nabla \mathbf{E}) - \nabla^2 \mathbf{E} = - \partial_t \left( \frac{1}{c^2} 
\partial_t \mathbf{E} + \mu_0 \mathbf{J}\right) = - \frac{1}{c^2} \partial_t^2 \mathbf{E} - \mu_0  \partial_t\mathbf{J}$$

where term $\nabla (\nabla \mathbf{E})$ describes contributions of charge density perturbations (plasma waves) which 
we may neglect, to obtain:

$$ \nabla^2 \mathbf{E} - \frac{1}{c^2}  \partial_t^2 \mathbf{E} - \mu_0  \partial_t\mathbf{J} = 0$$

### Temporal representation and propagation

Let use write wave equation terms as:


$$ \partial_z^2 E =  c^{-2}\partial_t^2 E -  \partial_\perp^2 E\,. $$

In the temporal representation, the full field $E(t,x,y,z)$ is considered at a fixed $z=z_0$, and is measured in a finite $(x,y)$-plane over a time interval $t\in[t_{min},t_{max}]$, and we can write this input as $E_0 = E(t,x,y,z=z_0)$. 

The RHS of wave equation can be linearized in Fourier space, and for  the temporal representation we can define it as:
$$ E = \sum_{\omega,k_x,k_y} \; \hat{E}_{\omega,k_x,k_y}  \exp\big[ -i (\omega t - k_x x - k_y y)\big]  $$

For propagation in positive direction along $z$ axis, the solution to the wave equation gives:
$$ \hat{E}(z) =  \hat{E}_0 \;  \exp\left[ i (z-z_0) \; \sqrt{\omega^2/c^2 - k_x^2 - k_y^2} \;\right]$$

Go to RZ geometry and consider only polarisation component:

___
$$ \partial_z^2 E = \frac{1}{c^2} \; \partial_t^2 E - \nabla_\perp^2 E + \mu_0  \partial_t J$$
___
