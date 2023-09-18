For optical propagation we need to describe evolution of electromagnetic field from the state defined at some time and space region to the state that it has after some time and in a different space region. Assuming all charges of the system to be free, the distribution and evolution of electromagnetic field in space and time is described by the Maxwell equations written for electric field and the magnetic induction vectors:
```math
\begin{aligned}
 &\nabla \times \mathbf{B} = \frac{1}{c^2} \partial_t \mathbf{E} + \mu_0 \mathbf{J}\\
 &\nabla \times \mathbf{E} = - \partial_t \mathbf{B}\,, \; \nabla E = \rho/\epsilon_0
\end{aligned}
```

In most problems of the optical propagation of laser pulse, it is enough to describe evolution of its electric components and we can find the proper equation by combining the above ones:
```math
\begin{aligned}
 & \nabla^2 \mathbf{E} = \frac{1}{c^2} \partial_t^2 \mathbf{E} + \mu_0  \partial_t\mathbf{J}  + \nabla \rho/\epsilon_0 
\end{aligned}
```

The last term in the right hend side of this equation corresponds to the contribution of charge density perturbations (plasma waves), and may be neglected. Indeed, considering a single component of the laser field, e.g. `x`, and with the help of the continuity equation for the charge we may write the last two _source_ terms as:
```math
\begin{aligned}
 & \mu_0  \partial_t J_x  + \partial_x \rho/\epsilon_0 =  \frac{1}{\epsilon_0 } \int_{-\infty}^{t} dt' (\frac{1}{c^2} \partial_t^2 J_x -  \partial_x^2 J_x )
\end{aligned}
```
The terms under the integral in the Fourier space read, $`-\omega^2/c^2 J_x`$ and $`-k_x^2 J_x`$, and most optical problems for the directed pulse propagation $`\omega^2/c^2 \gg k_x^2`$ .


In optical problems the state of the field is usually described 

initial 

With proper initial and boundary conditions and knowledge of cahrges motion thi

 This computation follows from the Maxwell equations for electric and magnetic fields:

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


___________________________
## [Return to README](https://github.com/hightower8083/axiprop/blob/new-docs/README.md#documentation)
