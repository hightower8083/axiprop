In the optical propagation simulation we describe evolution of electromagnetic field from a state defined at some time and space region to another state that it takes after some time and in a different space region. This problem can be described by the Maxwell equations: 
```math
\begin{aligned}
 &\nabla \times \mathbf{E} = - \partial_t \mathbf{B}\,,\\
 &\nabla \times \mathbf{H} = \partial_t \mathbf{D} + \mathbf{J}\,,\\
 & \nabla \mathbf{D} = \rho\,, \nabla  \mathbf{B} =0\,,\\
 & \mathbf{D}  = \epsilon \mathbf{E} \,,\; \mathbf{B}  = \mu \mathbf{H} \,.
\end{aligned}
```

Assuming all charges of the system to be free, i.e. $`\epsilon=\epsilon_0`$ and $`\mu=\mu_0`$, we can write a pair of separate equations for the fields:
```math
\begin{aligned}
 & \nabla^2 \mathbf{E} - \frac{1}{c^2} \partial_t^2 \mathbf{E} = \mu_0  \partial_t\mathbf{J}  + \nabla \rho/\epsilon_0  \,, \\
 & \nabla^2 \mathbf{H} - \frac{1}{c^2} \partial_t^2 \mathbf{H} =  - \nabla \times \mathbf{J}\,.
\end{aligned}
```

The terms in the last term in the right hand sides of these equations are the _sources_, and they generate the field of response of the media via the current and density modulations. For most cases it is practical to consider the electric field as it drives the current and is used for the calculations: 
```math
\begin{aligned}
 & \nabla^2 \mathbf{E} - \frac{1}{c^2} \partial_t^2 \mathbf{E} = \mu_0  \partial_t\mathbf{J}  \,.
\end{aligned}
```

Note, that here we have dropped the last term $`\nabla \rho/\epsilon_0`$ as it is purely electrostatic ($`\nabla\times\nabla \rho \equiv 0`$), and it does not propagate with the electromagnetic field. 

Physically, the contribution of this term can be estimated by re-writing the full source term as:
```math
\begin{aligned}
 & \mu_0  \partial_t\mathbf{J}  + \nabla \rho/\epsilon_0  =  \frac{1}{\epsilon_0 } \int_{-\infty}^{t} dt' (\frac{1}{c^2} \partial_t^2 \mathbf{J} -   \nabla (\nabla \mathbf{J}) )\,,
\end{aligned}
```
and comparing the terms under the intergral in the Fourier space, where we have them as $`\omega^2/c^2 \hat{\mathbf{J}}`$ and $`\mathbf{k}\cdot (\mathbf{k} \cdot \hat{\mathbf{J}})`$. Assuming the case of a linearly polarized laser field, we need to consider only one transverse current component (e.g. $`\mathbf{J} \parallel \mathbf{E} \parallel \mathbf{e}_x`$), which leads to the ratio between these terms $`k_x^2 c^2 / \omega^2`$. The optical field is typically contained in the spectral region, $`\omega\in [omega_0\pm \pi/\tau]`$ with $`k_x, k_y \in [\pm \pi/R]`$, where $`\omega_0`$, $`\tau`$ and $`R`$ are the central frequency, duration and the transverse size of the pulse or the field feature. 

In most practical situations (*except for few-cycle and/or diffraction-limit focused pulses*) we have $`omega_0 w_0/c>>\pi`$ and it is safe to 




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
