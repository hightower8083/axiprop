### Field representation

#### Temporal representation

We assumed the field propagation along z-axis is  and the temporal representation. The later means that the full field $E(t,x,y,z)$ is considered at a fixed $z=z_0$, and is measured on a finite $(x,y)$-plane over a time interval $t\in[t_{min},t_{max}]$, so the input as $E_0 = E(t,x,y,z=z_0)$. 

The field equation can be written as following:

$$ \partial_z^2 E =  \frac{1}{c^2}  \partial_t^2 E - \nabla_{\perp}^2 E + \mu_0  \partial_t J   $$

#### Geometries

Two main geometries are usually considered
- cartesian $(x, y, t)$, where transverse Laplace operator is $\nabla_\perp^2 = \partial_x^2 + \partial_y^2$
- cylindrical $(r, \theta, t)$, where $\nabla_{\perp}^2 = r^{-1} \partial_r (r \partial_r) +  r^{-2} \partial_\theta^2$. 

#### Envelope

If one considers the field around its central frequency $\omega_0$ as:

$$ E(t) = \mathrm{Re}[\hat{E}(t) \exp(- i \omega_0 t) ], $$

in many practical cases the complex function $\hat{E}(t)$ can be assumed to be much slower than the actual $E(t)$. This presentation called envelope is often preferrable for analysis, and is also used in `Lasy`

## [Return to README](https://github.com/hightower8083/axiprop/blob/new-docs/README.md#documentation)