import numpy as np
from propagate_lib import *
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.constants import c

# Laser
lam0 = 0.8e-6           # wavelength [m]
tau_fwhm = 35e-15       # FWHM duration (intensity) [s]
Nfreq = 128             # frequency grid number

R_las = 38.1e-3          # Radial size of laser [m]
Nr_init = 14000          # radial grid number

# Parabola
f0 = 0.4 #0.4
d0 = 0.015 #0.015

# add a hole
add_hole = False
R_hole = 8e-3

# propagation
z_load = min(f0, f0 + d0)
L_propag = 22e-3
N_z = 500
headstart = -2e-3 #adjust the `load` point

# process parameters
tau = tau_fwhm / (2*np.log(2))**0.5
freq_width = 2 / tau
Nfreq = 2*(Nfreq//2)+1

R_mirror = 1.03 * R_las
R_las_ext = R_las * 1.02

freq, wvnum, wvlgth, freq_symm = init_freq_axis(lam0, freq_width, Nfreq)
r_init, k_r =  init_radial_axes(Nr_init, R_mirror)

tau_ret = np.zeros((Nr_init))
a_hole = 1/d0 * np.log(R_las/R_hole)

mirror_phase = get_mirror_phase(f0, d0, r_init, R_las, wvnum, a_hole, freq_symm, tau_ret, method='approx')

dz_prop = L_propag/N_z
z_start = z_load + headstart
dz_steps = dz_prop*np.ones(N_z)
dz_steps[0] = z_start

# grid for the field around axis (this you can play with)
r_grid_prop = 0.002*r_init[::8] 


# generate the laser and reflect it from mirror
A_init_r = np.exp(- (r_init / R_las) ** 40)
A_init_r *= (r_init<R_las_ext) \
    * ( (r_init<R_las) + (r_init>R_las)*np.cos(np.pi/2*(r_init-R_las)/(R_las_ext-R_las))**2 )

if add_hole:
    A_init_r -= np.exp(- (r_init / R_hole) ** 15)

A_init_freq = np.exp(-(freq_symm/freq_width)**2 - (0.25*freq_symm/freq_width)** 24) / freq_width
A_init_freq_r = (A_init_freq[:,None] * A_init_r[None,:]) * mirror_phase


# run the simulation
if __name__ == '__main__':
    TM, invTM = dht_init(r_init, r_grid_prop, k_r)
    A_multi = dht_propagate_multi(A_init_freq_r, wvnum, 
                                  dz_steps, TM, invTM, k_r)

    np.save("result.npy",  A_multi)
    np.save("dz_steps.npy",  dz_steps)
