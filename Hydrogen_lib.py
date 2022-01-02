import scipy.special as special
import scipy.constants as const

import numpy as np


def laguerre_polynomial_builder(r,s):
    def laguerre_polynomial_callback(x):
        L_rs = 0
        k=0
        while k <= (r-s):
            L_rs += np.power(-1, k+s) * np.power(np.math.factorial(r), 2) / (np.math.factorial(k) * np.math.factorial(k+s) * np.math.factorial(r-k-s)) * np.power(x, k)
            k += 1
        return L_rs
    return laguerre_polynomial_callback

# returns spherical harmonics based on l, m, theta, phi
# Unit: none
# Value: complex
def spherical_harmonics(l: int, m: int, theta: float, phi: float):
    Y_lm = special.sph_harm(m, l, theta, phi)
    return Y_lm


# returns a callable radial wave function dependent only on r
def radial_wave_function_builder(z: int, n: int, l: int):
    a = const.physical_constants["Bohr radius"][0]
    k = z / (n*a)
    laguerre_polynomial = laguerre_polynomial_builder(n+l, 2*l + 1)
    def radial_wave_function_callback(r):
        func = -1 * np.power((np.math.factorial(n-l-1)*np.power(2*k, 3)) / (2 * n * np.power(np.math.factorial(n+l), 3)), 1/2) * np.power(2*k*r, l) * np.exp(-1 * k * r) * laguerre_polynomial(2 * k * r)
        return func

    return radial_wave_function_callback

def radial_wave_function_builder_eth(z, n, l):
    mu = const.electron_mass * const.proton_mass / (const.electron_mass + const.proton_mass)
    a = const.physical_constants["Bohr radius"][0]*const.electron_mass/mu
    laguerre_polynomial = special.genlaguerre(n-l-1, 2*l+1)
    def fun(r):
        val = np.sqrt((np.math.factorial(n-l-1))/(2*n*np.math.factorial(n+l))) * np.power((2*z)/(n*a), 3/2) * np.power((2*z*r)/(n*a),l) * np.exp((-z*r)/(n*a)) * laguerre_polynomial((2*z*r)/(n*a))
        return val

    return fun

# Calculates eigenvalues of coulomb-potential based on z, n, electron mass, proton mass
# Unit: eV, electron volt
# Value: real
def eva_coulomb_potential_electronVolt(z: int, n: int, mass_e: float, mass_p: float):
    # Reduzierte Masse
    mu = mass_e * mass_p / (mass_e + mass_p)
    # Energieeigenwerte Coulomb-Potential in eV
    e = - (mu * const.c ** 2) / (2 * const.e) * const.alpha ** 2 * z ** 2 / n ** 2
    return e

# Calculates eigenvalues of coulomb-potential based on z, n, electron mass, proton mass
# Unit: J, joules
# Value: real
def eva_coulomb_potential_joule(z: int, n: int, mass_e: float, mass_p: float):
    # reduced mass
    mu = mass_e * mass_p / (mass_e + mass_p)
    # eigenvalues coulomb potential in joules
    e = - mu * const.c ** 2 / 2 * const.alpha ** 2 * z ** 2 / n ** 2
    return e