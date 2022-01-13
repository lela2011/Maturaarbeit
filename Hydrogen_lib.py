import numpy as np
import scipy.constants as const
import scipy.special as special


def spherical_harmonics_builder(l: int, m: int):
    '''
    Creates spherical harmonic function dependent on theta and phi.

    :param l: Azimuthal quantum number
    :type l: int
    :param m: Magnetic quantum number
    :type m: int
    :return: returns a spherical harmonic for given quantum numbers
    :rtype: functionType
    '''
    def spherical_harmonic(theta, phi):
        '''
        Spherical harmonic dependent on quantum numbers l and m.

        :param theta: azimuthal angle in spherical coordinate system, 0 ≤ theta ≤ π
        :type theta: float
        :param phi: polar angle in spherical coordinate system, 0 ≤ phi < 2π
        :type phi: float
        :return: Complex number
        :rtype: complex
        '''
        return special.sph_harm(m, l, theta, phi)
    return spherical_harmonic


def radial_wave_function_builder(z: int, n: int, l: int):
    '''
    Creates callable wavefunction dependent on r.

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param l: Azimuthal quantum number
    :type l: int
    :return: Radial wave function dependent on r
    :rtype: FunctionType
    '''

    # Bohr radius
    a = const.physical_constants["Bohr radius"][0]
    k = z / (n*a)
    # generalized laguerre polynomial
    laguerre_polynomial = special.genlaguerre(n-l-1, 2*l+1)
    # Normalisation constant A
    norm_const = ((np.math.factorial(n - l - 1) * (2 * k) ** 3) / (2 * n * (np.math.factorial(n + l)) ** 3)) ** (1 / 2)

    def radial_wave_function(r):
        '''
        Returns the value of the radial wave function dependent on r

        :param r: distance to origin
        :type r: float
        :return: value of radial wave function
        '''

        return -1 * norm_const * (2*k*r) ** l * np.e ** (-1 * k * r) * laguerre_polynomial(2 * k * r)

    return radial_wave_function


def eva_coulomb_potential_electronVolt(z, n, mass_e, mass_p):
    '''
    Calculates the eigenvalues for the coulomb potential of a hydrogen Atom dependent on z, n, electron mass and proton mass

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param mass_e: Mass of electron
    :type mass_e: float
    :param mass_p: Mass of proton
    :type mass_p: float
    :return: Energy eigenvalue of hydrogen atom for given z, n, electron mass and proton mass in electron volts
    :rtype: float
    '''

    # reduced mass
    mu = mass_e * mass_p / (mass_e + mass_p)
    # Energieeigenwerte Coulomb-Potential in eV
    return - (mu * const.c ** 2) / (2 * const.e) * const.alpha ** 2 * z ** 2 / n ** 2


def eva_coulomb_potential_joule(z, n, mass_e, mass_p):
    '''
    Calculates the eigenvalues for the coulomb potential of a hydrogen Atom dependent on z, n, electron mass and proton mass

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param mass_e: Mass of electron
    :type mass_e: float
    :param mass_p: Mass of proton
    :type mass_p: float
    :return: Energy eigenvalue of hydrogen atom for given z, n, electron mass and proton mass in joules
    :rtype: float
    '''

    # reduced mass
    mu = mass_e * mass_p / (mass_e + mass_p)
    # eigenvalues coulomb potential in joules
    return - mu * const.c ** 2 / 2 * const.alpha ** 2 * z ** 2 / n ** 2