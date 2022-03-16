import numpy
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


def laguerre_polynomial_builder(n: int, m: int):
    '''
    Creates callable Laguerre-polynomial for given n and m

    :param n: parameter 1
    :type n: int
    :param m: parameter 2
    :type m: int
    :return: Laguerre Polynomial dependent on x
    :rtype: FunctionType
    '''
    def laguerre_polynomial(x: float):
        # Start values
        L = 0
        j = 0
        # Sum
        while j <= (n-m):
            L = L + (-1)**(j+m) * np.math.factorial(n)**2 / (np.math.factorial(j) * np.math.factorial(j+m) * np.math.factorial(n-j-m)) * x**j
            j += 1

        return L

    return laguerre_polynomial


def radial_wave_function_builder(z: int, n: int, l: int, m1: float, m2: float):
    '''
    Creates callable wavefunction dependent on r.

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param l: Azimuthal quantum number
    :type l: int
    :param m1: Electron mass
    :type m1: float
    :param m2: Nucleus mass
    :type m2: float
    :return: Radial wave function dependent on r
    :rtype: FunctionType
    '''

    # elementary charge
    e_0 = const.elementary_charge
    # Vacuum permittivity
    epsilon_0 = const.epsilon_0
    # reduced plank constant
    hbar = const.hbar
    # mu
    mu = m1 * m2 / ( m1 + m2 )

    k = (mu * e_0**2 * z) / ( 4 * numpy.pi * epsilon_0 * hbar ** 2 * n )

    # generalized laguerre polynomial
    laguerre_polynomial = laguerre_polynomial_builder(n+l, 2*l+1)
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


def radial_wave_function_dim_builder(z: int, n: int, l: int, m1: float, m2: float):
    '''
    Creates callable wavefunction dependent on dimensionless r.

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param l: Azimuthal quantum number
    :type l: int
    :param m1: Mass of electron
    :type m1: float
    :param m2: Mass of nucleus
    :type m2: float
    :return: Radial wave function dependent on r
    :rtype: FunctionType
    '''

    # reduced mass
    mu = (m1*m2)/(m1+m2)
    # electron mass
    m_e = const.electron_mass

    kappa = (mu * z) / (m_e * n)
    # generalized laguerre polynomial
    laguerre_polynomial = laguerre_polynomial_builder(n + l, 2 * l + 1)
    # Normalisation constant A
    norm_const = ( ( np.math.factorial(n-l-1) ) / ( 2 * n * ( np.math.factorial(n+l) ) ** 3 ) )**(1/2)

    def radial_wave_function(r):
        '''
        Returns the value of the radial wave function dependent on r

        :param r: distance to origin
        :type r: float
        :return: value of radial wave function
        '''

        return -1 * norm_const * (2*kappa) ** (3/2) * (2*kappa*r) ** l * np.e ** (-1*kappa*r) * laguerre_polynomial(2 * kappa * r)

    return radial_wave_function


def eigenenergy(z, n, m1, m2):
    '''
    Calculates the eigenvalues for the coulomb potential of a hydrogen Atom dependent on z, n, electron mass and proton mass

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param m1: Mass of electron
    :type m1: float
    :param m2: Mass of nucleus
    :type m2: float
    :return: Energy eigenvalue of hydrogen atom for given z, n, electron mass and proton mass in joules
    :rtype: float
    '''

    # reduced mass
    mu = m1 * m2 / (m1 + m2)
    # reduced plank constatn
    hbar = const.hbar
    # elementary charge
    e_0 = const.elementary_charge
    # Vacuum permittivity
    epsilon_0 = const.epsilon_0

    E_n = - 1/2 * (mu * e_0**4)/( (4*np.pi*epsilon_0)**2 * hbar ** 2 ) * z**2 / (n**2)

    # Eigenenergy for electron for given Z and n
    return E_n


def eigenenergy_dim(z, n, m1, m2):
    '''
    Calculates the eigenvalues for the coulomb potential of a hydrogen Atom dependent on z, n, electron mass and proton mass

    :param z: Atomic number
    :type z: int
    :param n: Principle quantum number
    :type n: int
    :param m1: Mass of electron
    :type m1: float
    :param m2: Mass of nucleus
    :type m2: float
    :return: Energy eigenvalue of hydrogen atom for given z, n, electron mass and proton mass in joules
    :rtype: float
    '''

    # reduced mass
    mu = m1 * m2 / (m1 + m2)
    # electron mass
    m_e = const.electron_mass

    E_n = - mu / m_e * z**2 / (n**2)

    # Eigenenergy for electron for given Z and n
    return E_n