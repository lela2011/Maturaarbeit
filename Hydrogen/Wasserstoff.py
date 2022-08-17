import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const
import scipy.integrate

from Hydrogen import Hydrogen_lib as hydlib


def plotR_nl(z, n, l, m1, m2):
    x = np.linspace(0, 20, 500)
    fig, axis = plt.subplots(1)
    radial_wave_function = hydlib.radial_wave_function_dim_builder(z, n, l, m1, m2)
    axis.plot(x, x ** 2 * radial_wave_function(x) ** 2, "b")
    axis.set_title("n: {}    l: {}".format(n, l))

    axis.set_ylabel("$R(r)^2 r^2$", loc="center")
    axis.set_xlabel("r [$a_0$]", loc="center")

    axis.spines['left'].set_position('zero')
    axis.spines['bottom'].set_position('zero')
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')

    plt.show()


def plotR_nl_123(z, n, m1, m2, ub):
    font = {'family': 'serif',
            'size': 13}
    plt.rc('font', **font)
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axis = plt.subplots(1,3)
    for i in range(0,n):
        x = np.linspace(0, ub[i], 500)
        for l in range(0,i+1):
            radial_wave_function = hydlib.radial_wave_function_dim_builder(z, i+1, l, m1, m2)
            axis[i].plot(x, x ** 2 * radial_wave_function(x) ** 2, label="n: {}; l: {}".format(i+1, l))

        axis[i].set_ylabel(r'$|R(r)|^2 r^2 dr$', loc="center")
        axis[i].set_xlabel(r'r [$a_0$]', loc="center")
        axis[i].legend()

        axis[i].spines['left'].set_position('zero')
        axis[i].spines['bottom'].set_position('zero')
        axis[i].spines['right'].set_color('none')
        axis[i].spines['top'].set_color('none')

    plt.show()


def plotR_nl_non_prob_123(z, n, m1, m2, ub):
    font = {'family': 'serif',
            'size': 13}
    plt.rc('font', **font)
    plt.rcParams["mathtext.fontset"] = "stix"

    fig, axis = plt.subplots(1,3)
    for i in range(0,n):
        x = np.linspace(0, ub[i], 500)
        for l in range(0,i+1):
            radial_wave_function = hydlib.radial_wave_function_dim_builder(z, i+1, l, m1, m2)
            axis[i].plot(x, radial_wave_function(x), label="n: {}; l: {}".format(i+1, l))

        axis[i].set_ylabel("R(r)", loc="center")
        axis[i].set_xlabel(r'r [$a_0$]', loc="center")
        axis[i].legend()

        axis[i].spines['left'].set_position('zero')
        axis[i].spines['bottom'].set_position('zero')
        axis[i].spines['right'].set_color('none')
        axis[i].spines['top'].set_color('none')

    plt.show()


def integrate_radial(z, m1, m2):
    for n in range(1, 8):
        for l in range(n):
            radial = hydlib.radial_wave_function_dim_builder(z, n, l, m1, m2)
            radial_prob = lambda r: radial(r)**2 * r**2
            integral = scipy.integrate.quad(radial_prob, 0, np.infty)[0]
            print(f"n: {n}\nl: {l}\nintegral: {integral}\n\n...................................\n")


if __name__ == '__main__':
    m1 = const.electron_mass
    m2 = const.proton_mass
    plotR_nl_123(1, 3, m1, m2, (10, 20, 30))
    plotR_nl_non_prob_123(1, 3, m1, m2, (10,20,30))
    #integrate_radial(1, m1, m2)
