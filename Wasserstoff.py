import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate

import Hydrogen_lib as hydlib


def plotR_nl(z, n, l):
    x = np.linspace(0, 50 * a, 500)
    fig, axis = plt.subplots(1)
    radial_wave_function_3_0_schwabl = hydlib.radial_wave_function_builder(z, n, l)
    axis.plot(x, (x) ** 2 * radial_wave_function_3_0_schwabl(x) ** 2, "b")
    axis.set_title("n: {} l: {}".format(n, l))

    axis.set_ylabel("(r)^2+R_nl^2(r)", loc="center")
    axis.set_xlabel("r", loc="center")

    axis.spines['left'].set_position('zero')
    axis.spines['bottom'].set_position('zero')
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')

    plt.show()


def plotY_lm(l, m):
    x = np.linspace(0, 2 * np.pi, 100)
    y = np.linspace(0, np.pi, 50)

    X, Y = np.meshgrid(x, y)

    fig3D = plt.figure()
    ax = plt.axes(projection="3d")
    ax.plot_surface(X, Y, (np.power(hydlib.spherical_harmonics(l, m, X, Y), 2)).real, rstride=1, cstride=1, cmap="jet",
                    edgecolor="none")

    plt.show()


def integrate_radial(z):
    for n in range(1, 8):
        for l in range(n):
            radial = hydlib.radial_wave_function_builder(z, n, l)
            func = lambda r: radial(r * a) * radial(r * a) * (r * a) * (r * a)
            integral = scipy.integrate.quad(func, 0, np.infty)[0] * a
            print(f"n: {n}\nl: {l}\nintegral: {integral}\n\n...................................\n")


if __name__ == '__main__':
    print("Code here")
