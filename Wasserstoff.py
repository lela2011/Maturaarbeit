import Hydrogen_lib as hydlib

import scipy.constants as const

import matplotlib.pyplot as plt
import numpy as np

if __name__ == '__main__':

    mu = const.electron_mass * const.proton_mass / (const.electron_mass + const.proton_mass)
    a = const.physical_constants["Bohr radius"][0] * const.electron_mass / mu

    # Input of values
    z = int(input("Ordnungszahl: "))
    n = int(input("Hauptquantenzahl: "))
    l = int(input("Nebenquantenzahl: "))
    m = int(input("Magnetquantenzahl: "))

    k = z / (n * a)

    x = np.linspace(0, 50, 500)
    fig, axis = plt.subplots(1)
    radial_wave_function_3_0_eth = hydlib.radial_wave_function_builder_eth(z, n, l)
    radial_wave_function_3_0_schwabl = hydlib.radial_wave_function_builder(z, n, l)
    axis.plot(x, radial_wave_function_3_0_eth(x*a) * np.power(a, 3/2), "r")
    axis.plot(x, radial_wave_function_3_0_schwabl(x*a) * np.power(a, 3/2), "b")
    axis.set_title("n: {}".format(n))

    axis.set_ylabel("R_nl(r)*a^(3/2)", loc="top")
    axis.set_xlabel("r/a", loc="right")

    axis.spines['left'].set_position('zero')
    axis.spines['bottom'].set_position('zero')
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')

    plt.show()
