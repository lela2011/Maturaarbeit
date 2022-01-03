import matplotlib.pyplot as plt
import numpy as np
import scipy.constants as const

import Hydrogen_lib as hydlib

if __name__ == '__main__':

    a = const.physical_constants["Bohr radius"][0]

    # Input of values
    z = int(input("Ordnungszahl: "))
    n = int(input("Hauptquantenzahl: "))
    l = int(input("Nebenquantenzahl: "))
    m = int(input("Magnetquantenzahl: "))

    k = z / (n * a)

    x = np.linspace(0, 50*a, 500)
    fig, axis = plt.subplots(1)
    radial_wave_function_3_0_eth = hydlib.radial_wave_function_builder_eth(z, n, l)
    radial_wave_function_3_0_schwabl = hydlib.radial_wave_function_builder(z, n, l)
    axis.plot(x, x**2 * radial_wave_function_3_0_eth(x)**2, "r")
    axis.plot(x, x**2 * radial_wave_function_3_0_schwabl(x)**2, "b")
    axis.set_title("n: {} l: {}".format(n, l))

    axis.set_ylabel("R_nl(r)", loc="center")
    axis.set_xlabel("r", loc="center")

    axis.spines['left'].set_position('zero')
    axis.spines['bottom'].set_position('zero')
    axis.spines['right'].set_color('none')
    axis.spines['top'].set_color('none')

    plt.show()
