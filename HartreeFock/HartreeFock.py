import numpy as np

from Objects.Atom import Atom
from Objects.Molecule import Molecule

from RHF import RHF

import matplotlib.pyplot as plt

if __name__ == "__main__":

    # Define carbon atom
    c = Atom("C", (0.0, 0.0, 0.0), 6)

    # Define array that stores energy based on bond length
    energies_core_co2 = []

    # Generate array that holds bond lengths that are to be calculated
    x = np.arange(0.5,3, 0.05)

    # keep track of progress
    i = 1

    # loop over bond lengths
    for pos in x:

        # Define oxygen atoms
        o_1 = Atom("O", (0.0, 0.0, -pos), 8)
        o_2 = Atom("O", (0.0, 0.0, pos), 8)

        # Define molecule
        co_2_mol = Molecule([o_1, c, o_2], [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0])

        # define RHF calculation
        rhf_co_2 = RHF(co_2_mol)

        # run SCF algorithm
        results_co2 = rhf_co_2.calculate()

        # stores returned values in variables
        energy_co2 = results_co2[0]
        iterations_co2 = results_co2[3]
        energies_core_co2.append(energy_co2)

        # prints progress
        print("----------")
        print("CO_2 --- {} of {} done --- Energy: {:.2f} with Iterations: {}".format(i, len(x), energy_co2, iterations_co2))
        print("----------\n")

        # updates iteration counter
        i += 1

    # plots graph
    plt.plot(x, energies_core_co2,'g-')
    plt.show()