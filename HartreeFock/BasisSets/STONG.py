import numpy as np
from scipy.special import factorial2

from HartreeFock.BasisSets import BasisFunction as Bf


class STONG:
    _angular_momentum_combinations_dic = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
        3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)]
    }

    def __init__(self, import_set, center):
        self.functions = self._build_basis_functions(import_set, center)

    def _build_basis_functions(self, import_set, center):
        functions = []

        elements = import_set["elements"]
        element = elements[list(elements.keys())[0]]
        electron_shells = element["electron_shells"]
        for electron_shell in electron_shells:
            angular_momenta = electron_shell["angular_momentum"]
            exponents = list(map(float, electron_shell["exponents"]))
            for m in range(len(angular_momenta)):
                angular_momentum = angular_momenta[m]
                coefficients = list(map(float, electron_shell["coefficients"][m]))
                angular_momentum_combinations = self._angular_momentum_combinations_dic[angular_momentum]
                for angular_momentum_combination in angular_momentum_combinations:

                    i = angular_momentum_combination[0]
                    j = angular_momentum_combination[1]
                    k = angular_momentum_combination[2]

                    primitive_normalization_constants = []

                    for n in range(len(exponents)):
                        primitive_normalization_constant = ((2 * exponents[n] / np.pi) ** (3 / 2) * (
                                    4 * exponents[n]) ** angular_momentum / (factorial2(2 * i - 1) * factorial2(
                            2 * j - 1) * factorial2(2 * k - 1))) ** (1 / 2)

                        primitive_normalization_constants.append(primitive_normalization_constant)

                    function = Bf.BasisFunction(exponents, coefficients, primitive_normalization_constants,
                                                angular_momentum_combination, center)
                    functions.append(function)

        return functions
