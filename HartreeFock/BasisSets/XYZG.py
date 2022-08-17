import numpy as np
from scipy.special import factorial2

from HartreeFock.BasisSets import BasisFunction as Bf


class XYZG:
    _angular_momentum_combinations_dic = {
        0: [(0, 0, 0)],
        1: [(1, 0, 0), (0, 1, 0), (0, 0, 1)],
        2: [(2, 0, 0), (0, 2, 0), (0, 0, 2), (1, 1, 0), (1, 0, 1), (0, 1, 1)],
        3: [(3, 0, 0), (0, 3, 0), (0, 0, 3), (2, 1, 0), (2, 0, 1), (1, 2, 0), (0, 2, 1), (1, 0, 2), (0, 1, 2), (1, 1, 1)]
    }

    def __init__(self, import_set):
        self.functions = self._build_basis_functions(import_set)

    def _build_basis_functions(self, import_set):
        functions = []

        name = import_set["name"]
        core_structure, valence_structure = name[:-1].split("-")
        structure = [int(core_structure)] + list(map(int, list(valence_structure)))

        elements = import_set["elements"]
        element = elements[list(elements.keys())[0]]
        electron_shells = element["electron_shells"]

        i = 0
        while i < len(electron_shells):

            electron_shell = electron_shells[i]

            exponents = list(map(float, electron_shell["exponents"]))

            if len(exponents) == structure[0]:
                functions += self._build_core_functions(electron_shell)
                i += 1
            elif len(exponents) == structure[1]:
                next_electron_shell = electron_shells[i + 1]
                functions += self._build_valence_functions(electron_shell, next_electron_shell)
                i += 2

        return functions

    def _build_core_functions(self, electron_shell):
        functions = []

        angular_momenta = electron_shell["angular_momentum"]
        exponents = list(map(float, electron_shell["exponents"]))
        for angular_momentum in angular_momenta:
            coefficients = list(map(float, electron_shell["coefficients"][angular_momentum]))
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

                contracted_normalization_sum = 0
                for a in range(len(exponents)):
                    for b in range(len(exponents)):
                        coefficient_a = coefficients[a]
                        coefficient_b = coefficients[b]
                        normalization_a = primitive_normalization_constants[a]
                        normalization_b = primitive_normalization_constants[b]
                        exponent_a = exponents[a]
                        exponent_b = exponents[b]

                        combined_exponent = exponent_a + exponent_b

                        contracted_normalization_sum_element = coefficient_a * coefficient_b * normalization_a * normalization_b * (
                                    np.pi / combined_exponent) ** (3 / 2) * factorial2(
                            2 * i - 1) * factorial2(2 * j - 1) * factorial2(2 * k - 1) / ((2 * combined_exponent) ** angular_momentum)
                        contracted_normalization_sum += contracted_normalization_sum_element

                contracted_normalization = contracted_normalization_sum ** (-1 / 2)

                function = Bf.BasisFunction(exponents, coefficients, primitive_normalization_constants,
                                            angular_momentum_combination, contracted_normalization)
                functions.append(function)

        return functions

    def _build_valence_functions(self, electron_shell, next_electron_shell):
        functions = []

        angular_momenta = electron_shell["angular_momentum"]
        exponents_1 = list(map(float, electron_shell["exponents"]))
        exponents_2 = list(map(float, next_electron_shell["exponents"]))
        for m in range(len(angular_momenta)):
            angular_momentum = angular_momenta[m]
            coefficients_1 = list(map(float, electron_shell["coefficients"][m]))
            coefficients_2 = list(map(float, next_electron_shell["coefficients"][m]))
            angular_momentum_combinations = self._angular_momentum_combinations_dic[angular_momentum]
            for angular_momentum_combination in angular_momentum_combinations:

                functions_1 = self._build_individual_valence_function(coefficients_1, exponents_1, angular_momentum, angular_momentum_combination)
                functions_2 = self._build_individual_valence_function(coefficients_2, exponents_2, angular_momentum, angular_momentum_combination)

                functions += [functions_1, functions_2]

        return functions

    def _build_individual_valence_function(self, coefficients, exponents, angular_momentum, angular_momentum_combination):

        i = angular_momentum_combination[0]
        j = angular_momentum_combination[1]
        k = angular_momentum_combination[2]

        primitive_normalization_constants = []

        for n in range(len(exponents)):
            primitive_normalization_constant = ((2 * exponents[n] / np.pi) ** (3 / 2) * (
                    4 * exponents[n]) ** angular_momentum / (factorial2(2 * i - 1) * factorial2(
                2 * j - 1) * factorial2(2 * k - 1))) ** (1 / 2)

            primitive_normalization_constants.append(primitive_normalization_constant)

        contracted_normalization_sum = 0
        for a in range(len(exponents)):
            for b in range(len(exponents)):
                coefficient_a = coefficients[a]
                coefficient_b = coefficients[b]
                normalization_a = primitive_normalization_constants[a]
                normalization_b = primitive_normalization_constants[b]
                exponent_a = exponents[a]
                exponent_b = exponents[b]

                combined_exponent = exponent_a + exponent_b

                contracted_normalization_sum_element = coefficient_a * coefficient_b * normalization_a * normalization_b * (
                        np.pi / combined_exponent) ** (3 / 2) * factorial2(
                    2 * i - 1) * factorial2(2 * j - 1) * factorial2(2 * k - 1) / (
                                                               (2 * combined_exponent) ** angular_momentum)
                contracted_normalization_sum += contracted_normalization_sum_element

        contracted_normalization = contracted_normalization_sum ** (-1 / 2)

        function = Bf.BasisFunction(exponents, coefficients, primitive_normalization_constants,
                                       angular_momentum_combination, contracted_normalization)

        return function
