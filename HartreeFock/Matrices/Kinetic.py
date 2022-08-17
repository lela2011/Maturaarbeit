import numpy as np
from scipy.special import factorial2

from HartreeFock.BasisSets import BasisFunction


class Kinetic:
    def __init__(self, basis_functions):
        self.matrix = self._build_matrix(basis_functions)

    def _build_matrix(self, basis_functions):
        kinetic_matrix = np.ones((len(basis_functions), len(basis_functions)))

        i = 0
        while i < len(basis_functions):
            j = i
            function_1 = basis_functions[i]
            while j < len(basis_functions):
                function_2 = basis_functions[j]

                kinetic_element = self._calculate(function_1, function_2)
                kinetic_matrix[i, j] = kinetic_element
                kinetic_matrix[j, i] = kinetic_element

                j += 1

            i += 1

        return kinetic_matrix

    def _calculate(self, function_1: BasisFunction, function_2: BasisFunction):

        i_1 = function_1.angular_momentum[0]
        i_2 = function_2.angular_momentum[0]

        j_1 = function_1.angular_momentum[1]
        j_2 = function_2.angular_momentum[1]

        k_1 = function_1.angular_momentum[2]
        k_2 = function_2.angular_momentum[2]

        i = i_1 + i_2
        j = j_1 + j_2
        k = k_1 + k_2

        if i % 2 == 1 or j % 2 == 1 or k % 2 == 1:
            return 0
        else:
            exponents_1 = function_1.exponents
            exponents_2 = function_2.exponents

            coefficients_1 = function_1.coefficients
            coefficients_2 = function_2.coefficients

            primitive_normalizations_1 = function_1.primitive_normalization_constants
            primitive_normalizations_2 = function_2.primitive_normalization_constants

            normalization_1 = function_1.normalization
            normalization_2 = function_2.normalization

            angular_momentum = i + j + k

            element_sum = 0

            for a in range(len(coefficients_1)):
                for b in range(len(coefficients_2)):
                    exponent_a = exponents_1[a]
                    exponent_b = exponents_2[b]
                    exponent = exponent_a + exponent_b
                    primitive_normalization_a = primitive_normalizations_1[a]
                    primitive_normalization_b = primitive_normalizations_2[b]
                    coefficient_a = coefficients_1[a]
                    coefficient_b = coefficients_2[b]

                    element_x1 = i_1 * i_2 * factorial2(
                        i - 3) * factorial2(j - 1) * factorial2(k - 1) / ((2 * exponent) ** ((angular_momentum - 2) / 2)) * (
                                          np.pi / exponent) ** (3 / 2)

                    element_x2 = 4 * exponent_a * exponent_b * factorial2(
                        i + 1) * factorial2(j - 1) * factorial2(k - 1) / ((2 * exponent) ** ((angular_momentum + 2) / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_x3 = 2 * (exponent_a * i_2 + exponent_b * i_1) * factorial2(
                        i - 1) * factorial2(j - 1) * factorial2(k - 1) / ((2 * exponent) ** (angular_momentum / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_x = element_x1 + element_x2 - element_x3

                    element_y1 = j_1 * j_2 * factorial2(
                        i - 1) * factorial2(j - 3) * factorial2(k - 1) / (
                                             (2 * exponent) ** ((angular_momentum - 2) / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_y2 = 4 * exponent_a * exponent_b * factorial2(
                        i - 1) * factorial2(j + 1) * factorial2(k - 1) / (
                                             (2 * exponent) ** ((angular_momentum + 2) / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_y3 = 2 * (exponent_a * j_2 + exponent_b * j_1) * factorial2(
                        i - 1) * factorial2(j - 1) * factorial2(k - 1) / ((2 * exponent) ** (angular_momentum / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_y = element_y1 + element_y2 - element_y3

                    element_z1 = k_1 * k_2 * factorial2(
                        i - 1) * factorial2(j - 1) * factorial2(k - 3) / (
                                         (2 * exponent) ** ((angular_momentum - 2) / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_z2 = 4 * exponent_a * exponent_b * factorial2(
                        i - 1) * factorial2(j - 1) * factorial2(k + 1) / (
                                         (2 * exponent) ** ((angular_momentum + 2) / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_z3 = 2 * (exponent_a * k_2 + exponent_b * k_1) * factorial2(
                        i - 1) * factorial2(j - 1) * factorial2(k - 1) / ((2 * exponent) ** (angular_momentum / 2)) * (
                                         np.pi / exponent) ** (3 / 2)

                    element_z = element_z1 + element_z2 - element_z3

                    element_sum += primitive_normalization_a * primitive_normalization_b * coefficient_a * coefficient_b * (element_x + element_y + element_z)

            element_sum = normalization_1 * normalization_2 * element_sum
            return element_sum
