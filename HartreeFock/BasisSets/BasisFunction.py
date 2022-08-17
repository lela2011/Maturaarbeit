import numpy as np


class BasisFunction:

    def __init__(self, exponents, coefficients, primitive_normalization_constants, angular_momentum, center):
        self.exponents = exponents
        self.coefficients = coefficients
        self.primitive_normalization_constants = primitive_normalization_constants
        self.angular_momentum = angular_momentum
        self.center = center

    def calculate(self, x: float, y: float, z: float):
        primitive_sum = 0

        i = self.angular_momentum[0]
        j = self.angular_momentum[1]
        k = self.angular_momentum[2]

        center_x = self.center[0]
        center_y = self.center[1]
        center_z = self.center[2]

        for a in range(len(self.coefficients)):
            coefficient = self.coefficients[a]
            exponent = self.exponents[a]
            primitive_normalization = self.primitive_normalization_constants[a]

            primitive = coefficient * primitive_normalization * np.exp(- exponent * ((x - center_x) ** 2 + (y - center_y) ** 2 + (z - center_z) ** 2))
            primitive_sum = primitive_sum + primitive

        angular = (x - center_x) ** i * (y - center_y) ** j * (z - center_z) ** k

        return self.normalization * angular * primitive_sum
