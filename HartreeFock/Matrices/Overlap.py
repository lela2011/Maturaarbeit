import numpy as np
from scipy.special import comb, factorial2

from HartreeFock.BasisSets import BasisFunction


class Overlap:
    def __init__(self, basis_functions):
        self.matrix = self._build_matrix(basis_functions)

