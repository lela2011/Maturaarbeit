import numpy as np

class Overlap:
    '''Object that stores the overlap matrix
    '''

    def __init__(self, matrix: np.ndarray) -> None:
        '''Creates an instance of the Overlap object 

        Parameters
        ----------
        matrix : np.ndarray
            Overlap integrals provided py pyscf in the form of a :math:`n^2` matrix
        '''

        # Overlap Matrix
        self.matrix : np.ndarray = matrix
