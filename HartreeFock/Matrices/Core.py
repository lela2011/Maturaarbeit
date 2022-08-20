import numpy as np

class Core:
    '''Object that stores the core hamiltonian which is later used to generate the Fock-Matrix.
    '''

    def __init__(self, kinetic_matrix : np.ndarray, core_potential_matrix : np.ndarray) -> None:
        '''Creates an instance of the Core object

        Parameters
        ----------
        kinetic_matrix : np.ndarray
            kinetic energy integrals of electrons provided by pyscf in the form of a :math:`n^2` matrix
        core_potential_matrix : _type_
            potential energy integrals between electrons and cores provided by pyscf in the form of a :math:`n^2` matrix
        '''

        # Generates core hamiltonian matrix by adding the kinetic energy matrix and the core potential matrix
        self.matrix: np.ndarray = np.add(kinetic_matrix, core_potential_matrix)