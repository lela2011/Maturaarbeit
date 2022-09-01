import numpy as np

from Matrices.Core import Core
from Matrices.Density import Density

class Fock:
    '''Object that stores the Fock matrix
    '''

    def __init__(self, core : Core, density : Density, ERIs : np.ndarray) -> None:
        '''Creates an instance of the Fock Object

        Parameters
        ----------
        core : Core
            Core matrix that contains the kinetic matrix and potential matrix
        density : Density
            Density matrix that is used for calculation of the two electron part
        ERIs : np.ndarray
            4D Matrix that stores the electron electron repulsion integrals. Inegrals are stored as ( r_1 r_1 | r_2 r_2 ) (chemist notation) and can also be accessed that way
        '''
        
        # builds the Fock matrix from passed arguments
        self.matrix : np.ndarray = self._build_matrix(core.matrix, ERIs, density.matrix)


    def _build_matrix(self, core : np.ndarray, ERIs : np.ndarray, density: np.ndarray) -> np.ndarray:
        '''Builds the fock matrix from passed arguments. Symmetry of the matrix is not taken into account

        Parameters
        ----------
        core : np.ndarray
            Core matrix that contains the kinetic matrix and potential matrix
        ERIs : np.ndarray
            4D Matrix that stores the electron electron repulsion integrals. Inegrals are stored as ( r_1 r_1 | r_2 r_2 ) / (ik|jl) (chemist notation) and can also be accessed that way. Physics notation would be ( r_1 r_2 | r_1 r_2 ) / (ij|kl)

        Returns
        -------
        np.ndarray
            calculated Fock matrix
        '''

        # get shape of the core matrix
        i, k = core.shape

        # copy shape of core matrix also for iteration of basis functions dependent on r_2
        j = l = i

        # define empty square matrix where calculated values can be stored in
        two_electron_matrix = np.empty((i,i))

        # iterate over each row
        for i_iter in range(i):

            # iterate over each column
            for k_iter in range(k):

                # define variable that stores double sum over basis functions dependent on r_2
                iter_sum = 0

                # iterate over first basis function dependent on r_2
                for j_iter in range(j):

                    # iterate over second basis function dependent on r_2
                    for l_iter in range(l):

                        # retireve electron electron repulsion integral (ik|jl) (chemist notation) / (ij|kl) (physics notation)
                        # --> coulomb integral
                        coulomb_integral = ERIs[i_iter, k_iter, j_iter, l_iter]
                        # retireve electron electron repulsion integral (il|jk) (chemist notation) / (ij|lk) (physics notation)
                        # --> exchange integral
                        exchange_integral = ERIs[i_iter, l_iter, j_iter, k_iter]
                        
                        # add calculated combination of density matrix element and integrals to summation variable
                        iter_sum += density[j_iter, l_iter] * (coulomb_integral - 0.5 * exchange_integral)
                
                # store summation over both basis functions dependent on r_2 in their corresponding matrix position
                two_electron_matrix[i_iter, k_iter] = iter_sum

        # calculate Fock matrix by combining the core part and two electron part
        fock_matrix = np.add(core, two_electron_matrix)

        return fock_matrix