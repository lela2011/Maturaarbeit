import numpy as np

from Matrices.Core import Core
from Matrices.Coefficient import Coefficient

class Fock:
    '''Object that stores the Fock matrix
    '''

    def __init__(self, core : Core, coefficient : Coefficient, ERIs : np.ndarray) -> None:
        '''Creates an instance of the Fock Object

        Parameters
        ----------
        core : Core
            Core matrix that contains the kinetic matrix and potential matrix
        coefficient : Coefficient
            Coefficient matrix that contains the expansion coefficients for each molecular shell. Each row resembles one shell
        ERIs : np.ndarray
            4D Matrix that stores the electron electron repulsion integrals. Inegrals are stored as ( r_1 r_1 | r_2 r_2 ) (chemist notation) and can also be accessed that way
        '''

        # stores the coefficient matrix to be later used to calculate the density matrix elements
        self._coefficient : Coefficient = coefficient
        
        # builds the Fock matrix from passed arguments
        self.matrix : np.ndarray = self._build_matrix(core.matrix, ERIs)


    def _build_matrix(self, core : np.ndarray, ERIs : np.ndarray) -> np.ndarray:
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

                        # calculate element of density matrix for given basis functions j and l
                        density_matrix_element = self._calculate_density_matrix_element(j_iter, l_iter)

                        # retireve electron electron repulsion integral (ik|jl) (chemist notation) / (ij|kl) (physics notation)
                        # --> coulomb integral
                        coulomb_integral = ERIs[i_iter, k_iter, j_iter, l_iter]
                        # retireve electron electron repulsion integral (il|jk) (chemist notation) / (ij|lk) (physics notation)
                        # --> exchange integral
                        exchange_integral = ERIs[i_iter, l_iter, j_iter, k_iter]
                        
                        # add calculated combination of density matrix element and integrals to summation variable
                        iter_sum += density_matrix_element * (coulomb_integral - 0.5 * exchange_integral)
                
                # store summation over both basis functions dependent on r_2 in their corresponding matrix position
                two_electron_matrix[i_iter, k_iter] = iter_sum

        # calculate Fock matrix by combining the core part and two electron part
        fock_matrix = np.add(core, two_electron_matrix)

        return fock_matrix



    def _calculate_density_matrix_element(self, j : int, l : int):
        '''Calculates the density matrix element for a specific position in the Fock matrix

        Parameters
        ----------
        j : int
            index of the first basis function dependent on r_2
        l : int
            index of the second basis function dependent on r_2

        Returns
        -------
        _type_
            _description_
        '''

        # Select all coefficients for the j-th basis function
        row_j = self._coefficient.matrix[j]
        # Select all coefficients for the l-th basis function
        row_l = self._coefficient.matrix[l]

        # Element wise multiplication of all the coefficients
        product_array = np.multiply(row_j, row_l)

        # Sum all products and multiply by 2 to generate a density matrix element
        density_matrix_element = 2 * np.sum(product_array)

        return density_matrix_element
