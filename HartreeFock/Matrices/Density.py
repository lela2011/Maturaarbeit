import numpy as np

class Density:
    '''Object that stores the density matrix
    '''

    def __init__(self, function_num : int, old_matrix : np.ndarray = []) -> None:
        '''Creates an instance of the density object

        Parameters
        ----------
        function_num : int
            amount of basis functions used to describe the molecule
        old_matrix : np.ndarray, optional
            coefficient matrix as the result of the previous iteration. Leave empty if it's the initial guess
        '''

        # Checks if this is the initial guess
        # if yes -> creates a 2-D square matrix with zeors
        if old_matrix == []:
            self.matrix : np.ndarray = np.zeros((function_num, function_num))
            
        # if no -> stores the result of the last calucaltion in the new matrix
        else:
            self.matrix : np.ndarray = self._build_density_matrix(old_matrix)

    def _build_density_matrix(self, old_matrix: np.ndarray):
        '''Calculates the density matrix

        Parameters
        ----------
        old_matrix : np.ndarray
            last coefficient matrix

        Returns
        -------
        np.ndarray
            density matrix
        '''

        # get shape for density matrix
        i_range, k_range = old_matrix.shape

        # generate empty matrix to store values in later
        matrix = np.zeros((i_range, k_range))

        for i in range(i_range):
            for k in range(k_range):
                # Select all coefficients for the j-th basis function
                row_i = old_matrix[i]
                # Select all coefficients for the l-th basis function
                row_k = old_matrix[k]

                # Element wise multiplication of all the coefficients
                product_array = np.multiply(row_i, row_k)

                # Sum all products and multiply by 2 to generate a density matrix element
                density_matrix_element = 2 * np.sum(product_array)

                # store calculated value in matrix
                matrix[i, k] = density_matrix_element

        return matrix