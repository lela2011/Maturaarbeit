import imp
import numpy as np

from Matrices.Overlap import Overlap

class Transformation:
    '''Object that stores the transformation matrix used for symmetric diagonalization
    '''

    def __init__(self, overlap: Overlap) -> None:
        '''Creates an instance of the Transformation object 

        Parameters
        ----------
        matrix : Overlap
            Overlap matrix calculated from overlap integrals
        '''

        # Find eigenvalues and eigenvectors of the overlap matrix.
        # Eigenvalues are returned as a 1-D array. 
        # Eigenvectors are returned as a 2-D array (matrix) where each column resembles one eigenvector
        eigen_val, vector_matrix = np.linalg.eigh(overlap.matrix)

        # Generates a diagonal matrix constructed form the inverse square roots of the overlap matrix' eigenvalues
        diagonal_matrix = np.diag(eigen_val ** -0.5)

        # Transposes the matrix of eigenvectors to be later used to calculate the transformation matrix
        transposed_vector_matrix = np.transpose(vector_matrix)

        # Calculates the transformation_matrix by concatenating two matrix multiplications
        transformation_matrix = np.linalg.multi_dot([vector_matrix, diagonal_matrix, transposed_vector_matrix]) 

        self.matrix : np.ndarray = transformation_matrix