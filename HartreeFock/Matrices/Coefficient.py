import numpy as np

class Coefficient:
    '''Object that stores the expansion coefficients of the basis functions where each row corresponds to one shell wavefunction
    '''

    def __init__(self, function_num : int, old_matrix : np.ndarray = []) -> None:
        '''Creates an instance of the Coefficient object

        Parameters
        ----------
        function_num : int
            amount of basis functions used to describe the molecule
        old_matrix : np.ndarray, optional
            coefficient matrix as the result of the previous iteration. Leave empty if it's the initial guess
        '''

        # Checks if this is the initial guess
        # if yes -> creates a 2-D square matrix with zeors or random matrix between -100 and 100 as the initial guess
        if old_matrix == []:
            # self.matrix : np.ndarray = np.zeros((function_num, function_num))
            self.matrix :np.ndarray = np.random.uniform(low=-100, high=100, size=(function_num, function_num))
            
        # if no -> stores the result of the last calucaltion in the new matrix
        else:
            self.matrix : np.ndarray = old_matrix