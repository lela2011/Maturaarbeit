from typing import Tuple


class Atom:
    '''Object that stores all the necessary information about an Atom
    '''

    def __init__(self, element: str, position: Tuple[float, float, float]):
        '''Creates an instance of an atom used for HartreeFock calculations

        Parameters
        ----------
        element : string
            Atomic symbol of atom, PySCF will use this information to generate the basis functions
        position : Tuple[float, float, float]
            Position of the nucleus in vector space in the form (x, y, z). Lengths are given in Bohr-radii
        '''

        self.element = element
        self.position = position

    def x(self) -> float:
        '''Cartesian coordinate x

        Returns
        -------
        float
            the x coordinate of the atomic nucleus
        '''

        return self.position[0]

    def y(self) -> float:
        '''Cartesian coordinate y

        Returns
        -------
        float
            the y coordinate of the atomic nucleus
        '''

        return self.position[1]

    def z(self) -> float:
        '''Cartesian coordinate z

        Returns
        -------
        float
            the z coordinate of the atomic nucleus
        '''

        return self.position[2]

    def pyscf_atom_name(self) -> str:
        '''Returns the string used in pyscf's Mole object atom attribute

        Returns
        -------
        str
            string used in pyscf's Mole object atom attribute
        '''

        return self.element + " " + str(self.x()) + " " + str(self.y()) + " " + str(self.z())
