from typing import List

from .Atom import Atom
import numpy as np


class Molecule:
    '''Object that holds all the necessary infromation of the molecule that is to be calculated
    '''

    def __init__(self, atoms : np.ndarray, shell_occupancy: np.ndarray):
        '''Generates instance of a Molecule

        Parameters
        ----------
        atoms : np.ndarray, optional
            list of Atoms that make up the Molecule, by default []
        '''

        self.atoms = atoms
        self.shell_occupancy = shell_occupancy

    def add_atoms(self, atoms: np.ndarray):
        '''adds atoms to the molecule after instance of Molecule has been initialized

        Parameters
        ----------
        atoms : np.ndarray
            List of atoms that make up the Molecule, by default []
        '''
        self.atoms.append(atoms)

    def pyscf_molecule_string(self) -> str:
        '''Generates string that describes molecule to be passed in pyscf's Mole object

        Returns
        -------
        str
            string describing molecule used in pyscf
        '''

        # Create list of pyscf atom names
        pyscf_atom_strings = [atom.pyscf_atom_name() for atom in self.atoms]

        # Combine strings into one string separated by "; "
        pyscf_molecule_string = "; ".join(pyscf_atom_strings)
        
        return pyscf_molecule_string