from typing import List

from HartreeFock.Objects.Atom import Atom


class Molecule:
    '''Object that holds all the necessary infromation of the molecule that is to be calculated
    '''

    def __init__(self, atoms : List[Atom] = []):
        '''Generates instance of a Molecule

        Parameters
        ----------
        atoms : list, optional
            list of Atoms that make up the Molecule, by default []
        '''

        self.atoms = atoms

    def add_atoms(self, atoms: List[Atom]):
        '''adds atoms to the molecule after instance of Molecule has been initialized

        Parameters
        ----------
        atoms : List[Atom]
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