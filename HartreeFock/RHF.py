from typing import Tuple
from pyscf import gto
import numpy as np

from Matrices.Density import Density
from Matrices.Transformation import Transformation
from Matrices.Core import Core
from Matrices.Coefficient import Coefficient
from Matrices.Fock import Fock

from Objects.Atom import Atom
from Objects.Molecule import Molecule

class RHF():
    '''Object that handles all the restricted Hartree Fock calculations and keeps track of atoms / molecules used in calculation
    '''


    def __init__(self, particle, max_iterations = 1000) -> None:
        '''Generates RHF object for molecule or atom

        Parameters
        ----------
        particle : Atom or Molecule
            Molecule or atom that should be calculated
        max_iterations: int
            Maximum amount of iterations after which the algorithm calls for non convergence, default 1000
        '''

        self.particle = particle
        self.max_iterations = max_iterations

    def calculate(self) -> Tuple:
        '''Generates all necessary integrals and runs the SCF Algorithm until values converge

        Returns
        -------
        Tuple
            tuple of total energy, orbital energy, final expansion coefficients, the amount of iterations necessary and whether the energy converged
        '''

        # generates PySCF molecule to be later used in integral generation
        mol = gto.M(atom = self.particle.pyscf_name(), basis="STO-3G", unit="Bohr")

        # Generate integrals used for caluclations
        s = mol.intor("int1e_ovlp", hermi=1)
        kin = mol.intor("int1e_kin", hermi=1)
        nuc = mol.intor("int1e_nuc", hermi=1)
        ERIs = mol.intor("int2e")

        # Calculate transformation matrix based on overlap matrix
        transformation = Transformation(s)

        # Generate Core-Hamiltonian based on kinetic integrals and nuclear attraction integrals
        core = Core(kin, nuc)

        # if particle is not a molecule, generate shell occupancy for atom
        if type(self.particle) is Molecule:
            shell_occupancy = self.particle.shell_occupancy
        else:
            shell_occupancy = np.full(mol.nao, 2)

        # Initial Guess
        last_coefficient = Coefficient(mol.nao)

        # Define starting energy
        last_energy : float = 0

        # set variable used to check for convergence
        converges = False

        # keep track of iterations
        iterations = 1

        # iterate until energies converge or maximum amount of iterations is reached
        while not converges or iterations > self.max_iterations:
            
            # generate density matrix
            density = Density(mol.nao, shell_occupancy, last_coefficient.matrix)

            # generate Fock matrix
            fock = Fock(core, density, ERIs)

            # transform Fock matrix
            transformed_fock = np.linalg.multi_dot([transformation.matrix.transpose(), fock.matrix, transformation.matrix])

            # diagonalize tranformed Fock matrix to find eigenenergy and eigenvectors
            eigen_energy, transformed_eigen_vec = np.linalg.eigh(transformed_fock)

            # order eigenenergies and corresponding eigenvectors
            idx = eigen_energy.argsort()
            eigen_energy = eigen_energy[idx]
            transformed_eigen_vec = transformed_eigen_vec[:,idx]

            # transform coefficients to untransformed state
            coefficient = np.dot(transformation.matrix, transformed_eigen_vec)

            # sum energy to be used to check convergence
            energy = np.sum(eigen_energy)

            # calculate energy difference
            energy_diff = np.abs(energy - last_energy)

            # store energy for next iteration
            last_energy = energy

            # generate new coefficient matrix
            last_coefficient = Coefficient(mol.nao, old_matrix = coefficient)

            # check for convergence
            if 0 <= energy_diff <= 10e-14: # np.linalg.norm(density_diff)
                converges = True

            # update iteration counter
            iterations += 1

        # define variable to store core repulsion energy
        V_eff = 0

        # calculate over each combination of core interaction
        if type(self.particle) == Molecule:
            atoms = self.particle.atoms
            for a in range(len(atoms)):
                for b in range(a+1, len(atoms)):
                    atom_a = atoms[a]
                    atom_b = atoms[b]
                    # Calculate distance between cores
                    r = np.sqrt((atom_a.x() - atom_b.x()) ** 2 + (atom_a.y() - atom_b.y()) ** 2 + (atom_a.z() - atom_b.z()) ** 2)
                    # calculate repulsion energy
                    V_eff += (atom_a.core_charge * atom_b.core_charge) / r

        # generate density matrix based on final iteration
        final_density = Density(mol.nao, shell_occupancy, last_coefficient.matrix)

        # generate Fock matrix based on final iteration
        final_fock = Fock(core, final_density, ERIs)

        # Calculate a matrix that combines the Fock matrix and core matrix.
        fock_core_combination = core.matrix + final_fock.matrix

        # Divide density matrix by 2
        half_density_matrix = 1/2 * final_density.matrix

        # Element wise multiplication of half density matrix and core fock combination matrix
        energy_matrix = np.multiply(half_density_matrix, fock_core_combination)

        # Sum over each element in matrix
        electron_energy = np.sum(energy_matrix)
        
        # combine core repulsion energy and electron energy
        e_tot = V_eff + electron_energy

        # return total energy, orbital energy, final expansion coefficients and the amount of iterations
        return (e_tot, eigen_energy, last_coefficient, iterations, converges)