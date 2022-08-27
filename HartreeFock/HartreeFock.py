from operator import truediv
from pyscf import gto
import numpy as np

from Matrices.Transformation import Transformation
from Matrices.Core import Core
from Matrices.Coefficient import Coefficient
from Matrices.Fock import Fock

from Objects.Atom import Atom
from Objects.Molecule import Molecule

if __name__ == "__main__":

    # Define Atoms
    f = Atom("F", (0.0, 0.0, 0.0))
    h = Atom("H", (0.0, 0.0, 1.73))

    # Define Molecule
    molecule = Molecule([f, h])

    # Generate pyscf string to build pyscf molecule
    pyscf_name = molecule.pyscf_molecule_string()

    # Build pyscf molecule
    mol = gto.M(atom = pyscf_name, basis="STO-3G")

    # Generate integrals used for caluclations
    s = mol.intor("int1e_ovlp", hermi=1)
    kin = mol.intor("int1e_kin", hermi=1)
    nuc = mol.intor("int1e_nuc", hermi=1)
    ERIs = mol.intor("int2e")

    # Calculate transformation matrix based on overlap matrix
    transformation = Transformation(s)

    # Generate Core-Hamiltonian based on kinetic integrals and nuclear attraction integrals
    core = Core(kin, nuc)

    # Initial Guess
    last_coefficient = Coefficient(mol.nao)

    # Print initial guess to check if algorithm is messing up somewhere
    print("Initial guess: {}\n----------------------------".format(last_coefficient.matrix))

    # Set energy to "unreachable" value for first iteration so convergence check is not triggered
    last_energy : float = 1000

    # set variable used to check for convergence
    converges = False

    # iterate until energies converge
    while not converges:
        
        # generate Fock matrix
        fock = Fock(core, last_coefficient, ERIs)

        # transform Fock matrix
        transformed_fock = np.linalg.multi_dot([transformation.matrix.transpose(), fock.matrix, transformation.matrix])

        # diagonalize tranformed Fock matrix to find eigenenergy and eigenvectors
        eigen_energy, transformed_eigen_vec = np.linalg.eig(transformed_fock)

        # order eigenenergies and corresponding eigenvectors
        idx = eigen_energy.argsort()
        eigen_energy = eigen_energy[idx]
        transformed_eigen_vec = transformed_eigen_vec[:,idx]

        # transform coefficients to untransformed state
        coefficient = np.dot(transformation.matrix, transformed_eigen_vec)

        # sum energy to be used to check convergence
        energy = np.sum(eigen_energy)

        # calculate energy difference
        energy_diff = energy - last_energy

        # store energy for next iteration
        last_energy = energy

        # generate new coefficient matrix
        last_coefficient = Coefficient(mol.nao, old_matrix = coefficient)

        # check for convergence
        if 0 <= energy_diff <= 0.0001:
            converges = True

    #print found energies
    print(eigen_energy)

