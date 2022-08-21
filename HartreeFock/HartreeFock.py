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

    fluor_1 = Atom("F", (0.0, 0.0, 0.0))
    fluor_2 = Atom("F", (0.3, 0.0, 0.0))

    fluor_molecule = Molecule([fluor_1, fluor_2])

    pyscf_name = fluor_molecule.pyscf_molecule_string()

    mol = gto.M(atom = pyscf_name, basis="STO-3G")

    s = mol.intor("int1e_ovlp", hermi=1)
    kin = mol.intor("int1e_kin", hermi=1)
    nuc = mol.intor("int1e_nuc", hermi=1)
    ERIs = mol.intor("int2e")

    transformation = Transformation(s)
    core = Core(kin, nuc)

    converges = False

    last_coefficient = Coefficient(mol.nao)
    last_energy : float = 0

    while not converges:
        
        fock = Fock(core, last_coefficient, ERIs)

        transformed_fock = np.linalg.multi_dot([transformation.matrix.transpose(), fock.matrix, transformation.matrix])

        eigen_energy, transformed_eigen_vec = np.linalg.eig(transformed_fock)

        coefficient = np.dot(transformation.matrix, transformed_eigen_vec)

        energy = np.sum(eigen_energy)

        energy_diff = np.abs(last_energy - energy)

        last_energy = energy

        last_coefficient = Coefficient(mol.nao, old_matrix = coefficient)

        if energy_diff <= 0.1:
            converges = True

    print(eigen_energy)




