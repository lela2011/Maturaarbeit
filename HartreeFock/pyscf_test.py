import imp
from pyscf import gto, scf
from Matrices.Density import Density
from Matrices.Transformation import Transformation
from Matrices.Core import Core
from Matrices.Fock import Fock
import numpy as np

mol = gto.M(atom="H 0 0 0; F 0 0 1.73", basis="STO-3G", unit="Bohr")
mol.verbose = 0

rhf = scf.RHF(mol)
_ = rhf.kernel()

coeff = rhf.mo_coeff

dm = Density(mol.nao, coeff)
s = mol.intor("int1e_ovlp", hermi=1)
kin = mol.intor("int1e_kin", hermi=1)
nuc = mol.intor("int1e_nuc", hermi=1)
ERIs = mol.intor("int2e")

core = Core(kin, nuc)

transform = Transformation(s)

fock = Fock(core, dm, ERIs)
gen_fock = rhf.get_fock()

transformed_fock = np.linalg.multi_dot([transform.matrix.transpose(), gen_fock, transform.matrix])

print("Stop")