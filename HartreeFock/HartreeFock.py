import imp
from pyscf import gto
import numpy as np

from Matrices.Transformation import Transformation
from Matrices.Core import Core
from Matrices.Coefficient import Coefficient
from Matrices.Fock import Fock

from Objects.Atom import Atom

if __name__ == "__main__":
    Neon = Atom("Ne", (0.0, 0.0, 0.0))
    pyscf_name = Neon.pyscf_atom_name()
    mol = gto.M(atom = pyscf_name, basis="STO-3G")

    s = mol.intor("int1e_ovlp", hermi=1)
    kin = mol.intor("int1e_kin", hermi=1)
    nuc = mol.intor("int1e_nuc", hermi=1)
    ERIs = mol.intor("int2e")

    transformation = Transformation(s)
    core = Core(kin, nuc)
    coefficient = Coefficient(mol.nao)
    fock = Fock(core, coefficient, ERIs)


