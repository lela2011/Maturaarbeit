class Electron:
    def __init__(self, charge: int, mass: float):
        """
        Creates an instance of an electron used for HartreeFock calculations

        :param charge: charge of the electron in elementary charge
        :type charge: int
        :param mass: mass of the electron in kilo-gramms
        :type mass: float
        """
        self.charge = charge
        self.mass = mass
