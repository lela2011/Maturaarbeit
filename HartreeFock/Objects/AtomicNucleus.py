import math


class AtomicNucleus:
    def __init__(self, charge: int, position: (float, float, float)):
        """
        Creates an instance of an atomic nucleus used for HartreeFock calculations

        :param charge: charge of the atomic nucleus in elementary charge
        :type charge: int
        :param position: three-dimensional position of the atomic nucleus in the form (x,y,z)
        :type position: tuple
        """
        self.charge = charge
        self.__position = position

    def x(self) -> float:
        """
        Cartesian coordinates

        :return: the x coordinate of the atomic nucleus
        """
        return self.__position[0]

    def y(self) -> float:
        """
        Cartesian coordinates

        :return: the y coordinate of the atomic nucleus
        """
        return self.__position[1]

    def z(self) -> float:
        """
        Cartesian Coordinates

        :return: the z coordinate of the atomic nucleus
        """
        return self.__position[2]

    def r(self) -> float:
        """
        Spherical Coordinates

        :return: the radial distance from origin of the atomic nucleus
        """
        return math.sqrt(self.x() ^ 2 + self.y() ^ 2 + self.z() ^ 2)

    def phi(self) -> float:
        """
        Spherical Coordinates

        :return: the azimuthal angle of the atomic nucleus with respect to the origin between 0 and 2π
        """
        return math.atan(self.y() - self.x())

    def theta(self) -> float:
        """
        Spherical Coordinates

        :return: the polar angle of the atomic nucleus with respect to the origin between 0 and π
        """
        return math.acos(self.z() / self.r())
