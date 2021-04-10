from typing import List

import numpy as np
from abc import ABC as AbstractBaseClass, abstractmethod


class BaseGeometry(AbstractBaseClass):
    """
    This class represents a geometric object in world space.

    Derive from this class and provide the generator methods for
    (A) A n x 3 array for n (x, y, z) coordinates in the real world
    (B) A list of lists for indices of coordinates, specified in (A), which are connected
    """
    def __init__(self) -> None:
        super().__init__()
        self._world_points = self.provide_world_points()
        self._connections = self.provide_connection_list()

    @property
    def world_points(self):
        """
        Access the geometries 3d world coordinates
        :return:
        """
        return self._world_points

    @property
    def connections(self):
        """
        Access the geometries index list, which is a list of list of world_point indices
        :return:
        """
        return self._connections

    @abstractmethod
    def provide_world_points(self) -> np.array:
        """
        Constructs the list of 3d world points
        :return: The actual geometry coordinates in world space
        """
        pass

    @abstractmethod
    def provide_connection_list(self) -> List[List[int]]:
        """
        Constructs the list of list of world point indices
        :return: The actual connection list of lists.
        """
        pass
