import itertools
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


class DenseGeometry(BaseGeometry):
    """
    A DenseGeometry samples the bounding volume for a given geometry in
    equidistant spaces. This is used for evaluation of re-projection errors.
    """
    def __init__(self, other: BaseGeometry, n_samples: int) -> None:
        self.other = other
        self.n_samples = n_samples
        super().__init__()

    def provide_world_points(self) -> np.array:
        min_pts = np.min(self.other.world_points, axis=0)
        max_pts = np.max(self.other.world_points, axis=0)
        steps = np.linspace(min_pts, max_pts, self.n_samples)
        return np.array([list(value) for value in list(itertools.product(*steps.T))])

    def provide_connection_list(self) -> list:
        return []


class PlaneGeometry(BaseGeometry):
    """
    A PlaneGeometry samples an orthogonal slice in equidistant spaces. This is used for evaluation of re-projection
    errors.
    """
    def __init__(self, other: BaseGeometry, height: float, n_samples: int) -> None:
        self.other = other
        self.n_samples = n_samples
        self.height = height
        super().__init__()

    def provide_world_points(self) -> np.array:
        min_pts = np.min(self.other.world_points, axis=0)
        max_pts = np.max(self.other.world_points, axis=0)

        xz_steps = np.linspace([min_pts[0], min_pts[2]], [max_pts[0], max_pts[2]], self.n_samples)
        xz_product = np.array([list(value) for value in list(itertools.product(*xz_steps.T))])
        final_product = np.insert(xz_product, 1, self.height, axis=1)

        return final_product

    def provide_connection_list(self) -> list:
        return []
