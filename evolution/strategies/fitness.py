import numpy as np
import cv2 as cv

from evolution.base.base_strategies import FitnessStrategy


class DistanceMap(FitnessStrategy):

    class DistanceType:
        L1 = cv.DIST_L1
        L2 = cv.DIST_L2

    def __init__(self, distance_type: DistanceType = DistanceType.L2, log_div: float = 0.1) -> None:
        super().__init__()
        self._distance_type = distance_type
        self._log_div = log_div

    def create_fitness(self, edge_image: np.array) -> np.array:
        fitness_map = cv.distanceTransform(~edge_image, self._distance_type, maskSize=3)
        cv.normalize(fitness_map, fitness_map, 0.0, 1.0, cv.NORM_MINMAX)

        fitness_map = np.log1p(fitness_map) / self._log_div
        fitness_map[fitness_map > 1.0] = 1.0
        return 1-fitness_map

    def printable_identifier(self):
        return "DistanceMap"


class DistanceMapWithPunishment(DistanceMap):

    def create_fitness(self, edge_image: np.array) -> np.array:
        fitness_map = super().create_fitness(edge_image)
        fitness_map = (2 * fitness_map) - 1
        return fitness_map

    def printable_identifier(self):
        return "DistanceMapWithPunishment"
