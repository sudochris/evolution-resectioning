from typing import List

import numpy as np
import cv2 as cv

from evolution.base.base_algorithm import BaseAlgorithm
from evolution.base.base_genome import BaseGenome
from evolution.base.base_geometry import BaseGeometry
from evolution.base.base_strategies import Population
from evolution.camera.camera_genome_factory import CameraGenomeFactory
from evolution.camera.camera_genome_parameters import CameraGenomeParameters
from evolution.camera.camera_rendering import render_geometry_with_camera
from evolution.camera.camera_translator import CameraTranslator
from evolution.strategies.strategy_bundle import StrategyBundle


class GeneticCameraAlgorithm(BaseAlgorithm):
    def __init__(self,
                 genome_parameters: CameraGenomeParameters,
                 strategy_bundle: StrategyBundle,
                 edge_image: np.array,
                 geometry: BaseGeometry,
                 headless=True) -> None:
        super().__init__(CameraTranslator(),
                         CameraGenomeFactory(genome_parameters),
                         strategy_bundle.populate_strategy,
                         strategy_bundle.selection_strategy,
                         strategy_bundle.crossover_strategy,
                         strategy_bundle.mutation_strategy,
                         strategy_bundle.termination_strategy)
        h, w = edge_image.shape
        self._headless = headless
        self._fitness_map = strategy_bundle.fitness_strategy.create_fitness(edge_image)

        self._geometry = geometry
        self._display_image = np.zeros((h, w, 3))
        self._render_image = np.zeros_like(self._fitness_map, dtype=np.uint8)
        self._current_best_genome = None

    def fitness(self, genome) -> float:
        self._render_image[:] = 0
        camera_matrix, t_vec, r_vec, d_vec = self.translator.translate_genome(genome)
        render_geometry_with_camera(self._render_image, self._geometry, camera_matrix, t_vec, r_vec, d_vec, (255,), 2)
        fitness_lookup = cv.bitwise_and(self._fitness_map, self._fitness_map, mask=self._render_image)
        return fitness_lookup.sum().astype(float)

    def on_display_population(self, current_generation, population: Population, population_fitness: List[float]):
        if not self._headless:
            super().on_display_population(current_generation, population, population_fitness)
            self._display_image[:] = 0

            if self._current_best_genome is not None:
                A_best, t_best, r_best, d_best = self.translator.translate_genome(self._current_best_genome)
                render_geometry_with_camera(self._display_image, self._geometry, A_best, t_best, r_best, d_best,
                                            (255, 0, 0), 2)

            for camera_genome in population:
                camera_matrix, t_vec, r_vec, d_vec = self.translator.translate_genome(camera_genome)
                render_geometry_with_camera(self._display_image, self._geometry, camera_matrix, t_vec, r_vec, d_vec,
                                            (0, 0, 255), 1)

            cv.putText(self._display_image, f"{current_generation=}", (0, 32), cv.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0))
            cv.imshow("camera_algorithm", self._display_image)
            cv.waitKey(1)

    def on_best_genome_found(self, new_best: BaseGenome, genome_fitness: float):
        super().on_best_genome_found(new_best, genome_fitness)
        self._current_best_genome = new_best
