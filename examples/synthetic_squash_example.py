from typing import Tuple

import numpy as np
import cv2 as cv

from evolution.base.base_genome import BaseGenome
from evolution.base.base_geometry import BaseGeometry
from evolution.camera.camera_algorithm import GeneticCameraAlgorithm
from evolution.camera.camera_genome_factory import CameraGenomeFactory
from evolution.camera.camera_genome_parameters import CameraGenomeParameters
from evolution.camera.camera_rendering import render_geometry_with_camera
from evolution.camera.camera_translator import CameraTranslator
from evolution.camera.object_geometry import ObjGeometry
from evolution.strategies.crossover import TwoPoint
from evolution.strategies.fitness import DistanceMapWithPunishment, DistanceMap
from evolution.strategies.mutation import BoundedUniformMutation
from evolution.strategies.populate import ValueUniformPopulation
from evolution.strategies.selection import Tournament
from evolution.strategies.strategy_bundle import StrategyBundle
from evolution.strategies.termination import NoImprovement


def synthetic_target_edge_image(shape: Tuple[int, int], target_geometry: BaseGeometry,
                                target_genome: BaseGenome) -> np.array:
    edge_image = np.zeros(shape, dtype=np.uint8)
    A, t, r, d = CameraTranslator().translate_genome(target_genome)
    render_geometry_with_camera(edge_image, target_geometry, A, t, r, d, (255,))
    return edge_image


def synthetic_target_dna(shape: Tuple[int, int]):
    fu = max(shape) - 100
    fv = fu
    h, w = shape
    cx, cy = w // 2, h // 2
    tx, ty, tz = 0.00, 2.35, 8.40
    rx, ry, rz = 0.29, 0.00, 0.00
    d0, d1, d2, d3, d4 = np.zeros(5)
    camera_dna = np.array([fu, fv, cx, cy, tx, ty, tz, rx, ry, rz, d0, d1, d2, d3, d4])
    return camera_dna


if __name__ == '__main__':
    # 1. Specify all parameters
    image_shape = (image_height, image_width) = 600, 800
    parameters_file = "data/synth/squash_parameters.json"
    geometry_file = "data/synth/squash_court.obj"

    # 2. Construct the needed classes and objects for the algorithm
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    camera_genome_factory = CameraGenomeFactory(genome_parameters)
    camera_translator = CameraTranslator()

    fitting_geometry = ObjGeometry(geometry_file)

    # ######## Create a synthetic, perfect squash court (edge) image ########
    # This step is only needed for synthetic experiments. In a real world application
    # one would extract the edges as binary image of the geometric object instead of
    # rendering the scene! (see real_squash_example)
    real_geometry = fitting_geometry  # Ideally, the target geometry is the fitting geometry
    real_dna = synthetic_target_dna(image_shape)
    real_genome = camera_genome_factory.create(real_dna, "target_camera")

    extracted_edge_image = synthetic_target_edge_image(image_shape, real_geometry, real_genome)
    # "extracted_edge_image" is the binary image which will be used for the algorithm
    # ########################################################################

    # 3. Define your actual strategy
    start_dna = synthetic_target_dna(image_shape)
    random_range = np.array(
        [[-100, -100, -32, -32, -0.1, -1.0, -1.00, np.deg2rad(-20), np.deg2rad(-10), np.deg2rad(-10), -0, -0, -0, -0,
          -3],
         [+100, +100, +32, +32, +0.1, +1.0, +1.00, np.deg2rad(+20), np.deg2rad(+10), np.deg2rad(+10), +0, +0, +0, +0,
          +3]])

    start_dna += np.random.uniform(low=random_range[0], high=random_range[1])

    start_genome = camera_genome_factory.create(start_dna, "opt_camera")

    population_strategy = ValueUniformPopulation(8)
    fitness_strategy = DistanceMapWithPunishment(DistanceMap.DistanceType.L2, .3)
    selection_strategy = Tournament(4)
    crossover_strategy = TwoPoint()
    mutation_strategy = BoundedUniformMutation(genome_parameters)
    termination_strategy = NoImprovement(300)

    strategy_bundle = StrategyBundle(population_strategy,
                                     fitness_strategy,
                                     selection_strategy,
                                     crossover_strategy,
                                     mutation_strategy,
                                     termination_strategy)

    # 4. Construct and run the optimization algorithm
    camera_algorithm = GeneticCameraAlgorithm(genome_parameters, strategy_bundle,
                                              extracted_edge_image, fitting_geometry, headless=False)

    result = camera_algorithm.run(start_dna)
    best_genome, best_fitness = result.best_genome

    # =========== Present the results ================
    result_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)

    A_real, t_real, r_real, d_real = camera_translator.translate_genome(real_genome)
    A_start, t_start, r_start, d_start = camera_translator.translate_genome(start_genome)
    A_best, t_best, r_best, d_best = camera_translator.translate_genome(best_genome)

    print(A_start, t_start, r_start, d_start)
    print(A_real, t_real, r_real, d_real)
    print(A_best, t_best, r_best, d_best)

    render_geometry_with_camera(result_image, real_geometry, A_real, t_real, r_real, d_real, (0, 200, 0), 8)
    render_geometry_with_camera(result_image, fitting_geometry, A_start, t_start, r_start, d_start, (255, 0, 0), 2)
    render_geometry_with_camera(result_image, fitting_geometry, A_best, t_best, r_best, d_best, (0, 0, 255), 2)

    cv.imshow("I", result_image)
    cv.waitKey(0)
