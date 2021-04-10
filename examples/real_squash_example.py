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


def target_edge_image(file_name: str) -> np.array:
    court_bgr = cv.imread(file_name)

    hsv_image = cv.cvtColor(court_bgr, cv.COLOR_BGR2HSV)
    running = True
    cv.namedWindow("Court")
    noop = lambda i: i
    cv.createTrackbar("hmin", "Court", 120, 180, noop)
    cv.createTrackbar("hmax", "Court", 180, 180, noop)

    result = None
    while running:
        h = hsv_image[:, :, 0]
        h_min = cv.getTrackbarPos("hmin", "Court")
        h_max = cv.getTrackbarPos("hmax", "Court")
        h = cv.inRange(h, h_min, h_max)
        cv.imshow("Court", h)
        key = cv.waitKey(1)
        if key == ord('q'):
            result = h
            running = False

    return result

if __name__ == '__main__':
    # 1. Specify all parameters
    image_shape = (image_height, image_width) = 1080, 1920
    parameters_file = "data/real/squash_parameters2.json"
    geometry_file = "data/real/squash_court2.obj"
    image_file_name = "data/court_1080p.png"

    # 2. Construct the needed classes and objects for the algorithm
    image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    camera_genome_factory = CameraGenomeFactory(genome_parameters)
    camera_translator = CameraTranslator()

    fitting_geometry = ObjGeometry(geometry_file)

    # ######## Load the squash court (edge) image ########
    extracted_edge_image = target_edge_image(image_file_name)
    # "extracted_edge_image" is the binary image which will be used for the algorithm
    # ########################################################################

    # 3. Define your actual strategy
    fu, fv = 1000, 1000
    cx, cy = 1920//2, 1080 //2
    tx, ty, tz = -.01, 1.02, -.789
    rx, ry, rz = np.deg2rad(0), np.deg2rad(0), np.deg2rad(0)
    d0, d1, d2, d3, d4 = np.zeros(5)
    start_dna = np.array([fu, fv, cx, cy, tx, ty, tz, rx, ry, rz, d0, d1, d2, d3, d4])

    start_genome = camera_genome_factory.create(start_dna, "opt_camera")

    population_strategy = ValueUniformPopulation(start_dna, 8)
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
                                              extracted_edge_image, fitting_geometry,
                                              False)

    result = camera_algorithm.run()
    best_genome, best_fitness = result.best_genome

    # =========== Present the results ================
    result_image = cv.imread(image_file_name)

    A_start, t_start, r_start, d_start = camera_translator.translate_genome(start_genome)
    A_best, t_best, r_best, d_best = camera_translator.translate_genome(best_genome)

    print(A_best, t_best, r_best, d_best)
    render_geometry_with_camera(result_image, fitting_geometry, A_start, t_start, r_start, d_start, (255, 0, 0), 2)
    render_geometry_with_camera(result_image, fitting_geometry, A_best, t_best, r_best, d_best, (0, 255, 0), 2)

    cv.imshow("I", result_image)
    cv.waitKey(0)
