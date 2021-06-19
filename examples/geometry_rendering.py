from typing import Tuple

import numpy as np

from evolution.camera import ObjGeometry, render_geometry_with_camera, CameraGenomeParameters, CameraGenomeFactory, \
    CameraTranslator

import cv2 as cv
def synthetic_target_dna(shape: Tuple[int, int]):
    fu = max(shape)
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
    image_shape = (image_height, image_width) = 228, 322
    parameters_file = "data/synth/squash_parameters.json"
    squash_geometry_file = "data/synth/squash_court.obj"

    # 2. Construct the needed classes and objects for the algorithm
    genome_parameters = CameraGenomeParameters(parameters_file, image_shape)
    camera_genome_factory = CameraGenomeFactory(genome_parameters)
    camera_translator = CameraTranslator()

    squash_geometry = ObjGeometry(squash_geometry_file)

    # =========== Present the results ================
    squash_image = np.full((image_height, image_width, 3), (255, 255, 255), dtype=np.uint8)

    real_dna = synthetic_target_dna(image_shape)
    real_genome = camera_genome_factory.create(real_dna, "target_camera")
    A_real, t_real, r_real, d_real = camera_translator.translate_genome(real_genome)

    render_geometry_with_camera(squash_image, squash_geometry, A_real, t_real, r_real, d_real, (0, 0, 0), 1, cv.MARKER_TILTED_CROSS, 8)

    cv.namedWindow("squash_image", cv.WINDOW_KEEPRATIO)
    cv.imshow("squash_image", squash_image)
    cv.resizeWindow("squash_image", 800, 600)
    cv.waitKey(0)
