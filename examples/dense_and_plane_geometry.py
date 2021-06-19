from typing import Tuple

import numpy as np

from evolution.base import DenseGeometry, PlaneGeometry
from evolution.camera import CameraGenomeParameters, CameraGenomeFactory, CameraTranslator, ObjGeometry, \
    render_geometry_with_camera
import cv2 as cv


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

    real_geometry = ObjGeometry(geometry_file)
    dense_geometry = DenseGeometry(real_geometry, 16)
    plane_geometry = PlaneGeometry(real_geometry, 0, 16)

    # =========== Present the results ================
    real_image = np.zeros((image_height, image_width, 3), dtype=np.uint8)
    dense_image = np.zeros_like(real_image)
    plane_image = np.zeros_like(real_image)

    real_dna = synthetic_target_dna(image_shape)
    real_genome = camera_genome_factory.create(real_dna, "target_camera")
    A_real, t_real, r_real, d_real = camera_translator.translate_genome(real_genome)

    render_geometry_with_camera(real_image, real_geometry, A_real, t_real, r_real, d_real, (0, 200, 0), 1, cv.MARKER_TILTED_CROSS)
    render_geometry_with_camera(dense_image, dense_geometry, A_real, t_real, r_real, d_real, (255, 0, 0), 1, cv.MARKER_TILTED_CROSS)
    render_geometry_with_camera(plane_image, plane_geometry, A_real, t_real, r_real, d_real, (0, 0, 255), 1, cv.MARKER_TILTED_CROSS)

    cv.imshow("real_image", real_image)
    cv.imshow("dense_image", dense_image)
    cv.imshow("plane_image", plane_image)
    cv.waitKey(0)
