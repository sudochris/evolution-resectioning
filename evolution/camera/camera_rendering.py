from typing import Tuple

import numpy as np
import cv2 as cv

from evolution.base.base_geometry import BaseGeometry


def project_points(object_points: np.array, camera_matrix: np.array, t_vector: np.array, r_vector: np.array,
                   d_vector: np.array):
    """ Projects world points using a pinhole camera model specified by the intrinsic and extrinsic camera parameters.

    Uses OpenCV's projectPoints internally. May be changed to custom implementation later, while preserving
    the functionality.

    :param object_points: World point coordinates
    :param camera_matrix: 3x3 Intrinsic camera matrix
    :param t_vector: 3 component extrinsic translation vector
    :param r_vector: 3 component extrinsic rotation vector
    :param d_vector: 5 component distortion coefficients
    :return:
    """
    projected_points, _ = cv.projectPoints(object_points, r_vector, t_vector, camera_matrix, d_vector)
    return projected_points


def project_geometry(geometry: BaseGeometry, camera_matrix: np.array, t_vector: np.array, r_vector: np.array,
                     d_vector: np.array):
    """ Projects a given geometry using a pinhole camera model specified by the intrinsic and extrinsic camera
    parameters.

    Basically extracts the geometry's world points and calls project_points.

    :param geometry: The geometry
    :param camera_matrix: 3x3 Intrinsic camera matrix
    :param t_vector: 3 component extrinsic translation vector
    :param r_vector: 3 component extrinsic rotation vector
    :param d_vector: 5 component distortion coefficients
    :return:
    """
    return project_points(geometry.world_points, camera_matrix, t_vector, r_vector, d_vector)


def render_geometry_with_camera(image: np.array,
                                geometry: BaseGeometry,
                                camera_matrix: np.array,
                                t_vector: np.array,
                                r_vector: np.array,
                                d_vector: np.array,
                                line_color: Tuple,
                                line_thickness: int = 2,
                                marker_type=None,
                                marker_size=16):
    """ Renders a geometry to an image using a pinhole camera model specified by the intrinsic and extrinsic camera
    parameters.

    Uses project_geometry to project the geometry's world points and connects them  by using
    geometry's connections attribute.

    Uses OpenCV internally, so you may provide the line_color as BGR

    :param with_points:
    :param image: The image to draw on
    :param geometry: The geometry
    :param camera_matrix: 3x3 Intrinsic camera matrix
    :param t_vector: 3 component extrinsic translation vector
    :param r_vector: 3 component extrinsic rotation vector
    :param d_vector: 5 component distortion coefficients
    :param line_color: The line color
    :param line_thickness: The line's thickness
    """
    projected_points = project_geometry(geometry, camera_matrix, t_vector, r_vector, d_vector)
    image_height, image_width = image.shape[:2]
    for p_idx in geometry.connections:
        poly_line_points = [[int(projected_points[idx][0][0]), int(projected_points[idx][0][1])] for idx in p_idx]
        poly_line_points = np.clip(poly_line_points, (0, 0), (image_width, image_height))
        cv.polylines(image, np.array([poly_line_points], dtype=np.int64), False, line_color, line_thickness)

    if marker_type:
        for (x, y) in projected_points.reshape(-1, 2):
            cv.drawMarker(image, (int(x), int(y)), line_color, marker_type, marker_size, line_thickness)
