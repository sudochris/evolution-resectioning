import numpy as np

from evolution.base.base_genome import BaseGenome
from evolution.base.base_translator import BaseTranslator


class CameraTranslator(BaseTranslator):
    def translate_genome(self, genome: BaseGenome, *args, **kwargs):
        """
        Splits the camera genome into essential pinhole camera parts

        Constructs a camera matrix for with intrinsic parameters, a translation vector,
        a rotation vector and the distortion coefficients. These array may be used to
        evaluate the camera's fitness and project a geometry onto an image plane.

        :param genome: The camera genome with 14 dna elements
        :return: A tuple with camera_matrix, translation vector, rotation vector, distortion coefficients
        """
        camera_matrix = np.array([
            [genome.dna[0], 0, genome.dna[2]],
            [0, genome.dna[1], genome.dna[3]],
            [0, 0, 1]], dtype=np.float32)

        t_vec = genome.dna[4:7]
        r_vec = genome.dna[7:10]
        d_vec = genome.dna[10:]

        return camera_matrix, t_vec, r_vec, d_vec
