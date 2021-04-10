from evolution.base.base_genome_factory import BaseGenomeFactory
from evolution.camera.camera_genome_parameters import CameraGenomeParameters


class CameraGenomeFactory(BaseGenomeFactory):
    def __init__(self, genome_parameters: CameraGenomeParameters) -> None:
        super().__init__(genome_parameters)
