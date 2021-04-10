from typing import Optional, Tuple

from evolution.base.base_genome_parameters import BaseGenomeParameters


class CameraGenomeParameters(BaseGenomeParameters):

    def __init__(self,
                 parameters_file: str,
                 image_shape: Tuple[int, int],
                 default_display_name: Optional[str] = None) -> None:
        super().__init__(parameters_file, default_display_name)
        self._image_shape = image_shape

    @property
    def image_width(self):
        return self._image_shape[1]

    @property
    def image_height(self):
        return self._image_shape[0]

    @property
    def image_rows(self):
        return self.image_height

    @property
    def image_cols(self):
        return self.image_width
