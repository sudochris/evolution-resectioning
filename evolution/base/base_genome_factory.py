from typing import Optional
import numpy as np
from abc import ABC as AbstractBaseClass

from evolution.base.base_genome import BaseGenome
from evolution.base.base_genome_parameters import BaseGenomeParameters


class BaseGenomeFactory(AbstractBaseClass):

    def __init__(self, genome_parameters: BaseGenomeParameters) -> None:
        """
        Creates a genome factory which is used to create genomes during runtime
        """
        super().__init__()
        self._genome_parameters = genome_parameters

    def validate_bounds(self, genome: BaseGenome, bounds: np.array):
        """
        Validates the genomes dna based on the bounds array.
        This method clips the individual values to their corresponding bounds in place.
        :param genome: The genome which should be validated
        :param bounds: 2 x n_genes bounds array. First row is used for lower, second for upper bounds.
        """
        lower, upper = bounds
        genome.dna = np.clip(genome.dna, lower, upper)

    def create(self, dna: np.array, display_name: Optional[str] = None) -> BaseGenome:
        """
        Creates a genome with given genes / dna
        :param dna: The actual genes / dna array
        :param display_name: Optinoal display name
        :return: A new genome
        """
        return BaseGenome(dna, display_name)

    def initial_genome(self, display_name: Optional[str] = None) -> BaseGenome:
        """
        Constructs a genome with the dna provided by provide_initial_dna
        :param display_name: Optional display name
        :return: A new genome
        """
        return self.create(self._genome_parameters.initial_dna, display_name)

    def empty_genome(self) -> "BaseGenome":
        """
        Creates a genome with all genes set to zero.
        :return: A new genome
        """
        empty_dna = np.zeros((self._genome_parameters.n_genes))
        return self.create(empty_dna, self._genome_parameters._default_display_name)

    @property
    def genome_bounds(self):
        return self._genome_parameters.genome_bounds
