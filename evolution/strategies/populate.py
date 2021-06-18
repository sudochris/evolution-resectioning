import numpy as np

from evolution.base.base_genome_factory import BaseGenomeFactory
from evolution.base.base_strategies import PopulateStrategy


class BoundedUniformPopulation(PopulateStrategy):
    def __init__(self, population_size: int = 16)  -> None:
        super().__init__()
        self.population_size = population_size

    def populate(self, genome_factory: BaseGenomeFactory, start_dna: np.array):
        lower_bounds, upper_bounds = genome_factory.genome_bounds
        return [genome_factory.create(np.random.uniform(lower_bounds, upper_bounds)) for _ in range(self.population_size)]

    def printable_identifier(self):
        return "BoundedUniformPopulation(n={})".format(self.population_size)


class ValueUniformPopulation(PopulateStrategy):
    def __init__(self, population_size: int = 16) -> None:
        super().__init__()
        self.population_size = population_size

    def populate(self, genome_factory: BaseGenomeFactory, start_dna: np.array):
        _random_range = np.array(
        [[-100, -100, -10, -10, -0.1, -0.1, -0.50, np.deg2rad(-1), np.deg2rad(-1), np.deg2rad(-1), -0, -0, -0, -0, -0],
         [+100, +100, +10, +10, +0.1, +0.1, +0.50, np.deg2rad(+1), np.deg2rad(+1), np.deg2rad(+1), +0, +0, +0, +0, +0]])

        return [genome_factory.create(start_dna + np.random.uniform(_random_range[0], _random_range[1]))
                for _ in range(self.population_size)]

    def printable_identifier(self):
        return "ValueUniformPopulation(n={})".format(self.population_size)
