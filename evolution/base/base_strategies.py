import numpy as np
from abc import ABC, abstractmethod
from typing import List, Tuple

from evolution.base.base_genome import BaseGenome
from evolution.base.base_genome_factory import BaseGenomeFactory

Population = List[BaseGenome]


class Strategy(ABC):
    @abstractmethod
    def printable_identifier(self):
        raise NotImplementedError


class PopulateStrategy(Strategy):
    @abstractmethod
    def populate(self, genome_factory: BaseGenomeFactory, start_dna: np.array):
        raise NotImplementedError


class FitnessStrategy(Strategy):
    @abstractmethod
    def create_fitness(self, edge_image: np.array) -> np.array:
        raise NotImplementedError


class SelectionStrategy(Strategy):
    @abstractmethod
    def select(self, population: Population, population_fitness: List[float]) -> Tuple[BaseGenome, BaseGenome]:
        """
        Select a new population subset from a given population.
        Indices in population and fitness_lookup match, so that no separate call to fitness is needed.
        :param population_fitness:
        :param population: A given population to select from
        :return: The new population
        """
        raise NotImplementedError


class CrossoverStrategy(Strategy):
    @abstractmethod
    def crossover(self,
                  genome_factory: BaseGenomeFactory,
                  genome_a: BaseGenome,
                  genome_b: BaseGenome) -> Tuple[BaseGenome, BaseGenome]:
        raise NotImplementedError


class MutationStrategy(Strategy):
    @abstractmethod
    def mutate(self, genome_factory: BaseGenomeFactory, genome: BaseGenome) -> None:
        """ Mutates the genome's dna in-place

        First selects mutation_values from the specified mutation interval for every dna entry.
        Then creates an array, based on the mutation probability, which specifies whether a
        single dna element should mutate. Then every offset which should not mutate is set to 0.
        This ensures that the dna element is offset by 0, so no mutation happens. Finally, the
        mutation_values are added in place to the dna.
        If bounds are specified, the dna is clipped to their bounds!
        """
        raise NotImplementedError


class TerminationStrategy(Strategy):
    @abstractmethod
    def should_terminate(self, current_generation: int, best_fitness: float) -> bool:
        raise NotImplementedError
