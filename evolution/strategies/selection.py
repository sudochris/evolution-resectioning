from random import choices
from typing import List, Tuple
import numpy as np

from evolution.base.base_genome import BaseGenome
from evolution.base.base_strategies import SelectionStrategy, Population


class RouletteWheel(SelectionStrategy):
    def select(self, population: Population, population_fitness: List[float]) -> Tuple[BaseGenome, BaseGenome]:
        pf = np.array(population_fitness)
        pf = (pf - np.min(pf))
        return choices(population, weights=pf, k=2)

    def printable_identifier(self):
        return "RouletteWheel"


class Tournament(SelectionStrategy):

    def __init__(self, tournament_size, p=0.5) -> None:
        super().__init__()
        self._k = tournament_size
        a = np.arange(tournament_size)
        self.probabilities = p * ((1-p)**a)

    def select(self, population: Population, population_fitness: List[float]) -> Tuple[BaseGenome, BaseGenome]:
        tournament = sorted(np.random.choice(len(population), self._k, replace=False))
        first, second = choices(tournament, weights=self.probabilities, k=2)
        return population[first], population[second]

    def printable_identifier(self):
        return "Tournament(k={})".format(self._k)


class Random(Tournament):
    def __init__(self) -> None:
        """
        A Random tournament is basically a Tournament of size 1
        """
        super().__init__(1, 1)

    def printable_identifier(self):
        return "Random"
