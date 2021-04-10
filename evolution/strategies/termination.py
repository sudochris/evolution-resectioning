import numpy as np

from evolution.base.base_strategies import TerminationStrategy


class MaxIteration(TerminationStrategy):
    def __init__(self, max_generations: int) -> None:
        super().__init__()
        self._max_generations = max_generations

    def should_terminate(self, current_generation: int, best_fitness: float) -> bool:
        return current_generation >= self._max_generations

    def printable_identifier(self):
        return "MaxIteration(n={})".format(self._max_generations)


class Or(TerminationStrategy):

    def __init__(self, *strategies) -> None:
        super().__init__()
        self._strategies = strategies


    def should_terminate(self, current_generation: int, best_fitness: float) -> bool:
        for strategy in self._strategies:
            if strategy.should_terminate(current_generation, best_fitness):
                return True
        return False

    def printable_identifier(self):
        pi = "|".join([s.printable_identifier() for s in self._strategies])
        return "[" + pi + "]"


class And(TerminationStrategy):
    def __init__(self, *strategies) -> None:
        super().__init__()
        self._strategies = strategies

    def should_terminate(self, current_generation: int, best_fitness: float) -> bool:
        for strategy in self._strategies:
            if not strategy.should_terminate(current_generation, best_fitness):
                return False
        return True

    def printable_identifier(self):
        pi = "&".join([s.printable_identifier() for s in self._strategies])
        return "[" + pi + "]"


class NoImprovement(TerminationStrategy):
    def __init__(self, n_generations_without_improvement: int) -> None:
        super().__init__()
        self._best_fitness = -np.inf
        self._n_generations_without_improvement = n_generations_without_improvement
        self._counter = 0

    def should_terminate(self, current_generation: int, best_fitness: float) -> bool:
        if best_fitness == self._best_fitness:
            self._counter += 1
        else:
            self._best_fitness = best_fitness
            self._counter = 0

        return self._counter >= self._n_generations_without_improvement

    def printable_identifier(self):
        return "NoImprovement(n={})".format(self._n_generations_without_improvement)


class FitnessReached(TerminationStrategy):
    def __init__(self, needed_fitness: float) -> None:
        super().__init__()
        self._needed_fitness = needed_fitness

    def should_terminate(self, current_generation: int, best_fitness: float) -> bool:
        return best_fitness >= self._needed_fitness

    def printable_identifier(self):
        return "FitnessReached(n={})".format(self._needed_fitness)
