import numpy as np


class BaseResult:
    def __init__(self) -> None:
        super().__init__()
        self._best_fitnesses = []
        self._best_genome = None
        self._best_fitness = -np.inf

    def add_generation(self, generation_num, mean_genome, mean_fitness, best_genome, best_fitness):
        self._best_fitnesses.append(best_fitness)

        if best_fitness > self._best_fitness:
            self._best_genome = best_genome
            self._best_fitness = best_fitness

    @property
    def best_genome(self):
        return self._best_genome, self._best_fitness

    @property
    def best_fitnesses(self):
        return self._best_fitnesses

    @property
    def n_generations(self):
        return len(self._best_fitnesses)
