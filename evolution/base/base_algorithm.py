from abc import ABC, abstractmethod
from typing import List

import numpy as np

from evolution.base.base_genome import BaseGenome
from evolution.base.base_genome_factory import BaseGenomeFactory
from evolution.base.base_result import BaseResult
from evolution.base.base_strategies import PopulateStrategy, SelectionStrategy, CrossoverStrategy, MutationStrategy, \
    TerminationStrategy, Population
from evolution.base.base_translator import BaseTranslator


class BaseAlgorithm(ABC):
    def __init__(self,
                 translator: BaseTranslator,
                 genome_factory: BaseGenomeFactory,
                 populate_strategy: PopulateStrategy,
                 selection_strategy: SelectionStrategy,
                 crossover_strategy: CrossoverStrategy,
                 mutation_strategy: MutationStrategy,
                 termination_strategy: TerminationStrategy,
                 print_info: bool = False) -> None:
        """
        Instantiates a new algorithm with a given translator and genome factory.
        The translator will be used to transform the raw genome data to meaningful variables.
        :param translator: The translator for transforming dna to meaningful variables
        :param genome_factory: A factory for creating genomes
        """
        super().__init__()

        self.populate_strategy = populate_strategy
        self.selection_strategy = selection_strategy
        self.crossover_strategy = crossover_strategy
        self.mutation_strategy = mutation_strategy
        self.termination_strategy = termination_strategy

        self.translator = translator
        self.genome_factory = genome_factory
        self._best_fitness = -np.inf

        self.print_info = print_info

    @abstractmethod
    def fitness(self, genome: BaseGenome) -> float:
        """
        Calculates the fitness for a given genome.
        Warning: May be an expensive in terms of resources and / or computation time!
        """
        raise NotImplemented

    def run(self, start_dna: np.array) -> BaseResult:
        """
        Stars and runs the algorithm. Calls all installed callbacks.
        :return:
        """
        population = self.populate_strategy.populate(self.genome_factory, start_dna)

        current_generation = 0

        result = BaseResult()
        while not self.termination_strategy.should_terminate(current_generation, self._best_fitness):
            if self.print_info:
                print("Running generation No.{:4}".format(current_generation))

            population_fitness = [self.fitness(genome) for genome in population]
            population_fitness, population = (list(t) for t in
                                              zip(*sorted(zip(population_fitness, population), reverse=True)))

            current_best_fitness = population_fitness[0]

            result.add_generation(current_generation, population[0], population_fitness[0], population[0],
                                  population_fitness[0])

            if current_best_fitness > self._best_fitness:
                self._best_fitness = current_best_fitness
                self.on_best_genome_found(population[0], population_fitness[0])

            self.on_display_population(current_generation, population, population_fitness)

            next_generation = population[:2]

            for j in range((len(population) // 2) - 1):
                parent_a, parent_b = self.selection_strategy.select(population, population_fitness)
                offspring_a, offspring_b = self.crossover_strategy.crossover(self.genome_factory, parent_a, parent_b)

                self.mutation_strategy.mutate(self.genome_factory, offspring_a)
                self.mutation_strategy.mutate(self.genome_factory, offspring_b)

                next_generation += [offspring_a, offspring_b]

            population = next_generation
            current_generation += 1

        return result

    # ####################### Callbacks ########################

    def on_display_population(self, current_generation: int, population: Population, population_fitness: List[float]):
        """
        This callback is called at the beginning of every new generation after calculating the populations fitness
        values.

        Note that the population is sorted w.r.t. the populations fitness. The fitness value the ith element in
        population can be found at the ith location in population_fitness.

        :param current_generation: Current number of generation / iteration
        :param population: Fresh population, sorted w.r.t. the fitness value
        :param population_fitness: List of the fitness values for the population
        :return:
        """
        pass

    def on_best_genome_found(self, genome: BaseGenome, genome_fitness: float):
        """
        This callback is called if a new best genome was found.

        :param genome: The new best genome / solution for the problem
        :param genome_fitness: The fitness
        :return:
        """
        pass
