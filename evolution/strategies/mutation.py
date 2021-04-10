import numpy as np

from evolution.base.base_genome import BaseGenome
from evolution.base.base_genome_factory import BaseGenomeFactory
from evolution.base.base_genome_parameters import BaseGenomeParameters
from evolution.base.base_strategies import MutationStrategy


class BoundedUniformMutation(MutationStrategy):
    def __init__(self, genome_parameters: BaseGenomeParameters) -> None:
        super().__init__()
        self.mutation_min, self.mutation_max, self.mutation_probability = genome_parameters.mutation_table
        self.genome_bounds = genome_parameters.genome_bounds

    def mutate(self, genome_factory: BaseGenomeFactory, genome: BaseGenome) -> None:
        mutation_values = np.random.uniform(self.mutation_min, self.mutation_max)
        mutation_selector = np.random.random_sample(len(genome)) <= self.mutation_probability
        mutation_values[~mutation_selector] = 0
        genome.dna += mutation_values
        if self.genome_bounds is not None:
            genome_factory.validate_bounds(genome, self.genome_bounds)

    def printable_identifier(self):
        return "BoundedUniformMutation"


class BoundedDistributionBasedMutation(MutationStrategy):
    def __init__(self, genome_parameters: BaseGenomeParameters) -> None:
        super().__init__()
        self.mutation_min, self.mutation_max, self.mutation_probability = genome_parameters.mutation_table
        self.distributions = genome_parameters.distributions
        self.genome_bounds = genome_parameters.genome_bounds

    def mutate(self, genome_factory: BaseGenomeFactory, genome: BaseGenome) -> None:
        mutation_selector = np.random.random_sample(len(genome)) <= self.mutation_probability
        mutation_values = np.zeros(len(genome))
        for idx, do_mutation in enumerate(mutation_selector):
            if do_mutation:
                if "uniform" in self.distributions[idx]:
                    mutation_values[idx] = np.random.uniform(self.distributions[idx]["uniform"]["low"],
                                                             self.distributions[idx]["uniform"]["high"])
                elif "normal" in self.distributions[idx]:
                    mutation_values[idx] = np.random.normal(self.distributions[idx]["normal"]["mu"],
                                                            self.distributions[idx]["normal"]["sigma"])
                elif "lognormal" in self.distributions[idx]:
                    mutation_values[idx] = np.random.normal(self.distributions[idx]["lognormal"]["mu"],
                                                            self.distributions[idx]["lognormal"]["sigma"]) \
                                           + self.distributions[idx]["lognormal"]["offset"]

        genome.dna += mutation_values
        if self.genome_bounds is not None:
            genome_factory.validate_bounds(genome, self.genome_bounds)

    def printable_identifier(self):
        return "BoundedDistributionBasedMutation"
