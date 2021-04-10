from typing import Tuple

import numpy as np

from evolution.base.base_genome import BaseGenome
from evolution.base.base_genome_factory import BaseGenomeFactory
from evolution.base.base_strategies import CrossoverStrategy


class Uniform(CrossoverStrategy):

    def __init__(self, crossover_probabilties: np.array, identifier_suffix="") -> None:
        super().__init__()
        self._crossover_probabilties = crossover_probabilties
        self.identifier_suffix = identifier_suffix

    def crossover(self,
                  genome_factory: BaseGenomeFactory,
                  genome_a: BaseGenome,
                  genome_b: BaseGenome) -> Tuple[BaseGenome, BaseGenome]:
        """
        Parent A    : XXXXXXXX
        Parent B    : --------
        Offspring0  : XX--X--X
        Offspring1  : --XX-XX-
        Based on probabilities
        """
        s = np.random.random_sample(genome_a.dna.size) > self._crossover_probabilties

        dna_child_a = np.array([dba_a if i else dba_b for (dba_a, dba_b, i) in zip(genome_a.dna, genome_b.dna, s)])
        dna_child_b = np.array([dba_b if i else dba_a for (dba_a, dba_b, i) in zip(genome_a.dna, genome_b.dna, s)])

        child_a = genome_factory.create(dna_child_a)
        child_b = genome_factory.create(dna_child_b)
        return child_a, child_b

    def printable_identifier(self):
        return f"Uniform{self.identifier_suffix}"


class SinglePoint(CrossoverStrategy):

    def crossover(self,
                  genome_factory: BaseGenomeFactory,
                  genome_a: BaseGenome,
                  genome_b: BaseGenome) -> Tuple[BaseGenome, BaseGenome]:
        """
        Parent A    : XXXXXXXX
        Parent B    : --------
        Offspring0  : XXX-----
        Offspring1  : ---XXXXX
        """
        point = np.random.randint(1, genome_a.dna.size)

        # dna_child_a = np.append(genome_a.dna[:point], genome_b.dna[point:])
        # dna_child_b = np.append(genome_b.dna[:point], genome_a.dna[point:])

        dna_child_a = np.concatenate((genome_a.dna[:point], genome_b.dna[point:]))
        dna_child_b = np.concatenate((genome_b.dna[:point], genome_a.dna[point:]))
        child_a = genome_factory.create(dna_child_a)
        child_b = genome_factory.create(dna_child_b)
        return child_a, child_b

    def printable_identifier(self):
        return "SinglePoint"


class TwoPoint(CrossoverStrategy):
    def crossover(self,
                  genome_factory: BaseGenomeFactory,
                  genome_a: BaseGenome,
                  genome_b: BaseGenome) -> Tuple[BaseGenome, BaseGenome]:
        """
        Parent A    : XXXXXXXX
        Parent B    : --------
        Offspring0  : XXX---XX
        Offspring1  : ---XXX--
        """
        point_1, point_2 = sorted(np.random.choice(genome_a.dna.size - 1, 2, replace=False) + 1)

        dna_child_a = np.concatenate((genome_a.dna[:point_1], genome_b.dna[point_1:point_2], genome_a.dna[point_2:]))
        dna_child_b = np.concatenate((genome_b.dna[:point_1], genome_a.dna[point_1:point_2], genome_b.dna[point_2:]))

        child_a = genome_factory.create(dna_child_a)
        child_b = genome_factory.create(dna_child_b)
        return child_a, child_b

    def printable_identifier(self):
        return "TwoPoint"
