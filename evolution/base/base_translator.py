from abc import ABC as AbstractBaseClass, abstractmethod
from evolution.base.base_genome import BaseGenome


class BaseTranslator(AbstractBaseClass):
    @abstractmethod
    def translate_genome(self, genome: BaseGenome, *args, **kwargs):
        """
        Translates a given genome to understandable variables.

        As a genome is made of genes, which is basically a float array a method is needed to transform
        the genome to another representation.
        :param genome: The translatable geome
        :param args:
        :param kwargs:
        :return:
        """
        raise NotImplementedError
