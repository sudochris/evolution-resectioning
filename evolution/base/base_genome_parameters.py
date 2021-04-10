import json as json
from functools import cached_property
from typing import Optional

import numpy as np


class BaseGenomeParameters:
    def __init__(self,
                 parameters_file: str,
                 default_display_name: Optional[str] = None) -> None:
        super().__init__()
        with open(parameters_file, 'r') as file:
            self._parameters = json.load(file)
        self._default_display_name = default_display_name

    @property
    def default_display_name(self):
        return self._default_display_name

    @cached_property
    def n_genes(self):
        return len(self._parameters["dna"])

    @cached_property
    def mutation_table(self):
        dna_parameters = self._parameters["dna"]
        dna_mutation = [g["mutation"] for g in dna_parameters]
        return np.array([[m["low"], m["high"], m["probability"]] for m in dna_mutation]).T

    @cached_property
    def distributions(self):
        dna_parameters = self._parameters["dna"]
        dna_mutation = [g["mutation"] for g in dna_parameters]
        distributions = []
        for m in dna_mutation:
            if "distribution_parameters" in m:
                distributions.append({m["distribution"]: m["distribution_parameters"]})
            else:
                distributions.append({m["distribution"]: {"low": m["low"], "high": m["high"]}})

        return distributions

    @cached_property
    def genome_bounds(self):
        dna_parameters = self._parameters["dna"]
        dna_bounds = [g["bounds"] for g in dna_parameters]
        return np.array([[m["low"], m["high"]] for m in dna_bounds]).T
