import numpy as np
from typing import Optional


class BaseGenome:
    def __init__(self, dna: np.array,
                 display_name: Optional[str] = None) -> None:
        super().__init__()
        self._dna: np.array = dna
        self._display_name: str = display_name

    def to_string(self):
        return " / ".join(map(str, self.dna))

    @property
    def dna(self):
        return self._dna

    @dna.setter
    def dna(self, dna: np.array):
        self._dna = dna

    def __lt__(self, other):
        return 0

    def __len__(self):
        return len(self._dna)

    def __copy__(self):
        return type(self)(self._dna.copy(), self._display_name)

    def __str__(self):
        result = "Genome"
        if self._display_name:
            result += " [" + self._display_name + "]"
        return result + " (" + self.to_string() + ")"
