from typing import List

import numpy as np

from evolution.base.base_geometry import BaseGeometry


class ObjGeometry(BaseGeometry):
    def __init__(self, obj_filename: str) -> None:
        self._lines_list = []
        self._vertex_positions = np.zeros((0, 3))
        self.load_data(obj_filename)
        super().__init__()

    def load_data(self, obj_filename):
        with open(obj_filename, 'r') as file:
            data = file.read().splitlines()

        parse_line_fn = {
            "v ": self._parse_vertex,
            "l ": self._parse_line,
            "# ": lambda x: x
        }
        for line in data:
            if len(line) > 2:
                if line[:2] in parse_line_fn:
                    parse_line_fn[line[:2]](line[2:])
                else:
                    print("Don't know how to parse '{}'".format(line[0]))

    def _parse_vertex(self, vertex_line: str):
        values = [float(e) for e in vertex_line.split(" ")]
        self._vertex_positions = np.vstack((self._vertex_positions, values))

    def _parse_line(self, line_line: str):
        # subtract 1 because obj starts counting with 1 and python with 0
        values = [int(e) - 1 for e in line_line.split(" ")]
        self._lines_list.append(values)

    def provide_world_points(self) -> np.array:
        return self._vertex_positions

    def provide_connection_list(self) -> List[List[int]]:
        return self._lines_list


if __name__ == '__main__':
    obj_geometry = ObjGeometry("data/synth/squash_court.obj")
    print(f"  pts: {obj_geometry.world_points}")
    print(f"lines: {obj_geometry.connections}")
