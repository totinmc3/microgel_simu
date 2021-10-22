import numpy as np
import math

from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M

class Microgel:
    def __init__(self,system,Nbead_edge,cell_unit):
        self.system = system
        self.Nbead_edge = Nbead_edge
        self.cell_unit = cell_unit

    def __repr__(self) -> str:
        return f'Microgel(system, {self.Nbead_edge})'

    def __str__(self) -> str:
        return f'Microgel created:\n\t\tNbead_edge = {self.Nbead_edge}'

    def __unit_cell(self, a, shift, bead_type):
        """
            a: lattice unit distance (float)
            shift: shift vector for the cell (list)
        """
        vec_pos_list = [[0, 0, 0],
                        [a/2, a/2, 0],
                        [a, a, 0],
                        [a/4, a/4, a/4],
                        [a/4+a/2, a/4+a/2, a/4],
                        [0, a/2, a/2],
                        [a/2, 0, a/2],
                        # [a/2, a, a/2],
                        # [a, a/2, a/2],
                        [a/4, a-a/4, a-a/4],
                        [a-a/4, a/4, a-a/4],
                        [0, a, a],
                        # [a/2, a/2, a],
                        [a, 0, a]]

        vec_pos_list = [[0,0,0], [0,2,2], [2,0,2], [2,2,0],[3,3,3], [3,1,1], [1,3,1], [1,1,3]]

        for vec_pos in vec_pos_list:
            self.system.part.add(pos=np.array(vec_pos)+np.array(shift), type=bead_type)


    def initialize_diamondLattice(self):
        a = self.cell_unit

        # shift_list = [[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]]
        shift_list = [[0, 0, 0], [a, 0, 0], [2*a, 0, 0], [0, a, 0], [a, a, 0], [2*a, a, 0]]
        for i,shift in enumerate(shift_list):
            self.__unit_cell(a, shift,i)
