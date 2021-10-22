import numpy as np
import math

from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M

class Microgel:
    def __init__(self,system,Nbeads_arm,cell_unit):
        self.system = system
        self.Nbeads_arm = Nbeads_arm
        self.cell_unit = cell_unit

    def __repr__(self) -> str:
        return f'Microgel(system, {self.Nbeads_arm})'

    def __str__(self) -> str:
        return f'Microgel created:\n\t\tNbead_edge = {self.Nbeads_arm}'

    def __unit_cell(self, a, shift, bead_type, id_num):
        """
            a: lattice unit distance (float)
            shift: shift vector for the cell (list)
            bead_type: type of bead
            id_num: bead id number
        """
        id_crosslinks_in_cell = []

        vec_pos_list = [[0, 0, 0],                  # 0
                        [a/2, a/2, 0],              # 1
                        [a, a, 0],                  # 2
                        [a/4, a/4, a/4],            # 3
                        [a/4+a/2, a/4+a/2, a/4],    # 4
                        [0, a/2, a/2],              # 5
                        [a/2, 0, a/2],              # 6
                        # [a/2, a, a/2],
                        # [a, a/2, a/2],
                        [a/4, a-a/4, a-a/4],        # 7
                        [a-a/4, a/4, a-a/4],        # 8
                        [0, a, a],                  # 9
                        # [a/2, a/2, a],
                        [a, 0, a]]                  # 10

        # vec_pos_list = [[0,0,0], [0,2,2], [2,0,2], [2,2,0],[3,3,3], [3,1,1], [1,3,1], [1,1,3]]

        for vec_pos in vec_pos_list:
            self.system.part.add(id=id_num, pos=np.array(vec_pos)+np.array(shift), type=bead_type)
            id_crosslinks_in_cell.append(id_num)
            id_num += 1
        return id_num, id_crosslinks_in_cell

    def __arms_unit_cell(self, id_crosslinks_in_cell, Nbeads_arm, id_num):
        """
            locate the polymer arm beads between the crosslinker bead given in id_crosslinks_in_cell
            for a single unit cell

            id_crosslinks_in_cell: id of the crosslinker bead in an unit cell
            Nbeads_arm: number of beads per arm between crosslinkers
            id_num: bead id number
        """
        id_armbeads_in_cell = []

        crosslinks_pairs = [[id_crosslinks_in_cell[0], id_crosslinks_in_cell[3]],
                            [id_crosslinks_in_cell[3], id_crosslinks_in_cell[1]],
                            [id_crosslinks_in_cell[1], id_crosslinks_in_cell[4]],
                            [id_crosslinks_in_cell[4], id_crosslinks_in_cell[2]],
                            [id_crosslinks_in_cell[3], id_crosslinks_in_cell[6]],
                            [id_crosslinks_in_cell[3], id_crosslinks_in_cell[5]],
                            [id_crosslinks_in_cell[5], id_crosslinks_in_cell[7]],
                            [id_crosslinks_in_cell[6], id_crosslinks_in_cell[8]],
                            [id_crosslinks_in_cell[7], id_crosslinks_in_cell[9]],
                            [id_crosslinks_in_cell[8], id_crosslinks_in_cell[10]]]

        for pairs in crosslinks_pairs:
            diff_vec = self.system.part[pairs[1]].pos - self.system.part[pairs[0]].pos

            for i in range(1,Nbeads_arm+1):
                vec_pos = self.system.part[pairs[0]].pos + diff_vec * i / (Nbeads_arm + 1)
                self.system.part.add(id=id_num, pos=vec_pos, type=10)
                id_armbeads_in_cell.append(id_num)
                id_num += 1

        return id_num, id_armbeads_in_cell

    # def __arms_unit_intercell(self, id_crosslinks_in_cell, cell_id):

    def initialize_diamondLattice(self):
        a = self.cell_unit
        id_num = 0

        # shift_list = [[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]]
        shift_list = [[0, 0, 0]]#, [a, 0, 0], [2*a, 0, 0], [0, a, 0], [a, a, 0], [2*a, a, 0]]
        for i,shift in enumerate(shift_list):
            id_num, id_crosslinks_in_cell = self.__unit_cell(a, shift, i, id_num)
            id_num, id_armbeads_in_cell = self.__arms_unit_cell(id_crosslinks_in_cell, self.Nbeads_arm, id_num)
