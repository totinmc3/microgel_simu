import numpy as np
from numpy import linalg as LA
import itertools
import math

from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M

class Microgel:
    def __init__(self, system, FENE_BOND_PARAMS, Nbeads_arm, cell_unit):
        self.system = system
        self.Nbeads_arm = Nbeads_arm
        self.cell_unit = cell_unit
        self.FENE_BOND_PARAMS = FENE_BOND_PARAMS

        self.fene = FeneBond(**self.FENE_BOND_PARAMS)
        self.system.bonded_inter.add(self.fene)

        self.bead_separation = self.cell_unit/(4*(self.Nbeads_arm+1))
        self.equal_criterion = 0.001 * self.bead_separation

    def __repr__(self) -> str:
        return f'Microgel(system, {self.Nbeads_arm})'

    def __str__(self) -> str:
        return f'Microgel created:\n\t\tNbead_edge = {self.Nbeads_arm}'

    def __unit_cell(self, a, shift, bead_type, id_num):
        """
            locate the crosslinker beads in a single unit cell

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
                        [a/2, 0, a/2],              # 5
                        [0, a/2, a/2],              # 6
                        [a/2, a, a/2],              # 7
                        [a, a/2, a/2],              # 8
                        [a/4, a-a/4, a-a/4],        # 9
                        [a-a/4, a/4, a-a/4],        # 10
                        [0, a, a],                  # 11
                        [a/2, a/2, a],              # 12
                        [a, 0, a]]                  # 13

        for vec_pos in vec_pos_list:
            self.system.part.add(id=id_num, pos=np.array(vec_pos)+np.array(shift), type=bead_type)
            id_crosslinks_in_cell.append(id_num)
            id_num += 1
        return id_num, id_crosslinks_in_cell

    def __arms_unit_cell(self, id_crosslinks_in_cell, Nbeads_arm, id_num):
        """
            locate the arm polymer beads between the crosslinker beads given in id_crosslinks_in_cell
            for a single unit cell

            id_crosslinks_in_cell: id of crosslinker beads in an unit cell
            Nbeads_arm: number of beads per arm between crosslinkers
            id_num: bead id number
        """

        crosslinks_pairs = [[id_crosslinks_in_cell[0], id_crosslinks_in_cell[3]],
                            [id_crosslinks_in_cell[3], id_crosslinks_in_cell[1]],
                            [id_crosslinks_in_cell[1], id_crosslinks_in_cell[4]],
                            [id_crosslinks_in_cell[4], id_crosslinks_in_cell[2]],
                            [id_crosslinks_in_cell[3], id_crosslinks_in_cell[5]],
                            [id_crosslinks_in_cell[3], id_crosslinks_in_cell[6]],
                            [id_crosslinks_in_cell[4], id_crosslinks_in_cell[8]],
                            [id_crosslinks_in_cell[4], id_crosslinks_in_cell[7]],
                            [id_crosslinks_in_cell[5], id_crosslinks_in_cell[10]],
                            [id_crosslinks_in_cell[6], id_crosslinks_in_cell[9]],
                            [id_crosslinks_in_cell[7], id_crosslinks_in_cell[9]],
                            [id_crosslinks_in_cell[8], id_crosslinks_in_cell[10]],
                            [id_crosslinks_in_cell[10], id_crosslinks_in_cell[13]],
                            [id_crosslinks_in_cell[10], id_crosslinks_in_cell[12]],
                            [id_crosslinks_in_cell[9], id_crosslinks_in_cell[12]],
                            [id_crosslinks_in_cell[9], id_crosslinks_in_cell[11]]]

        for pairs in crosslinks_pairs:
            diff_vec = self.system.part[pairs[1]].pos - self.system.part[pairs[0]].pos

            iter_init = 1
            iter_end = Nbeads_arm+1
            for i in range(iter_init,iter_end):
                vec_pos = self.system.part[pairs[0]].pos + diff_vec * i / (Nbeads_arm + 1)
                self.system.part.add(id=id_num, pos=vec_pos, type=10)
                id_num += 1

        return id_num

    def __remove_double_particles(self):
        id_list = self.system.part[:].id
        # print(id_list)
        repeated_part_list = []

        for i,j in itertools.combinations(id_list, 2):
            if LA.norm(self.system.part[i].pos-self.system.part[j].pos) < self.equal_criterion:
                repeated_part_list.append(i)
        self.system.part[repeated_part_list].remove()


    def initialize_diamondLattice(self):
        a = self.cell_unit
        id_num = 0
        id_crosslinks_matrix = []

        # shift_list = [[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]]
        shift_list = [[0, 0, 0], [a, 0, 0], [0, a, 0]]#, [2*a, 0, 0], [a, a, 0], [2*a, a, 0]]
        for i,shift in enumerate(shift_list):
            id_num, id_crosslinks_in_cell = self.__unit_cell(a, shift, i, id_num)
            id_crosslinks_matrix.append(id_crosslinks_in_cell)
        
        for id_crosslinks_in_cell in id_crosslinks_matrix:
            id_num = self.__arms_unit_cell(id_crosslinks_in_cell, self.Nbeads_arm, id_num)

        print(len(self.system.part[:].id))
        self.__remove_double_particles()
        print(len(self.system.part[:].id))