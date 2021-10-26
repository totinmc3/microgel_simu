import numpy as np
from numpy import linalg as LA
import itertools
import math

from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M

class Microgel:
    def __init__(self, system, FENE_BOND_PARAMS, PART_TYPE, NONBOND_WCA_PARAMS, Nbeads_arm, cell_unit):
        self.system = system
        self.Nbeads_arm = Nbeads_arm
        self.cell_unit = cell_unit
        self.FENE_BOND_PARAMS = FENE_BOND_PARAMS
        self.PART_TYPE = PART_TYPE
        self.NONBOND_WCA_PARAMS = NONBOND_WCA_PARAMS

        self.bead_separation = np.sqrt(3)*self.cell_unit/(4*(self.Nbeads_arm+1))
        self.equal_criterion = 0.001 * self.bead_separation
        self.bonding_criteria = self.bead_separation*1.3

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
            self.system.part.add(id=id_num, pos=np.array(vec_pos)+np.array(shift), type=self.PART_TYPE['crosslinker'])
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
                self.system.part.add(id=id_num, pos=vec_pos, type=self.PART_TYPE['polymer_arm'])
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

    def initialize_bonds(self):
        fene = FeneBond(**self.FENE_BOND_PARAMS)
        self.system.bonded_inter.add(fene)
        
        print(f'bonding_criteria = {self.bonding_criteria}')
        
        for i,j in itertools.combinations(self.system.part[:].id, 2):
            # print([i,j])
            print(LA.norm(self.system.part[i].pos-self.system.part[j].pos))
            if LA.norm(self.system.part[i].pos-self.system.part[j].pos) < self.bonding_criteria:
                # print("criteria satisfied")
                self.system.part[i].add_bond((fene, j))
        # print(self.system.part[:].bonds)

        # for part_pos in self.system.part.pairs():
        #     id_list = self.system.analysis.nbhood(pos=part_pos, r_catch=self.bonding_criteria)
        #     if 

    def initialize_internoelec(self,system):

        print("Define interactions (no electrostatic)")
        # Non-bonded Interactions:
        system.non_bonded_inter[self.PART_TYPE['polymer_arm'], self.PART_TYPE['polymer_arm']].wca.set_params(**self.NONBOND_WCA_PARAMS)
        system.non_bonded_inter[self.PART_TYPE['crosslinker'], self.PART_TYPE['crosslinker']].wca.set_params(**self.NONBOND_WCA_PARAMS)
        system.non_bonded_inter[self.PART_TYPE['polymer_arm'], self.PART_TYPE['crosslinker']].wca.set_params(**self.NONBOND_WCA_PARAMS)