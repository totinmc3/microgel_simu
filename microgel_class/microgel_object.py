import numpy as np
from numpy import linalg as LA
import itertools
import math
import random

from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M

class Microgel:
    def __init__(self, system, FENE_BOND_PARAMS, PART_TYPE, NONBOND_WCA_PARAMS, Nbeads_arm, cell_unit, N_cat, N_an):
        self.system = system
        self.Nbeads_arm = Nbeads_arm
        self.cell_unit = cell_unit
        self.FENE_BOND_PARAMS = FENE_BOND_PARAMS
        self.PART_TYPE = PART_TYPE
        self.NONBOND_WCA_PARAMS = NONBOND_WCA_PARAMS
        self.N_cat = N_cat
        self.N_an = N_an

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


    def __arms_unit_cell(self, id_crosslinks_in_cell, id_num):
        """
            locate the arm polymer beads between the crosslinker beads given in id_crosslinks_in_cell
            for a single unit cell

            id_crosslinks_in_cell: id of crosslinker beads in an unit cell
            Nbeads_arm: number of beads per arm between crosslinkers
            id_num: bead id number
        """

        diff_mod = self.Nbeads_arm + 1 # minimal crosslinker-crosslinker distance in units of sigma

        for i,j in itertools.combinations(id_crosslinks_in_cell, 2):
            diff_vec = self.system.part[i].pos-self.system.part[j].pos
            if LA.norm(diff_vec) < 1.01 * diff_mod:
                iter_init = 1
                iter_end = self.Nbeads_arm+1
                for l in range(iter_init,iter_end):
                    vec_pos = self.system.part[j].pos + diff_vec * l / (self.Nbeads_arm + 1)
                    self.system.part.add(id=id_num, pos=vec_pos, type=self.PART_TYPE['polymer_arm'])
                    id_num += 1

        return id_num


    def __remove_outterCrosslinker(self, radius, sphere_center, id_crosslinks_matrix):
        """
            The function removes the crosslinker beads that are located further than @radius form the network centre
            
            radius: critical radius
            sphere_center:  center of the critical sphere
            id_crosslinks_matrix: list with the id's of the crosslinkers
        """
        new_id_crosslinks_matrix = []
        sphere_center = np.array(sphere_center) # in units of a
        for id_crosslinks_list in id_crosslinks_matrix:
            iter_list = id_crosslinks_list
            for i in iter_list:
                if LA.norm(self.system.part[i].pos-sphere_center) > radius:
                    self.system.part[i].remove()
                    id_crosslinks_list.remove(i)
            new_id_crosslinks_matrix.append(id_crosslinks_list)
        return new_id_crosslinks_matrix


    def __remove_double_particles(self):
        """ For overlapping beads, it keeps only one and removes the rest """
        id_list = self.system.part[:].id
        # print(id_list)
        repeated_part_list = []

        for i,j in itertools.combinations(id_list, 2):
            if LA.norm(self.system.part[i].pos-self.system.part[j].pos) < self.equal_criterion:
                if i not in repeated_part_list:
                    repeated_part_list.append(i) 
        self.system.part[repeated_part_list].remove()
    
    # def __remove_deadendCrosslinker(self):
    #     for part in self.system.part[:]:
    #         if part.type  == self.PART_TYPE['crosslinker'] and len(part.bond)

    def __remove_deadendCrosslinker(self, id_crosslinks_matrix):
        """
            Remove deadend crosslinkers and returns an updated id-crosslinked list
        """

        new_id_crosslinks_matrix = []
        for id_crosslinks_list in id_crosslinks_matrix:
            iter_list = id_crosslinks_list
            for i in iter_list:
                if len(self.system.analysis.nbhood(pos=self.system.part[i].pos, r_catch=1.1))<3:
                    self.system.part[i].remove()
                    id_crosslinks_list.remove(i)
            new_id_crosslinks_matrix.append(id_crosslinks_list)

        return new_id_crosslinks_matrix


    def initialize_diamondLattice(self):
        """
            The function initializes a diamond lattice. It returns the number of crosslinkers of
            the network and the number of monomers
        '"""
        a = self.cell_unit
        id_num = 0
        id_crosslinks_matrix = [] # list containeng id-lists of crosslinkers of each unit cell

        # shift_list = [[0, 0, 0], [a, 0, 0], [0, a, 0], [0, 0, a]]
        center_shift = self.system.box_l[0] / 2 - a
        shift_list = [[0 + center_shift, 0 + center_shift, 0 + center_shift], 
                      [a + center_shift, 0 + center_shift, 0 + center_shift], 
                      [0 + center_shift, a + center_shift, 0 + center_shift], 
                      [a + center_shift, a + center_shift, 0 + center_shift], 
                      [0 + center_shift, 0 + center_shift, a + center_shift], 
                      [a + center_shift, 0 + center_shift, a + center_shift], 
                      [0 + center_shift, a + center_shift, a + center_shift], 
                      [a + center_shift, a + center_shift, a + center_shift]]#, [2*a, 0, 0], [a, a, 0], [2*a, a, 0]]

        for i,shift in enumerate(shift_list):
            id_num, id_crosslinks_in_cell = self.__unit_cell(a, shift, i, id_num)
            id_crosslinks_matrix.append(id_crosslinks_in_cell)
        
        # remove crosslinkers that are further than radius from the box centre
        sphere_center = self.system.box_l / 2
        radius = 1.7*a
        id_crosslinks_matrix = self.__remove_outterCrosslinker(radius, sphere_center, id_crosslinks_matrix)
        
        for id_crosslinks_in_cell in id_crosslinks_matrix:
            id_num = self.__arms_unit_cell(id_crosslinks_in_cell, id_num)

        id_crosslinks_matrix = self.__remove_deadendCrosslinker(id_crosslinks_matrix)
        self.__remove_double_particles()

        #------------- Reload particles for continuous id list -------------
        crosslinker_pos_list = []
        arm_pos_list = []
        for part in self.system.part[:]:
            if part.type == self.PART_TYPE['crosslinker']:
                crosslinker_pos_list.append(part.pos)
            else:
                arm_pos_list.append(part.pos)
        # print(f"# crosslinkers = {len(crosslinker_pos_list)}")
        # print(f"# arm beads = {len(arm_pos_list)}")
        self.system.part[:].remove()
        # print(f"###### # of particles  = {len(self.system.part[:])}")
        self.system.part.add(pos=crosslinker_pos_list, type=[self.PART_TYPE['crosslinker']]*len(crosslinker_pos_list))
        self.system.part.add(pos=arm_pos_list, type=[self.PART_TYPE['polymer_arm']]*len(arm_pos_list))
        # print(self.system.part[:].id)

        return len(crosslinker_pos_list), len(arm_pos_list)



    def initialize_bonds(self):
        fene = FeneBond(**self.FENE_BOND_PARAMS)
        self.system.bonded_inter.add(fene)
        
        print(f'bonding_criteria = {self.bonding_criteria}')
        
        for i,j in itertools.combinations(self.system.part[:].id, 2):
            if LA.norm(self.system.part[i].pos-self.system.part[j].pos) < self.bonding_criteria:
                self.system.part[i].add_bond((fene, j))

        # for part_pos in self.system.part.pairs():
        #     id_list = self.system.analysis.nbhood(pos=part_pos, r_catch=self.bonding_criteria)


    def initialize_internoelec(self):

        print("Define interactions (non electrostatic)")
        # Non-bonded Interactions:
        for i,j in itertools.combinations_with_replacement([x for x in self.PART_TYPE], 2):
            self.system.non_bonded_inter[self.PART_TYPE[i], self.PART_TYPE[j]].wca.set_params(**self.NONBOND_WCA_PARAMS)


    def __insert_ions(self, N_ions, particle_type, particle_charge):
        self.system.part.add(pos=np.random.random((N_ions, 3)) * self.system.box_l, type=[particle_type] * N_ions, q=[particle_charge] * N_ions)
        print(f" # of particles = {len(self.system.part[:])}")


    def charge_beads_homo(self):
        """
            This function picks randomly beads from the particle list and charge them negatively with valence q = 1. It also adds
            the corresponding counterions, for which wca interection is also set.
        
        """
        print('Charge microgel homogeneously')
        part_rdm_list = []
        j = 0
        while j < self.N_an:
            rdm_num = random.randint(0,len(self.system.part[:])-1)
            if rdm_num not in part_rdm_list:
                part_rdm_list.append(rdm_num)
                j += 1
        # for i,part in enumerate(self.system.part[:]):
        #     print(part.id)
        #     if part.q != 0:
        #         print(f"prematurely charged particle : {part.id}")
        #     if part.id in part_rdm_list:
        #         part.q = -1
        #         part.type = self.PART_TYPE['anion']
        #         print(f'total charge = {sum(self.system.part[:].q)}   {part.id}')
        self.system.part[part_rdm_list].q = [-1] * self.N_an
        self.system.part[part_rdm_list].type = [self.PART_TYPE['anion']] * self.N_an
        self.__insert_ions(self.N_an, self.PART_TYPE["ion_cat"], +1)
        assert abs(sum(self.system.part[:].q)) < 1e-10


        
