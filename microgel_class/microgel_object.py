import numpy as np
from numpy import linalg as LA
import itertools
import math
import random
import sys


from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M
from system_parameters import N_an

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
        repeated_part_list = []

        for i,j in itertools.combinations(id_list, 2):
            if LA.norm(self.system.part[i].pos-self.system.part[j].pos) < self.equal_criterion:
                if i not in repeated_part_list:
                    repeated_part_list.append(i) 
        self.system.part[repeated_part_list].remove()
    
    
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

        cell_repeat = 2 # number of diamond lattice cells per axis
        remove_ref = {2: 1.7, 3: 1.9} # criterium radius for bead removal depending on cell_repeat

        center_shift = 0.95*self.system.box_l[0] / 2 - a * cell_repeat / 2

        shift_list = []
        for i,j,k in itertools.product([x for x in range(cell_repeat)], repeat=3):
            shift_list.append([i*a + center_shift, j*a + center_shift, k*a + center_shift])

        for i,shift in enumerate(shift_list):
            id_num, id_crosslinks_in_cell = self.__unit_cell(a, shift, i, id_num)
            id_crosslinks_matrix.append(id_crosslinks_in_cell)
        
        # remove crosslinkers that are further than radius from the box centre
        sphere_center = self.system.box_l / 2
        radius = remove_ref[cell_repeat]*a
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
        self.system.part[:].remove()
        self.system.part.add(pos=crosslinker_pos_list, type=[self.PART_TYPE['crosslinker']]*len(crosslinker_pos_list))
        self.system.part.add(pos=arm_pos_list, type=[self.PART_TYPE['polymer_arm']]*len(arm_pos_list))
        
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
            the corresponding counterions, for which wca interaction is also set.
        
        """
        print('Charge microgel homogeneously')
        part_rdm_list = random.sample(range(len(self.system.part[:])-1), self.N_an)
        self.system.part[part_rdm_list].q = [-1] * self.N_an
        self.system.part[part_rdm_list].type = [self.PART_TYPE['anion']] * self.N_an
        self.__insert_ions(self.N_an, self.PART_TYPE["ion_cat"], +1)
        assert abs(sum(self.system.part[:].q)) < 1e-10

    def charge_beads_shell(self):
        """
            This function picks randomly beads from the particle list for beads with a distance larged than internal radius b from the
            network COM and charge them negatively with valence q such that the total charged is Z_an. It also adds
            the corresponding monovalent counterions, for which wca interaction is also set.
        
        """

        print('Charge microgel shell')

        b = 1.1 * 3*self.cell_unit/4
        if b > self.cell_unit:
            sys.exit("Error: Internal radius b larger than unit cell size a_cell (for 2x2x2 unit cell gel): take b <= a_cell")
        com = self.system.analysis.center_of_mass(p_type=self.PART_TYPE['polymer_arm'])
        dist_to_com = LA.norm( self.system.part[:].pos - com, axis=1)
        index = np.array([x for x in range(len(self.system.part[:]))])
        zeros = np.zeros_like(index)-1
        outter_bead_list = np.where(dist_to_com > b, index, zeros)
        outter_bead_list = outter_bead_list[outter_bead_list !=-1]
        n_outter_beads = len(outter_bead_list)
        print(f'{n_outter_beads=}')

        if self.N_an > n_outter_beads:
            charge_per_bead = self.N_an / n_outter_beads
        else:
            outter_bead_list = random.sample(list(outter_bead_list), self.N_an)
            charge_per_bead = 1
            n_outter_beads = len(outter_bead_list)
        
        self.system.part[outter_bead_list].q = [-charge_per_bead] * n_outter_beads
        self.system.part[outter_bead_list].type = [self.PART_TYPE['anion']] * n_outter_beads
        self.__insert_ions(self.N_an, self.PART_TYPE["ion_cat"], +1)
        assert abs(sum(self.system.part[:].q)) < 1e-10



