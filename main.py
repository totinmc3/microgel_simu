import builtins
import numpy as np
from numpy import linalg as LA, number, string_
import math
import argparse
import itertools
import os

import espressomd
from espressomd import electrostatics
from espressomd.interactions import *
from espressomd import polymer
from espressomd import visualization
from espressomd.pair_criteria import DistanceCriterion
from espressomd.cluster_analysis import ClusterStructure

from system_parameters import *
from microgel_class import microgel_object
from handling import handler


def system_info(dir_name_var):
    with open(dir_name_var + "/system_info.txt", "w") as info_file:
        print("L = {:.2f}".format(box_l), file=info_file)
        print("kBT = {:.2f}".format(kBT), file=info_file)
        print("# beads per arm = {:d}".format(Nbeads_arm), file=info_file)
        print("# number of cataionic beads in microgel network = {:d}".format(N_cat), file=info_file)   
        print("# number of anionic beads in microgel network = {:d}".format(N_an), file=info_file)   


###########################################################################################
###################                                                     ###################
#############################             MAIN           ##################################
###################                                                     ###################
###########################################################################################
if __name__ == "__main__":
    print("System initialization")

    dir_name_var = "results/"
    if not os.path.exists(dir_name_var):
        os.mkdir(dir_name_var)

    system_info(dir_name_var)
    
    system = espressomd.System(box_l=[box_l,box_l,box_l])
    system.periodicity = [True, True, True]
    system.time_step = dt
    system.cell_system.skin = skin

    microgel = microgel_object.Microgel(system, FENE_BOND_PARAMS, PART_TYPE, NONBOND_WCA_PARAMS, Nbeads_arm, cell_unit, N_cat, N_an)
    microgel.initialize_diamondLattice()

    microgel.initialize_bonds()
    microgel.initialize_internoelec()
    microgel.charge_beads_homo()
    handler.remove_overlap(system,STEEPEST_DESCENT_PARAMS)
    
    if N_cat != 0 or N_an !=0:
        handler.initialize_elec(system,P3M_PARAMS)

    system.thermostat.set_langevin(**LANGEVIN_PARAMS)
    
    with open(dir_name_var + "/system_info.txt", "a") as info_file:
        number_monomers = len([x for x in system.part[:] if x.type == PART_TYPE['polymer_arm']])
        number_crosslink = len([x for x in system.part[:] if x.type == PART_TYPE['crosslinker']])
        print("# of polymer monomers = {:d}".format(number_monomers), file=info_file)
        print("# of crosslinkers = {:d}".format(number_crosslink), file=info_file)
        print("# of chains = {:d}".format(int(number_monomers/Nbeads_arm)), file=info_file)

    # handler.warmup(system,warm_n_times,warm_steps,dir_name_var,TUNE_SET,TUNE_SKIN_PARAM)

    visualizer = visualization.openGLLive(system)
    visualizer.run()
    # visualizer.screenshot("screenshot_finconfig.png")
