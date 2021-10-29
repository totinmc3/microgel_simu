import builtins
import numpy as np
from numpy import linalg as LA, string_
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


###########################################################################################
###################                                                     ###################
#############################             MAIN           ##################################
###################                                                     ###################
###########################################################################################
if __name__ == "__main__":
    print("System initialization")

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
    
    visualizer = visualization.openGLLive(system)
    visualizer.run()
    # visualizer.screenshot("screenshot_finconfig.png")
