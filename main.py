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

    microgel = microgel_object.Microgel(system,Nbeads_arm,cell_unit)
    microgel.initialize_diamondLattice()

    visualizer = visualization.openGLLive(system)
    visualizer.run()