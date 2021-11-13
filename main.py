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

    parser = argparse.ArgumentParser(description='Process running parameters.')
    parser.add_argument('box_size', metavar='box_size', type=int, help='box size')
    argm = parser.parse_args()

    box_l = argm.box_size

    dir_name_var = "results/"
    if not os.path.exists(dir_name_var):
        os.mkdir(dir_name_var)

    system_info(dir_name_var)
    
    system = espressomd.System(box_l=[box_l,box_l,box_l])
    system.periodicity = [True, True, True]
    system.time_step = dt
    system.cell_system.skin = skin

    microgel = microgel_object.Microgel(system, FENE_BOND_PARAMS, PART_TYPE, NONBOND_WCA_PARAMS, Nbeads_arm, cell_unit, N_cat, N_an)
    number_crosslink, number_monomers = microgel.initialize_diamondLattice()
    
    with open(dir_name_var + "system_info.txt", "a") as info_file:
        # number_monomers = len([x for x in system.part[:] if x.type == PART_TYPE['polymer_arm']])
        # number_crosslink = len([x for x in system.part[:] if x.type == PART_TYPE['crosslinker']])
        print("# of polymer monomers = {:d}".format(number_monomers), file=info_file)
        print("# of crosslinkers = {:d}".format(number_crosslink), file=info_file)
        print("# of chains = {:d}".format(int(number_monomers/Nbeads_arm)), file=info_file)


    microgel.initialize_bonds()
    microgel.initialize_internoelec()
    # microgel.charge_beads_homo()
    handler.remove_overlap(system,STEEPEST_DESCENT_PARAMS)
    
    if N_cat != 0 or N_an !=0:
        handler.initialize_elec(system,P3M_PARAMS)

    system.thermostat.set_langevin(**LANGEVIN_PARAMS)

    system.time = 0
    handler.warmup(system,warm_n_times,warm_steps,dir_name_var,TUNE_SET,TUNE_SKIN_PARAM)


    energies_tot = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_kin = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_nonbon = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_bon = np.zeros((int_n_times*int_uncorr_times, 2))
    system.time = 0
    counter_energy = 0
    for j in range(int_uncorr_times):
        counter_energy = handler.main_integration(system, int_n_times, int_steps, energies_tot, energies_kin, energies_nonbon, energies_bon, counter_energy)
        gyr_tens = system.analysis.gyration_tensor(p_type=[PART_TYPE['crosslinker'], PART_TYPE['polymer_arm']])
        shape_list = gyr_tens["shape"]
        print('%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e' % (
                gyr_tens["Rg^2"], shape_list[0], shape_list[1], shape_list[2], gyr_tens["eva0"][0], gyr_tens["eva1"][0], gyr_tens["eva2"][0]),
                file = open(dir_name_var + "gyration_tensor.dat", "a"))
    
    visualizer = visualization.openGLLive(system)
    # visualizer.run()
    visualizer.screenshot("results/screenshot_finconfig.png")
