import numpy as np
from numpy import linalg as LA, number, string_
import argparse
import os
import itertools
import math
from tqdm import tqdm

import espressomd
from espressomd import electrostatics
from espressomd.interactions import *
from espressomd import polymer
from espressomd.interactions import FeneBond
from espressomd.electrostatics import P3M
# from espressomd import visualization
from espressomd.pair_criteria import DistanceCriterion
from espressomd.cluster_analysis import ClusterStructure
import espressomd.io.writer.vtf
from espressomd import checkpointing
from espressomd import MDA_ESP

from system_parameters import *
from microgel_class import microgel_object
from handling import handler
from analysis import densityProfile_calc as dp
from analysis import part_counter as pc
from analysis import com as com_mod
import microgel_class.read_vtf_file as read_vtf
import MDAnalysis as mda
from MDAnalysis.coordinates.XTC import XTCWriter

if ION_PROFILE_BOOL:
    N_bins = int(box_l) #  number of bins of calculation of ion profile in cartesion coord
HAS_A_CHECKPOINT = os.path.exists(CHECK_NAME)

def system_info(dir_name_var):
    with open(dir_name_var + "/system_info.txt", "w") as info_file:
        print("L = {:.2f}".format(box_l), file=info_file)
        print("kBT = {:.2f}".format(kBT), file=info_file)
        print("c_salt = {:.6f} Molar".format(c_salt_molar), file=info_file)
        print("# beads per arm = {:d}".format(Nbeads_arm), file=info_file)
        print("# number of cationic beads in microgel network = {:d}".format(N_cat), file=info_file)   
        print("# number of anionic beads in microgel network = {:d}".format(N_an), file=info_file)   


###########################################################################################
###################                                                     ###################
#############################             MAIN           ##################################
###################                                                     ###################
###########################################################################################
if __name__ == "__main__":
    print("System initialization")

    parser = argparse.ArgumentParser(description='Process running parameters.')
    parser.add_argument('vtf_file', metavar='vtf_file', type=str, help='VTF file')
    argm = parser.parse_args()

    vtf_file = argm.vtf_file
    print('{vtf_file=}')

    dir_name_var = os.path.abspath('.') + '/'
    if not os.path.exists(dir_name_var):
        os.mkdir(dir_name_var)

    box_l,ids,types,bonds,positions = read_vtf.read_vtf_file(vtf_file)

    system = espressomd.System(box_l=box_l)
    system.periodicity = [True, True, True]
    system.time_step = dt
    system.cell_system.skin = skin

    charges = np.zeros_like(types, dtype=int)
    charges[types==PART_TYPE['cation']] = 1
    charges[types==PART_TYPE['ion_cat']] = 1
    charges[types==PART_TYPE['anion']] = -1
    charges[types==PART_TYPE['ion_an']] = -1

    system.part.add(id=ids, pos=positions[:, 1:], type=types, q=charges)

    fene = FeneBond(**FENE_BOND_PARAMS)
    system.bonded_inter.add(fene)

    for bond in bonds:
        system.part[bond[0]].add_bond((fene, bond[1]))

    print("Define interactions (non electrostatic)")
    # Non-bonded Interactions:
    for i,j in itertools.combinations_with_replacement([x for x in PART_TYPE], 2):
        system.non_bonded_inter[PART_TYPE[i], PART_TYPE[j]].wca.set_params(**NONBOND_WCA_PARAMS)

    # center particles in the middle of simulation box
    com_vec = com_mod.com_calculation(system, PART_TYPE['polymer_arm'],PART_TYPE['cation'], PART_TYPE['anion'])
    diff = com_vec - system.box_l/2.
    system.part[:].pos -= diff

    handler.remove_overlap(system,STEEPEST_DESCENT_PARAMS)

    N_an = len(system.part.select(type=PART_TYPE['anion']))
    N_cat = len(system.part.select(type=PART_TYPE['cation']))
    if N_cat != 0 or N_an !=0:
            handler.initialize_elec(system,P3M_PARAMS)

    system.thermostat.set_langevin(**LANGEVIN_PARAMS)
    system.time = 0


    print("Short warmup integration") # it appears just the first time the function is called
    iter_warmup = 0
    short_warm_n_times = 100
    pbar = tqdm(desc='Warmup loop', total=short_warm_n_times)
    while (iter_warmup < short_warm_n_times):
        system.integrator.run(warm_steps)  # Default: velocity Verlet algorithm
        print("\r\trun %d at time=%.0f " % (iter_warmup, system.time), end='')
        iter_warmup += 1
        pbar.update(1)

    pbar.close()

    # Export trajectory to vtf file
    fp_0 = open('trajectory_0.vtf', mode='w+t')
    espressomd.io.writer.vtf.writevsf(system, fp_0)
    espressomd.io.writer.vtf.writevcf(system, fp_0)
    fp_0.close()
   
    print("\nEnd short warmup")

    print("Tune skin")
    system.cell_system.tune_skin(**TUNE_SKIN_PARAM)
    
    #----------------------------------------------------------------------------------

    energies_tot = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_kin = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_nonbon = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_bon = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_coul = np.zeros((int_n_times*int_uncorr_times, 2))
    system.time = 0
    counter_energy = 0

    if TRAJECTORY_BOOL:
    ##### Set up output of trajectory
      f = open('conf.gro', 'w')
      f.write('Macroscopic gel network\n')
      f.write(str(len(system.part[:]))+'\n')
      for i in range(len(system.part[:])):
        index = i
        f.write("{: >5}".format(str(1)))# f.write("{: >5}".format(str(np.int(system.part[index].mol_id)))) # residue number (5 positions, integer) 
        f.write("{: >5}".format("GEL")) # residue name (5 characters)
        f.write("{: >5}".format(str(system.part[index].type))) # atom name (5 characters)
        f.write("{: >5}".format(str(system.part[index].id))) # atom number (5 positions, integer)
        f.write("{: >8}".format(str("{:.3f}".format(system.part[index].pos[0])))) # position (x y z in 3 columns, each 8 positions with 3 decimal places)
        f.write("{: >8}".format(str("{:.3f}".format(system.part[index].pos[1])))) # position (x y z in 3 columns, each 8 positions with 3 decimal places)
        f.write("{: >8}".format(str("{:.3f}".format(system.part[index].pos[2])))) # position (x y z in 3 columns, each 8 positions with 3 decimal places)
        f.write("\n")

      f.write("{: >10}".format(str("{:.5f}".format(system.box_l[0]))))
      f.write("{: >10}".format(str("{:.5f}".format(system.box_l[1]))))
      f.write("{: >10}".format(str("{:.5f}".format(system.box_l[2]))))
      f.close()

      eos = MDA_ESP.Stream(system)  # create the stream
      u = mda.Universe(eos.topology, eos.trajectory)  # create the MDA universe

      # open the trajectory file
      W = XTCWriter("trajectories.xtc", n_atoms=len(system.part))
      

    # Initialize observable and accumulator for density profile calculation
    # neutral polymer beads
    observ_neutral = dp.define_density_obs(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker']], N_bins)
    accumulator_neutral = espressomd.accumulators.MeanVarianceCalculator(obs=observ_neutral)
    # cation polymer beads
    observ_cation = dp.define_density_obs(system, PART_TYPE['cation'], N_bins)
    accumulator_cation = espressomd.accumulators.MeanVarianceCalculator(obs=observ_cation)
    # anion polymer beads
    observ_anion = dp.define_density_obs(system, PART_TYPE['anion'], N_bins)
    accumulator_anion = espressomd.accumulators.MeanVarianceCalculator(obs=observ_anion)
    # ion_cat polymer beads
    observ_ion_cat = dp.define_density_obs(system, PART_TYPE['ion_cat'], N_bins)
    accumulator_ion_cat = espressomd.accumulators.MeanVarianceCalculator(obs=observ_ion_cat)
    # ion_an polymer beads
    observ_ion_an = dp.define_density_obs(system, PART_TYPE['ion_an'], N_bins)
    accumulator_ion_an = espressomd.accumulators.MeanVarianceCalculator(obs=observ_ion_an)
    # Whole microgel
    observ_whole = dp.define_density_obs(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker'], PART_TYPE['cation'], PART_TYPE['anion']], N_bins)
    accumulator_whole = espressomd.accumulators.MeanVarianceCalculator(obs=observ_whole)


    for j in range(int_uncorr_times):
        counter_energy = handler.main_integration(system, int_n_times, int_steps, energies_tot, energies_kin, energies_nonbon, energies_bon, energies_coul, counter_energy)
        com = com_mod.com_calculation(system, PART_TYPE['polymer_arm'],PART_TYPE['cation'], PART_TYPE['anion'])
        #print('%.5e\t%.5e\t%.5e' % (com[0], com[1], com[2]), file = open(dir_name_var + "center_of_mass.dat", "a"))
        gyr_tens = system.analysis.gyration_tensor(p_type=[PART_TYPE['crosslinker'], PART_TYPE['polymer_arm'], PART_TYPE['cation'], PART_TYPE['anion']])
        shape_list = gyr_tens["shape"]
        #print('%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e' % (
        #        gyr_tens["Rg^2"], shape_list[0], shape_list[1], shape_list[2], gyr_tens["eva0"][0], gyr_tens["eva1"][0], gyr_tens["eva2"][0]),
        #        file = open(dir_name_var + "gyration_tensor.dat", "a"))
        if TRAJECTORY_BOOL:
            u.load_new(eos.trajectory)  # load the frame to the MDA universe
            W.write(u)  # append it to the trajectory
        if ION_PROFILE_BOOL:
            print("Compute density profiles")
            # Shifting the COM of the cluster ot the center of the box
            print("\tShift COM")
            com_vec = com_mod.com_calculation(system, PART_TYPE['polymer_arm'],PART_TYPE['cation'], PART_TYPE['anion'])
            diff = com_vec - system.box_l/2.
            system.part[:].pos -= diff

            print("\tExplicit calculation")
            # option (a)
            accumulator_neutral.update()
            accumulator_cation.update()
            accumulator_anion.update()
            accumulator_ion_cat.update()
            accumulator_ion_an.update()
            accumulator_whole.update()
            print("\tCount inner ions")
            obs_data, obs_bins = dp.particle_density_profile(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker']], N_bins)
            polymerProfile = obs_data
            R_mic, size_index = dp.microgel_radius(*dp.profile_spher_transf(polymerProfile, int(N_bins/2), system.box_l[0]))
            N_coun_in = pc.count_interior_ions(system, PART_TYPE['ion_cat'], R_mic)
            N_coun_in_weight = pc.count_interior_ions_weighted(system, PART_TYPE['ion_cat'], R_mic)
            print('%.5e\t%.5e\t%.5e' % (N_coun_in, R_mic, N_coun_in_weight), file = open(dir_name_var + "N_counterions_cutoff0-01_.dat", "a"))
    if ION_PROFILE_BOOL: # tranformation from cartesian to spherical coordinates
        # option (a)
        microionProfile_cations = accumulator_ion_cat.mean()
        microionProfile_anions = accumulator_ion_an.mean()
        polymerProfile = accumulator_neutral.mean()
        cationProfile = accumulator_cation.mean()
        anionProfile = accumulator_anion.mean()
        microgelProfile = accumulator_whole.mean()

        prof_list = [microionProfile_cations, microionProfile_anions, polymerProfile, cationProfile, anionProfile, microgelProfile]
        averaged_profile_list = []
        profile_sph_realiz = []
        for count,profile in enumerate(prof_list):
            averaged_profile = profile/int_uncorr_times

            Nbins = int(N_bins/2)
            intensities, r = dp.profile_spher_transf(averaged_profile, Nbins, system.box_l[0])
            profile_sph_realiz.append(intensities)
        profile_sph_realiz.append(r)
        intensity_stack = profile_sph_realiz
        np.savetxt("averaged_profiles_b.txt", np.transpose(intensity_stack), fmt='%.4e', delimiter='\t')
        ''' File columns
        1. microionProfile_cations
        2. microionProfile_anions
        3. polymerProfile
        4. cationProfile
        5. anionProfile
        6. microgelProfile
        7. r
        '''


            
    # save data
    string1 = dir_name_var + "positions.dat"
    Npart_tot = len(system.part[:])
    i = np.arange(0,Npart_tot,1)
    position_matrix = np.asarray(system.part[:].pos_folded)
    particle_type = np.asarray(system.part[:].type)
    np.savetxt(string1, np.column_stack((i,particle_type,position_matrix[:,0],position_matrix[:,1],
                                            position_matrix[:,2])),fmt='%d\t%d\t%.6f\t%.6f\t%.6f', delimiter='\t')

    string1 = dir_name_var + '/energies.dat'
    #np.savetxt(string1, np.column_stack((energies_tot[:, 0], energies_tot[:, 1], energies_kin[:, 1], energies_bon[:, 1], 
    #                                     energies_nonbon[:, 1], energies_coul[:, 1])),fmt='%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e', delimiter='\t')
    ''' File columns
    1. time
    2. total energy
    3. kinetic energy
    4. bonded energy
    5. non-bonded energy
    6. coulomb energy
    '''

    # Export trajectory to vtf file
    fp = open('trajectory.vtf', mode='w+t')
    # write structure block as header
    espressomd.io.writer.vtf.writevsf(system, fp)
    # write final positions as coordinate block
    espressomd.io.writer.vtf.writevcf(system, fp)

    # # visualizer = visualization.openGLLive(system)
    # # visualizer.run()
    # # # visualizer.screenshot("results/screenshot_finconfig.png")
