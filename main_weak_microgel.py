import numpy as np
from numpy import linalg as LA, number, string_
import argparse
import os
import math
from tqdm import tqdm
from scipy import interpolate

import espressomd
from espressomd import electrostatics
from espressomd.interactions import *
from espressomd import polymer
# from espressomd import visualization
from espressomd.pair_criteria import DistanceCriterion
from espressomd.cluster_analysis import ClusterStructure
import espressomd.io.writer.vtf
from espressomd import reaction_ensemble
from espressomd import checkpointing

from system_parameters import *
from microgel_class import microgel_object
from handling import handler
from analysis import densityProfile_calc as dp
from analysis import com as com_mod


HAS_A_CHECKPOINT = os.path.exists(CHECK_NAME)

def system_info(dir_name_var):
    with open(dir_name_var + "/system_info.txt", "w") as info_file:
        print("L = {:.2f}".format(box_l), file=info_file)
        print("kBT = {:.2f}".format(kBT), file=info_file)
        print("c_salt = {:.2f} Molar".format(c_salt_molar), file=info_file)
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
    parser.add_argument('box_size', metavar='box_size', type=float, help='box size')
    # parser.add_argument('N_an', metavar='N_an', type=int, help='Number of anionic beads per microgel')
    #parser.add_argument('alpha_an', metavar='alpha_an', type=float, help='anionic ionization degree')
    parser.add_argument('pH_res', metavar='pH_res', type=float, help='pH in the reservoir')
    # parser.add_argument('N_cat', metavar='N_cat', type=int, help='Number of cationic beads per microgel')
    argm = parser.parse_args()

    box_l = argm.box_size
    PH_VALUE_RES = argm.pH_res
    #alpha_an = argm.alpha_an
    # N_cat = argm.N_cat

    dir_name_var = os.path.abspath('.') + '/'
    if not os.path.exists(dir_name_var):
        os.mkdir(dir_name_var)

    if not HAS_A_CHECKPOINT:

        ##### creating checkpointing
        checkpoint = checkpointing.Checkpoint(checkpoint_id = CHECK_NAME, checkpoint_path = '.')

        system_info(dir_name_var)

        system = espressomd.System(box_l=[box_l,box_l,box_l])
        system.periodicity = [True, True, True]
        system.time_step = dt
        system.cell_system.skin = skin

        microgel = microgel_object.Microgel(system, FENE_BOND_PARAMS, PART_TYPE, NONBOND_WCA_PARAMS, Nbeads_arm, cell_unit, N_cat, N_an, c_salt)
        # number_crosslink, number_monomers = microgel.initialize_diamondLattice()
        number_crosslink, number_monomers = microgel.initialize_from_file()
        alpha_HH =  1 / (1 + 10**(pKa+1.5-PH_VALUE_RES))
        N_an = int(alpha_HH * (number_crosslink + number_monomers))
        #print(system.part)
        #exit()
        microgel.N_an = N_an
        #for p in system.part:
        #    print(p.type)



        # gyr_tens = system.analysis.gyration_tensor(p_type=[PART_TYPE['crosslinker'], PART_TYPE['polymer_arm']])
        # shape_list = gyr_tens["shape"]
        # print('%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e' % (
        #         gyr_tens["Rg^2"], shape_list[0], shape_list[1], shape_list[2], gyr_tens["eva0"][0], gyr_tens["eva1"][0], gyr_tens["eva2"][0]),
        #         file = open(dir_name_var + "gyration_tensor.dat", "a"))


        with open(dir_name_var + "system_info.txt", "a") as info_file:
            print("# of polymer monomers = {:d}".format(number_monomers), file=info_file)
            print("# of crosslinkers = {:d}".format(number_crosslink), file=info_file)
            print("# of chains = {:d}".format(int(number_monomers/Nbeads_arm)), file=info_file)

        # microgel.initialize_bonds()
        microgel.initialize_internoelec()
        if N_cat != 0 or N_an !=0:
            microgel.charge_beads_homo()
            # microgel.charge_beads_shell()

        if c_salt != 0:
            print("Add salt to the system")
            N_salt_ion_pairs = microgel.add_salt()
            with open(dir_name_var + "system_info.txt", "a") as info_file:
                print("# of salt anions = {:d}".format(N_salt_ion_pairs), file=info_file)
                print("# of salt cations = {:d}".format(N_salt_ion_pairs), file=info_file)

        
        handler.remove_overlap(system,STEEPEST_DESCENT_PARAMS)

        if N_cat != 0 or N_an !=0:
            handler.initialize_elec(system,P3M_PARAMS)

        fp_ic = open('trajectory_init_cond.vtf', mode='w+t')
        espressomd.io.writer.vtf.writevsf(system, fp_ic)
        espressomd.io.writer.vtf.writevcf(system, fp_ic)
        fp_ic.close()

        # Export init condition to pdb file
        #if True:
        if False:
            import MDAnalysis as mda
            import espressomd.MDA_ESP

            eos = espressomd.MDA_ESP.Stream(system)
            u = mda.Universe(eos.topology, eos.trajectory)
            u.atoms.write("trajectory_init_cond.pdb")
            print("===> The initial configuration has been writen to trajectory_init_cond.pdb ")

        system.thermostat.set_langevin(**LANGEVIN_PARAMS)

        system.time = 0

        iter_warmup = 0

        energies_tot_warm = np.zeros((warm_n_times, 2))
        
        checkpoint.register("system")
        checkpoint.register("dir_name_var")
        checkpoint.register("iter_warmup")
        checkpoint.register("energies_tot_warm")


    elif HAS_A_CHECKPOINT:
        checkpoint = checkpointing.Checkpoint(checkpoint_id = CHECK_NAME, checkpoint_path = '.')   
        checkpoint.load()

        print("Loaded checkpoint.\n")

    ##### Setting up the chemical reactions

    # Determine the reservoir composition
    ionic_strength, excess_chemical_potential_monovalent_pairs_in_bulk_data, bjerrums,excess_chemical_potential_monovalent_pairs_in_bulk_data_error =np.loadtxt("/data/dbeyer/brush_titration/simulation_data/excess_chemical_potential.dat", unpack=True) #remember, excess chemical potential does not know about types
    excess_chemical_potential_monovalent_pairs_in_bulk=interpolate.interp1d(ionic_strength, excess_chemical_potential_monovalent_pairs_in_bulk_data)

    KT = LANGEVIN_PARAMS["kT"]
    C_SALT_RES_SIM = c_salt
    sigma = 3.55e-10 # Sigma in SI units
    avo = 6.022e+23 # Avogadro's number in SI units
    PREF = 1/(10**3 * avo * sigma**3) # Prefactor to mol/L
    CREF_IN_MOL_PER_L = 1.0 #in mol/l
    KW=10**-14 #dimensionless dissociation constant Kw=relative_activity(H)*relative_activity(OH)


    def determine_bulk_concentrations_selfconsistently(cH_bulk_in_mol_per_l, cs_bulk):
        global KW, CREF_IN_MOL_PER_L, PREF 
        #calculate initial guess for concentrations
        cOH_bulk=(KW/(cH_bulk_in_mol_per_l/CREF_IN_MOL_PER_L))*CREF_IN_MOL_PER_L/PREF
        cH_bulk=cH_bulk_in_mol_per_l/PREF
        cNa_bulk=None
        cCl_bulk=None
        if (cOH_bulk>=cH_bulk):
            #there is excess OH in the bulk
            #electro-neutralize this excess OH with Na+ (in the bulk)
            cNa_bulk=cs_bulk+(cOH_bulk-cH_bulk)
            cCl_bulk=cs_bulk
        else:
            #there is excess H
            #electro-neutralize this excess H with Cl- (in the bulk)
            cCl_bulk=cs_bulk+(cH_bulk-cOH_bulk)
            cNa_bulk=cs_bulk

            
        def calculate_concentrations_self_consistently(cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk):
            global KT, MAX_SELF_CONSISTENT_RUNS, self_consistent_run
            if(self_consistent_run<MAX_SELF_CONSISTENT_RUNS):
                self_consistent_run+=1
                ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
                cOH_bulk=(KW/(cH_bulk_in_mol_per_l/CREF_IN_MOL_PER_L)*np.exp(-(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk))/KT))*CREF_IN_MOL_PER_L/PREF
                if (cOH_bulk>=cH_bulk):
                    #there is excess OH in the bulk
                    #electro-neutralize this excess OH with Na+ (in the bulk)
                    cNa_bulk=cs_bulk+(cOH_bulk-cH_bulk)
                    cCl_bulk=cs_bulk
                else:
                    #there is excess H
                    #electro-neutralize this excess H with Cl- (in the bulk)
                    cCl_bulk=cs_bulk+(cH_bulk-cOH_bulk)
                    cNa_bulk=cs_bulk
                return calculate_concentrations_self_consistently(cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk)
            else:
                return np.array([cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk])
        return calculate_concentrations_self_consistently(cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk)


    def check_concentrations():
        if(abs(KW-cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/KT) * PREF**2/CREF_IN_MOL_PER_L**2)>1e-15):
            raise RuntimeError("Kw incorrect")
        if(abs(cNa_bulk+cH_bulk-cOH_bulk-cCl_bulk)>1e-14):
            raise RuntimeError("bulk is not electroneutral")
        if(abs(PH_VALUE_RES-determined_pH)>1e-5):
            raise RuntimeError("pH is not compatible with ionic strength and bulk H+ concentration")
        if(abs(C_SALT_RES_SIM-min(cNa_bulk, cCl_bulk))>1e-14):
            raise RuntimeError("bulk salt concentration is not correct")
        if(abs(PH_VALUE_RES-7)<1e-14):
            if((cH_bulk/cOH_bulk-1)>1e-5):
                raise RuntimeError("cH and cOH need to be symmetric at pH 7")

    MAX_SELF_CONSISTENT_RUNS = 200
    self_consistent_run = 0
    cH_bulk_in_mol_per_l = 10**(-PH_VALUE_RES)*CREF_IN_MOL_PER_L #this is a guess, which is used as starting point of the self consistent optimization
    cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk = determine_bulk_concentrations_selfconsistently(cH_bulk_in_mol_per_l, C_SALT_RES_SIM)
    ionic_strength_bulk = 0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
    determined_pH = -np.log10(cH_bulk*PREF/CREF_IN_MOL_PER_L*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(2.0*KT)))

    while abs(determined_pH-PH_VALUE_RES)>1e-6:
        if(determined_pH)>PH_VALUE_RES:
            cH_bulk_in_mol_per_l=cH_bulk_in_mol_per_l*1.005
        else:
            cH_bulk_in_mol_per_l=cH_bulk_in_mol_per_l/1.003
        cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk = determine_bulk_concentrations_selfconsistently(cH_bulk_in_mol_per_l, C_SALT_RES_SIM)
        ionic_strength_bulk = 0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
        determined_pH = -np.log10(cH_bulk*PREF*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(2.0*KT)))
        self_consistent_run=0

            
    print("after self consistent concentration calculation: cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk", cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk)
    print("check KW: ",KW, cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/KT)*PREF**2/CREF_IN_MOL_PER_L**2)
    print("check electro neutrality bulk after", cNa_bulk+cH_bulk-cOH_bulk-cCl_bulk) #note that charges are neutral up to numerical precision. femto molar charge inequalities are not important in the bulk.
    print("check pH: input", PH_VALUE_RES, "determined pH", determined_pH)
    print("check cs bulk: input", C_SALT_RES_SIM, "determined cs_bulk", min(cNa_bulk, cCl_bulk))
    print("check cH_bulk/cOH_bulk:", cH_bulk/cOH_bulk)
    check_concentrations()

    # Determine the equilibrium constants
    K_W = KW / PREF ** 2
    K_XX = (cNa_bulk + cH_bulk) * (cCl_bulk + cOH_bulk) * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / KT)

    gamma_pK_acid = 10 ** (-pKa) / PREF 
    gamma_K_ACID_X = gamma_pK_acid * (cNa_bulk + cH_bulk) / cH_bulk


    # Add the chemical reactions
    RE = espressomd.reaction_ensemble.ReactionEnsemble(kT=KT, seed=LANGEVIN_PARAMS["seed"], exclusion_radius=1)

    # Coupling to reservoir of monovalent ions
    RE.add_reaction(
        gamma = K_XX,
        reactant_types = [],
        reactant_coefficients = [],
        product_types = [ PART_TYPE['ion_cat'], PART_TYPE['ion_an'] ],
        product_coefficients = [ 1, 1 ],
        default_charges = {
            PART_TYPE['ion_cat']: +1.0,
            PART_TYPE['ion_an']: -1.0 
        }
    )

    # Ionization of HA: HA = A + X+
    RE.add_reaction(
        gamma = gamma_K_ACID_X,
        reactant_types = [ PART_TYPE['polymer_arm'] ],
        reactant_coefficients = [ 1 ],
        product_types = [ PART_TYPE['anion'], PART_TYPE['ion_cat'] ],
        product_coefficients = [ 1, 1 ],
        default_charges = {
            PART_TYPE['polymer_arm']: 0.0,
            PART_TYPE['ion_cat']:  +1.0,
            PART_TYPE['anion']: -1.0 
        }
    )

    # X- + HA = A
    RE.add_reaction(
        gamma = gamma_K_ACID_X / K_XX,
        reactant_types = [ PART_TYPE['polymer_arm'], PART_TYPE['ion_an'] ],
        reactant_coefficients = [ 1, 1 ],
        product_types = [ PART_TYPE['anion'] ],
        product_coefficients = [ 1 ],
        default_charges = {
            PART_TYPE['polymer_arm']: 0.0,
            PART_TYPE['ion_an']:  -1.0,
            PART_TYPE['anion']: -1.0 
        }
    )

    RE.set_non_interacting_type(len(PART_TYPE)+1)

    # handler.warmup(system,warm_n_times,warm_steps,dir_name_var,TUNE_SET,TUNE_SKIN_PARAM, checkpoint, CHECKPOINT_PERIOD, iter_warmup)
    # Warmup --------------------------------------------------------------------------
    print("Warmup integration") # it appears just the first time the function is called
        
    warm_n_times = 500
    pbar = tqdm(desc='Warmup loop', total=warm_n_times)
    while (iter_warmup < warm_n_times):
        if iter_warmup%CHECKPOINT_PERIOD == 0:
            checkpoint.save()
        if (iter_warmup == TUNE_SET['i_val_1'] or iter_warmup == TUNE_SET['i_val_2']) and TUNE_SET['tune_bool']:
            system.cell_system.tune_skin(**TUNE_SKIN_PARAM)
        system.integrator.run(warm_steps)  # Default: velocity Verlet algorithm
        RE.reaction(warm_steps)
        print("\r\trun %d at time=%.0f " % (iter_warmup, system.time), end='')
        HA =  system.number_of_particles(type = PART_TYPE["polymer_arm"])
        A =  system.number_of_particles(type = PART_TYPE["anion"])
        print("Current degree of ionization:", A / (HA + A))
        #energies_tot_warm[iter_warmup] = (system.time, system.analysis.energy()['total'])
        iter_warmup += 1

        pbar.update(1)

    checkpoint.save()
    pbar.close()

    # Export trajectory to vtf file
    fp_0 = open('trajectory_0.vtf', mode='w+t')
    espressomd.io.writer.vtf.writevsf(system, fp_0)
    espressomd.io.writer.vtf.writevcf(system, fp_0)
    fp_0.close()
    # Export trajectory to pdb file
    if False:
        import MDAnalysis as mda
        import espressomd.MDA_ESP

        eos = espressomd.MDA_ESP.Stream(system)
        u = mda.Universe(eos.topology, eos.trajectory)
        u.atoms.write("trajectory_0.pdb")
        print("===> The initial configuration has been writen to trajectory_0.pdb ")
    
    print("\nEnd warmup")

    # save energy
    string1 = dir_name_var + '/TotEner_warmup.dat'
    np.savetxt(string1, np.column_stack((energies_tot_warm[:, 0], energies_tot_warm[:, 1])),fmt='%.5e', delimiter='\t')
    #----------------------------------------------------------------------------------

    energies_tot = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_kin = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_nonbon = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_bon = np.zeros((int_n_times*int_uncorr_times, 2))
    energies_coul = np.zeros((int_n_times*int_uncorr_times, 2))
    alphas = np.zeros((int_n_times*int_uncorr_times, 2))
    system.time = 0
    counter_energy = 0

    for j in range(int_uncorr_times):
        counter_energy = handler.main_integration(system, int_n_times, int_steps, energies_tot, energies_kin, energies_nonbon, energies_bon, energies_coul, counter_energy)
        HA =  system.number_of_particles(type = PART_TYPE["polymer_arm"])
        A =  system.number_of_particles(type = PART_TYPE["anion"])
        alphas.append(A / (HA + A))
        com = com_mod.com_calculation(system, PART_TYPE['polymer_arm'],PART_TYPE['cation'], PART_TYPE['anion'])
        print('%.5e\t%.5e\t%.5e' % (com[0], com[1], com[2]), file = open(dir_name_var + "center_of_mass.dat", "a"))
        gyr_tens = system.analysis.gyration_tensor(p_type=[PART_TYPE['crosslinker'], PART_TYPE['polymer_arm'], PART_TYPE['cation'], PART_TYPE['anion']])
        shape_list = gyr_tens["shape"]
        print('%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e' % (
                gyr_tens["Rg^2"], shape_list[0], shape_list[1], shape_list[2], gyr_tens["eva0"][0], gyr_tens["eva1"][0], gyr_tens["eva2"][0]),
                file = open(dir_name_var + "gyration_tensor.dat", "a"))
        if ION_PROFILE_BOOL:
            print("Compute density profiles")
            # Shifting the COM of the cluster ot the center of the box
            print("\tShift COM")
            com_vec = com_mod.com_calculation(system, PART_TYPE['polymer_arm'],PART_TYPE['cation'], PART_TYPE['anion'])
            diff = com_vec - system.box_l/2.
            system.part[:].pos -= diff

            print("\tExplicit calculation")
            if j==0:
                # neutral polymer beads
                obs_data, obs_bins = dp.particle_density_profile(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker']], N_bins)
                polymerProfile = obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['cation'], N_bins)
                cationProfile = obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['anion'], N_bins)
                anionProfile = obs_data
                # Cations
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['ion_cat'], N_bins)
                microionProfile_cations = obs_data
                # Anions
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['ion_an'], N_bins)
                microionProfile_anions = obs_data
                # Whole microgel
                obs_data, obs_bins = dp.particle_density_profile(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker'], PART_TYPE['cation'], PART_TYPE['anion']], N_bins)
                microgelProfile_anions = obs_data
            else:
                # neutral polymer beads
                obs_data, obs_bins = dp.particle_density_profile(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker']], N_bins)
                polymerProfile += obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['cation'], N_bins)
                cationProfile += obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['anion'], N_bins)
                anionProfile += obs_data
                # Cations
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['ion_cat'], N_bins)
                microionProfile_cations += obs_data
                # Anions
                obs_data, obs_bins = dp.particle_density_profile(system, PART_TYPE['ion_an'], N_bins)
                microionProfile_anions += obs_data
                # Whole microgel
                obs_data, obs_bins = dp.particle_density_profile(system, [PART_TYPE['polymer_arm'], PART_TYPE['crosslinker'], PART_TYPE['cation'], PART_TYPE['anion']], N_bins)
                microgelProfile_anions += obs_data
    if ION_PROFILE_BOOL: # tranformation from cartesian to spherical coordinates
        prof_list = [microionProfile_cations, microionProfile_anions, polymerProfile, cationProfile, anionProfile, microgelProfile_anions]
        averaged_profile_list = []
        profile_sph_realiz = []
        for count,profile in enumerate(prof_list):
            averaged_profile = profile/int_uncorr_times

            Nbins = int(N_bins/2)
            intensities = np.zeros(Nbins)
            box_part = len(averaged_profile)
            r = np.linspace(0, system.box_l[0]/2, Nbins)
            bin_size = r[1] - r[0]
            box_part2 = box_part/2
            for i in range(box_part):
                for j in range(box_part):
                    for k in range(box_part):
                        dist = math.sqrt((i - box_part2)**2 + (j - box_part2)**2 + (k - box_part2)**2)
                        if dist <= box_part2:
                            intensities[math.floor(dist/bin_size)] += averaged_profile[i, j, k]
            profile_sph_realiz.append(intensities)
        profile_sph_realiz.append(r)
        intensity_stack = profile_sph_realiz
        np.savetxt("averaged_profiles.txt", np.transpose(intensity_stack), fmt='%.4e', delimiter='\t')
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
    np.savetxt(string1, np.column_stack((energies_tot[:, 0], energies_tot[:, 1], energies_kin[:, 1], energies_bon[:, 1], 
                                         energies_nonbon[:, 1], energies_coul[:, 1])),fmt='%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e', delimiter='\t')
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
    
    # Export trajectory to pdb file
    if False:
        import MDAnalysis as mda
        import espressomd.MDA_ESP

        eos = espressomd.MDA_ESP.Stream(system)
        u = mda.Universe(eos.topology, eos.trajectory)
        u.atoms.write("trajectory.pdb")
        print("===> The initial configuration has been writen to trajectory.pdb ")


    # visualizer = visualization.openGLLive(system)
    # visualizer.run()
    # # visualizer.screenshot("results/screenshot_finconfig.png")
