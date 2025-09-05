import numpy as np
import argparse
import os
import math
from tqdm import tqdm

import espressomd
from espressomd.interactions import *
import espressomd.io.writer.vtf
from espressomd import checkpointing

from system_parameters import *
from microgel_class import microgel_object
from handling import handler
from analysis import densityProfile_calc as dp
from analysis import com as com_mod
from datetime import datetime


HAS_A_CHECKPOINT = os.path.exists(CHECK_NAME)


def system_info(dir_name_var):
    with open(dir_name_var + "/system_info.txt", "w") as info_file:
        print("L = {:.2f}".format(box_l), file=info_file)
        print("kBT = {:.2f}".format(kBT), file=info_file)
        print("c_salt = {:.2f} Molar".format(c_salt_molar), file=info_file)
        print("initial time os =", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file=info_file)


        
###########################################################################################
###################                                                     ###################
#############################             MAIN           ##################################
###################                                                     ###################
###########################################################################################
if __name__ == "__main__":
    print("System initialization")

    parser = argparse.ArgumentParser(description="Process running parameters.")
    # parser.add_argument('N_an', metavar='N_an', type=int, help='Number of anionic beads per microgel')
    parser.add_argument(
        "alpha_an", metavar="alpha_an", type=float, help="anionic ionization degree"
    )
    parser.add_argument(
        "box_l", metavar="box_l", type=float, help="Box size in sigma units"
    )
    parser.add_argument(
        "Nbeads_arm", metavar="Nbeads_arm", type=int, help="Number of beads per arm"
    )
    # parser.add_argument('N_cat', metavar='N_cat', type=int, help='Number of cationic beads per microgel')
    
    argm = parser.parse_args()
    
    alpha_an = argm.alpha_an
    box_l = argm.box_l
    nbeads_arm = argm.Nbeads_arm
    N_cat = 0
    N_an = 0 #luego cambia, sino no corre
    cell_unit = 4 * (nbeads_arm + 1) / np.sqrt(3)
    N_bins = int(box_l)

    dir_name_var = os.path.abspath(".") + "/"
    if not os.path.exists(dir_name_var):
        os.mkdir(dir_name_var)

    if not HAS_A_CHECKPOINT:
        ##### creating checkpointing
        checkpoint = checkpointing.Checkpoint(
            checkpoint_id=CHECK_NAME, checkpoint_path="."
        )

        system_info(dir_name_var)

        system = espressomd.System(box_l=[box_l, box_l, box_l])
        system.periodicity = [True, True, True]
        system.time_step = dt
        system.cell_system.skin = skin

        microgel = microgel_object.Microgel(
            system,
            FENE_BOND_PARAMS,
            PART_TYPE,
            NONBOND_WCA_PARAMS,
            nbeads_arm,
            cell_unit,
            N_cat,
            N_an,
            c_salt,
        )
        # number_crosslink, number_monomers = microgel.initialize_diamondLattice()
        number_crosslink, number_monomers = microgel.initialize_from_file(nbeads_arm)
        N_an = int(alpha_an * (number_crosslink + number_monomers))
        microgel.N_an = N_an

        # gyr_tens = system.analysis.gyration_tensor(p_type=[PART_TYPE['crosslinker'], PART_TYPE['polymer_arm']])
        # shape_list = gyr_tens["shape"]
        # print('%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e' % (
        #         gyr_tens["Rg^2"], shape_list[0], shape_list[1], shape_list[2], gyr_tens["eva0"][0], gyr_tens["eva1"][0], gyr_tens["eva2"][0]),
        #         file = open(dir_name_var + "gyration_tensor.dat", "a"))

        with open(dir_name_var + "system_info.txt", "a") as info_file:
            print(
                "# of polymer monomers = {:d}".format(number_monomers), file=info_file
            )
            print("# of crosslinkers = {:d}".format(number_crosslink), file=info_file)
            print(
                "# of chains = {:d}".format(int(number_monomers / nbeads_arm)),
                file=info_file,
            )
            print(
            "# number of cationic beads in microgel network = {:d}".format(N_cat),
            file=info_file,
        )
            print(
            "# number of anionic beads in microgel network = {:d}".format(0),
            file=info_file,
        )
            print("# beads per arm = {:d}".format(nbeads_arm), file=info_file)
        # microgel.initialize_bonds()
        microgel.initialize_internoelec()
        if N_cat != 0 or N_an != 0:
            microgel.charge_beads_homo()
            # microgel.charge_beads_shell()

        if c_salt != 0:
            print("Add salt to the system")
            N_salt_ion_pairs = microgel.add_salt()
            with open(dir_name_var + "system_info.txt", "a") as info_file:
                print(
                    "# of salt anions = {:d}".format(N_salt_ion_pairs), file=info_file
                )
                print(
                    "# of salt cations = {:d}".format(N_salt_ion_pairs), file=info_file
                )

        handler.remove_overlap(system, STEEPEST_DESCENT_PARAMS)

        if N_cat != 0 or N_an != 0:
            handler.initialize_elec(system, P3M_PARAMS)

        fp_ic = open("trajectory_init_cond.vtf", mode="w+t")
        espressomd.io.writer.vtf.writevsf(system, fp_ic)
        espressomd.io.writer.vtf.writevcf(system, fp_ic)
        fp_ic.close()

        system.thermostat.set_langevin(**LANGEVIN_PARAMS)

        system.time = 0

        iter_warmup = 0

        energies_tot_warm = np.zeros((warm_n_times, 2))

        checkpoint.register("system")
        checkpoint.register("dir_name_var")
        checkpoint.register("iter_warmup")
        checkpoint.register("energies_tot_warm")

    elif HAS_A_CHECKPOINT:
        checkpoint = checkpointing.Checkpoint(
            checkpoint_id=CHECK_NAME, checkpoint_path="."
        )
        checkpoint.load()

        print("Loaded checkpoint.\n")

    # Warmup --------------------------------------------------------------------------
    print("Warmup integration")  # it appears just the first time the function is called


    fp_time = open("tiempo_iteraciones.dat", mode="w+t")

    pbar = tqdm(desc="Warmup loop", total=warm_n_times)
    while iter_warmup < warm_n_times:
        if iter_warmup % CHECKPOINT_PERIOD == 0:
            checkpoint.save()
            fp_time.write("\trun %d at time=%.0f, local time=%s\n" % (iter_warmup, system.time, datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
        if (
            iter_warmup == TUNE_SET["i_val_1"] or iter_warmup == TUNE_SET["i_val_2"]
        ) and TUNE_SET["tune_bool"]:
            system.cell_system.tune_skin(**TUNE_SKIN_PARAM)
        system.integrator.run(warm_steps)  # Default: velocity Verlet algorithm
        energies_tot_warm[iter_warmup] = (
            system.time,
            system.analysis.energy()["total"],
        )

        iter_warmup += 1
        pbar.update(1)
    pbar.close()
    fp_time.close()

    # Export trajectory to vtf file
    fp_0 = open("trajectory_0.vtf", mode="w+t")
    espressomd.io.writer.vtf.writevsf(system, fp_0)
    espressomd.io.writer.vtf.writevcf(system, fp_0)
    fp_0.close()

    print("\nEnd warmup")

    # save energy
    string1 = dir_name_var + "/TotEner_warmup.dat"
    np.savetxt(
        string1,
        np.column_stack((energies_tot_warm[:, 0], energies_tot_warm[:, 1])),
        fmt="%.5e",
        delimiter="\t",
    )
    # ----------------------------------------------------------------------------------

    energies_tot = np.zeros((int_n_times * int_uncorr_times, 2))
    energies_kin = np.zeros((int_n_times * int_uncorr_times, 2))
    energies_nonbon = np.zeros((int_n_times * int_uncorr_times, 2))
    energies_bon = np.zeros((int_n_times * int_uncorr_times, 2))
    energies_coul = np.zeros((int_n_times * int_uncorr_times, 2))
    system.time = 0
    counter_energy = 0

    for j in range(int_uncorr_times):
        counter_energy = handler.main_integration(
            system,
            int_n_times,
            int_steps,
            energies_tot,
            energies_kin,
            energies_nonbon,
            energies_bon,
            energies_coul,
            counter_energy,
        )
        com = com_mod.com_calculation(
            system, PART_TYPE["polymer_arm"], PART_TYPE["cation"], PART_TYPE["anion"]
        )
        print(
            "%.5e\t%.5e\t%.5e" % (com[0], com[1], com[2]),
            file=open(dir_name_var + "center_of_mass.dat", "a"),
        )
        gyr_tens = system.analysis.gyration_tensor(
            p_type=[
                PART_TYPE["crosslinker"],
                PART_TYPE["polymer_arm"],
                PART_TYPE["cation"],
                PART_TYPE["anion"],
            ]
        )
        shape_list = gyr_tens["shape"]
        print(
            "%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e"
            % (
                gyr_tens["Rg^2"],
                shape_list[0],
                shape_list[1],
                shape_list[2],
                gyr_tens["eva0"][0],
                gyr_tens["eva1"][0],
                gyr_tens["eva2"][0],
            ),
            file=open(dir_name_var + "gyration_tensor.dat", "a"),
        )
        if ION_PROFILE_BOOL:
            print("Compute density profiles")
            # Shifting the COM of the cluster ot the center of the box
            print("\tShift COM")
            com_vec = com_mod.com_calculation(
                system,
                PART_TYPE["polymer_arm"],
                PART_TYPE["cation"],
                PART_TYPE["anion"],
            )
            diff = com_vec - system.box_l / 2.0
            system.part.all().pos -= diff

            print("\tExplicit calculation")
            if j == 0:
                # neutral polymer beads
                obs_data, obs_bins = dp.particle_density_profile(
                    system, [PART_TYPE["polymer_arm"], PART_TYPE["crosslinker"]], N_bins
                )
                polymerProfile = obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["cation"], N_bins
                )
                cationProfile = obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["anion"], N_bins
                )
                anionProfile = obs_data
                # Cations
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["ion_cat"], N_bins
                )
                microionProfile_cations = obs_data
                # Anions
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["ion_an"], N_bins
                )
                microionProfile_anions = obs_data
                # Whole microgel
                obs_data, obs_bins = dp.particle_density_profile(
                    system,
                    [
                        PART_TYPE["polymer_arm"],
                        PART_TYPE["crosslinker"],
                        PART_TYPE["cation"],
                        PART_TYPE["anion"],
                    ],
                    N_bins,
                )
                microgelProfile_anions = obs_data
            else:
                # neutral polymer beads
                obs_data, obs_bins = dp.particle_density_profile(
                    system, [PART_TYPE["polymer_arm"], PART_TYPE["crosslinker"]], N_bins
                )
                polymerProfile += obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["cation"], N_bins
                )
                cationProfile += obs_data
                # cation beads
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["anion"], N_bins
                )
                anionProfile += obs_data
                # Cations
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["ion_cat"], N_bins
                )
                microionProfile_cations += obs_data
                # Anions
                obs_data, obs_bins = dp.particle_density_profile(
                    system, PART_TYPE["ion_an"], N_bins
                )
                microionProfile_anions += obs_data
                # Whole microgel

                microgelProfile_anions += obs_data
    if ION_PROFILE_BOOL:  # tranformation from cartesian to spherical coordinates
        prof_list = [
            microionProfile_cations,
            microionProfile_anions,
            polymerProfile,
            cationProfile,
            anionProfile,
            microgelProfile_anions,
        ]
        averaged_profile_list = []
        profile_sph_realiz = []
        for count, profile in enumerate(prof_list):
            averaged_profile = profile / int_uncorr_times

            Nbins = int(N_bins / 2)
            intensities = np.zeros(Nbins)
            box_part = len(averaged_profile)
            r = np.linspace(0, system.box_l[0] / 2, Nbins)
            bin_size = r[1] - r[0]
            box_part2 = box_part / 2
            for i in range(box_part):
                for j in range(box_part):
                    for k in range(box_part):
                        dist = math.sqrt(
                            (i - box_part2) ** 2
                            + (j - box_part2) ** 2
                            + (k - box_part2) ** 2
                        )
                        if dist <= box_part2:
                            intensities[math.floor(dist / bin_size)] += (
                                averaged_profile[i, j, k]
                            )
            profile_sph_realiz.append(intensities)
        profile_sph_realiz.append(r)
        intensity_stack = profile_sph_realiz
        np.savetxt(
            "averaged_profiles.txt",
            np.transpose(intensity_stack),
            fmt="%.4e",
            delimiter="\t",
        )
        """ File columns
        1. microionProfile_cations
        2. microionProfile_anions
        3. polymerProfile, No cuenta cargadas
        4. cationProfile
        5. anionProfile
        6. microgelProfile
        7. r
        """

    # save data
    string1 = dir_name_var + "positions.dat"
    Npart_tot = len(system.part.all())
    i = np.arange(0, Npart_tot, 1)
    position_matrix = np.asarray(system.part.all().pos_folded)
    particle_type = np.asarray(system.part.all().type)
    np.savetxt(
        string1,
        np.column_stack(
            (
                i,
                particle_type,
                position_matrix[:, 0],
                position_matrix[:, 1],
                position_matrix[:, 2],
            )
        ),
        fmt="%d\t%d\t%.6f\t%.6f\t%.6f",
        delimiter="\t",
    )

    string1 = dir_name_var + "/energies.dat"
    np.savetxt(
        string1,
        np.column_stack(
            (
                energies_tot[:, 0],
                energies_tot[:, 1],
                energies_kin[:, 1],
                energies_bon[:, 1],
                energies_nonbon[:, 1],
                energies_coul[:, 1],
            )
        ),
        fmt="%.5e\t%.5e\t%.5e\t%.5e\t%.5e\t%.5e",
        delimiter="\t",
    )
    """ File columns
    1. time
    2. total energy
    3. kinetic energy
    4. bonded energy
    5. non-bonded energy
    6. coulomb energy
    """
    with open(dir_name_var + "/system_info.txt", "a") as info_file:
        print("final time os =", datetime.now().strftime("%Y-%m-%d %H:%M:%S"), file=info_file)
    # Export trajectory to vtf file
    fp = open("trajectory.vtf", mode="w+t")
    # write structure block as header
    espressomd.io.writer.vtf.writevsf(system, fp)
    # write final positions as coordinate block
    espressomd.io.writer.vtf.writevcf(system, fp)
    fp.close()