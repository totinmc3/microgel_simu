import numpy as np
from tqdm import tqdm
from espressomd.electrostatics import P3M


def remove_overlap(system, sd_params):
    print("Remove overlap")
    # Removes overlap by steepest descent until forces or energies converge
    # Set up steepest descent integration
    system.integrator.set_steepest_descent(
        f_max=0,
        gamma=sd_params["damping"],
        max_displacement=sd_params["max_displacement"],
    )

    # Initialize integrator to obtain initial forces
    system.integrator.run(0)
    maxforce = np.max(np.linalg.norm(system.part.all().f, axis=1))
    energy = system.analysis.energy()["total"]

    i = 0
    while i < sd_params["max_steps"] // sd_params["emstep"]:
        prev_maxforce = maxforce
        prev_energy = energy
        system.integrator.run(sd_params["emstep"])
        maxforce = np.max(np.linalg.norm(system.part.all().f, axis=1))
        relforce = np.abs((maxforce - prev_maxforce) / prev_maxforce)
        energy = system.analysis.energy()["total"]
        relener = np.abs((energy - prev_energy) / prev_energy)
        if i > 1 and (i + 1) % 4 == 0:
            print(
                f"minimization step: {(i + 1) * sd_params['emstep']:4.0f}"
                f"    max. rel. force change:{relforce:+3.3e}"
                f"    rel. energy change:{relener:+3.3e}"
            )
        if relforce < sd_params["f_tol"] or relener < sd_params["e_tol"]:
            break
        i += 1

    system.integrator.set_vv()  # switch back to velocity Verlet (default integrator)


def warmup(system, warm_n_times, warm_steps, dir_name_var, TUNE_SET, TUNE_SKIN_PARAM):
    print("Warmup integration")  # it appears just the first time the function is called

    energies_tot_warm = np.zeros((warm_n_times, 2))

    i = 0
    while i < warm_n_times:
        if (i == TUNE_SET["i_val_1"] or i == TUNE_SET["i_val_2"]) and TUNE_SET[
            "tune_bool"
        ]:
            system.cell_system.tune_skin(**TUNE_SKIN_PARAM)
        system.integrator.run(warm_steps)  # Default: velocity Verlet algorithm
        print("\r\trun %d at time=%.0f " % (i, system.time), end="")
        energies_tot_warm[i] = (system.time, system.analysis.energy()["total"])
        i += 1

    print("\nEnd warmup")

    # save energy
    string1 = dir_name_var + "/TotEner_warmup.dat"
    np.savetxt(
        string1,
        np.column_stack((energies_tot_warm[:, 0], energies_tot_warm[:, 1])),
        fmt="%.5e",
        delimiter="\t",
    )


def warmup_chkpnt(
    system,
    warm_n_times,
    warm_steps,
    dir_name_var,
    TUNE_SET,
    TUNE_SKIN_PARAM,
    checkpoint,
    CHECKPOINT_PERIOD,
    iter_warmup,
):
    """
    warmup with checkpointing
    """
    print("Warmup integration")  # it appears just the first time the function is called

    energies_tot_warm = np.zeros((warm_n_times, 2))

    pbar = tqdm(desc="Warmup loop", total=warm_n_times)
    while iter_warmup < warm_n_times:
        if iter_warmup % CHECKPOINT_PERIOD == 0:
            checkpoint.save()
        if (
            iter_warmup == TUNE_SET["i_val_1"] or iter_warmup == TUNE_SET["i_val_2"]
        ) and TUNE_SET["tune_bool"]:
            system.cell_system.tune_skin(**TUNE_SKIN_PARAM)
        system.integrator.run(warm_steps)  # Default: velocity Verlet algorithm
        print("\r\trun %d at time=%.0f " % (iter_warmup, system.time), end="")
        energies_tot_warm[iter_warmup] = (
            system.time,
            system.analysis.energy()["total"],
        )
        iter_warmup += 1

        pbar.update(1)

    pbar.close()

    print("\nEnd warmup")

    # save energy
    string1 = dir_name_var + "/TotEner_warmup.dat"
    np.savetxt(
        string1,
        np.column_stack((energies_tot_warm[:, 0], energies_tot_warm[:, 1])),
        fmt="%.5e",
        delimiter="\t",
    )


def main_integration(
    system,
    int_n_times,
    int_steps,
    energies_tot,
    energies_kin,
    energies_nonbon,
    energies_bon,
    energies_coul,
    counter_energy=0,
):
    """
    main_integration function perform the main integartion for the calculation of observables

    system: system instance
    int_steps: integration steps per call of integrator
    int_n_times: number of iterations within correlated configurations (# of calls of integrator)
    energy_*: matrices to store the different energies -np.zeros((int_n_times*int_uncorr_times, 2))-
    counter_energy = counter of calls of main_integration function
    """
    print("Main Integration")
    for i in range(int_n_times):
        print("\rrun %d at time=%.0f " % (i, system.time), end="")
        system.integrator.run(int_steps)  # Default: velocity Verlet algorithm
        energies_tot[counter_energy] = (system.time, system.analysis.energy()["total"])
        energies_kin[counter_energy] = (
            system.time,
            system.analysis.energy()["kinetic"],
        )
        energies_nonbon[counter_energy] = (
            system.time,
            system.analysis.energy()["non_bonded"],
        )
        energies_bon[counter_energy] = (system.time, system.analysis.energy()["bonded"])
        energies_coul[counter_energy] = (
            system.time,
            system.analysis.energy()["coulomb"],
        )
        counter_energy += 1

    return counter_energy


def initialize_elec(system, P3M_PARAMS):
    print("Define electrostatic interactions")
    solver = P3M(**P3M_PARAMS)
    system.electrostatics.solver = solver
