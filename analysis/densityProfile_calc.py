import espressomd.observables

def particle_density_profile(system,  particle_types, N_bins: int):
    """
        The funtion calculates the density profiles for particles of type in @particle_types
        binning the space in @N_bins in the three axes.
    """

    id_list = []
    if isinstance(particle_types,int):
        for part in system.part[:]:
            if part.type == particle_types:
                id_list.append(part.id)
    else:
        for part in system.part[:]:
            for particle_type in particle_types:
                if part.type == particle_type:
                    id_list.append(part.id)

    density_profile = espressomd.observables.DensityProfile(
        ids=id_list,
        n_x_bins=N_bins, min_x=0.0, max_x=system.box_l[0],
        n_y_bins=N_bins, min_y=0.0, max_y=system.box_l[0],
        n_z_bins=N_bins, min_z=0.0, max_z=system.box_l[0])
        
    obs_data = density_profile.calculate()
    obs_bins = density_profile.bin_centers()

    return obs_data, obs_bins