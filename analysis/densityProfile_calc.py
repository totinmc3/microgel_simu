import espressomd.observables
import math
import numpy as np

def define_density_obs(system,  particle_types, N_bins: int):
    """
        The function initializes the DensityProfile observable
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

    return density_profile

def particle_density_profile(system, particle_types, N_bins: int):
    """
        The funtion calculates the density profiles for particles of type in @particle_types
        binning the space in @N_bins in the three axes.
    """

    density_profile = define_density_obs(system,  particle_types, N_bins)
        
    obs_data = density_profile.calculate()
    obs_bins = density_profile.bin_centers()

    return obs_data, obs_bins


def profile_spher_transf(averaged_profile, Nbins, box_l):
    """
        Transformation of the cartesian density profile to (rotation invariant) spherical 
        profile with center at the simulation box.

        averaged_profile: cartesian profile - averaged_profile[i, j, k] -
        Nbins: number of bins in r component 
        box_l: box_size
    """

    intensities = np.zeros(Nbins)
    half_box_l = box_l / 2
    box_part = len(averaged_profile) # unit of bin size
    r = np.linspace(0, half_box_l, Nbins) # unit of sigma
    bin_size = r[1] - r[0] # unit of sigma
    box_part2 = box_part/2 # unit of bin size
    for i in range(box_part):
        for j in range(box_part):
            for k in range(box_part):
                dist = math.sqrt((i - box_part2)**2 + (j - box_part2)**2 + (k - box_part2)**2) # unit of bin size
                if dist < box_part2:
                    intensities[math.floor(dist)] += averaged_profile[i, j, k]

    return intensities, r


def microgel_radius(polym_prof, r, cutoff=0.01):
    """
        Determination of microgel radius based on cutoff criterion.

        polym_prof: polymer density profile
        r: bining of the profile
        cutoff: intensity cutoff criterion to determine microgel radius
    """

    max_val = np.max(polym_prof)
    polym_prof_copy = np.copy(polym_prof)
    polym_prof_copy [:np.argmax(polym_prof)]= max_val
    filtered_prof = polym_prof_copy - cutoff
    profile_sign = np.trim_zeros(np.sign(filtered_prof)+1)
    size_index = len(profile_sign) # this is the entry index in the profile vectors that corresponds to the microgel equilibrium radius, according to cut_criterion
    R_mic = r[size_index] # microgel radius

    return R_mic, size_index
