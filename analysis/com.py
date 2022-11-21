import numpy as np

def com_calculation(system, *types):
    """
        Calculation of center of mass for a group of atoms fo different types
    """

    com = np.zeros((3), dtype=float)
    total_number_part = 0
    for type in types:
        N_type = len(system.part.select(type=type))
        if N_type==0:
            com_type = np.array([0, 0, 0])
        else:
            com_type = np.array(system.analysis.center_of_mass(p_type=type))

        com += N_type * com_type
        total_number_part += N_type

    return com/total_number_part