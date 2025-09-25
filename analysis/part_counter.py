import math


def count_interior_ions(system, part_type, R_mic, x_center=-1, y_center=-1, z_center=-1):
    """
        Counting of how many particles of type part_type there are inside a sphere of
        radius R_mic with center at (x_center, y_center, z_center)

        system: system class
        part_type: type of particles of interest
        R_mic: microgel radius
        x_center, y_center, z_center: coordinates of the microgel COM
    """

    if x_center==-1 and y_center==-1 and z_center==-1:
        x_center = system.box_l[0]/2
        y_center = system.box_l[1]/2
        z_center = system.box_l[2]/2
    
    counter = 0
    for part in system.part.select(type=part_type):
        x_part = part.pos_folded[0]
        y_part = part.pos_folded[1]
        z_part = part.pos_folded[2]
        dist = math.sqrt((x_part - x_center)**2 + (y_part - y_center)**2 + (z_part - z_center)**2)
        if dist <= R_mic:
            counter += 1
    
    return counter


def count_interior_ions_weighted(system, part_type, R_mic, x_center=-1, y_center=-1, z_center=-1):
    """
        Weighted count of particles of type part_type inside a sphere of
        radius R_mic with center at (x_center, y_center, z_center)

        system: system class
        part_type: type of particles of interest
        R_mic: microgel radius
        x_center, y_center, z_center: coordinates of the microgel COM
    """

    if x_center==-1 and y_center==-1 and z_center==-1:
        x_center = system.box_l[0]/2
        y_center = system.box_l[1]/2
        z_center = system.box_l[2]/2
    
    counter = 0
    for part in system.part.select(type=part_type):
        x_part = part.pos_folded[0]
        y_part = part.pos_folded[1]
        z_part = part.pos_folded[2]
        dist = math.sqrt((x_part - x_center)**2 + (y_part - y_center)**2 + (z_part - z_center)**2)
        if dist <= R_mic:
            counter += dist**2
    
    return counter