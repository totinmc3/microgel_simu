import numpy as np
# system parameters: reduced units
eps = 1.0                   # energy: amplitude epsilon (eps) from the Weeks-Chandler-Andersen potential
sgm = 1.0                   # length: particle diameter sigma (sgm)
m = 1.0                     # mass: particle mass m
Q_E = 1.0                   # elenetary charge
# time: derived quantity
box_l = 100                  # box size
dt = 0.01                   # time step
skin = 0.4                  # skin of Verlet list


Nbeads_arm = 8
cell_unit = 4 * (Nbeads_arm + 1) / np.sqrt(3)

# Interaction and bonds:
# FENE bond
r_inf = 1.8                              # max extention
FENE_BOND_PARAMS = {'k': 30,             # FENE constant
                    'd_r_max' : r_inf,   # max extention
                    'r_0' : 0.0}

# Particle types
PART_TYPE = {'crosslinker' : 0,
            'polymer_arm' : 1}

# Langevin thermostat parameters
kBT = 1.0                       # temperature
LANGEVIN_PARAMS = {'kT': kBT,
                   'gamma': 1.0,    # gamma thermostat
                   'seed': int(10**6*np.random.random())}      # seed
                   
# Non-bonded interactions
# in WCA pot: cutoff=2^(1/6)*sigma and  shift="auto"
NONBOND_WCA_PARAMS = {'epsilon' :eps,
                      'sigma' : sgm}