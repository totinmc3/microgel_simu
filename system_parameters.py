import numpy as np
# system parameters: reduced units
eps = 1.0                   # energy: amplitude epsilon (eps) from the Weeks-Chandler-Andersen potential
sgm = 1.0                   # length: particle diameter sigma (sgm)
m = 1.0                     # mass: particle mass m
Q_E = 1.0                   # elenetary charge
conversion_factor = 6.02214/10 # If the density is in molar, multiplication by this factor gives density in number of particles per nm^3
sigma = 0.355               # sigma in nm for molar units
# time: derived quantity
box_l = 400                 # box size
dt = 0.01                   # time step
skin = 0.4                  # skin of Verlet list
pKa = 4.0


Nbeads_arm = 10
cell_unit = 4 * (Nbeads_arm + 1) / np.sqrt(3)
N_cat = 0   # number of cationic beads in microgel network
N_an = 10   # number of anionic beads in microgel network

c_salt_molar = 0.001    # salt concentration in molar
c_salt = c_salt_molar * conversion_factor * sigma**3      # salt-ion-pair concentration in units of sigma^3

# Interaction and bonds:
# FENE bond
r_inf = 1.8                              # max extention
FENE_BOND_PARAMS = {'k': 30,             # FENE constant
                    'd_r_max' : r_inf,   # max extention
                    'r_0' : 0.0}

# Particle types
PART_TYPE = {'crosslinker' : 0,
            'polymer_arm' : 1,
            'cation' : 2,
            'anion' : 3,
            'ion_cat' : 4,
            'ion_an' : 5,
            'salt_cat': 6,
            'salt_an': 7,
            'OH': 8}

# Langevin thermostat parameters
kBT = 1.0                       # temperature
LANGEVIN_PARAMS = {'kT': kBT,
                   'gamma': 1.0,    # gamma thermostat
                   'seed': int(10**6*np.random.random())}      # seed
                   
# Non-bonded interactions
# in WCA pot: cutoff=2^(1/6)*sigma and  shift="auto"
NONBOND_WCA_PARAMS = {'epsilon' :eps,
                      'sigma' : sgm}

# Steepest decendent parameters
STEEPEST_DESCENT_PARAMS = {'f_tol': 1e-2,
                           'e_tol': 1e-5,
                           'damping': 30,
                           'max_steps': 10000,
                           'max_displacement': 0.01,
                           'emstep': 10}

# Electrostatic
Bjerrum_length = 2.0*sgm    # Bjerrum length
P3M_PARAMS = {'prefactor': kBT * Bjerrum_length,
              'accuracy': 1e-3}

# Warmup integration
warm_steps = int(1.0/dt)
warm_n_times = 5
warmup_loop = 1                     # number of warmup function iteractions (= calls)
warmup_counter = 0                  # counter of warmup function calls
energies_tot_warm_bool = True       # calculation of warmup energy

# Integration
int_steps = int(1.0/dt)     # integration steps (chosen such that after int_steps, one unit of time is achieved)
int_n_times = 5         # number of iterations within correlated configurations
int_uncorr_times = 1       # number of iterations for uncorrelated configurations

TUNE_SET = {'tune_bool' : True,
             'i_val_1' : 15000,
             'i_val_2' : 80000}
             
TUNE_SKIN_PARAM = {'min_skin' : 0.1, 
                   'max_skin': 1.0, 
                   'tol' : 0.05, 
                   'int_steps' : 50*int_steps, 
                   'adjust_max_skin' : True}

# Density profiles
ION_PROFILE_BOOL = True
N_bins = int(box_l)

# Checkpointing
CHECK_NAME = "chk-res"
CHECKPOINT_PERIOD = 25
