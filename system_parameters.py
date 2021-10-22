import numpy as np
# system parameters: reduced units
eps = 1.0                   # energy: amplitude epsilon (eps) from the Weeks-Chandler-Andersen potential
sgm = 1.0                   # length: particle diameter sigma (sgm)
m = 1.0                     # mass: particle mass m
Q_E = 1.0                   # elenetary charge
# time: derived quantity
box_l = 15                 # box size
dt = 0.01                   # time step
skin = 0.4                  # skin of Verlet list


Nbeads_arm = 5
cell_unit = Nbeads_arm + 1