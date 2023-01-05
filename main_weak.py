from __future__ import print_function
import numpy as np
import sys

import espressomd
from espressomd import code_info
from espressomd import analyze
from espressomd import integrate
from espressomd import reaction_ensemble
from espressomd import electrostatics
from espressomd import diamond
from espressomd.interactions import *
from scipy import interpolate
from espressomd import minimize_energy
from espressomd.io.writer import vtf 

from statistic import *
import csv

import gzip
import pickle
import os
import time

required_features = ["ELECTROSTATICS", "EXTERNAL_FORCES", "LENNARD_JONES"]
espressomd.assert_features(required_features)

# print help message if proper command-line arguments are not provided
if (len(sys.argv) != 8):
    print("\nGot ", str(len(sys.argv) - 1), " arguments, need 7\n\nusage:" +
          sys.argv[0] + " final_box_l MPC Kcideal_in_mol_per_l cs_bulk bjerrum pH run_id\n")
    sys.exit()

# System parameters
#############################################################



#example call pypresso weak-gel.py 20 39 1e-4 0.00269541778 2 1e-9 0 #Note that the scripts becomes more unstable as the number of monomers per chain become smaller (MPC)
final_box_l=float(sys.argv[1]) #20
MPC=int(sys.argv[2]) #39
Kcideal_in_mol_per_l=float(sys.argv[3]) #1e-4
cs_bulk=float(sys.argv[4]) #0.00269541778 #in units of 1/sigma**3, sigma=3.55Angstrom #0.00269541778/sigma**3=0.1 mol/l
bjerrum=float(sys.argv[5]) #2.0
pH=float(sys.argv[6]) #pH in bulk in mol/l from 1 to 13 works good
run_id=int(sys.argv[7])

conversion_factor_from_1_per_sigma_3_to_mol_per_l=37.1
temperature = 1.0


ionic_strength, excess_chemical_potential_monovalent_pairs_in_bulk_data, bjerrums,excess_chemical_potential_monovalent_pairs_in_bulk_data_error =np.loadtxt("../excess_chemical_potential.dat", unpack=True) #remember, excess chemical potential does not know about types
excess_chemical_potential_monovalent_pairs_in_bulk=interpolate.interp1d(ionic_strength, excess_chemical_potential_monovalent_pairs_in_bulk_data)


Kw=10**-14 #dimensionless dissociation constant Kw=relative_activity(H)*relative_activity(OH)
cref_in_mol_per_l=1.0 #in mol/l

def determine_bulk_concentrations_selfconsistently(cH_bulk_in_mol_per_l, cs_bulk):
    global Kw, cref_in_mol_per_l, conversion_factor_from_1_per_sigma_3_to_mol_per_l
    #calculate initial guess for concentrations
    cOH_bulk=(Kw/(cH_bulk_in_mol_per_l/cref_in_mol_per_l))*cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l
    cH_bulk=cH_bulk_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l
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

        
    #ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
    #print("before self consistent concentration calculation cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk", cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk)
    #print("check bef: ",Kw, cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/temperature)*conversion_factor_from_1_per_sigma_3_to_mol_per_l**2)
    #print("electro neutrality bulk before", cNa_bulk+cH_bulk-cOH_bulk-cCl_bulk) 
    
    
    def calculate_concentrations_self_consistently(cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk):
        global temperature, max_self_consistent_runs, self_consistent_run
        if(self_consistent_run<max_self_consistent_runs):
            self_consistent_run+=1
            ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
            cOH_bulk=(Kw/(cH_bulk_in_mol_per_l/cref_in_mol_per_l)*np.exp(-(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk))/temperature))*cref_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l
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

#calculate concentrations self consistently
max_self_consistent_runs=200
self_consistent_run=0
cH_bulk_in_mol_per_l=10**(-pH)*cref_in_mol_per_l #this is a guess, which is used as starting point of the self consistent optimization
cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk =determine_bulk_concentrations_selfconsistently(cH_bulk_in_mol_per_l, cs_bulk)
ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
determined_pH=-np.log10(cH_bulk*conversion_factor_from_1_per_sigma_3_to_mol_per_l/cref_in_mol_per_l*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(2.0*temperature)))
while abs(determined_pH-pH)>1e-6:
    if(determined_pH)>pH:
        cH_bulk_in_mol_per_l=cH_bulk_in_mol_per_l*1.005
    else:
        cH_bulk_in_mol_per_l=cH_bulk_in_mol_per_l/1.003
    cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk =determine_bulk_concentrations_selfconsistently(cH_bulk_in_mol_per_l, cs_bulk)
    ionic_strength_bulk=0.5*(cNa_bulk+cCl_bulk+cOH_bulk+cH_bulk) #in units of 1/sigma^3
    determined_pH=-np.log10(cH_bulk*conversion_factor_from_1_per_sigma_3_to_mol_per_l*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/(2.0*temperature)))
    self_consistent_run=0

def check_concentrations():
    if(abs(Kw-cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/temperature)*conversion_factor_from_1_per_sigma_3_to_mol_per_l**2/cref_in_mol_per_l**2)>1e-15):
        raise RuntimeError("Kw incorrect")
    if(abs(cNa_bulk+cH_bulk-cOH_bulk-cCl_bulk)>1e-14):
        raise RuntimeError("bulk is not electroneutral")
    if(abs(pH-determined_pH)>1e-5):
        raise RuntimeError("pH is not compatible with ionic strength and bulk H+ concentration")
    if(abs(cs_bulk-min(cNa_bulk, cCl_bulk))>1e-14):
        raise RuntimeError("bulk salt concentration is not correct")
    if(abs(pH-7)<1e-14):
        if((cH_bulk/cOH_bulk-1)>1e-5):
            raise RuntimeError("cH and cOH need to be symmetric at pH 7")
        
print("after self consistent concentration calculation: cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk", cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk)
print("check KW: ",Kw, cOH_bulk*cH_bulk*np.exp((excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) )/temperature)*conversion_factor_from_1_per_sigma_3_to_mol_per_l**2/cref_in_mol_per_l**2)
print("check electro neutrality bulk after", cNa_bulk+cH_bulk-cOH_bulk-cCl_bulk) #note that charges are neutral up to numerical precision. femto molar charge inequalities are not important in the bulk.
print("check pH: input", pH, "determined pH", determined_pH)
print("check cs bulk: input", cs_bulk, "determinde cs_bulk", min(cNa_bulk, cCl_bulk))
print("check cH_bulk/cOH_bulk:", cH_bulk/cOH_bulk)
check_concentrations()


# Integration parameters
#############################################################
simulation_parameters=[final_box_l,MPC,Kcideal_in_mol_per_l, cs_bulk, bjerrum, pH, cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, ionic_strength_bulk ]

system = espressomd.System(box_l=[50.0, 50.0, 50.0])
system.set_random_state_PRNG()
#system.seed = system.cell_system.get_state()['n_nodes'] * [1234]
np.random.seed(seed=np.random.randint(2**30-1))

system.time_step = 0.01
system.cell_system.skin = 0.4

system.thermostat.set_langevin(kT=temperature, gamma=1.0)
system.cell_system.max_num_cells = 2744


#############################################################
#  Setup System                                             #
#############################################################

# Particle setup
#############################################################
type_H=0
type_A=1
type_Cl=2
type_Na=3
type_constraint=4
type_HA=5
type_OH=6
types=[type_H, type_A, type_Cl,type_Na, type_constraint, type_HA, type_OH]
charges_of_types={type_H:1, type_A:-1, type_Cl:-1, type_Na:1}

lj_eps = 1.0
lj_sig = 1.0
lj_cut = 2**(1.0 / 6) * lj_sig
for type_1 in types:
    for type_2 in types:
        system.non_bonded_inter[type_1, type_2].lennard_jones.set_params(
            epsilon=lj_eps, sigma=lj_sig,
            cutoff=lj_cut, shift="auto")

def calculate_current_end_to_end_distance():
    current_end_to_end_distances=[]
###code for identifying neighbouring nodes, intended to run with: pypresso weak-gel.py 45 20 1e-4 0.001 2 1e-4    (due to magic number 18.9 below)
#    connectivity_pairs=[]
#    for i in range(8):
#        for j in range(8):
#            if(np.abs(np.sqrt(system.analysis.min_dist2(system.part[i].pos, system.part[j].pos))-18.9)<0.001):
#                print(i,j, np.sqrt(system.analysis.min_dist2(system.part[i].pos, system.part[j].pos)))
#                if(i<j):
#                    connectivity_pairs.append({i:j})
#                else:
#                    connectivity_pairs.append({j:i})
#                current_end_to_end_distances.append(np.sqrt(system.analysis.min_dist2(system.part[i].pos, system.part[j].pos)))
#    connectivity_pairs=np.unique(connectivity_pairs)
#    print(connectivity_pairs, len(connectivity_pairs))

    connectivity_pairs=[{0: 1},{0: 5},{0: 6}, {0: 7}, {1: 2}, {1: 3}, {1: 4}, {2: 5}, {2: 6}, {2: 7}, {3: 5}, {3: 6}, {3: 7}, {4: 5}, {4: 6}, {4: 7}] #describes between which nodes there are connecting chains
    for pair in connectivity_pairs:
        key,val=pair.keys()[0], pair.values()[0]
        current_end_to_end_distances.append(np.sqrt(system.analysis.min_dist2(system.part[key].pos, system.part[val].pos)))
    return np.mean(current_end_to_end_distances)


############
#setup weak gel

#diamond uses bond with id 0
fene = FeneBond(k=30.0, d_r_max=1.5, r_0=0)
system.bonded_inter.add(fene)

# length for Kremer-Grest chain
bond_length = 0.9

# The physical distance beween nodes such that a line of monomers "fit" needs to be worked out.
# This is done via the unit diamond lattice size parameter "a".
a = (MPC + 1) * bond_length / (0.25 * np.sqrt(3))

# Lastly, the created periodic connections requires a specific simulation box.
system.box_l = [a, a, a]

# We can now call diamond to place the monomers, crosslinks and bonds.
diamond.Diamond(a=a, bond_length=bond_length, MPC=MPC)
system.part[:].type=type_HA
num_gel_particles=len(system.part[:].type)
gel_ids=range(0,num_gel_particles)
print("number of gel particles", num_gel_particles)
bonds_from=[]
bonds_to=[]
for i in range(num_gel_particles):
    for j in range(len(system.part[i].bonds)):
        bonds_from.append(i)
        bonds_to.append(system.part[i].bonds[j][1])

def calculate_average_bond_length(bonds_from,bonds_to):
    current_bond_lengths=[]
    for i,j in zip(bonds_from, bonds_to):
        current_bond_lengths.append(np.sqrt(system.analysis.min_dist2(system.part[i].pos_folded, system.part[j].pos_folded)))
    return np.mean(current_bond_lengths)

# because stretched polymers are not too impressive...
print("simulating a slow compression...")
step_size_box_l_compression=1 # in units of sigma
steps_needed_until_we_have_final_compression=int((a-final_box_l)/step_size_box_l_compression)
energy_minimizer= minimize_energy.MinimizeEnergy(f_max = 2.0, gamma= 10.0, max_steps= 1e8,max_displacement=0.05)
for d in np.arange(0, steps_needed_until_we_have_final_compression):
    system.change_volume_and_rescale_particles(
        d_new=system.box_l[0] - step_size_box_l_compression, dir='xyz')
    print("box now at ", system.box_l)
    system.integrator.run(steps=1000)
    print(system.analysis.min_dist())  
#    energy_minimizer.minimize()
system.change_volume_and_rescale_particles(d_new=final_box_l, dir='xyz')

outfile = open('diamond_before_warmup_'+str(final_box_l)+'.vtf', 'w')
vtf.writevsf(system, outfile)
vtf.writevcf(system, outfile)
outfile.close()

###########################################
# set up reactions

def MC_swap_A_HA_particles(type_HA, type_A):
    As=system.part.select(type=type_A)
    HAs=system.part.select(type=type_HA)
    ids_A=As.id
    ids_HA=HAs.id
    if(len(ids_A)>0 and len(ids_HA)>0):
        #choose random_id_A, choose_random_id_HA
        random_id_A=ids_A[np.random.randint(0,len(ids_A))]
        random_id_HA=ids_HA[np.random.randint(0,len(ids_HA))]
        
        old_energy=system.analysis.energy()["total"]
        #adapt type and charge
        system.part[random_id_A].type=type_HA
        system.part[random_id_A].q=0
        system.part[random_id_HA].type=type_A
        system.part[random_id_HA].q=-1
        new_energy=system.analysis.energy()["total"]
        #apply metropolis criterion, accept or reject based on energetic change
        if(np.random.random()<min(1,np.exp(-(new_energy-old_energy)/temperature))):
            #accept
            pass            
        else:
            #reject
            system.part[random_id_A].type=type_A
            system.part[random_id_A].q=-1
            system.part[random_id_HA].type=type_HA
            system.part[random_id_HA].q=0

RE = reaction_ensemble.ReactionEnsemble(temperature=temperature, exclusion_radius=0.9)
#coupling to the Na, OH, Cl, reservoir
RE.add_reaction(gamma=cNa_bulk*cCl_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature), reactant_types=[], reactant_coefficients=[], product_types=[type_Na, type_Cl], product_coefficients=[1, 1], default_charges={type_Na: 1, type_Cl: -1})
RE.add_reaction(gamma=cOH_bulk/cCl_bulk, reactant_types=[type_Cl], reactant_coefficients=[1], product_types=[type_OH], product_coefficients=[1], default_charges={type_OH: -1, type_Cl: -1})
RE.add_reaction(gamma=cH_bulk/cNa_bulk, reactant_types=[type_Na], reactant_coefficients=[1], product_types=[type_H], product_coefficients=[1], default_charges={type_Na: 1, type_H: 1})

RE.add_reaction(gamma=cNa_bulk*cOH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature), reactant_types=[], reactant_coefficients=[], product_types=[type_Na, type_OH], product_coefficients=[1, 1], default_charges={type_Na: 1, type_OH: -1})
RE.add_reaction(gamma=cH_bulk*cCl_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature), reactant_types=[], reactant_coefficients=[], product_types=[type_H, type_Cl], product_coefficients=[1, 1], default_charges={type_H: 1, type_Cl: -1})
RE.add_reaction(gamma=cOH_bulk*cH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature), reactant_types=[], reactant_coefficients=[], product_types=[type_H, type_OH], product_coefficients=[1,1], default_charges={type_OH: -1, type_H: 1})

#implementing the reactions
##HA <-> A- + H+
RE.add_reaction(gamma=Kcideal_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l ,reactant_types=[type_HA], reactant_coefficients=[1], product_types=[type_A, type_H], product_coefficients=[1, 1], default_charges={type_HA: 0, type_H: 1, type_A: -1})
##HA+ OH- <-> A-
RE.add_reaction(gamma=(Kcideal_in_mol_per_l/Kw)*conversion_factor_from_1_per_sigma_3_to_mol_per_l ,reactant_types=[type_HA, type_OH], reactant_coefficients=[1,1], product_types=[type_A], product_coefficients=[1], default_charges={type_HA: 0, type_OH: -1, type_A: -1}) #cH*cA/cHA /(cOH*cH)=cA/(cOH*cHA)
## HA <-> A- +Na+
#Kcideal_in_mol_per_l*K_NaOH/Kw
K_NaOH=cNa_bulk*cOH_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature)
Kw_cref_squared_in_1_div_sigma_to_power_six=Kw*cref_in_mol_per_l**2/conversion_factor_from_1_per_sigma_3_to_mol_per_l**2
RE.add_reaction(gamma=Kcideal_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l*(K_NaOH/Kw_cref_squared_in_1_div_sigma_to_power_six) ,reactant_types=[type_HA], reactant_coefficients=[1], product_types=[type_A, type_Na], product_coefficients=[1, 1], default_charges={type_HA: 0, type_Na: 1, type_A: -1})
## HA + Cl- <-> A-
#Kcideal_in_mol_per_l/K_HCl
K_HCl=cH_bulk*cCl_bulk * np.exp(excess_chemical_potential_monovalent_pairs_in_bulk(ionic_strength_bulk) / temperature)
RE.add_reaction(gamma=Kcideal_in_mol_per_l/conversion_factor_from_1_per_sigma_3_to_mol_per_l*1/K_HCl ,reactant_types=[type_HA, type_Cl], reactant_coefficients=[1,1], product_types=[type_A], product_coefficients=[1], default_charges={type_HA: 0, type_Cl: -1, type_A: -1})

system.setup_type_map(types)
RE.set_volume(np.prod(system.box_l))
print(RE.get_status())

def reaction(steps):
    global type_HA, type_A, MPC
    RE.reaction(int(steps))
    for k in range(int(MPC/5.0)):
        MC_swap_A_HA_particles(type_HA, type_A)

#setup widom insertion
Widom = reaction_ensemble.WidomInsertion(temperature=temperature, exclusion_radius=0.0)
Widom.add_reaction(gamma=np.nan, reactant_types=[], reactant_coefficients=[], product_types=[
                type_Na, type_Cl], product_coefficients=[1, 1], default_charges={type_Cl: -1, type_Na: +1})
Widom.add_reaction(gamma=np.nan, reactant_types=[], reactant_coefficients=[], product_types=[
                type_Na], product_coefficients=[1], default_charges={type_Na: +1},check_for_electroneutrality=False)                
Widom.add_reaction(gamma=np.nan, reactant_types=[], reactant_coefficients=[], product_types=[
                type_Cl], product_coefficients=[1], default_charges={type_Cl: -1},check_for_electroneutrality=False)   

reaction(10000+16*2*MPC)

def create_particle_in_safe_dist(part_type, part_charge):
    min_dist=2**(1.0/6.0) #in units of sigma
    # add particle after last known particle
    pos = system.box_l[0]*np.random.random(3)
    part=system.part.add(pos=pos, type=part_type, q=part_charge)
    while(system.analysis.dist_to(id=part.id)<min_dist*1.1):
        pos = system.box_l[0]*np.random.random(3)
        system.part[part.id].pos=pos
    return part.id

#add salt for p3m tuning, this is removed automatically by grandcanonical moves if the salt needs to be removed
for i in range(16*MPC+int(0.6*cs_bulk*final_box_l**3)):
    create_particle_in_safe_dist(type_Cl, -1)
    create_particle_in_safe_dist(type_Na, 1)

tuned_skin=system.cell_system.tune_skin(min_skin=1.0, max_skin=1.6, tol=0.05, int_steps=200)
print("tuned_skin", tuned_skin)

system.integrator.run(steps=1000)

p3m = electrostatics.P3M(prefactor=bjerrum*temperature, accuracy=1e-3)
system.actors.add(p3m)
p3m_params = p3m.get_params()
for key in list(p3m_params.keys()):
    print("{} = {}".format(key, p3m_params[key]))

tuned_skin=system.cell_system.tune_skin(min_skin=1.0, max_skin=1.6, tol=0.05, int_steps=200)
print("tuned_skin", tuned_skin)

# Warmup
#############################################################
# warmup integration (with capped LJ potential)
warm_n_times = 500 #XXX 100
# do the warmup until the particles have at least the distance min_dist
# set LJ cap
lj_cap = 50
system.force_cap = lj_cap
system.time_step = 0.01


RE.reaction(80000)

# Warmup Integration Loop
i = 0

start_warmup = time.clock() 
while (i < warm_n_times):
    print(i, "warmup")
    reaction(MPC*20+300)
    system.integrator.run(steps=1000+MPC*2)
    print("mindist", system.analysis.min_dist(), "N_tot", len(system.part[:].q))    
    i += 1
    #Increase LJ cap
    lj_cap = lj_cap + 10
    system.force_cap = lj_cap
end_warmup=time.clock()
elapsed_time_in_minutes = (end_warmup - start_warmup)/60.0 #in seconds
time_per_cyle=elapsed_time_in_minutes/warm_n_times #in minutes/cycle
nr_of_cylces_in_15_minutes=int((15.-5)/time_per_cyle) #minus 5 minute to make sure the last checkpoint is written before the time slot on bee ends
print("nr_of_cylces_in_15_minutes", nr_of_cylces_in_15_minutes)

# remove force capping
system.force_cap = 0
system.time_step = 0.01
#MC warmup
reaction(5*MPC*16)
RE.reaction(120000)

#tuned_skin=system.cell_system.tune_skin(min_skin=1.0, max_skin=1.6, tol=0.05, int_steps=200)
#print("tuned_skin", tuned_skin)

##retune electrostatics after warmup
#p3m.tune(accuracy=1e-3)

outfile = open('diamond_after_warmup_'+str(final_box_l)+'.vtf', 'w')
vtf.writevsf(system, outfile)
vtf.writevcf(system, outfile)
outfile.close()

end_to_end_distances=[]
num_Hs= []
num_OHs= []
num_Nas= []
num_Cls= []
num_As=[]
total_isotropic_pressures=[]
end_to_end_distances=[]
total_stress_tensors=[]
kinetic_stress_tensors=[]
coulomb_stress_tensors=[]
bonded_stress_tensors=[]
nonbonded_stress_tensors=[]
bond_lengths=[]
gel_charges=[]

if(os.path.exists("checkpoint_"+str(final_box_l)+".pgz")):
    #try to load safed statistical data
    with gzip.GzipFile("checkpoint_"+str(final_box_l)+".pgz", 'rb') as fcheck:
        print("loading checkpoint")
        data = pickle.load(fcheck)
        num_Hs=data[0]
        num_OHs=data[1]
        num_Nas=data[2]
        num_Cls=data[3]
        num_As=data[4]
        total_isotropic_pressures=data[5]
        stress_tensor=data[6]
        total_stress_tensors=data[7]
        kinetic_stress_tensors=data[8]
        coulomb_stress_tensors=data[9]
        bonded_stress_tensors=data[10]
        nonbonded_stress_tensors=data[11]
        gel_charges=data[12]
        end_to_end_distances=data[13]
        bond_lengths=data[14]

filename="observables_run_"+str(run_id)+"box_l_"+str(final_box_l)
np.savetxt(filename+"_bonds.out", np.c_[bonds_from,bonds_to])

scalar_observables=[num_Hs, num_OHs, num_Nas, num_Cls, num_As, total_isotropic_pressures, end_to_end_distances, bond_lengths] #collect references to scalar observables

box_l=final_box_l
volume=final_box_l**3
for i in range(10000):
    N_steps=len(num_Hs)+1
    for k in range(2):
        Widom.measure_excess_chemical_potential(0)  # 0 for insertion reaction
        Widom.measure_excess_chemical_potential(2)  #insertion Na
        Widom.measure_excess_chemical_potential(4)  #insertion Cl
    reaction(MPC*20+300)
    system.integrator.run(steps=1000+MPC*2) #XXX 1000
    num_Hs.append(system.number_of_particles(type=type_H))
    num_OHs.append(system.number_of_particles(type=type_OH))
    num_Nas.append(system.number_of_particles(type=type_Na))
    num_Cls.append(system.number_of_particles(type=type_Cl))
    num_As.append(system.number_of_particles(type=type_A))
    total_isotropic_pressures.append(system.analysis.pressure()["total"])
    stress_tensor=system.analysis.stress_tensor()
    total_stress_tensors.append(stress_tensor["total"])
    kinetic_stress_tensors.append(stress_tensor["kinetic"])
    coulomb_stress_tensors.append(stress_tensor["coulomb"])
    bonded_stress_tensors.append(stress_tensor["bonded"])
    nonbonded_stress_tensors.append(stress_tensor["non_bonded"])
    
    gel_charges.append(system.part[0:num_gel_particles].q)
    end_to_end_distances.append(calculate_current_end_to_end_distance())
    
    bond_lengths.append(calculate_average_bond_length(bonds_from, bonds_to))

    if(i % nr_of_cylces_in_15_minutes == 0):
        
        scalar_observables_means=[]
        scalar_observables_errors=[]
        for scalar_observable_i in range(len(scalar_observables)):
            mean_scalar_observable, correlation_corrected_error =calc_error(scalar_observables[scalar_observable_i])
            scalar_observables_means.append(mean_scalar_observable)
            scalar_observables_errors.append(correlation_corrected_error)

        #save degree_of_dissociation
        degree_of_dissociation=scalar_observables_means[4]/(16.0*MPC+8)
        err_degree_of_dissociation=scalar_observables_errors[4]/(16.0*MPC+8)
        scalar_observables_means.append(degree_of_dissociation)
        scalar_observables_errors.append(err_degree_of_dissociation)
        
        #save excess chemical potential
        excess_chemical_potential_pair,std_error_excess_chemical_potential_pair=Widom.measure_excess_chemical_potential(0)
        scalar_observables_means.append(excess_chemical_potential_pair)
        scalar_observables_errors.append(std_error_excess_chemical_potential_pair)
        excess_chemical_potential_Na,std_error_excess_chemical_potential_Na=Widom.measure_excess_chemical_potential(2)
        scalar_observables_means.append(excess_chemical_potential_Na)
        scalar_observables_errors.append(std_error_excess_chemical_potential_Na)
        excess_chemical_potential_Cl,std_error_excess_chemical_potential_Cl=Widom.measure_excess_chemical_potential(4)
        scalar_observables_means.append(excess_chemical_potential_Cl)
        scalar_observables_errors.append(std_error_excess_chemical_potential_Cl)        

        
        output_list=simulation_parameters+[N_steps]+scalar_observables_means+scalar_observables_errors
        
        with open(filename+"_pressure.out", mode="w") as f:
            writer=csv.writer(f)
            header="# final_box_l,MPC,Kcideal_in_mol_per_l, cs_bulk, bjerrum, pH, cH_bulk, cOH_bulk, cNa_bulk, cCl_bulk, ionic_strength_bulk, Nsteps , num_Hs, num_OHs, num_Nas, num_Cls, num_As, total_isotropic_pressures, end_to_end_distances, bond_lengths, degree_of_dissociation, excess_chemical_potential_pair, excess_chemical_potential_Na, excess_chemical_potential_Cl, err_num_Hs, err_num_OHs, err_num_Nas, err_num_Cls, err_num_As, err_total_isotropic_pressures, err_end_to_end_distances, err_bond_lengths, err_degree_of_dissociation, std_err_excess_chemical_potential_pair, std_err_excess_chemical_potential_Na, std_err_excess_chemical_potential_Cl "
            writer.writerow(header.split(','))
            writer.writerow(output_list)
        print("N_tot", len(system.part[:].q),"HA", system.number_of_particles(type=type_HA), "A-", system.number_of_particles(type=type_A), "H+",system.number_of_particles(type=type_H), "total charge", np.sum(system.part[:].q), "end_to_end_distance", np.mean(end_to_end_distances))
        
        
        #save tensorial observables
        mean_total_stress_tensor=np.mean(total_stress_tensors, axis=0)
        std_error_mean_total_stress_tensor=np.std(total_stress_tensors,axis=0, ddof=1)/np.sqrt(N_steps)
        np.savetxt(filename+str("_total_stress.out"), mean_total_stress_tensor)
        np.savetxt(filename+str("_total_stress_err.out"), std_error_mean_total_stress_tensor)
        mean_kinetic_stress_tensor=np.mean(kinetic_stress_tensors, axis=0)
        std_error_mean_kinetic_stress_tensor=np.std(kinetic_stress_tensors,axis=0, ddof=1)/np.sqrt(N_steps)
        np.savetxt(filename+str("_kinetic_stress.out"), mean_kinetic_stress_tensor)
        np.savetxt(filename+str("_kinetic_stress_err.out"), std_error_mean_kinetic_stress_tensor)
        mean_coulomb_stress_tensor=np.mean(coulomb_stress_tensors, axis=0)
        std_error_mean_coulomb_stress_tensor=np.std(coulomb_stress_tensors,axis=0, ddof=1)/np.sqrt(N_steps)
        np.savetxt(filename+str("_coulomb_stress.out"), mean_coulomb_stress_tensor)
        np.savetxt(filename+str("_coulomb_stress_err.out"), std_error_mean_coulomb_stress_tensor)
        mean_bonded_stress_tensor=np.mean(bonded_stress_tensors, axis=0)
        std_error_mean_bonded_stress_tensor=np.std(bonded_stress_tensors,axis=0, ddof=1)/np.sqrt(N_steps)
        np.savetxt(filename+str("_bonded_stress.out"), mean_bonded_stress_tensor)
        np.savetxt(filename+str("_bonded_stress_err.out"), std_error_mean_bonded_stress_tensor)             
        mean_nonbonded_stress_tensor=np.mean(nonbonded_stress_tensors, axis=0)
        std_error_mean_nonbonded_stress_tensor=np.std(nonbonded_stress_tensors,axis=0, ddof=1)/np.sqrt(N_steps)
        np.savetxt(filename+str("_nonbonded_stress.out"), mean_nonbonded_stress_tensor)   
        np.savetxt(filename+str("_nonbonded_stress_err.out"), std_error_mean_nonbonded_stress_tensor)
        

        np.savetxt(filename+str("_widom.out"), np.c_[excess_chemical_potential_pair,std_error_excess_chemical_potential_pair], header="1)excess_chemical_potential_pair 2)std_error_excess_chemical_potential_pair")

        #save gel charges
        mean_gel_charges=np.mean(gel_charges,axis=0)
        std_error_gel_charges=np.std(gel_charges,axis=0, ddof=1)/np.sqrt(N_steps)
        np.savetxt(filename+"_gel_charges.out", np.c_[gel_ids, mean_gel_charges, std_error_gel_charges], header="id, average_charge, std_error_mean_charge")
        np.savetxt(filename+"_acceptance_rates.out",np.c_[RE.get_acceptance_rate_reaction(0), RE.get_acceptance_rate_reaction(1), RE.get_acceptance_rate_reaction(2), RE.get_acceptance_rate_reaction(3),RE.get_acceptance_rate_reaction(4), RE.get_acceptance_rate_reaction(5)])


        outfile = open('diamond_sim_'+str(final_box_l)+'.vtf', 'w')
        vtf.writevsf(system, outfile)
        vtf.writevcf(system, outfile)
        outfile.close()
        with gzip.GzipFile("checkpoint_"+str(final_box_l)+".pgz", 'wb') as fcheck:
            print("writing checkpoint")
            data=[num_Hs, num_OHs, num_Nas, num_Cls, num_As, total_isotropic_pressures, stress_tensor, total_stress_tensors, kinetic_stress_tensors, coulomb_stress_tensors, bonded_stress_tensors, nonbonded_stress_tensors, gel_charges,end_to_end_distances, bond_lengths]
            pickle.dump(data, fcheck)
