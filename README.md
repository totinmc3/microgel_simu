# Microgel Simulation

Program for simulating a single ionic microgel in a cell to explore swelling properties of microgels:
- The microgel consists of a finite diamond polymer network of tetrafunctional nodes
- Polymer are simulated via WCA beads connected with FENE bonds (generic bead-spring model)
- Electrostatic is incorporated explicitly by charged polymer beads and free WCA co- and counterions 

## Cloning Repository

To clone the repository, run:

```bash
git clone https://github.com/mebrito/microgel_simu.git
```
## Execution

1. Set both system and simulation parameters in `system_parameter.py` file
2. For running a simulation with single core:

```bash
<Path/to/ESPResSo>/pypresso <Path/to/repository>/microgel_simu/main.py <param1> <param2> ...
```

where `<param1>`, `<param2>`, ... are the parsed parameters required. Check `parser.add_argument()`
in `main.py`.
For running on several MPI ranks:

```bash
mpiexec -n <N_ranks> <Path/to/ESPResSo>/pypresso <Path/to/repository>/microgel_simu/main.py <param1> <param2> ...
```

with `<N_ranks> ` being the number of ranks.
3.  Bash scripts can be used for iterating over given parameters, or further automation.

## Core Components

### main.py file

The main simulation script that orchestrates the entire microgel simulation process for a NVT simulation 
of a single microgel with strong charges and desired extra salt concentration. Key features:

- **Command-line arguments**: Accepts box size and anionic ionization degree (`alpha_an`) as parameters
- **System initialization**: Sets up the ESPResSo system with periodic boundary conditions
- **Microgel creation**: Instantiates the Microgel object and initializes the polymer network 
(`microgel = microgel_object.Microgel(**args)`). The microgel can be created from pre-existing trajectories
(`microgel.initialize_from_file()`) or created from scratch (`microgel.initialize_diamondLattice()`).
- **Checkpointing**: Handles simulation restarts using ESPResSo's checkpointing system
- **Simulation flow**: Manages the complete simulation pipeline from initialization to data collection
- **Output management**: Creates system information files and handles trajectory output

The script follows this general workflow:
1. Parse command-line arguments (box size, ionization degree)
2. Initialize or restore from checkpoint
3. Create microgel structure (diamond lattice or from file)
4. Set up interactions and electrostatics
5. Run warmup and main sampling loops
6. Output trajectories and analysis data

### Microgel Class

Located in `microgel_class/microgel_object.py`, this class encapsulates all microgel-related functionality:
#### Instantiation Example

```python
from microgel_class import microgel_object

# Create microgel instance
microgel = microgel_object.Microgel(
    system,                # ESPResSo system object
    FENE_BOND_PARAMS,     # FENE bond parameters dictionary
    PART_TYPE,            # Particle type definitions dictionary
    NONBOND_WCA_PARAMS,   # WCA interaction parameters dictionary
    Nbeads_arm,           # Number of beads per polymer arm
    cell_unit,            # Unit cell size
    N_cat,                # Number of cationic beads
    N_an,                 # Number of anionic beads
    c_salt                # Salt concentration (optional, default=0)
)
```

#### Key Methods:
- **`initialize_diamondLattice()`**: Creates a diamond lattice polymer network structure
- **`initialize_from_file()`**: Loads pre-generated microgel configurations from PDB files
- **`initialize_bonds()`**: Sets up FENE bonds between polymer beads
- **`initialize_internoelec()`**: Configures non-electrostatic interactions (WCA potential)
- **`charge_beads_homo()`**: Homogeneously distributes charges across the microgel
- **`charge_beads_shell()`**: Creates shell-like charge distribution
- **`add_salt()`**: Introduces salt ions to the system
- **`add_cell_boundary()`**: Sets up spherical cell boundaries

#### Properties:
- Manages polymer arm length (`Nbeads_arm`)
- Controls charge distribution (cationic/anionic beads)
- Handles salt concentration
- Maintains interaction parameters (FENE bonds, WCA potential)

### Extra simulation tools

The `handling/handler.py` module provides essential simulation management functions:

#### Functions:
- **`remove_overlap()`**: Eliminates particle overlaps using steepest descent minimization
- **`warmup()`**: Performs system equilibration with gradual parameter tuning
- **`warmup_chkpnt()`**: Handles warmup when restarting from checkpoints
- **`main_integration()`**: Executes the main simulation loop with data collection
- **`initialize_elec()`**: Sets up P3M electrostatic solver

These functions manage:
- Energy minimization to resolve initial overlaps
- System thermalization and equilibration
- Electrostatic interaction setup
- Simulation progress monitoring

### Analysis ftools

The `analysis/` directory contains specialized analysis tools:

#### `autocorr.py`
- **`autocor()`**: Calculates autocorrelation functions for time series data
- **`fit_correlation_time()`**: Fits exponential decay to determine correlation times
- Used for analyzing energy fluctuations and system equilibration

#### `com.py`
- **`com_calculation()`**: Computes center of mass for specified particle types
- Supports multi-type center of mass calculations
- Essential for tracking microgel translational movement

#### `densityProfile_calc.py`
- **`particle_density_profile()`**: Calculates 3D density profiles
- Creates spatial binning of particle distributions
- Useful for analyzing microgel structure and ion distributions
- Supports both single and multiple particle types

These analysis tools enable:
- Structural characterization of microgels
- Monitoring of simulation equilibration
- Spatial distribution analysis
- Time-dependent property tracking