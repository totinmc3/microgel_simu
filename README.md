Program for simulating a single ionic microgel in a cell to explore swelling properties of microgels with networks of different polymer blocks:
- The microgel consists of a finite diamond polymer network of tetrafunctional nodes
- Polymer are simulated via WCA potential with FENE bonds
- Electrostatic is incorporated explicitly by charged polymer beads and free co- and counterions

Execution:
1- Set both system and simulation parameters in system_parameter.py file
2- If iterating over given parameter:
    a- set parsing of that parameter in main.py
    b- set folder name for the parameter in microgel_exe.sh (plabel variable)
    c- configure variation range of parameter in parameter_list.txt: one value per line
3- Create log folder in the directory of execution
4- Submit microgel_condor.cmd to condor