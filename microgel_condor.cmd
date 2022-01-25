####################
##
## Condor command file
##
## Iteration over parameter in parameter_list.txt
##
####################

universe	= vanilla

initialdir	= .
executable	= microgel_exe.sh
arguments	= $(arg)

output		= log/condor_$(arg).out
error		= log/condor_$(arg).err
log		= log/condor_$(arg).log

requirements    = FileSystemDomain == "icp.uni-stuttgart.de" && UidDomain == "icp.uni-stuttgart.de"
should_transfer_files = IF_NEEDED
when_to_transfer_output = ON_EXIT

request_cpus = 1

queue arg from parameter_list.txt
