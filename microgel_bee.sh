#!/bin/bash

# BASH program to execute a single pypresso simulation in bee

# execute by: $ ./microgel_bee.sh

p_label="ionDegree" 	# parameter label
p_values="0.0"
wdir_path=`pwd`
queueing_exe=${wdir_path}/../microgel_slurm

for p_val in $p_values; do
	dir_name=""$p_label"_$p_val"
	
	echo $dir_name

	if [ -d "$dir_name" ]
	then
		cd "$dir_name"
	else
		mkdir "$dir_name" && cd "$_"
	fi

	sbatch $queueing_exe $p_val
	cd ..
done

exit
