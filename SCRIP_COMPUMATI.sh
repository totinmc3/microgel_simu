#!/bin/bash 


declare -a Alpha_n=(0.0 0.05 0.1 0.015 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

declare -a Box_l=(309.33 389.73 491.04)
b=${Box_l[0]}

declare -a NBeads=(10 20 40)
j=${NBeads[0]}

for k in 0 1 2 3 4 5 6 7 8 9 10 11;
do
	cd /home/tobias/trabajo/resultados

	if [ -d N_beads"$j" ]
	then
		cd N_beads"$j"
	else
		mkdir N_beads"$j" && cd "$_"
	fi

	if [ -d Alpha_n${Alpha_n[k]} ]
	then
		cd Alpha_n${Alpha_n[k]}
	else
		mkdir Alpha_n${Alpha_n[k]} && cd "$_"
	fi


	echo Alpha_${Alpha_n[k]}
	echo Box_length_$b
	
	pwd

	cat /home/tobias/trabajo/microgel_simu/AN_BOX.sh | sed "s/ALPHA/${Alpha_n[k]}/" > script_aux.in


	cat script_aux.in | sed s/BOX/$b/ > sub_script.in


	cat sub_script.in | sed s/NBEADS/$j/ > script_aux.in


	mv script_aux.in sub_script.sh


	rm  sub_script.in

	#qsub ./sub_script.sh
	chmod +x sub_script.sh

	. sub_script.sh
cd ~/trabajo
done

