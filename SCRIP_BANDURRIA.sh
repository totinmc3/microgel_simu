#!/bin/bash 


declare -a Alpha_n=(0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9)

declare -a Box_l=(309.33)
b=${Box_l[0]}

declare -a NBeads=(10 20 40)
j=${NBeads[0]}

for k in "${!Alpha_n[@]}"
do
	cd /nashome/tloggia/trabajo/resultados

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
	echo Box_length_${Box_l[0]}
	
	pwd

	cat /nashome/tloggia/trabajo/microgel_simu/microgel_simu/AN_BOX.sh | sed "s/ALPHA/${Alpha_n[k]}/" > script_aux.in
	cat script_aux.in | sed s/BOX/${Box_l[0]}/ > sub_script.sh
	cat sub_script.sh | sed s/NBEADS/$j/ > script_aux.in
	mv script_aux.in sub_script.sh

	#rm  script_aux.in

	qsub ./sub_script.sh
done

