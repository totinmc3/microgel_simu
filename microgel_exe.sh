#!/usr/bin/env bash

# BASH program to execute a a single pypresso process for given parameter p value and process number proc_iter
# p and proc_iter must be parsed
# p can be: boxsize L, number of anionic (cationic) groups N_an (N_cat), salinity, etc. 


# execute by: $ ./microgel_exe.sh <p> <proc_iter>


LOGFILE="log.log"
j="$1"		# p value
plabel="alpha_n" # p label

 echo "$plabel = $j"
 if [ -d ""$plabel"_$j" ]
 then
    cd "$plabel"_$j
 else
    mkdir "$plabel"_$j && cd "$_"
 fi

 /home/tobias/espresso_md/build_Feb25/pypresso /home/tobias/trabajo_afa/microgel_simu/main.py $j

exit
