#!/usr/bin/env bash

# BASH program to execute a a single pypresso process for given parameter p value and process number proc_iter
# p and proc_iter must be parsed
# p can be: boxsize L, number of anionic (cationic) groups N_an (N_cat), salinity, etc. 


# execute by: $ ./microgel_exe.sh <p> <proc_iter>

LOGFILE="log.log"
plabel="alpha_n" # etiqueta para el parámetro
NPROCS=6 # Número de procesos MPI (puede cambiarse según necesidad)
# Check if mpiexec exists

# Verificar que mpiexec esté disponible
command -v mpiexec &> /dev/null || {
    echo "Error: mpiexec no encontrado. Instala MPI." >&2
    exit 1
}

PY_SCRIPT="/home/tobias/trabajo/geles/microgel_simu/main.py"
[ -f "$PY_SCRIPT" ] || {
    echo "Error: No se encontró el script $PY_SCRIPT" >&2
    exit 1
}

plabel="alpha_n"
LOGFILE="simu.log"

for j in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9; do
    echo "$plabel = $j"
    dir="${plabel}_$j"
    mkdir -p "$dir" && cd "$dir" || {
        echo "Error: No se pudo acceder a $dir" >&2
        exit 1
    }

    mpiexec -n 6 /home/tobias/trabajo/espresso_md/espresso/build/pypresso "$PY_SCRIPT" "$j" >> "../$LOGFILE" 2>&1

    cd ..
done

