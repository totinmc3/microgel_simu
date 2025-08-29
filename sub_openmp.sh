#!/bin/bash
#
#====================== Directivas SGE =======================
#
# Tiempo máximo de CPU
#$ -l h_rt=119:59:59
#
# Solicitar entre 2 y 4 CPUs (paralelismo shared memory)
#$ -pe smp 2-4
#
# Exportar variable de hilos OpenMP
#$ -v OMP_NUM_THREADS=$NSLOTS
#
# Archivo de salida
#$ -o output.log
#
# Usar directorio actual
#$ -cwd
#
#=============================================================

# Asegurar número de threads
export OMP_NUM_THREADS=$NSLOTS

# ---------------------- Configuración -----------------------

LOGFILE="simu.log" # archivo de log (general)
PY_SCRIPT="/nashome/tloggia/trabajo/microgel_simu/main.py"
PYPRESSO="/nashome/tloggia/espresso_md/espresso/build/pypresso"

# Verifico que exista el script de Python
[ -f "$PY_SCRIPT" ] || {
    echo "Error: No se encontró el script $PY_SCRIPT" >&2
    exit 1
}

 $PYPRESSO "$PY_SCRIPT" ALPHA BOX NBEADS >> "../$LOGFILE" 2>&1
exit

