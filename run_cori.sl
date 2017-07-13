#!/bin/bash
#SBATCH -p regular
#SBATCH -N 4
#SBATCH -C haswell
#SBATCH -t 00:10:00
#SBATCH -J wss_giant
#SBATCH -L SCRATCH
#SBATCH -e giant_job_%j.err
#SBATCH -o giant_job_%j.out

PROJ_HOME="$SCRATCH/SparkGiant"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
DATA_FILE="$PROJ_HOME/data/YearPredictionMSD"

NUM_SPLITS="16"

module load python/3.5-anaconda
module load spark
ulimit -s unlimited
start-all.sh

spark-submit \
    --class "distopt.Quadratic" \
    $JAR_FILE $DATA_FILE $NUM_SPLITS 1E-6 50 
  
stop-all.sh
