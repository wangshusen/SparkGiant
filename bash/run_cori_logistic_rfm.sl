#!/bin/bash
#SBATCH -p debug
#SBATCH -N 2
#SBATCH -C haswell
#SBATCH -t 00:15:00
#SBATCH -J wss_giant
#SBATCH -L SCRATCH
#SBATCH -e giant_job_%j.err
#SBATCH -o giant_job_%j.out

PROJ_HOME="$SCRATCH/SparkGiant"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
DATA_FILE="$PROJ_HOME/data/covtype_perm"

NUM_SPLITS="31"
NUM_FEATURE="50"

module load spark
ulimit -s unlimited
start-all.sh

spark-submit \
    --class "distopt.logistic.ExperimentRfm" \
    --num-executors $NUM_SPLITS \
    $JAR_FILE $DATA_FILE $NUM_FEATURE $NUM_SPLITS
  
stop-all.sh
