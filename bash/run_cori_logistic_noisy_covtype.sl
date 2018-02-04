#!/bin/bash
#SBATCH -p regular
#SBATCH -N 75
#SBATCH -C haswell
#SBATCH -t 5:00:00
#SBATCH -J giant_logistic
#SBATCH -L SCRATCH
#SBATCH -e giant_covtype_job_%j.err
#SBATCH -o giant_covtype_job_%j.out

PROJ_HOME="$SCRATCH/SparkGiant"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
DATA_FILE1="$PROJ_HOME/data/covtype_train_noise"
DATA_FILE2="$PROJ_HOME/data/covtype_test_noise"

NUM_SPLITS="449"
NUM_FEATURES="10000"

module load spark
ulimit -s unlimited
start-all.sh

spark-submit \
    --class "distopt.logistic.ExperimentCovtype" \
    --num-executors $NUM_SPLITS \
    --driver-cores 5 \
    --executor-cores 5 \
    --driver-memory 20G \
    --executor-memory 20G \
    $JAR_FILE $DATA_FILE1 $DATA_FILE2 $NUM_FEATURES $NUM_SPLITS
  
stop-all.sh
