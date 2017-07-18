#!/usr/bin/env bash

PROJ_HOME="$HOME/Code/SparkGiant"
SPARK_HOME="$HOME/local/spark-2.1.1"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
NUM_SPLITS="128"
MASTER="local["$NUM_SPLITS"]"

DATA_FILE="$PROJ_HOME/data/YearPredictionMSD"

$SPARK_HOME/bin/spark-submit \
    --class "distopt.quadratic.Experiment" \
    --master $MASTER \
    --driver-memory 3G \
    --executor-cores 1 \
    --executor-memory 1G \
    $JAR_FILE $DATA_FILE $NUM_SPLITS 1E-3 30 \
    > result.out
  
  