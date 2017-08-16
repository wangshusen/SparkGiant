#!/usr/bin/env bash

PROJ_HOME="$HOME/Code/SparkGiant"
SPARK_HOME="$HOME/local/spark-2.1.1"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
NUM_SPLITS="63"
MASTER="local["$NUM_SPLITS"]"

DATA_TRAIN_FILE="$PROJ_HOME/data/YearPredictionMSD"
DATA_TEST_FILE="$PROJ_HOME/data/YearPredictionMSD.t"

$SPARK_HOME/bin/spark-submit \
    --class "distopt.quadratic.Experiment" \
    --master $MASTER \
    --driver-memory 8G \
    --executor-cores 1 \
    --executor-memory 8G \
    --num-executors $NUM_SPLITS \
    $JAR_FILE $DATA_TRAIN_FILE $DATA_TEST_FILE $NUM_SPLITS \
    > Result_Quadratic_FEATURE"$NUM_FEATURE".out
  
  