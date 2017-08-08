#!/usr/bin/env bash

PROJ_HOME="$HOME/Code/SparkGiant"
SPARK_HOME="$HOME/local/spark-2.1.1"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
NUM_SPLITS="4"
MASTER="local["$NUM_SPLITS"]"
NUM_FEATURE="1000"

DATA_FILE="$PROJ_HOME/data/covtype_perm"

$SPARK_HOME/bin/spark-submit \
    --class "distopt.logistic.ExperimentRfm" \
    --master $MASTER \
    --driver-memory 8G \
    --executor-cores 1 \
    --executor-memory 8G \
    --num-executors $NUM_SPLITS \
    $JAR_FILE $DATA_FILE $NUM_FEATURE $NUM_SPLITS \
    > Result_FEATURE"$NUM_FEATURE".out
  
  