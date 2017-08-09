#!/usr/bin/env bash

export PROJ_HOME="/root/SparkGiant"
export JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
export DATA_FILE=$PROJ_HOME"/data/covtype_perm"
export SPARK_HOME="/root/spark-2.1.0-bin-hadoop2.7"
NUM_SPLITS="4"
MASTER="local["$NUM_SPLITS"]"
NUM_FEATURE="200"

$SPARK_HOME/bin/spark-submit \
    --class "distopt.logistic.ExperimentRfm" \
    --master $MASTER \
    --num-executors $NUM_SPLITS \
    $JAR_FILE $DATA_FILE $NUM_FEATURE $NUM_SPLITS
  