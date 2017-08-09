#!/usr/bin/env bash

PROJ_HOME="/root/SparkGiant"
JAR_FILE="$PROJ_HOME/target/scala-2.11/giant_2.11-1.0.jar"
DATA_FILE=$PROJ_HOME"/data/covtype"
SPARK_HOME="/root/spark-2.1.0-bin-hadoop2.7"
NUM_SPLITS="4"
MASTER="local["$NUM_SPLITS"]"
NUM_FEATURE="200"

DATA_FILE="$PROJ_HOME/data/covtype"

$SPARK_HOME/bin/spark-submit \
    --class "distopt.logistic.ExperimentRfm" \
    --master $MASTER \
    --num-executors $NUM_SPLITS \
    $JAR_FILE $DATA_FILE $NUM_FEATURE $NUM_SPLITS
  