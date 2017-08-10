#!/usr/bin/env bash

PROJ_HOME="/root/SparkGiant"

export JAR_DIR="$PROJ_HOME/target/scala-2.11"
export JAR_FILE="$JAR_DIR/giant_2.11-1.0.jar"
cp $JAR_FILE /root/share/
/root/spark-ec2/copy-dir /root/share/

export DATA_FILE_HDFS="hdfs://"`cat /root/spark-ec2/masters`":9000/covtype_perm"

NUM_SPLITS="3"
NUM_FEATURE="500"

/root/spark/bin/spark-submit \
    --class "distopt.logistic.ExperimentRfm" \
    --master `cat /root/spark-ec2/cluster-url` \
    --num-executors $NUM_SPLITS \
    --driver-memory 6G \
    --executor-memory 6G \
    --executor-cores 2 \
    $JAR_FILE $DATA_FILE_HDFS $NUM_FEATURE $NUM_SPLITS \
    > Result_FEATURE"$NUM_FEATURE".out

  
  