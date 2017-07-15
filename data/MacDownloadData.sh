#!/usr/bin/env bash

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2" -o "YearPredictionMSD.bz2"
bzip2 -d YearPredictionMSD.bz2

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2" -o "YearPredictionMSD.t.bz2"
bzip2 -d YearPredictionMSD.t.bz2

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2" -o "covtype.bz2"
bzip2 -d covtype.bz2