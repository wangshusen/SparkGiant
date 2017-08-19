#!/usr/bin/env bash

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2" -o "YearPredictionMSD.bz2"
bzip2 -d YearPredictionMSD.bz2

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2" -o "YearPredictionMSD.t.bz2"
bzip2 -d YearPredictionMSD.t.bz2


curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2" -o "covtype.bz2"
bzip2 -d covtype.bz2

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist.scale.bz2" -o "mnist.bz2"
bzip2 -d mnist.bz2

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a" -o "a9a"
curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a.t" -o "a9a.t"

curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a" -o "w8a"
curl "http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a.t" -o "w8a.t"


