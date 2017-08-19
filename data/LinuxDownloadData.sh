#!/usr/bin/env bash

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
bzip2 -d YearPredictionMSD.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.t.bz2
bzip2 -d YearPredictionMSD.t.bz2

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
mv covtype.libsvm.binary.scale.bz2 covtype.bz2
bzip2 -d covtype.bz2

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/multiclass/mnist8m.scale.bz2
mv mnist8m.scale.bz2 mnist.bz2
bzip2 -d mnist.bz2

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.bz2
bzip2 -d epsilon_normalized.bz2
wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/epsilon_normalized.t.bz2
bzip2 -d epsilon_normalized.t.bz2
