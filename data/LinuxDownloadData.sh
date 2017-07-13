#!/usr/bin/env bash

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/regression/YearPredictionMSD.bz2
bzip2 -d YearPredictionMSD.bz2

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/covtype.libsvm.binary.scale.bz2
mv covtype.libsvm.binary.scale.bz2 covtype.bz2
bzip2 -d covtype.bz2

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/w8a

wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/a9a
