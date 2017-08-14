#!/usr/bin/env bash

cd data/
bash MacDownloadData.sh
python RandPerm.py
cd ..
