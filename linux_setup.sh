#!/usr/bin/env bash

cd data/
bash LinuxDownloadData.sh
python RandPerm.py
cd ..
