#!/bin/sh
set -e
set -x

./run_train.sh
./run_export.sh
./run_convert_to_tflite.sh

./run_train.sh -q
./run_export.sh -q
./run_convert_to_tflite.sh -q


./run_posttrain_quant.sh


