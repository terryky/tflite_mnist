#!/bin/sh
set -e
set -x

make clean

make TFLITE_VERSION=1_12
mv mnist_infer mnist_infer_112

make TFLITE_VERSION=1_13
mv mnist_infer mnist_infer_113

make TFLITE_VERSION=1_14
mv mnist_infer mnist_infer_114


