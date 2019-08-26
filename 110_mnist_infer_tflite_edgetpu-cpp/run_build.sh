#!/bin/sh
set -e
set -x

make clean

make TFLITE_VERSION=1_14
mv mnist_infer mnist_infer_edgetpu


