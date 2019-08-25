#!/bin/sh
set -e
set -x

BASEDIR=../001_mnist_train

# ----------------------------------------------------------------
# TF1.12 trained model
# ----------------------------------------------------------------
./mnist_infer_112 ${BASEDIR}/tf112/mnist_frozengraph_float.tflite
./mnist_infer_112 ${BASEDIR}/tf112/mnist_frozengraph_quant.tflite

./mnist_infer_113 ${BASEDIR}/tf112/mnist_frozengraph_float.tflite
./mnist_infer_113 ${BASEDIR}/tf112/mnist_frozengraph_quant.tflite

./mnist_infer_114 ${BASEDIR}/tf112/mnist_frozengraph_float.tflite
./mnist_infer_114 ${BASEDIR}/tf112/mnist_frozengraph_quant.tflite



# ----------------------------------------------------------------
# TF1.13 trained model
# ----------------------------------------------------------------
./mnist_infer_112 ${BASEDIR}/tf113/mnist_frozengraph_float.tflite
./mnist_infer_112 ${BASEDIR}/tf113/mnist_frozengraph_quant.tflite

./mnist_infer_113 ${BASEDIR}/tf113/mnist_frozengraph_float.tflite
./mnist_infer_113 ${BASEDIR}/tf113/mnist_frozengraph_quant.tflite

./mnist_infer_114 ${BASEDIR}/tf113/mnist_frozengraph_float.tflite
./mnist_infer_114 ${BASEDIR}/tf113/mnist_frozengraph_quant.tflite



# ----------------------------------------------------------------
# TF1.14 trained model
# ----------------------------------------------------------------
./mnist_infer_112 ${BASEDIR}/tf114/mnist_frozengraph_float.tflite
./mnist_infer_112 ${BASEDIR}/tf114/mnist_frozengraph_quant.tflite

./mnist_infer_113 ${BASEDIR}/tf114/mnist_frozengraph_float.tflite
./mnist_infer_113 ${BASEDIR}/tf114/mnist_frozengraph_quant.tflite

./mnist_infer_114 ${BASEDIR}/tf114/mnist_frozengraph_float.tflite
./mnist_infer_114 ${BASEDIR}/tf114/mnist_frozengraph_quant.tflite



