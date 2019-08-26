#!/bin/sh
set -e
set -x

BASEDIR=../007_mnist_train

# ----------------------------------------------------------------
# TF1.12 trained model
# ----------------------------------------------------------------
./mnist_infer_edgetpu ${BASEDIR}/tf112/mnist_frozengraph_quant_edgetpu.tflite


# ----------------------------------------------------------------
# TF1.13 trained model
# ----------------------------------------------------------------
./mnist_infer_edgetpu ${BASEDIR}/tf113/mnist_frozengraph_quant_edgetpu.tflite


# ----------------------------------------------------------------
# TF1.14 trained model
# ----------------------------------------------------------------
./mnist_infer_edgetpu ${BASEDIR}/tf114/mnist_frozengraph_quant_edgetpu.tflite
