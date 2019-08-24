#!/bin/sh
set -e
set -x


# floatでトレーニングするか、quantでトレーニングするか
FLAG_QUANT=false
while getopts q OPT
do
  case $OPT in
    "q" ) FLAG_QUANT=true  ;;
    "f" ) FLAG_QUANT=false ;;
  esac
done


BASE_DIR=./
#BASE_DIR=./1.12/


# =================
#   QUANT
# =================
if $FLAG_QUANT ; then

  # --------------------------------------------------------
  # freeze_graph
  # --------------------------------------------------------
  PB_FILE=${BASE_DIR}/graph_def_quant_export/model_graph.pb
  CKPT_FILE=${BASE_DIR}/checkpoint_quant/mymodel
  OUTNODE_NAME=softmax/Softmax
  FROZEN_PB_FILE=${BASE_DIR}/graph_def_quant_export/frozen.pb

  ../convert_script/freeze_pb_graph.sh ${PB_FILE} ${CKPT_FILE} ${OUTNODE_NAME} ${FROZEN_PB_FILE}


  # --------------------------------------------------------
  # TFLiteに変換 (Frozen GraphDef)
  # --------------------------------------------------------
  INNODE_NAME=Placeholder
  INNODE_SHAPE=1,784

  ../convert_script/convert_pb_to_tflite_quant.sh ${FROZEN_PB_FILE} ${INNODE_NAME} ${OUTNODE_NAME} ${INNODE_SHAPE}
  mv ${BASE_DIR}/graph_def_quant_export/frozen.pb.tf_quant.tflite ${BASE_DIR}/mnist_frozengraph_quant.tflite


  # --------------------------------------------------------
  # TFLiteに変換 (Saved Model)
  # --------------------------------------------------------
  SAVEDMODEL_DIR=${BASE_DIR}/saved_model_quant
  INNODE_NAME=Placeholder
  INNODE_SHAPE=1,784

  #../convert_script/convert_savedmodel_to_tflite_quant.sh ${SAVEDMODEL_DIR} ${INNODE_NAME} ${OUTNODE_NAME} ${INNODE_SHAPE}
  #mv ./saved_model.tf_quant.tflite ./mnist_savedmodel_quant.tflite


# =================
#   FLOAT
# =================
else

  # --------------------------------------------------------
  # freeze_graph
  # --------------------------------------------------------
  PB_FILE=${BASE_DIR}/graph_def_export/model_graph.pb
  CKPT_FILE=${BASE_DIR}/checkpoint/mymodel
  OUTNODE_NAME=softmax/Softmax
  FROZEN_PB_FILE=${BASE_DIR}/graph_def_export/frozen.pb

  ../convert_script/freeze_pb_graph.sh ${PB_FILE} ${CKPT_FILE} ${OUTNODE_NAME} ${FROZEN_PB_FILE}


  # --------------------------------------------------------
  # TFLiteに変換 (Frozen GraphDef)
  # --------------------------------------------------------
  INNODE_NAME=Placeholder
  INNODE_SHAPE=1,784

  ../convert_script/convert_pb_to_tflite_float.sh ${FROZEN_PB_FILE} ${INNODE_NAME} ${OUTNODE_NAME} ${INNODE_SHAPE}
  mv ${BASE_DIR}/graph_def_export/frozen.pb.tf_float.tflite ${BASE_DIR}/mnist_frozengraph_float.tflite


  # --------------------------------------------------------
  # TFLiteに変換 (Saved Model)
  # --------------------------------------------------------
  SAVEDMODEL_DIR=${BASE_DIR}/saved_model

  ../convert_script/convert_savedmodel_to_tflite_float.sh ${SAVEDMODEL_DIR} ${INNODE_NAME} ${OUTNODE_NAME} ${INNODE_SHAPE}
  mv saved_model.tf_float.tflite ${BASE_DIR}/mnist_savedmodel_float.tflite

fi

