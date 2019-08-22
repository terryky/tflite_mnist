#!/bin/bash
set -e


#
# [usage] 
#   > ./freeze_pb_graph pb_fname ckpt_fname outnode_name outpb_fname
#   > ./freeze_pb_graph inference.pb ./checkpoint/model.ckpt fc2/add inference_freeze.pb
#

# 引数が4個でなければエラー
if [ $# -ne 4 ]
then
  echo "[usage] freeze_pb_graph pb_fname ckpt_fname outnode_name outpb_fname"
  exit 0
fi


PB_FILE=$1
CKPT_FILE=$2
OUTNODE_NAME=$3
FROZEN_FILE=$4


echo '---------------------------------'
 echo "import pb   file :" $PB_FILE
 echo "import ckpt file :" $CKPT_FILE
 echo "output node name :" $OUTNODE_NAME
 echo "output pb   file :" $FROZEN_FILE
echo '---------------------------------'


freeze_graph \
  --input_graph=${PB_FILE} \
  --input_checkpoint=${CKPT_FILE} \
  --checkpoint_version=2 \
  --input_binary=True \
  --output_graph=${FROZEN_FILE} \
  --output_node_names=${OUTNODE_NAME}
  


