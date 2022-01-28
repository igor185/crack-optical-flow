#!/usr/bin/env bash
cd code

export CUDA_VISIBLE_DEVICES="1"

echo "PWC"
bash eval/pwc.sh

echo "Flownet2"
bash eval/flownet2.sh

echo "MaskflowNet"
bash eval/maskflow.sh

echo "RAFT"
bash eval/raft.sh

echo "GMA"
bash eval/gma.sh