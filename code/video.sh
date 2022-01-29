#!/usr/bin/env bash
cd code

export CUDA_VISIBLE_DEVICES="1"

echo "PWC"
bash demo/video_demo.py --video eval/test.mp4 --config configs/pwcnet/pwcnet_sintel.py --checkpoint ckpt/pwc.pth --out eval/test_pwc_out_noisy.mp4 --device cuda:1

echo "Flownet2"
bash demo/video_demo.py --video eval/test.mp4 --config configs/flownet2/flownet2_sintel.py --checkpoint ckpt/flownet2.pth --out eval/test_flownet2_out_noise.mp4 --device cuda:1 --use-noisy

echo "MaskflowNet"
bash demo/video_demo.py --video eval/test.mp4 --config configs/maskflownet/maskflownet_sintel.py --checkpoint ckpt/maskflownet.pth --out eval/test_maskflow_out_noisy.mp4 --device cuda:1

echo "RAFT"
bash demo/video_demo.py --video eval/test.mp4 --config configs/raft/raft_sintel.py --checkpoint ckpt/raft.pth --out eval/test_raft_out_noisy.mp4 --device cuda:1

echo "GMA"
bash demo/video_demo.py --video eval/test.mp4 --config configs/gma/gma_sintel.py --checkpoint ckpt/gma.pth --out eval/test_gma_out_noisy.mp4 --device cuda:1 --use-noisy