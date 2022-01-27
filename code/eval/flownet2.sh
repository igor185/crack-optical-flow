export CUDA_VISIBLE_DEVICES="1"

python tools/test.py \
  configs/flownet2/flownet2_sintel.py \
  ckpt/flownet2.pth --eval EPE Fl

python tools/test.py \
  configs/flownet2/flownet2_fly.py \
  ckpt/flownet2.pth --eval EPE Fl

python tools/test.py \
  configs/flownet2/flownet2_kitti.py \
  ckpt/flownet2.pth --eval EPE Fl

