python tools/test.py \
  configs/flownet2/flownet2_sintel.py \
  ckpt/flownet2.pth --eval EPE Fl EPE-noise

python tools/test.py \
  configs/flownet2/flownet2_fly.py \
  ckpt/flownet2.pth --eval EPE Fl EPE-noise

python tools/test.py \
  configs/flownet2/flownet2_kitti.py \
  ckpt/flownet2.pth --eval EPE Fl EPE-noise

