python tools/test.py \
  configs/maskflownet/maskflownet_sintel.py \
  ckpt/maskflownet.pth --eval EPE Fl

python tools/test.py \
  configs/maskflownet/maskflownet_fly.py \
  ckpt/maskflownet.pth --eval EPE Fl

python tools/test.py \
  configs/maskflownet/maskflownet_kitti.py \
  ckpt/maskflownet.pth --eval EPE Fl

