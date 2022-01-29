python tools/test.py \
  configs/gma/gma_sintel.py \
  ckpt/gma.pth --eval EPE Fl EPE-noise

python tools/test.py \
  configs/gma/gma_fly.py \
  ckpt/gma.pth --eval EPE Fl EPE-noise

python tools/test.py \
  configs/gma/gma_kitti.py \
  ckpt/gma.pth --eval EPE Fl EPE-noise

