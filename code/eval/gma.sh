export CUDA_VISIBLE_DEVICES="1"

python tools/test.py \
  configs/gma/gma_sintel.py \
  ckpt/gma.pth --eval EPE Fl

python tools/test.py \
  configs/gma/gma_fly.py \
  ckpt/gma.pth --eval EPE Fl

python tools/test.py \
  configs/gma/gma_kitti.py \
  ckpt/gma.pth --eval EPE Fl

