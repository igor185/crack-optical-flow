export CUDA_VISIBLE_DEVICES="1"

python tools/test.py \
  configs/pwcnet/pwcnet_sintel.py \
  ckpt/pwc.pth --eval EPE Fl

python tools/test.py \
  configs/pwcnet/pwcnet_fly.py \
  ckpt/pwc.pth --eval EPE Fl

python tools/test.py \
  configs/pwcnet/pwcnet_kitti.py \
  ckpt/pwc.pth --eval EPE Fl

