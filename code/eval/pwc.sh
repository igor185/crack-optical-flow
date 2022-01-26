#export CUDA_VISIBLE_DEVICES="0" && python tools/test.py \
#configs/pwcnet/pwcnet_8x1_slong_flyingchairs_384x448_2.py \
#checkpoints/pwcnet_8x1_slong_flyingchairs_384x448.pth --eval EPE Fl

export CUDA_VISIBLE_DEVICES="0"

python tools/test.py \
  configs/pwcnet/pwcnet.py \
  ckpt/pwc.pth --eval EPE Fl

