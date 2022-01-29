python tools/test.py \
  configs/raft/raft_sintel.py \
  ckpt/raft.pth --eval EPE Fl EPE-noise

python tools/test.py \
  configs/raft/raft_fly.py \
  ckpt/raft.pth --eval EPE Fl EPE-noise

python tools/test.py \
  configs/raft/raft_kitti.py \
  ckpt/raft.pth --eval EPE Fl EPE-noise

