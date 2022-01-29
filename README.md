# Crack Optical Flow

This repo is a final project of [winter school](https://apps.ucu.edu.ua/en/machine-learning-research-winter-school/) "
How to run an effective machine learning research". Code is based on [mmflow repo](https://github.com/open-mmlab/mmflow)
.

## Goal

Research how stable SOTA approaches to optical flow calculation with noise and adversarial attacks.

## Installation

Follow [installation guide](code/docs/en/install.md), [dataset preparation](/code/docs/en/dataset_prepare.md) and
download models [ckpt](code/configs)

## Model comparison without noise

| Model                   | Trained on                                                      | Sintel Clean EPE / Fl / Noise | Sintel Fina EPE / Fl / Noise | FlyingChairs EPE / Fl / Noise | Kitti 2012 EPE / Fl/ Noise | Kitti 2015 EPE / Fl / Noise | 
|-------------------------|-----------------------------------------------------------------|-------------------------------|------------------------------|-------------------------------|----------------------------|-----------------------------|
| [PWC](PWC.md)           | FlyingChairs + FlyingThing3d subset + Sintel + KITTI2015 + HD1K | 1.89 / 6.26 / 0.35            | 2.39 / 8.23 / 0.58           | 2.91 / 11.83 / 0.50           | 2.27 / 7.78 / 0.127        | 2.54 / 8.7 / 0.16           | 
| [FlowNet2](FlowNet2.md) | FlyingThing3d subset                                            | 1.78 / 6.3  / 0.43            | 3.31 / 10.84 / 0.83          | 1.62 / 7.6 / 0.33             | 3 / 13.8 / 0.13            | 8 / 25.1 / 0.4              |
| MaskFlowNet             | Flying Chairs + Flying Thing3d subset                           | 2.29 /7.88 / 0.34             | 3.7 / 11.9 / 0.53            | 1.85 / 9.6 / 0.21             | 3.82/ 17.6 / 0.123         | 9.7 / 29.27 / 0.3           |
| [RAFT](RAFT.md)         | FlyingChairs, FlyingThing3d, Sintel, KITTI2015, and HD1K.       | 0.73 / 2.86 / 0.1             | 1.48 / 5.27 / 0.29           | 1.25 / 4.3 / 0.14             | 1.26 / 4.46 / 0.04         | 1.76 / 6.17 / 0.07          |
| [GMA](GMA.md)           | FlyingChairs, FlyingThing3d, Sintel, KITTI2015, and HD1K.       | 0.63 / 2.62 / 0.09            | 0.94 / 4.33 / 0.22           | 1.27 / 4.52 / 0.18            | 1.67 / 6.62 / 0.16         | 2.78 / 9.34 / 0.51          |

To get same results run `bash code/metrics.sh`

## Model samples

All models [output](https://drive.google.com/drive/folders/1VZjwkBinIB2MSfiGVBuJPCV_Z1r0nwfP?usp=sharing)

