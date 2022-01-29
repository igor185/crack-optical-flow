import os.path as osp
from argparse import ArgumentParser
from typing import Sequence

import cv2
import mmcv
import numpy as np
import torch
import torch.optim as optim
from matplotlib import pyplot as plt
from numpy import ndarray
from tqdm import tqdm

from mmflow.apis import inference_model, init_model
from mmflow.datasets import visualize_flow, write_flow
torch.autograd.set_detect_anomaly(True)

def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--config', help='Config file')
    parser.add_argument('--checkpoint', help='Checkpoint file')
    parser.add_argument(
        '--device', default='cuda:0', help='Device used for inference')
    args = parser.parse_args()
    return args


def apply(img1, img2, noise, box, with_noise=False):
    if with_noise:
        img1[:, :, box[0]:box[2], box[1]:box[3]] = noise.reshape(1, 3, 200, 200)
        img2[:, :, box[0]:box[2], box[1]:box[3]] = noise.reshape(1, 3, 200, 200)
    return torch.concat([img1, img2], 1)


def optimize(img1, img2, noise, box, model):
    optimizer = optim.Adam([noise])

    frames = []
    for i in tqdm(range(10001)):
        b1 = img1.clone()
        b2 = img2.clone()
        with torch.no_grad():
            noise.data.clamp_(0, 1)
        optimizer.zero_grad()

        output = model(apply(b1, b2, noise, box, with_noise=False))[0]["flow"]
        output_noisy = model(apply(b1, b2, noise, box, with_noise=True))[0]["flow"]
        on = output_noisy.clone()
        on[box[0]:box[2], box[1]:box[3]] = output[box[0]:box[2], box[1]:box[3]]

        loss = -torch.square(output - on).mean()
        loss.backward()
        optimizer.step()
        if i % 50 == 0 and i != 0:
            # print(i, loss)
            img_full = apply(b1, b2, noise, box, with_noise=True)[0].permute(1, 2, 0).detach().cpu().numpy()
            img1t = img_full[:, :, :3]
            fl1 = visualize_flow(output.detach().cpu().numpy()) / 255.
            fl2 = visualize_flow(output_noisy.detach().cpu().numpy()) / 255.
            result = np.clip(np.concatenate([img1t, fl1, fl2], 1), a_min=0, a_max=1)
            frames.append(result)
            plt.imshow(result)
            plt.show()
    create_video(frames, "result.mp4", cv2.VideoWriter_fourcc(*'mp4v'), 10, (frames[0].shape[1], frames[0].shape[0]))
    with torch.no_grad():
        noise = noise.data.clamp_(0, 1)

    return noise


def main(args):
    # build the model from a config file and a checkpoint file
    model = init_model(args.config, args.checkpoint, device=args.device)
    im1 = cv2.imread("./eval/frame-001.jpg")
    im2 = cv2.imread("./eval/frame-002.jpg")
    im1 = cv2.resize(im1, (576, 1024)) / 255.
    im2 = cv2.resize(im2, (576, 1024)) / 255.
    b1 = torch.tensor(im1).permute(2, 0, 1).to(args.device).float().unsqueeze(0)
    b2 = torch.tensor(im2).permute(2, 0, 1).to(args.device).float().unsqueeze(0)
    # batch = torch.concat([b1, b2], 1)
    # flow = model(batch, test_mode=True)[0]["flow"]
    # result = visualize_flow(flow)

    box = [400, 200, 600, 400]
    noise = torch.tensor(np.random.normal(0, 1, 200 * 200 * 3))
    noise.requires_grad_(True)
    model.requires_grad_(False)
    b1.requires_grad_(False)
    b2.requires_grad_(False)
    optimize(b1, b2, noise, box, model)


def create_video(frames: Sequence[ndarray], out: str, fourcc: int, fps: int,
                 size: tuple) -> None:
    """Create a video to save the optical flow.

    Args:
        frames (list, tuple): Image frames.
        out (str): The output file to save visualized flow map.
        fourcc (int): Code of codec used to compress the frames.
        fps (int):      Framerate of the created video stream.
        size (tuple): Size of the video frames.
    """
    # init video writer
    video_writer = cv2.VideoWriter(out, fourcc, fps, size, True)

    for frame in frames:
        video_writer.write((frame * 255).astype("uint8"))
    video_writer.release()


if __name__ == '__main__':
    args = parse_args()
    main(args)
