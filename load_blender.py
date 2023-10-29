import os
import torch
import numpy as np
import imageio
import json
import torch.nn.functional as F
import cv2
from run_nerf_helpers import *

trans_t = lambda t: torch.Tensor([
    [1, 0, 0, 0],
    [0, 1, 0, 0],
    [0, 0, 1, t],
    [0, 0, 0, 1]]).float()

rot_phi = lambda phi: torch.Tensor([
    [1, 0, 0, 0],
    [0, np.cos(phi), -np.sin(phi), 0],
    [0, np.sin(phi), np.cos(phi), 0],
    [0, 0, 0, 1]]).float()

rot_theta = lambda th: torch.Tensor([
    [np.cos(th), 0, -np.sin(th), 0],
    [0, 1, 0, 0],
    [np.sin(th), 0, np.cos(th), 0],
    [0, 0, 0, 1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi / 180. * np.pi) @ c2w
    c2w = rot_theta(theta / 180. * np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
    return c2w


def load_blender_data(basedir, divide_fac=1, testskip=1):
    splits = ['train', 'val', 'test']
    metas = {}
    for s in splits:
        with open(os.path.join(basedir, 'transforms_{}.json'.format(s)), 'r') as fp:
            metas[s] = json.load(fp)

    all_imgs = []
    all_poses = []
    all_times = []
    counts = [0]
    for s in splits:
        meta = metas[s]
        imgs = []
        poses = []
        times = []
        if s == 'train' or testskip == 0:
            skip = 1
        else:
            skip = testskip

        for t, frame in enumerate(meta['frames'][::skip]):
            fname = os.path.join(basedir, frame['file_path'] + '.png')
            img = imageio.imread(fname)

            # create channel
            cur_time = frame['time'] if 'time' in frame else float(t) / (len(meta['frames'][::skip]) - 1)
            time_channel = np.full_like(img[:, :, 0], cur_time, dtype=np.float32)  # Assuming img is (H, W, 4) for RGBA

            img_with_time = np.dstack((img, time_channel))
            imgs.append(img_with_time)
            poses.append(np.array(frame['transform_matrix']))
            times.append(cur_time)

        assert times[0] == 0, "Time must start at 0"

        imgs = np.array(imgs).astype(np.float32)  # keep all 5 channels (RGBA)
        imgs[:, :, :, :4] = imgs[:, :, :, :4] / 255.
        poses = np.array(poses).astype(np.float32)
        times = np.array(times).astype(np.float32)
        counts.append(counts[-1] + imgs.shape[0])
        all_imgs.append(imgs)
        all_poses.append(poses)
        all_times.append(times)

    i_split = [np.arange(counts[i], counts[i + 1]) for i in range(3)]

    imgs = np.concatenate(all_imgs, 0)
    poses = np.concatenate(all_poses, 0)
    times = np.concatenate(all_times, 0)

    H, W = imgs[0].shape[:2]
    camera_angle_x = float(meta['camera_angle_x'])
    focal = .5 * W / np.tan(.5 * camera_angle_x)

    render_poses = torch.stack([pose_spherical(angle, -30.0, 4.0) for angle in np.linspace(-180, 180, 40 + 1)[:-1]], 0)
    render_times = torch.linspace(0., 1., render_poses.shape[0])

    if divide_fac != 1:
        H = H // divide_fac
        W = W // divide_fac
        focal = focal / divide_fac

        imgs_half_res = np.zeros((imgs.shape[0], H, W, 5))
        for i, img in enumerate(imgs):
            imgs_half_res[i] = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
        imgs = imgs_half_res
        print(imgs.dtype)
        imgs = imgs.astype(np.float32)

    return imgs, poses, times, render_poses, render_times, [H, W, focal], i_split