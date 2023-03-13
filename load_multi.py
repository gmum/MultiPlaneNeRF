import os
import torch
import numpy as np
import imageio 
import json
import torch.nn.functional as F
import cv2
import glob
from os.path import join
from multiplane_helpers import ImagePlanes

trans_t = lambda t : torch.Tensor([
    [1,0,0,0],
    [0,1,0,0],
    [0,0,1,t],
    [0,0,0,1]]).float()

rot_phi = lambda phi : torch.Tensor([
    [1,0,0,0],
    [0,np.cos(phi),-np.sin(phi),0],
    [0,np.sin(phi), np.cos(phi),0],
    [0,0,0,1]]).float()

rot_theta = lambda th : torch.Tensor([
    [np.cos(th),0,-np.sin(th),0],
    [0,1,0,0],
    [np.sin(th),0, np.cos(th),0],
    [0,0,0,1]]).float()


def pose_spherical(theta, phi, radius):
    c2w = trans_t(radius)
    c2w = rot_phi(phi/180.*np.pi) @ c2w
    c2w = rot_theta(theta/180.*np.pi) @ c2w
    c2w = torch.Tensor(np.array([[-1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1]])) @ c2w
    return c2w


def get_train_ids():
    base = [0, 1, 3, 4, 6,
            7, 8, 10, 11, 13,
            14, 15, 17, 18, 20,
            21, 22, 24, 25, 27,
            28, 29, 31, 32, 34]
    for i in base:
        yield i
        yield i + 36
        yield i + 72
        yield i + 108
        
def get_test_ids():
    base = [2, 5, 
            9, 12, 
            16, 19, 
            23, 26, 
            30, 33, 35]
    for i in base:
        yield i
        yield i + 36
        yield i + 72
        yield i + 108

def load_many_data(basedir):
    splits = ['train', 'val', 'test']
    focal = torch.Tensor([277.77])
    objects = {}
    test_objects = {}
    
    for file in glob.glob(join(basedir, 'train', '*.npz')):
        object = {}
        loaded = np.load(file)
        
        p = np.array(loaded['cam_poses']).astype(np.float32)
        imgs = (np.array(loaded['images'])).astype(np.float32)
        image_plane = ImagePlanes(focal, p[list(get_train_ids())], imgs[list(get_train_ids())], 100)
        
        object['images'] = imgs[list(get_train_ids())]
        object['poses'] = p[list(get_train_ids())]
        object['images_test'] = imgs[list(get_test_ids())]
        object['poses_test'] = p[list(get_test_ids())]
        object['image_plane'] = image_plane
        objects[file] = object
        
    for file in glob.glob(join(basedir, 'test', '*.npz')):
        object = {}
        loaded = np.load(file)
        
        p = np.array(loaded['cam_poses']).astype(np.float32)
        imgs = (np.array(loaded['images'])).astype(np.float32)
        image_plane = ImagePlanes(focal, p[list(get_train_ids())], imgs[list(get_train_ids())], 100)
        
        object['images'] = imgs[list(get_train_ids())]
        object['poses'] = p[list(get_train_ids())]
        object['images_test'] = imgs[list(get_test_ids())]
        object['poses_test'] = p[list(get_test_ids())]
        object['image_plane'] = image_plane
        test_objects[file] = object

    render_poses = torch.stack([pose_spherical(angle, -30.0, 3.8) for angle in np.linspace(-180,180,40+1)[:-1]], 0)
    
    return objects, test_objects, render_poses, [200, 200, focal]