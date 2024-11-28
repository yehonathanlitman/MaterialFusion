# Copyright (c) 2020-2022, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import glob
import json
import imageio.v2 as imageio

import torch
import numpy as np

from render import util

from dataset import Dataset
from torchvision import transforms

###############################################################################
# NERF image based dataset (synthetic)
###############################################################################

def _load_img(path):
    files = glob.glob(path + '.*')
    assert len(files) > 0, "Tried to find image file for: %s, but found 0 files" % (path)
    img = util.load_image_raw(files[0])
    if img.dtype != np.float32: # LDR image
        img = torch.tensor(img / 255, dtype=torch.float32)
        img[..., 0:3] = util.srgb_to_rgb(img[..., 0:3])
    else:
        img = torch.tensor(img, dtype=torch.float32)
    return img

class DatasetNERFTrain(Dataset):
    def __init__(self, cfg_path, args, examples=None):
        self.examples = examples
        self.base_dir = os.path.dirname(cfg_path)
        self.limiter = 10 if "train" in cfg_path else 3
        self.args = args

        # Load config / transforms
        self.cfg = json.load(open(cfg_path, 'r'))
        self.n_images = len(self.cfg['frames'])

        # Determine resolution & aspect ratio
        self.resolution = _load_img(os.path.join(self.base_dir, self.cfg['frames'][0]['file_path'])).shape[0:2]
        self.aspect = self.resolution[1] / self.resolution[0]
        
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((args.resolution, args.resolution), antialias = True),
                transforms.Normalize([0.5], [0.5])
            ]
        )
        self.resize = transforms.Resize((args.resolution, args.resolution), antialias = True)
        self.normalize = transforms.Normalize([0.5], [0.5])

        print("DatasetNERFTrain: %d images with shape [%d, %d]" % (self.n_images, self.resolution[0], self.resolution[1]))

        if args.pre_load:
            # Pre-load from disc to avoid slow png parsing
            self.preloaded_data = []
            for i in range(self.n_images):
                self.preloaded_data += [self._parse_frame(self.cfg, i)]
                if (i+1) % 200 == 0:
                    print("Preloading images: %d/%d" % (i+1, self.n_images))

    def _parse_frame(self, cfg, idx, cam_near_far=[0.1, 1000.0]):
        # Config projection matrix (static, so could be precomputed)
        fovy   = util.fovx_to_fovy(cfg['camera_angle_x'], self.aspect)
        proj   = util.perspective(fovy, self.aspect, cam_near_far[0], cam_near_far[1])

        # Load image data and modelview matrix
        img    = _load_img(os.path.join(self.base_dir, cfg['frames'][idx]['file_path']))
        albedo    = imageio.imread(os.path.join(self.base_dir, cfg['frames'][idx]['file_path_albedo'] + ".png"))
        orm    = imageio.imread(os.path.join(self.base_dir, cfg['frames'][idx]['file_path_orm'] + ".png"))
        mask = img[..., 3:]
        img = img[..., :3]
        albedo = albedo[..., :3]
        orm = orm[..., :3]

        img = self.normalize(img.permute(2, 0, 1))
        albedo = self.transform(albedo)
        orm = self.transform(orm)
        mask = self.resize(mask.permute(2, 0, 1))

        envmap = cfg['frames'][idx]['envmap']
        mv     = torch.linalg.inv(torch.tensor(cfg['frames'][idx]['transform_matrix'], dtype=torch.float32))
        mv     = mv @ util.rotate_x(-np.pi / 2)

        campos = torch.linalg.inv(mv)[:3, 3]
        mvp    = proj @ mv

        return img, albedo, orm, mask, envmap, mv[None, ...], mvp[None, ...], campos[None, ...]

    def getMesh(self):
        return None

    def __len__(self):
        return self.n_images // self.limiter if self.examples is None else self.examples

    def __getitem__(self, itr):
        img      = []
        # fovy     = util.fovx_to_fovy(self.cfg['camera_angle_x'], self.aspect)

        if self.args.pre_load:
            img, albedo, orm, mask, envmap, mv, mvp, campos = self.preloaded_data[itr % self.n_images]
        else:
            img, albedo, orm, mask, envmap, mv, mvp, campos = self._parse_frame(self.cfg, itr % self.n_images)

        return {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : [self.args.resolution, self.args.resolution],
            'spp' : 16,
            'img' : img,
            'albedo' : albedo,
            'orm' : orm,
            'mask' : mask,
            'envmap' : envmap
        }
    
    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        out_batch = {
            'image': torch.stack([b['img'] for b in batch]),
            'albedo': torch.stack([b['albedo'] for b in batch]),
            'orm': torch.stack([b['orm'] for b in batch]),
            'mask': torch.stack([b['mask'] for b in batch]),
            'mvp': torch.stack([b['mvp'] for b in batch]),
            'campos': torch.stack([b['campos'] for b in batch]),
            'resolution': iter_res,
            'spp': iter_spp,
            'envmap': [b['envmap'] for b in batch]
        }
        return out_batch