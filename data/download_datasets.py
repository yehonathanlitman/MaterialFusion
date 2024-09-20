# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import io
import zipfile
import requests
import tarfile

import gdown
import wget
import imageio

import numpy as np
import torch

def download_nerf_synthetic():
    TMP_ARCHIVE = "nerf_synthetic.zip"
    print("------------------------------------------------------------")
    print(" Downloading NeRF synthetic dataset")
    print("------------------------------------------------------------")
    nerf_synthetic_url = "https://drive.google.com/file/d/1qGP4rbKRJk1LLtffWmTpaZspDHjl1y_o/view?usp=sharing"
    gdown.download(url=nerf_synthetic_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting NeRF synthetic dataset")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('nerf_synthetic/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

def download_nerfactor():
    TMP_ARCHIVE = "nerfactor.zip"
    print("------------------------------------------------------------")
    print(" Downloading NeRFactor dataset")
    print("------------------------------------------------------------")
    hotdog_url = "https://drive.google.com/file/d/1INgHfdHlBCuiHO0DyfLvwTS8-tTpPUAd/view?usp=sharing"
    drums_url  = "https://drive.google.com/file/d/12_jxpTYgfh1P0pGiV_Yu5YqDps0cdDId/view?usp=sharing"
    ficus_url  = "https://drive.google.com/file/d/1djxkqif32eWdITjpJST4S-BDB3j5bqn9/view?usp=sharing"
    lego_url   = "https://drive.google.com/file/d/14sxlUJzTTz31Yj_njS2-lq5QeLiTC1cv/view?usp=sharing"

    nerfactor_scenes = [hotdog_url, drums_url, ficus_url, lego_url]
    scene_name = ["hotdog_2163", "drums_3072", "ficus_2188", "lego_3072"]

    for scene, name in zip(nerfactor_scenes, scene_name):
        gdown.download(url=scene, output=TMP_ARCHIVE, quiet=False, fuzzy=True)
        print("------------------------------------------------------------")
        print(" Extracting NeRFactor dataset ", name)
        print("------------------------------------------------------------")
        archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
        archive.extractall(os.path.join("nerfactor", name))
        archive.close()
        os.remove(TMP_ARCHIVE)

def download_materialfusion_dataset():
    TMP_ARCHIVE = "materialfusion_dataset.zip"
    print("------------------------------------------------------------")
    print(" Downloading MaterialFusion dataset")
    print("------------------------------------------------------------")
    materialfusion_dataset_url = "https://drive.google.com/file/d/1xcmn6AO1KL22qcWjWc8cm31D8xR4LBrQ/view?usp=sharing"
    gdown.download(url=materialfusion_dataset_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting MaterialFusion dataset")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('materialfusion_dataset/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

def download_stablematerial_dataset():
    TMP_ARCHIVE = "stablematerial_dataset.zip"
    print("------------------------------------------------------------")
    print(" Downloading StableMaterial dataset")
    print("------------------------------------------------------------")
    stablematerial_dataset_url = "https://drive.google.com/file/d/169jJsEji7BW4QXQasqK0ReLwJQC7tnck/view?usp=sharing"
    gdown.download(url=stablematerial_dataset_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting StableMaterial dataset")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('stablematerial_dataset/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

def download_stanford_orb():
    print("------------------------------------------------------------")
    print(" Downloading Stanford-ORB dataset")
    print("------------------------------------------------------------")

    url = 'https://downloads.cs.stanford.edu/viscam/StanfordORB/blender_LDR.tar.gz'
    fname = wget.download(url)

    print("------------------------------------------------------------")
    print(" Extracting Stanford-ORB dataset")
    print("------------------------------------------------------------")

    if fname.endswith("tar.gz"):
        tar = tarfile.open(fname, "r:gz")
        tar.extractall()
        tar.close()

download_stablematerial_dataset()
download_materialfusion_dataset()
download_nerf_synthetic()
download_nerfactor()
download_stanford_orb()
print("Completed")