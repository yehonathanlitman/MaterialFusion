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

import gdown
import imageio

import numpy as np
import torch

def download_stablematerial():
    TMP_ARCHIVE = "stablematerial-model.zip"
    print("------------------------------------------------------------")
    print(" Downloading StableMaterial checkpoint")
    print("------------------------------------------------------------")
    stablematerial_model_url = "https://drive.google.com/file/d/1RDk3cvci1BPAVDWrk2ZzLLPwIlTLkt_4/view?usp=sharing"
    gdown.download(url=stablematerial_model_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting StableMaterial checkpoint")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('stablematerial-model/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

def download_stablematerial_mv():
    TMP_ARCHIVE = "stablematerial-mv-model.zip"
    print("------------------------------------------------------------")
    print(" Downloading StableMaterial-MV checkpoint")
    print("------------------------------------------------------------")
    stablematerial_mv_model_url = "https://drive.google.com/file/d/1J6EmhgXyjKgNa1prz7PckrsV5GsfvFXl/view?usp=sharing"
    gdown.download(url=stablematerial_mv_model_url, output=TMP_ARCHIVE, quiet=False, fuzzy=True)

    print("------------------------------------------------------------")
    print(" Extracting StableMaterial-MV checkpoint")
    print("------------------------------------------------------------")
    archive = zipfile.ZipFile(TMP_ARCHIVE, 'r')
    for zipinfo in archive.infolist():
        if zipinfo.filename.startswith('stablematerial-mv-model/'):
            archive.extract(zipinfo)
    archive.close()
    os.remove(TMP_ARCHIVE)

download_stablematerial()
download_stablematerial_mv()
print("Completed")