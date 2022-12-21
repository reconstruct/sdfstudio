# Copyright 2022 The Nerfstudio Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data parser for friends dataset"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Type

import numpy as np
import torch
import torchvision.transforms.functional as F
from PIL import Image
from rich.console import Console
from torchtyping import TensorType

from nerfstudio.cameras import camera_utils
from nerfstudio.cameras.cameras import Cameras, CameraType
from nerfstudio.data.dataparsers.base_dataparser import (
    DataParser,
    DataParserConfig,
    DataparserOutputs,
)
from nerfstudio.data.scene_box import SceneBox
from nerfstudio.utils.io import load_from_json

CONSOLE = Console()

def read_pfm(pfm_file_path: str)-> torch.Tensor:
    """parses PFM file into torch float tensor

    :param pfm_file_path: path like object that contains full path to the PFM file

    :returns: parsed PFM file of shape CxHxW
    """
    color = None
    width = None
    height = None
    scale = None
    data_type = None
    with open(pfm_file_path, 'rb') as file:
        header = file.readline().decode('UTF-8').rstrip()

        if header == 'PF':
            color = True
        elif header == 'Pf':
            color = False
        else:
            raise Exception('Not a PFM file.')
        dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('UTF-8'))

        if dim_match:
            width, height = map(int, dim_match.groups())
        else:
            raise Exception('Malformed PFM header.')

        # scale = float(file.readline().rstrip())
        scale = float((file.readline()).decode('UTF-8').rstrip())
        if scale < 0: # little-endian
            data_type = '<f'
        else:
            data_type = '>f' # big-endian
        data_string = file.read()
        data = np.fromstring(data_string, data_type)
        shape = (height, width, 3) if color else (height, width)
        data = np.reshape(data, shape)
        data = np.ascontiguousarray(np.flip(data, 0))
    return torch.from_numpy(data).view(height, width, -1).permute(2, 0, 1)



def get_image(image_filename, alpha_color=None) -> TensorType["image_height", "image_width", "num_channels"]:
    """Returns a 3 channel image.

    Args:
        image_idx: The image index in the dataset.
    """
    pil_image = Image.open(image_filename)
    np_image = np.array(pil_image, dtype="uint8")  # shape is (h, w, 3 or 4)
    assert len(np_image.shape) == 3
    assert np_image.dtype == np.uint8
    assert np_image.shape[2] in [3, 4], f"Image shape of {np_image.shape} is in correct."
    image = torch.from_numpy(np_image.astype("float32") / 255.0)
    if alpha_color is not None and image.shape[-1] == 4:
        assert image.shape[-1] == 4
        image = image[:, :, :3] * image[:, :, -1:] + alpha_color * (1.0 - image[:, :, -1:])
    else:
        image = image[:, :, :3]
    return image


def get_depths_and_normals(image_idx: int, depths, normals):
    """function to process additional depths and normal information

    Args:
        image_idx: specific image index to work with
        semantics: semantics data
    """

    # depth
    depth = depths[image_idx]
    # normal
    normal = normals[image_idx]

    return {"depth": depth, "normal": normal}


@dataclass
class ReconstructDataParserConfig(DataParserConfig):
    """Scene dataset parser config"""

    _target: Type = field(default_factory=lambda: ReconstructDataset)
    """target class to instantiate"""
    data: Path = Path("data/DTU/scan65")
    """Directory specifying location of data."""
    include_mono_prior: bool = False
    """whether or not to include loading of normal """
    skip_every_for_val_split: int = 1
    """sub sampling validation images"""
    auto_orient: bool = False


@dataclass
class ReconstructDataset(DataParser):
    """SDFStudio Dataset"""

    config: ReconstructDataParserConfig

    def _generate_dataparser_outputs(self, split="train"):  # pylint: disable=unused-argument,too-many-statements
        # load meta data
        meta = load_from_json(self.config.data / "transforms.json")

        indices = list(range(len(meta["frames"])))
        # subsample to avoid out-of-memory for validation set
        if split != "train" and self.config.skip_every_for_val_split >= 1:
            indices = indices[:: self.config.skip_every_for_val_split]

        image_filenames = []
        depth_images = []
        normal_images = []
        camera_to_worlds = []
        for i, frame in enumerate(meta["frames"]):
            if i not in indices:
                continue

            image_filename = self.config.data / frame["file_path"]

            camtoworld = torch.tensor(frame["transform_matrix"]).view(4, 4)

            # append data
            image_filenames.append(image_filename)
            camera_to_worlds.append(camtoworld)

            if self.config.include_mono_prior:
                assert ('normal_file_path' in frame) and ('depth_file_path' in frame)
                # load mono depth
                depth = read_pfm(self.config.data / frame["depth_file_path"])
                depth_images.append(torch.from_numpy(depth).float())

                # load mono normal
                with Image.open(self.config.data / frame["normal_file_path"]) as img:
                    normal = F.to_tensor(img).float()

                # transform normal to world coordinate system
                normal = normal * 2.0 - 1.0  # omnidata output is normalized so we convert it back to normal here

                rot = camtoworld[:3, :3]

                normal_map = normal.reshape(3, -1)
                normal_map = torch.nn.functional.normalize(normal_map, p=2, dim=0)

                normal_map = rot @ normal_map
                normal_map = normal_map.permute(1, 0).reshape(*normal.shape[1:], 3)
                normal_images.append(normal_map)

        camera_to_worlds = torch.stack(camera_to_worlds)

        # Convert from COLMAP's/OPENCV's camera coordinate system to nerfstudio
        camera_to_worlds[:, 0:3, 1:3] *= -1

        if self.config.auto_orient:
            camera_to_worlds, transform = camera_utils.auto_orient_and_center_poses(
                camera_to_worlds,
                method="up",
                center_poses=False,
            )

            # we should also transform normal accordingly
            normal_images_aligned = []
            for normal_image in normal_images:
                h, w, _ = normal_image.shape
                normal_image = transform[:3, :3] @ normal_image.reshape(-1, 3).permute(1, 0)
                normal_image = normal_image.permute(1, 0).reshape(h, w, 3)
                normal_images_aligned.append(normal_image)
            normal_images = normal_images_aligned

        # scene box from meta data
        aabb_scale = meta['aabb_scale']
        aabb = torch.tensor([
            [-1, -1, -1], [1, 1, 1]
        ]).float() * aabb_scale
        # aabb = torch.tensor(meta_scene_box["aabb"], dtype=torch.float32)
        scene_box = SceneBox( aabb=aabb)

        height, width = meta["h"], meta["w"]
        fx, fy = meta['fl_x'], meta['fl_y']
        cx, cy = meta['cx'], meta['cy']
        cameras = Cameras(
            fx=fx,
            fy=fy,
            cx=cx,
            cy=cy,
            height=height,
            width=width,
            camera_to_worlds=camera_to_worlds[:, :3, :4],
            camera_type=CameraType.PERSPECTIVE,
        )

        # TODO supports downsample
        # cameras.rescale_output_resolution(scaling_factor=1.0 / self.config.downscale_factor)

        if self.config.include_mono_prior:
            additional_inputs_dict = {
                "cues": {"func": get_depths_and_normals, "kwargs": {"depths": depth_images, "normals": normal_images}}
            }
        else:
            additional_inputs_dict = {}

        dataparser_outputs = DataparserOutputs(
            image_filenames=image_filenames,
            cameras=cameras,
            scene_box=scene_box,
            additional_inputs=additional_inputs_dict,
            depths=depth_images,
            normals=normal_images,
        )
        return dataparser_outputs
