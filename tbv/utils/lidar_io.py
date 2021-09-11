"""
Copyright (c) 2021
Argo AI, LLC, All Rights Reserved.

Notice: All information contained herein is, and remains the property
of Argo AI. The intellectual and technical concepts contained herein
are proprietary to Argo AI, LLC and may be covered by U.S. and Foreign
Patents, patents in process, and are protected by trade secret or
copyright law. This work is licensed under a CC BY-NC-SA 4.0 
International License.

Originating Authors: John Lambert
"""

from typing import Optional

import numpy as np
import torch

"""
LiDAR sweeps are stored on disk as dictionaries in Pytorch files. The encoding for (x,y,z,i,l)
is (x,y,z) as float16, and (i,l) as uint8.
"""

def load_tbv_sweep(pt_fpath: str, attrib_spec: str = "xyzil") -> Optional[np.ndarray]:
    """
    load_pytorch_encoded_point_cloud
    Convert to numpy, and convert to float32 where appropriate

    Args:
        pt_fpath: path to LiDAR file, stored as Pytorch files with a ".pt" suffix.
        attrib_spec: string of C characters, each char representing a desired point attribute
          x -> point x-coord
          y -> point y-coord
          z -> point z-coord
          i -> point intensity/reflectance
          l -> laser number of laser from which point was returned

    Returns:
        arr: Array of shape (N, C). If attrib_str is invalid, `None` will be returned
    """
    possible_attributes = ["x", "y", "z", "i", "l"]
    if not all([a in possible_attributes for a in attrib_spec]):
        return None

    load_types = {'x': torch.float32, 'y': torch.float32, 'z': torch.float32, 'i': torch.uint8, 'l': torch.uint8}
    pytorch_data = torch.load(pt_fpath)
    # now in Numpy
    attrib_dict = {k:v.type(load_types[k]).numpy() for k,v in pytorch_data.items()}
    # return only the requested point attributes
    attrib_arrs = [attrib_dict[a] for a in attrib_spec]
    # join arrays of the same shape along new dimension
    return np.stack(attrib_arrs, axis=1)
