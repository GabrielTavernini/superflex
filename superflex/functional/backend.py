# https://github.dev/mit-han-lab/pvcnn/blob/master/modules/functional/backend.py

import os
from torch.utils.cpp_extension import load

_src_path = os.path.dirname(os.path.abspath(__file__))
_backend = load(name='_pvcnn_backend',
                extra_cflags=['-O3', '-std=c++17'],
                sources=[os.path.join(_src_path,'src', f) for f in [
                    'voxelization/vox.cpp',
                    'voxelization/vox.cu',
                    'interpolate/trilinear_devox.cpp',
                    'interpolate/trilinear_devox.cu',
                    'bindings.cpp',
                ]]
                )

__all__ = ['_backend']