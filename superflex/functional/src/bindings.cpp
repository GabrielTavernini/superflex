#include <pybind11/pybind11.h>

#include "interpolate/trilinear_devox.hpp"
#include "voxelization/vox.hpp"

PYBIND11_MODULE(_pvcnn_backend, m) {
m.def("trilinear_devoxelize_forward", &trilinear_devoxelize_forward,
        "Trilinear Devoxelization forward (CUDA)");
  m.def("trilinear_devoxelize_backward", &trilinear_devoxelize_backward,
        "Trilinear Devoxelization backward (CUDA)");
  m.def("avg_voxelize_forward", &avg_voxelize_forward,
        "Voxelization forward with average pooling (CUDA)");
  m.def("avg_voxelize_backward", &avg_voxelize_backward,
        "Voxelization backward (CUDA)");
}
