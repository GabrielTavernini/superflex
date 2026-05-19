import random

import numpy as np
import colorsys
import torch
from plyfile import (PlyData, PlyElement)

def generate_ncolors(num):
    def get_n_hls_colors(num):
        hls_colors = []
        i = 0
        step = 360.0 / num
        while i < 360:
            h = i
            s = 90 + random.random() * 10
            l = 50 + random.random() * 10
            _hlsc = [h / 360.0, l / 100.0, s / 100.0]
            hls_colors.append(_hlsc)
            i += step
        return hls_colors
    rgb_colors = np.zeros((0,3), dtype=np.uint8)
    if num < 1:
        return rgb_colors
    hls_colors = get_n_hls_colors(num)
    for hlsc in hls_colors:
        _r, _g, _b = colorsys.hls_to_rgb(hlsc[0], hlsc[1], hlsc[2])
        r, g, b = [int(x * 255.0) for x in (_r, _g, _b)]
        rgb_colors = np.concatenate((rgb_colors,np.array([r,g,b])[np.newaxis,:]))
    return rgb_colors

def get_segmentation_from_assign_matrix(assign_matrix):
    if isinstance(assign_matrix, torch.Tensor):
        assign_matrix = assign_matrix.cpu().numpy()
    P = assign_matrix.shape[1]
    segmentation = np.argmax(assign_matrix, axis=1)
    colors = generate_ncolors(P)
    colored_pc = colors[segmentation]
    return colored_pc

def export_mesh_trimesh(mesh, filename):
    mesh.export(filename)

# def export_o3d_pc(pc, filename):
#     o3d.io.write_point_cloud(filename, pc)

def export_mesh(vertices, faces_idx, vertex_color, face_color, filename):
    if vertex_color is not None:
        vertex = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(vertices.shape[0]):
            vertex[i] = (vertices[i][0], vertices[i][1], vertices[i][2],vertex_color[i,0],vertex_color[i,1],vertex_color[i,2])
    else:
        vertex = np.zeros(vertices.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        for i in range(vertices.shape[0]):
            vertex[i] = (vertices[i][0], vertices[i][1], vertices[i][2])
    if face_color is not None:
        faces = np.zeros(faces_idx.shape[0], dtype=[('vertex_indices', 'i4', (3,)),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
        for i in range(faces_idx.shape[0]):
            faces[i] = ([faces_idx[i][0], faces_idx[i][1], faces_idx[i][2]],face_color[i,0],face_color[i,1],face_color[i,2])
    else:
        faces = np.zeros(faces_idx.shape[0], dtype=[('vertex_indices', 'i4', (3,))])
        for i in range(faces_idx.shape[0]):
            faces[i] = ([faces_idx[i][0], faces_idx[i][1], faces_idx[i][2]])

    ply_out = PlyData([PlyElement.describe(vertex, 'vertex', comments=['vertices']),
                       PlyElement.describe(faces, 'face')],text=True)
    ply_out.write(filename)
    return ply_out
