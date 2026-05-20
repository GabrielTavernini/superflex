import numpy as np
import torch
from typing import Dict
import trimesh

from superflex.utils.visualizations import generate_ncolors

def extend_dict(outdict):
    if isinstance(outdict['scale'], torch.Tensor): cls = torch
    else: cls = np
    
    B, N, _ = outdict['scale'].shape
    if 'tapering' not in outdict:
        outdict['tapering'] = cls.zeros((B, N, 2))
    if 'bending' not in outdict or outdict['bending'].shape[-1] != 6: 
        if 'bending_k' in outdict and 'bending_a' in outdict:
            outdict['bending'] = cls.stack([
                outdict['bending_k'][..., 0], outdict['bending_a'][..., 0],
                outdict['bending_k'][..., 1], outdict['bending_a'][..., 1],
                outdict['bending_k'][..., 2], outdict['bending_a'][..., 2]
            ], -1)
        else:
            outdict['bending'] = cls.zeros((B, N, 6))
    return outdict

class PointCloud:
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors

class PredictionHandler:
    def __init__(self, predictions: Dict[str, np.ndarray]):
        # P : number of primitives
        # N : number of points
        # BS: batch size
        self.names = predictions['names']
        self.pc = predictions['pc']                        # [BS, N, 3]
        self.assign_matrix = predictions['assign_matrix']  # [BS, N, P]
        self.scale = predictions['scale']                  # [BS, P, 3]
        self.rotation = predictions['rotation']            # [BS, P, 3, 3]
        self.translation = predictions['translation']      # [BS, P, 3]
        self.exponents = predictions['exponents']          # [BS, P, 2]
        self.exist = predictions['exist']                  # [BS, P]
        self.colors = generate_ncolors(self.translation.shape[1])  # Generate colors for each primitive

        # SuperFlex extension (bending and tapering)
        self.tapering = predictions['tapering']            # [BS, P, 2]
        self.bending = predictions['bending']              # [BS, P, 6] (kbz, alphaz, kbx, alphax, kby, alphay)
            
    def save_npz(self, filepath):
        """Save accumulated outputs to compressed npz file."""
        np.savez_compressed(
            filepath,
            names=np.array(self.names),
            pc=np.stack(self.pc),
            assign_matrix=np.stack(self.assign_matrix),
            scale=np.stack(self.scale),
            rotation=np.stack(self.rotation),
            translation=np.stack(self.translation),
            exponents=np.stack(self.exponents),
            exist=np.stack(self.exist),
            tapering=np.stack(self.tapering),
            bending=np.stack(self.bending),
        )
    
    @classmethod
    def from_npz(cls, path: str):
        data = np.load(path, allow_pickle=True)
        data_dict = {key: data[key] for key in data.files}
        return cls(extend_dict(data_dict))

    @classmethod # TODO test!!
    def from_outdict(cls, outdict, pcs, names):
        extend_dict(outdict)
        predictions = {
            'names': names, 
            'pc': pcs.cpu().numpy(), 
            'assign_matrix': outdict['assign_matrix'].cpu().numpy(), 
            'scale': outdict['scale'].cpu().numpy(), 
            'rotation': outdict['rotate'].cpu().numpy(),
            'translation': outdict['trans'].cpu().numpy(), 
            'exponents': outdict['shape'].cpu().numpy(), 
            'exist': outdict['exist'].cpu().numpy(),
            'tapering': outdict['tapering'].cpu().numpy(),
            'bending': outdict['bending'].cpu().numpy(), 
        }
        return cls(predictions)

    def append_outdict(self, outdict, pcs, names):
        extend_dict(outdict)
        self.names = np.concatenate((self.names, names), axis=0)
        self.pc = np.concatenate((self.pc, pcs.cpu().numpy()), axis=0)
        self.assign_matrix = np.concatenate((self.assign_matrix, outdict['assign_matrix'].cpu().numpy()), axis=0)
        self.scale = np.concatenate((self.scale, outdict['scale'].cpu().numpy()), axis=0)
        self.rotation = np.concatenate((self.rotation, outdict['rotate'].cpu().numpy()), axis=0)
        self.translation = np.concatenate((self.translation, outdict['trans'].cpu().numpy()), axis=0)
        self.exponents = np.concatenate((self.exponents, outdict['shape'].cpu().numpy()), axis=0)
        self.exist = np.concatenate((self.exist, outdict['exist'].cpu().numpy()), axis=0)
        self.tapering = np.concatenate((self.tapering, outdict['tapering'].cpu().numpy()), axis=0)
        self.bending = np.concatenate((self.bending, outdict['bending'].cpu().numpy()), axis=0)

    def get_segmented_pc(self, index):
        if isinstance(self.assign_matrix, torch.Tensor):
            assign_matrix = assign_matrix.cpu().numpy()[index]
        else:
            assign_matrix = self.assign_matrix[index]
        P = assign_matrix.shape[1]
        segmentation = np.argmax(assign_matrix, axis=1)
        colors = generate_ncolors(P)
        colored_pc = colors[segmentation]

        points = self.pc[index]
        colors = colored_pc / 255.0
        pc_obj = PointCloud(points=points, colors=colors)
        return pc_obj
    
    def get_segmented_pcs(self):
        pcs = []
        B = self.scale.shape[0]
        for b in range(B):
            pc = self.get_segmented_pc(b)
            pcs.append(pc)
        return pcs

    def get_meshes(self, resolution: int = 100, colors=True):
        meshes = []
        B = self.scale.shape[0]
        for b in range(B):
            try:
                mesh = self.get_mesh(b, resolution, colors)
                meshes.append(mesh)
            except Exception as e:
                print(f"Error generating mesh for index {b}: {e}")
                meshes.append(None)
        return meshes

    def get_mesh(self, index, resolution: int = 100, colors=True):
        P = self.scale.shape[1]

        vertices = []
        faces = []
        if colors:
            v_colors = []
            f_colors = []
        os_vertices = 0
        for p in range(P):
            if self.exist[index, p] > 0.5:
                mesh = self._superquadric_mesh(
                    self.scale[index, p], self.exponents[index, p],
                    self.rotation[index, p], self.translation[index, p], self.tapering[index, p], self.bending[index, p], resolution
                )
            
                vertices_cur, faces_cur = mesh

                vertices.append(vertices_cur)
                faces.append(faces_cur + os_vertices)

                if colors:
                    cur_color = self.colors[p]
                    v_colors.append(np.ones((vertices_cur.shape[0],3)) * cur_color)
                    f_colors.append(np.ones((faces_cur.shape[0],3)) * cur_color)

                os_vertices += len(vertices_cur)
        vertices = np.concatenate(vertices)
        faces = np.concatenate(faces)
        if colors:
            v_colors = np.concatenate(v_colors)/255.0
            f_colors = np.concatenate(f_colors)/255.0
            mesh = trimesh.Trimesh(vertices, faces, face_colors=f_colors, vertex_colors=v_colors)
        else:
            mesh = trimesh.Trimesh(vertices, faces)
                
        return mesh

    def apply_bending_axis(self, x, y, z, val_kb, val_alpha, axis):
        if np.abs(val_kb) < 1e-3: return x, y, z
        
        if axis == 'z':
            u, v_coord, w = x, y, z
        elif axis == 'x':
            u, v_coord, w = y, z, x
        elif axis == 'y':
            u, v_coord, w = z, x, y
        
        sin_alpha = np.sin(val_alpha)
        cos_alpha = np.cos(val_alpha)
        
        beta = np.arctan2(v_coord, u)
        r = np.sqrt(u**2 + v_coord**2) * np.cos(val_alpha - beta)

        # Clamp kb so 1/|kb| > r_max (prevents rho < 0 and self-intersection)
        # r_max = np.max(np.abs(r))
        # if r_max > 1e-6:
        #     kb_max = 0.95 / r_max
        #     if np.abs(val_kb) > kb_max:
        #         print(f"[bending {axis}] clamping |kb| from {np.abs(val_kb):.4f} to {kb_max:.4f} (r_max={r_max:.4f})")
        #         sign_kb = np.sign(val_kb)
        #         val_kb = sign_kb * kb_max
        #         if np.abs(val_kb) < 1e-3: return x, y, z
 
        inv_kb = 1.0 / val_kb
        gamma = w * val_kb
        rho = inv_kb - r
        R = inv_kb - rho * np.cos(gamma)

        expr = (R - r)
        u = u + expr * cos_alpha
        v_coord = v_coord + expr * sin_alpha
        w = rho * np.sin(gamma)
        
        if axis == 'z':
            return u, v_coord, w
        elif axis == 'x':
            return w, u, v_coord
        elif axis == 'y':
            return v_coord, w, u
 
    @staticmethod
    def _sample_superellipse_dc(a1, a2, e, theta_a, theta_b, N):
        """Divide-and-conquer arc-length sampling of a superellipse.
 
        Ported from superdec/fast_sampler. Distributes N angles between
        theta_a and theta_b proportionally to chord length so that the
        resulting points are approximately equally spaced in 2-D.
        """
        def fexp(x, p):
            return np.copysign(np.abs(x)**p, x)
 
        def xy(theta):
            return (a1 * fexp(np.cos(theta), e),
                    a2 * fexp(np.sin(theta), e))
 
        def dist(A, B):
            return np.sqrt((A[0] - B[0])**2 + (A[1] - B[1])**2)
 
        thetas = np.empty(N)
        thetas[0] = theta_a
        thetas[N - 1] = theta_b
 
        A = xy(theta_a)
        B = xy(theta_b)
        stack = [(A, B, theta_a, theta_b, N - 2, 1)]
 
        while stack:
            A, B, ta, tb, n, offset = stack.pop()
            if n <= 0:
                continue
            theta = (ta + tb) / 2
            C = xy(theta)
            dA = dist(A, C)
            dB = dist(C, B)
            total = dA + dB
            if total < 1e-12:
                nA = n // 2
            else:
                nA = int(round(dA / total * (n - 1)))
            nB = n - nA - 1
            thetas[nA + offset] = theta
            stack.append((A, C, ta, theta, nA, offset))
            stack.append((C, B, theta, tb, nB, offset + nA + 1))
 
        return thetas

    def _superquadric_mesh(self, scale, exponents, rotation, translation, tapering, bending, N):
        def f(o, m):
            sin_o = np.sin(o)
            return np.sign(sin_o) * np.abs(sin_o)**m
        def g(o, m):
            cos_o = np.cos(o)
            return np.sign(cos_o) * np.abs(cos_o)**m

        # Divide-and-conquer arc-length sampling (matches superdec/fast_sampler)
        # omega (u): azimuthal, superellipse in (x, y) plane
        u = self._sample_superellipse_dc(scale[0], scale[1], exponents[1], -np.pi, np.pi, N)
        # eta (v): polar, superellipse in (x, z) plane
        v = self._sample_superellipse_dc(scale[0], scale[2], exponents[0], -np.pi/2, np.pi/2, N)
        u = np.tile(u, N)
        v = np.repeat(v, N)
        if np.linalg.det(rotation) < 0:
            u = u[::-1]

        x = scale[0] * g(v, exponents[0]) * g(u, exponents[1])
        y = scale[1] * g(v, exponents[0]) * f(u, exponents[1])
        z = scale[2] * f(v, exponents[0])
        # Set poles to zero to account for numerical instabilities in f and g due to ** operator
        x[:N] = 0.0
        x[-N:] = 0.0

        kx, ky = tapering
        z_norm = z / scale[2]
        fx = kx * z_norm + 1
        fy = ky * z_norm + 1
        x = x*fx
        y = y*fy

        # Apply bending transformation
        x, y, z = self.apply_bending_axis(x, y, z, bending[4], bending[5], 'y')
        x, y, z = self.apply_bending_axis(x, y, z, bending[2], bending[3], 'x')
        x, y, z = self.apply_bending_axis(x, y, z, bending[0], bending[1], 'z')

        vertices = np.stack([x, y, z], axis=1) # Faster than concatenate expand_dims
        vertices = (rotation @ vertices.T).T + translation  

        # Create a grid of indices for the top-left corner of each quad
        i = np.arange(N - 1)
        j = np.arange(N - 1)
        J, I = np.meshgrid(j, i)
        
        # Indices for the quad vertices
        # p1: (i, j)     -> I*N + J
        # p2: (i, j+1)   -> I*N + (J+1)
        # p3: (i+1, j)   -> (I+1)*N + J
        # p4: (i+1, j+1) -> (I+1)*N + (J+1)
        p1 = I * N + J
        p2 = I * N + (J + 1)
        p3 = (I + 1) * N + J
        p4 = (I + 1) * N + (J + 1)
        t1 = np.stack([p1, p2, p3], axis=-1).reshape(-1, 3)
        t2 = np.stack([p3, p2, p4], axis=-1).reshape(-1, 3)        
        main_body_triangles = np.concatenate([t1, t2], axis=0)

        # Seam connection (wrapping u around): Connect last column to first column
        # Last vertex in row i: i*N + (N-1)
        # First vertex in row i: i*N
        # Last vertex in row i+1: (i+1)*N + (N-1)
        # First vertex in row i+1: (i+1)*N
        i_seam = np.arange(N - 1)
        p1_s = i_seam * N + (N - 1)
        p2_s = i_seam * N
        p3_s = (i_seam + 1) * N + (N - 1)
        p4_s = (i_seam + 1) * N
        t1_s = np.stack([p1_s, p2_s, p3_s], axis=-1)
        t2_s = np.stack([p3_s, p2_s, p4_s], axis=-1)
        seam_triangles = np.concatenate([t1_s, t2_s], axis=0)
        
        # Pole caps
        cap_triangles = np.array([
            [(N-1)*N+(N-1), (N-1)*N, (N-1)],
            [(N-1), (N-1)*N, 0]
        ])

        triangles = np.concatenate([main_body_triangles, seam_triangles, cap_triangles], axis=0)
        return np.array(vertices), np.array(triangles)
    