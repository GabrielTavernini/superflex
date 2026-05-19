from scipy.spatial import ConvexHull
import random
import numpy as np
from .transform import PointCloudsTransform
# from superoptim.utils import timing

def get_random_camera(points, phi_range=(0, np.pi)):
    """Sample a random camera position on a sphere around the point cloud.

    Args:
        points: (N, 3+) array of points.
        phi_range (tuple): (min_phi, max_phi) range for the polar angle in radians.
            Default (0, pi) covers the full sphere. Use e.g. (pi/4, 3*pi/4) to
            exclude the top and bottom caps.
    """
    # Handle empty points safely
    if points is None or points.shape[0] == 0:
        return np.array([0.0, 0.0, 0.0])

    bbox_min = points[:, :3].min(axis=0)
    bbox_max = points[:, :3].max(axis=0)
    bbox_center = (bbox_min + bbox_max) / 2
    bbox_size = np.linalg.norm(bbox_max - bbox_min)
    
    # Random camera on sphere around bbox
    theta = random.uniform(0, 2 * np.pi)
    phi = random.uniform(phi_range[0], phi_range[1])
    camera_pos = bbox_center + bbox_size * np.array([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi)
    ])
    return camera_pos

class BackFaceCulling(PointCloudsTransform):
    """Remove points that face away from a given camera viewpoint (back-face culling).

    Args:
        camera_position (tuple): (x, y, z) position of the camera viewpoint.
            If None, uses a random position on a sphere. Default: None.
        threshold (float): Dot product threshold for culling. Points with 
            dot(normal, view_direction) < threshold are removed. Default: 0.0.
        keep_points_without_normals (bool): If True, keeps points that don't have
            normal information. Default: True.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        camera_position=None,
        threshold=0.0,
        keep_points_without_normals=True,
        phi_range=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.camera_position = camera_position
        self.threshold = threshold
        self.keep_points_without_normals = keep_points_without_normals
        self.phi_range = phi_range if phi_range is not None else (0, np.pi)

    @property
    def targets_as_params(self):
        return ["points", "normals"]

    # @timing
    def get_params_dependent_on_targets(self, params):
        points = params["points"]
        # Safe early return for empty point clouds
        if points is None or points.shape[0] == 0:
            camera_pos = np.array(self.camera_position) if self.camera_position is not None else np.array([0.0, 0.0, 0.0])
            return {"mask": np.zeros(0, dtype=bool), "camera_position": camera_pos}

        normals = params.get("normals", None)
        if self.camera_position is None:
            camera_pos = get_random_camera(points, phi_range=self.phi_range)
        else:
            camera_pos = np.array(self.camera_position)
        
        # Compute view direction for each point
        view_directions = camera_pos - points[:, :3]
        view_directions = view_directions / (np.linalg.norm(view_directions, axis=1, keepdims=True) + 1e-8)
        
        # Compute mask based on normals
        if normals is not None and normals.shape[0] == points.shape[0]:
            # Dot product between normals and view directions
            dots = np.sum(normals[:, :3] * view_directions, axis=1)
            mask = dots >= self.threshold
        elif self.keep_points_without_normals:
            mask = np.ones(points.shape[0], dtype=bool)
        else:
            mask = np.zeros(points.shape[0], dtype=bool)
        
        return {"mask": mask, "camera_position": camera_pos}

    def apply(self, points, mask, **params):
        return points[mask]

    def apply_to_normals(self, normals, mask, **params):
        if normals is None:
            return None
        return normals[mask]

    def apply_to_features(self, features, mask, **params):
        if features is None:
            return None
        return features[mask]

    def apply_to_labels(self, labels, mask, **params):
        if labels is None:
            return None
        return labels[mask]

    def get_transform_init_args_names(self):
        return ("camera_position", "threshold", "keep_points_without_normals", "phi_range")


class RandomOcclusion(PointCloudsTransform):
    """Randomly occlude points by placing virtual occluders in the scene.

    Args:
        num_occluders (int or tuple): Number of occluders to place. If tuple (min, max),
            samples randomly. Default: (1, 3).
        occluder_type (str): Type of occluder - 'plane', 'sphere', or 'box'. Default: 'plane'.
        size_range (tuple): Range for occluder size as a FRACTION of object diagonal. 
            E.g., (0.1, 0.4) means 10% to 40% of object size. Default: (0.1, 0.5).
        camera_position (tuple): (x, y, z) camera position for occlusion. If None,
            uses random position. Default: None.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels
    """

    def __init__(
        self,
        num_occluders=(1, 2),
        occluder_type="sphere",
        size_range=(0.02,0.2),
        camera_position=None,
        phi_range=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        if isinstance(num_occluders, int):
            self.num_occluders = (num_occluders, num_occluders)
        else:
            self.num_occluders = num_occluders
        self.occluder_type = occluder_type
        self.size_range = size_range
        self.camera_position = camera_position
        self.phi_range = phi_range if phi_range is not None else (0, np.pi)

    @property
    def targets_as_params(self):
        return ["points"]

    def get_params_dependent_on_targets(self, params):
        points = params["points"]
        # Handle empty points safely
        if points is None or points.shape[0] == 0:
            camera_pos = np.array(self.camera_position) if self.camera_position is not None else np.array([0.0, 0.0, 0.0])
            return {"mask": np.zeros(0, dtype=bool), "occluders": [], "camera_position": camera_pos}

        bbox_min = points[:, :3].min(axis=0)
        bbox_max = points[:, :3].max(axis=0)
        bbox_center = (bbox_min + bbox_max) / 2
        # Use diagonal of the AABB as the scale reference
        object_scale = np.linalg.norm(bbox_max - bbox_min)
        
        # Get camera position
        if self.camera_position is None:
            camera_pos = get_random_camera(points, phi_range=self.phi_range)
        else:
            camera_pos = np.array(self.camera_position)
            
        forward = bbox_center - camera_pos
        forward /= (np.linalg.norm(forward) + 1e-8)
        
        # Calculate Up and Right vectors to shift perpendicular to view line
        # Use arbitrary vector to compute cross product
        tmp_up = np.array([0, 1, 0]) if abs(forward[1]) < 0.9 else np.array([1, 0, 0])
        right = np.cross(forward, tmp_up)
        right /= (np.linalg.norm(right) + 1e-8)
        up = np.cross(right, forward)
        
        # Sample number of occluders
        n_occluders = random.randint(self.num_occluders[0], self.num_occluders[1])
        if self.occluder_type == "plane":
            n_occluders = 1
        
        occluders = []
        for _ in range(n_occluders):
            t = random.uniform(0.4, 0.6)
            base_pos = camera_pos * (1 - t) + bbox_center * t
            
            # Shift allows the occluder to be off-center relative to the line of sight.
            shift_x = random.uniform(-0.1, 0.1) * object_scale
            shift_y = random.uniform(-0.1, 0.1) * object_scale
            position = base_pos + (right * shift_x) + (up * shift_y)
            
            # Scale factor based on user provided range (percentage)
            scale_factor = random.uniform(self.size_range[0], self.size_range[1])
            size = object_scale * scale_factor
            
            # Create Occluder Definition
            if self.occluder_type == "plane":
                # Random plane normal
                normal = np.random.randn(3)
                normal = normal / (np.linalg.norm(normal) + 1e-8)
                occluders.append({"type": "plane", "position": position, "normal": normal, "size": size})
            elif self.occluder_type == "sphere":
                occluders.append({"type": "sphere", "position": position, "radius": size/2})
            elif self.occluder_type == "box":
                # Random rotation
                angles = np.random.uniform(0, 2 * np.pi, 3)
                occluders.append({"type": "box", "position": position, "size": size, "angles": angles})
        
        mask = self._compute_occlusion_mask(points, occluders, camera_pos)
        return {"mask": mask, "occluders": occluders, "camera_position": camera_pos}

    def _compute_occlusion_mask(self, points, occluders, camera_pos):
        num_points = points.shape[0]
        mask = np.ones(num_points, dtype=bool)
        xyz = points[:, :3].astype(np.float32)
        camera_pos = camera_pos.astype(np.float32)
        
        # Precompute ray directions and distances once
        to_points = xyz - camera_pos # (N, 3)
        point_dist_sq = np.sum(to_points**2, axis=1)
        point_dist = np.sqrt(point_dist_sq)
        ray_dirs = to_points / (point_dist[:, None] + 1e-8) # (N, 3)

        for occ in occluders:
            occ_pos = occ["position"].astype(np.float32)
            if occ["type"] == "sphere":
                # Vector from camera to sphere center
                L = occ_pos - camera_pos
                tca = np.dot(ray_dirs, L)
                
                # If tca < 0, sphere is behind camera
                # d2 is the square distance from sphere center to the projection
                d2 = np.sum(L**2) - tca**2
                radius2 = occ["radius"]**2
                
                # Logic: Intersection exists if d2 < r2 AND tca > 0 
                # AND the sphere center is closer than the point
                occ_dist_sq = np.sum(L**2)
                occluded = (tca > 0) & (d2 < radius2) & (point_dist_sq > occ_dist_sq)
                mask &= ~occluded
            elif occ["type"] == "box":
                # Optimized Slab Method
                size = occ["size"]
                half_extents = size * 0.5
                
                # Get Inverse Rotation Matrix
                angles = occ["angles"]
                c, s = np.cos(angles), np.sin(angles)
                Rx = np.array([[1, 0, 0], [0, c[0], s[0]], [0, -s[0], c[0]]]) # Transposed = Inverse
                Ry = np.array([[c[1], 0, -s[1]], [0, 1, 0], [s[1], 0, c[1]]])
                Rz = np.array([[c[2], s[2], 0], [-s[2], c[2], 0], [0, 0, 1]])
                R_inv = Rx @ Ry @ Rz # Order matters: (Rz @ Ry @ Rx).T
                
                # Transform Camera and Rays into Box Local Space
                local_cam = (camera_pos - occ_pos) @ R_inv.T
                local_rays = ray_dirs @ R_inv.T
                
                # Slab Intersection
                inv_dir = 1.0 / (local_rays + 1e-8)
                t1 = (-half_extents - local_cam) * inv_dir
                t2 = (half_extents - local_cam) * inv_dir
                
                t_near = np.max(np.minimum(t1, t2), axis=1)
                t_far = np.min(np.maximum(t1, t2), axis=1)
                
                # Intersection if t_near < t_far and it's between camera and point
                # Since ray_dirs are normalized, t_near is actual distance
                occluded = (t_far >= t_near) & (t_far > 0) & (point_dist > t_near)
                mask &= ~occluded
            elif occ["type"] == "plane":
                # Standard Plane-Halfspace Check
                to_occ = occ_pos - camera_pos
                normal = occ["normal"]
                
                # dot(point - pos, normal)
                points_side = np.dot(xyz - occ_pos, normal)
                camera_side = np.dot(camera_pos - occ_pos, normal)
                
                # Occluded if signs differ (plane is between)
                # and point is further from camera than the plane is
                occ_dist = np.linalg.norm(to_occ)
                occluded = (np.sign(points_side) != np.sign(camera_side)) & (point_dist > occ_dist)
                mask &= ~occluded
        return mask

    def apply(self, points, mask, **params):
        return points[mask]

    def apply_to_normals(self, normals, mask, **params):
        if normals is None: return None
        return normals[mask]

    def apply_to_features(self, features, mask, **params):
        if features is None: return None
        return features[mask]

    def apply_to_labels(self, labels, mask, **params):
        if labels is None: return None
        return labels[mask]

    def get_transform_init_args_names(self):
        return ("num_occluders", "occluder_type", "size_range", "camera_position", "phi_range")


class HRPOcclusion(PointCloudsTransform):
    """HRP.

    Args:
        camera_position (tuple): (x, y, z) position of the camera. If None,
            uses random position. Default: None.
        radius_multiplier (float): Radius multiplier for the HPR algo. Default: 1e3.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        camera_position=None,
        radius_multiplier=1e3,
        phi_range=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.camera_position = camera_position
        self.radius_multiplier = radius_multiplier
        self.phi_range = phi_range if phi_range is not None else (0, np.pi)

    @property
    def targets_as_params(self):
        return ["points"]

    def get_params_dependent_on_targets(self, params):
        points = params["points"]
        # Handle empty points safely
        if points is None or points.shape[0] == 0:
            camera_pos = np.array(self.camera_position) if self.camera_position is not None else np.array([0.0, 0.0, 0.0])
            return {"mask": np.zeros(0, dtype=bool), "camera_position": camera_pos}

        # Get camera position
        if self.camera_position is None:
            camera_pos = get_random_camera(points, phi_range=self.phi_range)
        else:
            camera_pos = np.array(self.camera_position)
        
        # Compute depth buffer
        mask = self._compute_mask(points[:, :3], camera_pos)
        
        return {"mask": mask, "camera_position": camera_pos}

    # @timing
    def _compute_mask(self, points, camera_pos):
        diameter = np.linalg.norm( points.max(axis=0) - points.min(axis=0))
        radius = diameter * self.radius_multiplier # large multiplier since we have a dense pointcloud (can cause false positives)
        
        # Translate points relative to camera
        p = points - camera_pos
        norms = np.linalg.norm(p, axis=1)
        
        # Invert points
        valid = norms > 1e-8
        p_inv = np.zeros_like(p)
        p_inv[valid] = p[valid] * ((2 * radius - norms[valid]) / norms[valid])[:, None]
        
        # Add camera at origin
        p_inv_with_cam = np.vstack([p_inv, np.zeros((1, 3))])
        
        try:
            hull = ConvexHull(p_inv_with_cam)
            visible_indices = hull.vertices
            # Remove the camera index (which is the last element)
            visible_indices = visible_indices[visible_indices < len(points)]
        except Exception:
            # Fallback if convex hull fails (e.g., coplanar points)
            visible_indices = np.arange(len(points))
        
        mask = np.zeros(len(points), dtype=bool)
        mask[visible_indices] = True
        return mask

    def apply(self, points, mask, **params):
        return points[mask]

    def apply_to_normals(self, normals, mask, **params):
        if normals is None:
            return None
        return normals[mask]

    def apply_to_features(self, features, mask, **params):
        if features is None:
            return None
        return features[mask]

    def apply_to_labels(self, labels, mask, **params):
        if labels is None:
            return None
        return labels[mask]

    def get_transform_init_args_names(self):
        return ("camera_position", "radius_multiplier", "phi_range")