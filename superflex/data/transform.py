# from https://github.dev/kumuji/volumentations
import random
import math
from copy import deepcopy
from warnings import warn
import numpy as np

REPR_INDENT_STEP = 2

def move(points, offset=(0, 0, 0)):
    points[:, :3] = points[:, :3] + offset
    return points

def f_scale(points, scale_factor=(1, 1, 1)):
    transformation_matrix = np.eye(3)
    np.fill_diagonal(transformation_matrix, scale_factor)
    points[:, :3] = np.dot(points[:, :3], transformation_matrix)
    return points


def format_args(args_dict):
    formatted_args = []
    for k, v in args_dict.items():
        if isinstance(v, str):
            v = "'{}'".format(v)
        formatted_args.append("{}={}".format(k, v))
    return ", ".join(formatted_args)


class Transforms:
    def __init__(self, transforms):
        self.transforms = transforms
        self.start_end = self._find_dual_start_end(transforms)

    def _find_dual_start_end(self, transforms):
        dual_start_end = None
        last_dual = None
        for idx, transform in enumerate(transforms):
            if isinstance(transform, BaseCompose):
                inside = self._find_dual_start_end(transform)
                if inside is not None:
                    last_dual = idx
                    if dual_start_end is None:
                        dual_start_end = [idx]
        if dual_start_end is not None:
            dual_start_end.append(last_dual)
        return dual_start_end

    def get_always_apply(self, transforms):
        new_transforms = []
        for transform in transforms:
            if isinstance(transform, BaseCompose):
                new_transforms.extend(self.get_always_apply(transform))
            elif transform.always_apply:
                new_transforms.append(transform)
        return Transforms(new_transforms)

    def __getitem__(self, item):
        return self.transforms[item]

class BaseCompose:
    def __init__(self, transforms, p):
        self.transforms = Transforms(transforms)
        self.p = p

        self.replay_mode = False
        self.applied_in_replay = False

    def __getitem__(self, item):
        return self.transforms[item]

    def __repr__(self):
        return self.indented_repr()

    def indented_repr(self, indent=REPR_INDENT_STEP):
        args = {
            k: v
            for k, v in self._to_dict().items()
            if not (k.startswith("__") or k == "transforms")
        }
        repr_string = self.__class__.__name__ + "(["
        for t in self.transforms:
            repr_string += "\n"
            if hasattr(t, "indented_repr"):
                t_repr = t.indented_repr(indent + REPR_INDENT_STEP)
            else:
                t_repr = repr(t)
            repr_string += " " * indent + t_repr + ","
        repr_string += (
            "\n"
            + " " * (indent - REPR_INDENT_STEP)
            + "], {args})".format(args=format_args(args))
        )
        return repr_string

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def _to_dict(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "p": self.p,
            "transforms": [t._to_dict() for t in self.transforms],
        }

    def get_dict_with_id(self):
        return {
            "__class_fullname__": self.get_class_fullname(),
            "id": id(self),
            "params": None,
            "transforms": [t.get_dict_with_id() for t in self.transforms],
        }

    def add_targets(self, additional_targets):
        if additional_targets:
            for t in self.transforms:
                t.add_targets(additional_targets)

    def set_deterministic(self, flag, save_key="replay"):
        for t in self.transforms:
            t.set_deterministic(flag, save_key)


class Compose(BaseCompose):
    """Compose transforms and handle all transformations regrading bounding boxes.

    Args:
        transforms (list): list of transformations to compose.
        additional_targets (dict): Dict with keys - new target name,
            values - old target name. ex: {'image2': 'image'}
        p (float): probability of applying all list of transforms. Default: 1.0.
    """

    def __init__(
        self, transforms, additional_targets=None, p=1.0,
    ):
        super(Compose, self).__init__([t for t in transforms if t is not None], p)

        self.processors = {}

        if additional_targets is None:
            additional_targets = {}

        self.additional_targets = additional_targets

        for proc in self.processors.values():
            proc.ensure_transforms_valid(self.transforms)

        self.add_targets(additional_targets)

    def __call__(self, force_apply=False, **data):
        need_to_run = force_apply or (random.random() < self.p)
        for p in self.processors.values():
            p.ensure_data_valid(data)
        transforms = (
            self.transforms
            if need_to_run
            else self.transforms.get_always_apply(self.transforms)
        )
        dual_start_end = transforms.start_end if self.processors else None

        for idx, t in enumerate(transforms):
            if dual_start_end is not None and idx == dual_start_end[0]:
                for p in self.processors.values():
                    p.preprocess(data)

            data = t(force_apply=force_apply, **data)

            if dual_start_end is not None and idx == dual_start_end[1]:
                for p in self.processors.values():
                    p.postprocess(data)

        return data

    def _to_dict(self):
        dictionary = super(Compose, self)._to_dict()
        dictionary.update({"additional_targets": self.additional_targets})
        return dictionary


def to_tuple(param, low=None, bias=None):
    """Convert input argument to min-max tuple

    Args:
        param (scalar, tuple or list of 2+ elements): Input value.
            If value is scalar, return value would be (offset - value, offset + value).
            If value is tuple, return value would be value + offset (broadcasted).
        low:  Second element of tuple can be passed as optional argument
        bias: An offset factor added to each element
    """
    if low is not None and bias is not None:
        raise ValueError("Arguments low and bias are mutually exclusive")

    if param is None:
        return param

    if isinstance(param, (int, float)):
        if low is None:
            param = -param, +param
        else:
            param = (low, param) if low < param else (param, low)
    elif isinstance(param, (list, tuple)):
        param = tuple(param)
    else:
        raise ValueError("Argument param must be either scalar (int, float) or tuple")

    if bias is not None:
        return tuple(bias + x for x in param)

    return tuple(param)

def rotate_around_axis(points, axis, angle, center_point=None):
    """
    Rotate points around given axis by `angle` (radians).
    Works for shapes (..., 3) – arbitrary leading batch dims.
    """
    points = np.asarray(points)
    orig_shape = points.shape          # e.g. (B, N, 3)
    pts = points.reshape(-1, orig_shape[-1])  # (N_total, C), C >= 3

    if center_point is None:
        center_point = pts[:, :3].mean(axis=0).astype(pts.dtype)

    axis = np.asarray(axis, dtype=pts.dtype)
    axis = axis / np.linalg.norm(axis)

    a = np.cos(angle / 2.0)
    b, c, d = -axis * np.sin(angle / 2.0)
    aa, bb, cc, dd = a*a, b*b, c*c, d*d
    bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

    rotation_matrix = np.array(
        [
            [aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac)],
            [2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab)],
            [2 * (bd + ac),     2 * (cd - ab),     aa + dd - bb - cc],
        ],
        dtype=pts.dtype,
    )

    pts[:, :3] = pts[:, :3] - center_point
    pts[:, :3] = pts[:, :3] @ rotation_matrix.T
    pts[:, :3] = pts[:, :3] + center_point

    return pts.reshape(orig_shape)




class BasicTransform:
    call_backup = None

    def __init__(self, always_apply=False, p=0.5):
        self.p = p
        self.always_apply = always_apply
        self._additional_targets = {}

        # replay mode params
        self.deterministic = False
        self.save_key = "replay"
        self.params = {}
        self.replay_mode = False
        self.applied_in_replay = False

    def __call__(self, force_apply=False, **kwargs):
        if self.replay_mode:
            if self.applied_in_replay:
                return self.apply_with_params(self.params, **kwargs)

            return kwargs

        if (random.random() < self.p) or self.always_apply or force_apply:
            params = self.get_params()

            if self.targets_as_params:
                assert all(
                    key in kwargs for key in self.targets_as_params
                ), "{} requires {}".format(
                    self.__class__.__name__, self.targets_as_params
                )
                targets_as_params = {k: kwargs[k] for k in self.targets_as_params}
                params_dependent_on_targets = self.get_params_dependent_on_targets(
                    targets_as_params
                )
                params.update(params_dependent_on_targets)
            if self.deterministic:
                if self.targets_as_params:
                    warn(
                        self.get_class_fullname()
                        + " could work incorrectly in ReplayMode for other input data"
                        " because its' params depend on targets."
                    )
                kwargs[self.save_key][id(self)] = deepcopy(params)
            return self.apply_with_params(params, **kwargs)

        return kwargs

    def apply_with_params(
        self, params, force_apply=False, **kwargs
    ):  # skipcq: PYL-W0613
        if params is None:
            return kwargs
        params = self.update_params(params, **kwargs)
        res = {}
        for key, arg in kwargs.items():
            if arg is not None:
                target_function = self._get_target_function(key)
                target_dependencies = {
                    k: kwargs[k] for k in self.target_dependence.get(key, [])
                }
                res[key] = target_function(arg, **dict(params, **target_dependencies))
            else:
                res[key] = None
        return res

    def set_deterministic(self, flag, save_key="replay"):
        assert save_key != "params", "params save_key is reserved"
        self.deterministic = flag
        self.save_key = save_key
        return self

    def __repr__(self):
        state = self.get_base_init_args()
        state.update(self.get_transform_init_args())
        return "{name}({args})".format(
            name=self.__class__.__name__, args=format_args(state)
        )

    def _get_target_function(self, key):
        transform_key = key
        if key in self._additional_targets:
            transform_key = self._additional_targets.get(key, None)

        target_function = self.targets.get(transform_key, lambda x, **p: x)
        return target_function

    def apply(self, img, **params):
        raise NotImplementedError

    def get_params(self):
        return {}

    @property
    def targets(self):
        # you must specify targets in subclass
        # for example: ('image', 'mask')
        #              ('image', 'boxes')
        raise NotImplementedError

    def update_params(self, params, **kwargs):
        return params

    @property
    def target_dependence(self):
        return {}

    def add_targets(self, additional_targets):
        """Add targets to transform them the same way as one of existing targets
        ex: {'normals1': 'normals', 'normals2': 'normals'}

        Args:
            additional_targets (dict): keys - new target name, values -
            old target name. ex: {'normals2': 'normals'}
        """
        self._additional_targets = additional_targets

    @property
    def targets_as_params(self):
        return []

    def get_params_dependent_on_targets(self, params):
        raise NotImplementedError(
            "Method get_params_dependent_on_targets is not implemented in class "
            + self.__class__.__name__
        )

    @classmethod
    def get_class_fullname(cls):
        return "{cls.__module__}.{cls.__name__}".format(cls=cls)

    def get_transform_init_args_names(self):
        raise NotImplementedError(
            "Class {name} is not serializable because the `get_transform_init_args_names` method is not "
            "implemented".format(name=self.get_class_fullname())
        )

    def get_base_init_args(self):
        return {"always_apply": self.always_apply, "p": self.p}

    def get_transform_init_args(self):
        return {k: getattr(self, k) for k in self.get_transform_init_args_names()}

    def _to_dict(self):
        state = {"__class_fullname__": self.get_class_fullname()}
        state.update(self.get_base_init_args())
        state.update(self.get_transform_init_args())
        return state

    def get_dict_with_id(self):
        d = self._to_dict()
        d["id"] = id(self)
        return d


class PointCloudsTransform(BasicTransform):
    """Transform for point clouds."""

    @property
    def targets(self):
        return {
            "points": self.apply,
            "normals": self.apply_to_normals,
            "features": self.apply_to_features,
            "cameras": self.apply_to_camera,
            "bbox": self.apply_to_bboxes,
            "labels": self.apply_to_labels,
            "transform_info": self.apply_to_transform_info,
        }

    def apply_to_transform_info(self, transform_info, **params):
        return transform_info

    def apply_to_bboxes(self, bboxes, **params):
        return [self.apply_to_bbox(bbox, **params) for bbox in bboxes]

    def apply_to_bbox(self, bbox, **params):
        raise NotImplementedError(
            "Method apply_to_bbox is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_cameras(self, cameras, **params):
        return [self.apply_to_bbox(camera, **params) for camera in cameras]

    def apply_to_camera(self, camera, **params):
        raise NotImplementedError(
            "Method apply_to_camera is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_normals(self, normals, **params):
        raise NotImplementedError(
            "Method apply_to_normals is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_features(self, features, **params):
        raise NotImplementedError(
            "Method apply_to_features is not implemented in class "
            + self.__class__.__name__
        )

    def apply_to_labels(self, labels, **params):
        raise NotImplementedError(
            "Method apply_to_labels is not implemented in class "
            + self.__class__.__name__
        )

class Move3d(PointCloudsTransform):
    """Move point cloud on offset.

    Args:
        offset (float): coorinate where to move origin of coordinate frame. Default: 0.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(self, offset=(0, 0, 0), always_apply=False, p=0.5):
        super().__init__(always_apply, p)
        self.offset = offset

    def get_params(self):
        return {"offset": self.offset}

    def apply(self, points, offset, **params):
        return move(points, offset)

    def apply_to_transform_info(self, transform_info, offset, **params):
        if transform_info is None:
            return None
        
        cur_M = transform_info.get('matrix', np.eye(4))
        
        T_offset = np.eye(4)
        T_offset[:3, 3] = offset
        
        transform_info['matrix'] = T_offset @ cur_M
        
        return transform_info

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "offset": self.offset,
        }

class RandomMove3d(Move3d):
    """Move point cloud on random offset.

    Args:
        x_min (float): Minimum x coordinate. Default: -1.
        y_min (float): Minimum y coordinate. Default: -1.
        z_min (float): Minimum z coordinate. Default: -1.
        x_max (float): Maximum x coordinate. Default: 1.
        y_max (float): Maximum y coordinate. Default: 1.
        z_max (float): Maximum z coordinate. Default: 1.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        x_min=-1.0,
        y_min=-1.0,
        z_min=-1.0,
        x_max=1.0,
        y_max=1.0,
        z_max=1.0,
        offset=(0, 0, 0),
        always_apply=False,
        p=0.5,
    ):
        super().__init__(offset, always_apply, p)
        self.x_min = x_min
        self.y_min = y_min
        self.z_min = z_min
        self.x_max = x_max
        self.y_max = y_max
        self.z_max = z_max

    def get_params(self):
        offset = [
            random.uniform(self.x_min, self.x_max),
            random.uniform(self.y_min, self.y_max),
            random.uniform(self.z_min, self.z_max),
        ]
        return {"offset": offset}

    def get_transform_init_args_names(self):
        return {
            "offset": self.offset,
            "x_min": self.x_min,
            "y_min": self.y_min,
            "z_min": self.z_min,
            "x_max": self.x_max,
            "y_max": self.y_max,
            "z_max": self.z_max,
        }

class RandomJitter(PointCloudsTransform):
    """Add small clipped Gaussian noise to each point independently.

    Labels and normals are unchanged; only the xyz positions are perturbed.

    Args:
        sigma (float): standard deviation of the Gaussian noise. Default: 0.01.
        clip  (float): maximum absolute noise value (|noise| <= clip). Default: 0.05.
        p     (float): probability of applying the transform. Default: 1.0.

    Targets:
        points
        normals
        features
        labels
    """

    def __init__(self, sigma=0.01, clip=0.05, always_apply=False, p=1.0):
        super().__init__(always_apply, p)
        self.sigma = sigma
        self.clip = clip

    def get_params(self):
        return {"sigma": self.sigma, "clip": self.clip}

    def apply(self, points, sigma, clip, **params):
        points = np.asarray(points, dtype=np.float32)
        noise = np.clip(
            np.random.normal(0.0, sigma, size=points[:, :3].shape).astype(np.float32),
            -clip, clip,
        )
        out = points.copy()
        out[:, :3] += noise
        return out

    def apply_to_transform_info(self, transform_info, **params):
        # Jitter is not a linear transform; leave aug_matrix unchanged
        return transform_info

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {"sigma": self.sigma, "clip": self.clip}


class Scale3d(PointCloudsTransform):
    """Scale the input point cloud uniformly.

    Args:
        scale_limit (float): maximum scaling of input point cloud.
            Default: 0.5.
        bias (float): base scaling that is always applied.
            Default: 1.0.
        p (float): probability of applying the transform. Default: 1.0.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self, scale_limit=0.5, bias=1.0, always_apply=False, p=1.
    ):
        super().__init__(always_apply, p)
        self.scale_limit = to_tuple(scale_limit, bias=bias)

    def get_params(self):
        s = random.uniform(self.scale_limit[0], self.scale_limit[1])
        return {"scale": (s, s, s)}

    def apply(self, points, scale=(1, 1, 1), **params):
        return f_scale(points, scale)

    def apply_to_transform_info(self, transform_info, scale=(1, 1, 1), **params):
        if transform_info is None:
            return None
        
        S = np.eye(4)
        S[0, 0] = scale[0]
        S[1, 1] = scale[1]
        S[2, 2] = scale[2]
        
        cur_M = transform_info.get('matrix', np.eye(4))
        transform_info['matrix'] = S @ cur_M
        
        return transform_info

    def apply_to_normals(self, normals, **params):
        return normals

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {"scale_limit": self.scale_limit}


class RotateAroundAxis3d(PointCloudsTransform):
    """Rotate point cloud around axis on random angle.

    Args:
        rotation_limit (float): maximum rotation of the input point cloud. Default: (pi / 2).
        axis (list(float, float, float)): axis around which the point cloud is rotated. Default: (0, 0, 1).
        center_point (float, float, float): point around which to rotate. Default: mean points.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        points
        normals
        features
        labels

    """

    def __init__(
        self,
        rotation_limit=math.pi / 2,
        axis=(0, 0, 1),
        center_point=None,
        always_apply=False,
        p=0.5,
    ):
        super().__init__(always_apply, p)
        self.rotation_limit = to_tuple(rotation_limit, bias=0)
        self.axis = axis
        self.center_point = center_point

    def get_params(self):
        angle = random.uniform(self.rotation_limit[0], self.rotation_limit[1])
        return {"angle": angle, "axis": self.axis, "center_point": self.center_point}

    def apply(self, points, axis, angle, **params):
        return rotate_around_axis(points, axis, angle, center_point=self.center_point)

    @property
    def target_dependence(self):
        return {"transform_info": ["points"]}

    def apply_to_transform_info(self, transform_info, points, axis, angle, center_point, **params):
        if transform_info is None:
            return None
            
        points = np.asarray(points)
        orig_shape = points.shape
        pts = points.reshape(-1, orig_shape[-1])
        
        if center_point is None:
            center_point = pts[:, :3].mean(axis=0)

        # Build 4x4 transform for this rotation around center
        axis = np.asarray(axis)
        axis = axis / np.linalg.norm(axis)
        a = np.cos(angle / 2.0)
        b, c, d = -axis * np.sin(angle / 2.0)
        aa, bb, cc, dd = a*a, b*b, c*c, d*d
        bc, ad, ac, ab, bd, cd = b*c, a*d, a*c, a*b, b*d, c*d

        rot_3x3 = np.array(
            [
                [aa + bb - cc - dd, 2 * (bc + ad),     2 * (bd - ac)],
                [2 * (bc - ad),     aa + cc - bb - dd, 2 * (cd + ab)],
                [2 * (bd + ac),     2 * (cd - ab),     aa + dd - bb - cc],
            ]
        )

        # T_step = T(c) @ R @ T(-c)
        T_c = np.eye(4)
        T_c[:3, 3] = center_point
        
        T_neg_c = np.eye(4)
        T_neg_c[:3, 3] = -center_point
        
        R_4x4 = np.eye(4)
        R_4x4[:3, :3] = rot_3x3
        
        M_step = T_c @ R_4x4 @ T_neg_c
        
        cur_M = transform_info.get('matrix', np.eye(4))
        transform_info['matrix'] = M_step @ cur_M
        
        return transform_info

    def apply_to_normals(self, normals, axis, angle, **params):
        return rotate_around_axis(normals, axis, angle, center_point=(0, 0, 0))

    def apply_to_features(self, features, **params):
        return features

    def apply_to_labels(self, labels, **params):
        return labels

    def get_transform_init_args(self):
        return {
            "rotation_limit": to_tuple(self.rotation_limit, bias=0),
            "axis": self.axis,
        }