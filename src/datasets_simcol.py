"""
SimCol / SyntheticColon dataset loader for EndoMetric3D.

Supports the raw SimCol/SyntheticColon layout used in the TMI 2026 code style:

    SyntheticColon_I/Frames_S1/FrameBuffer_0000.png
    SyntheticColon_I/Frames_S1/Depth_0000.png
    SyntheticColon_I/SavedPosition_S1.txt
    SyntheticColon_I/SavedRotation_S1.txt      # if available

This loader is separate from the C3VD loader and returns the same training keys:

    image, image_01, image_next, image_next_01, image_prev, image_prev_01
    depth_gt                    # cm, [1,H,W]
    K                           # pixel-space 3x3 intrinsics after resize
    pose_gt                     # T_t_to_next, [4,4], translation in cm
    pose_gt_prev                # T_t_to_prev, [4,4], translation in cm

Depth convention from the TMI 2026 paper:
    normalized depth in [0,1] represents [0,20] cm.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


SIMCOL_TMI2026_SPLITS: Dict[str, List[str]] = {
    "train": [
        "S1", "S2", "S3", "S6", "S7", "S8", "S11", "S12", "S13",
        "B1", "B2", "B3", "B6", "B7", "B8", "B11", "B12", "B13",
    ],
    "val": ["S4", "S9", "S14", "B4", "B9", "B14"],
    "test": ["S5", "S10", "S15", "B5", "B10", "B15"],
    "test_iii": ["O1", "O2", "O3"],
}

_FRAME_RE = re.compile(r"FrameBuffer_(\d+)\.(png|jpg|jpeg)$", re.IGNORECASE)
_DEPTH_RE = re.compile(r"Depth_(\d+)\.(png|jpg|jpeg|tif|tiff|npy)$", re.IGNORECASE)
_SEQ_KEY_RE = re.compile(r"(?:Frames[_-]?)?([SBO]\d+)$", re.IGNORECASE)

POSE_FILE_CANDIDATES = [
    "pose.txt", "poses.txt", "Pose.txt", "Poses.txt",
    "CameraPose.txt", "camera_pose.txt", "camera_poses.txt", "CameraPoses.txt",
    "Trajectory.txt", "trajectory.txt", "slam_pose.txt", "SLAM.txt",
]


def _as_bool(x, default=False) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    return str(x).lower() in {"1", "true", "yes", "y", "on"}


def _opt_get(opt, key: str, default=None):
    if isinstance(opt, dict):
        return opt.get(key, default)
    return getattr(opt, key, default)


def _seq_key_from_dir(seq_dir: Path) -> Optional[str]:
    m = _SEQ_KEY_RE.match(seq_dir.name)
    if m:
        return m.group(1).upper()
    m = _SEQ_KEY_RE.match(seq_dir.parent.name)
    if m:
        return m.group(1).upper()
    return None


def _pose_search_dirs(seq_dir: Path) -> List[Path]:
    """Likely locations for SimCol pose files."""
    dirs: List[Path] = [seq_dir, seq_dir.parent]
    for base in [seq_dir, seq_dir.parent]:
        if not base.exists():
            continue
        for d in base.iterdir():
            if d.is_dir() and any(tok in d.name.lower() for tok in ["pose", "camera", "traj", "position", "rotation", "transform"]):
                dirs.append(d)
    out: List[Path] = []
    seen = set()
    for d in dirs:
        if d not in seen:
            out.append(d)
            seen.add(d)
    return out


def _pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _resize_rgb(img: Image.Image, size_hw: Tuple[int, int]) -> Image.Image:
    h, w = size_hw
    return img.resize((w, h), resample=Image.BILINEAR)


def _resize_depth_np(depth: np.ndarray, size_hw: Tuple[int, int]) -> np.ndarray:
    h, w = size_hw
    img = Image.fromarray(depth.astype(np.float32), mode="F")
    img = img.resize((w, h), resample=Image.NEAREST)
    return np.asarray(img, dtype=np.float32)


def _to_tensor_01(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img).astype(np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def _imagenet_normalize(img01: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor([0.485, 0.456, 0.406], dtype=img01.dtype).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=img01.dtype).view(3, 1, 1)
    return (img01 - mean) / std


def _read_depth_norm(path: Path) -> np.ndarray:
    """Read SimCol depth as normalized float depth in [0,1] when possible."""
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)

    with Image.open(path) as img:
        arr0 = np.asarray(img)

    if arr0.dtype == np.uint16:
        return arr0.astype(np.float32) / 65535.0
    if arr0.dtype == np.uint8:
        return arr0.astype(np.float32) / 255.0

    return arr0.astype(np.float32)


def _unit_scale_to_cm(unit: str) -> float:
    unit = unit.lower()
    if unit == "cm":
        return 1.0
    if unit == "mm":
        return 0.1
    if unit == "m":
        return 100.0
    raise ValueError(f"Unknown pose unit: {unit}")


def _quat_xyzw_to_mat(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
    q = np.array([qx, qy, qz, qw], dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)
    x, y, z, w = q
    R = np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=np.float32)
    return R


def _euler_to_mat(angles, order: str = "xyz", degrees: bool = True) -> np.ndarray:
    """Simple Euler-to-matrix converter.

    Default assumes SavedRotation rows are Unity-like Euler angles in degrees.
    If sanity plots/pose are wrong, try --simcol_rotation_mode euler_rad or
    --simcol_euler_order zyx.
    """
    a = np.asarray(angles, dtype=np.float64).reshape(3)
    if degrees:
        a = np.deg2rad(a)
    cx, cy, cz = np.cos(a)
    sx, sy, sz = np.sin(a)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=np.float64)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=np.float64)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=np.float64)
    mats = {"x": Rx, "y": Ry, "z": Rz}
    R = np.eye(3, dtype=np.float64)
    for ch in order.lower():
        if ch not in mats:
            raise ValueError(f"Invalid Euler order: {order}")
        R = mats[ch] @ R
    return R.astype(np.float32)


def _fix_matrix_layout(T: np.ndarray, layout: str = "auto") -> np.ndarray:
    T = np.asarray(T, dtype=np.float32).reshape(4, 4)
    if layout == "transpose":
        T = T.T
    elif layout == "auto":
        col_t = np.linalg.norm(T[:3, 3])
        row_t = np.linalg.norm(T[3, :3])
        if col_t < 1e-9 and row_t > 1e-9:
            T = T.T
    elif layout in {"standard", "colmajor", "rowmajor"}:
        pass
    else:
        raise ValueError(f"Unknown simcol_pose_layout={layout}")

    if not np.allclose(T[3], np.array([0, 0, 0, 1], dtype=np.float32), atol=1e-3):
        T[3] = np.array([0, 0, 0, 1], dtype=np.float32)
    return T.astype(np.float32)


def _relative_pose(T_abs_t: np.ndarray, T_abs_s: np.ndarray, convention: str) -> np.ndarray:
    """Return T_t_to_s: transforms 3D points from camera t coordinates to camera s coordinates."""
    convention = convention.lower()
    if convention == "c2w":
        return np.linalg.inv(T_abs_s) @ T_abs_t
    if convention == "w2c":
        return T_abs_s @ np.linalg.inv(T_abs_t)
    raise ValueError(f"Unknown simcol_pose_convention={convention}")


def _find_companion_file(seq_dir: Path, prefixes: List[str], seq_key: Optional[str]) -> Optional[Path]:
    names: List[str] = []
    if seq_key:
        for pref in prefixes:
            names += [f"{pref}_{seq_key}.txt", f"{pref}{seq_key}.txt", f"{seq_key}_{pref}.txt"]
    for pref in prefixes:
        names += [f"{pref}.txt", f"{pref}s.txt"]

    for base in _pose_search_dirs(seq_dir):
        for name in names:
            p = base / name
            if p.exists():
                return p
    return None


def _load_saved_position_rotation(
    seq_dir: Path,
    pos_path: Path,
    seq_key: Optional[str],
    trans_scale_cm: float,
    rotation_mode: str = "euler_deg",
    euler_order: str = "xyz",
    quat_format: str = "xyzw",
) -> Dict[int, np.ndarray]:
    pos = np.loadtxt(str(pos_path), dtype=np.float32)
    if pos.ndim == 1:
        pos = pos[None, :]

    if pos.shape[1] == 4:
        frame_ids = [int(round(float(x))) for x in pos[:, 0]]
        xyz = pos[:, 1:4]
    elif pos.shape[1] == 3:
        frame_ids = list(range(pos.shape[0]))
        xyz = pos[:, :3]
    else:
        raise RuntimeError(f"Unsupported SavedPosition format {pos.shape} in {pos_path}")

    rot_path = _find_companion_file(
        seq_dir,
        prefixes=[
            "SavedRotation", "SavedRotations", "SavedOrientation", "SavedOrientations",
            "SavedQuaternion", "SavedQuaternions", "SavedRotationQuaternion", "SavedRotationQuaternions",
            "Rotation", "Rotations", "Orientation", "Orientations", "Quaternion", "Quaternions",
        ],
        seq_key=seq_key,
    )

    rot = None
    if rot_path is not None:
        rot = np.loadtxt(str(rot_path), dtype=np.float32)
        if rot.ndim == 1:
            rot = rot[None, :]
        # If first column is frame id, drop it.
        if rot.shape[1] == 5:
            rot = rot[:, 1:]
        elif rot.shape[1] == 4:
            # Ambiguous: could be quaternion xyzw, or index+euler. If first col
            # looks like frame indices, treat it as index+Euler.
            first = rot[:, 0]
            if np.allclose(first, np.round(first), atol=1e-4) and len(np.unique(first[:min(20, len(first))])) > 1:
                rot = rot[:, 1:4]
        elif rot.shape[1] != 3:
            raise RuntimeError(f"Unsupported SavedRotation format {rot.shape} in {rot_path}")

    poses: Dict[int, np.ndarray] = {}
    for row_i, frame_idx in enumerate(frame_ids):
        T = np.eye(4, dtype=np.float32)
        T[:3, 3] = xyz[row_i].astype(np.float32) * trans_scale_cm

        if rot is not None and row_i < rot.shape[0]:
            rv = rot[row_i].astype(np.float32).reshape(-1)
            if rv.size == 3:
                mode = rotation_mode.lower()
                if mode in {"none", "identity"}:
                    Rm = np.eye(3, dtype=np.float32)
                elif mode == "euler_rad":
                    Rm = _euler_to_mat(rv, order=euler_order, degrees=False)
                else:
                    Rm = _euler_to_mat(rv, order=euler_order, degrees=True)
                T[:3, :3] = Rm
            elif rv.size == 4:
                if quat_format.lower() == "wxyz":
                    qw, qx, qy, qz = [float(x) for x in rv]
                else:
                    qx, qy, qz, qw = [float(x) for x in rv]
                T[:3, :3] = _quat_xyzw_to_mat(qx, qy, qz, qw)
        poses[int(frame_idx)] = T

    if rot_path is None:
        print(f"[SimCol WARN] Loaded {pos_path.name} but no SavedRotation companion was found; using identity rotations.")
    else:
        print(f"[SimCol] Loaded positions {pos_path.name} + rotations {rot_path.name}")
    return poses


def _load_pose_table(
    seq_dir: Path,
    layout: str,
    trans_scale_cm: float,
    rotation_mode: str = "euler_deg",
    euler_order: str = "xyz",
    quat_format: str = "xyzw",
) -> Optional[Dict[int, np.ndarray]]:
    pose_path = None
    seq_key = _seq_key_from_dir(seq_dir)
    names = list(POSE_FILE_CANDIDATES)
    if seq_key is not None:
        names += [
            f"pose_{seq_key}.txt", f"poses_{seq_key}.txt",
            f"Pose_{seq_key}.txt", f"Poses_{seq_key}.txt",
            f"CameraPose_{seq_key}.txt", f"CameraPoses_{seq_key}.txt",
            f"camera_pose_{seq_key}.txt", f"camera_poses_{seq_key}.txt",
            f"Trajectory_{seq_key}.txt", f"trajectory_{seq_key}.txt",
            f"SavedPosition_{seq_key}.txt", f"SavedPose_{seq_key}.txt",
            f"{seq_key}_pose.txt", f"{seq_key}_poses.txt",
            f"{seq_key}_CameraPose.txt", f"{seq_key}_Trajectory.txt",
        ]
    for base in _pose_search_dirs(seq_dir):
        for name in names:
            p = base / name
            if p.exists():
                pose_path = p
                break
        if pose_path is not None:
            break
    if pose_path is None:
        return None

    if "savedposition" in pose_path.name.lower() or "position" in pose_path.name.lower():
        return _load_saved_position_rotation(seq_dir, pose_path, seq_key, trans_scale_cm, rotation_mode, euler_order, quat_format)

    arr = np.loadtxt(str(pose_path), dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]

    poses: Dict[int, np.ndarray] = {}
    ncol = arr.shape[1]
    for row_i, row in enumerate(arr):
        frame_idx = row_i
        vals = row
        if ncol in {17, 13, 8}:
            frame_idx = int(round(float(row[0])))
            vals = row[1:]

        if vals.size == 16:
            T = _fix_matrix_layout(vals.reshape(4, 4), layout)
        elif vals.size == 12:
            T = np.eye(4, dtype=np.float32)
            T[:3, :4] = vals.reshape(3, 4)
            T = _fix_matrix_layout(T, layout)
        elif vals.size == 7:
            t = vals[:3].astype(np.float32)
            qx, qy, qz, qw = [float(x) for x in vals[3:]]
            T = np.eye(4, dtype=np.float32)
            T[:3, :3] = _quat_xyzw_to_mat(qx, qy, qz, qw)
            T[:3, 3] = t
        else:
            raise RuntimeError(
                f"Unsupported pose row length {vals.size} in {pose_path}. "
                "Expected matrix/quaternion pose columns, or SavedPosition_* with companion SavedRotation_*."
            )

        T = T.astype(np.float32)
        T[:3, 3] *= trans_scale_cm
        poses[frame_idx] = T
    return poses


def _load_per_frame_pose(seq_dir: Path, frame_idx: int, layout: str, trans_scale_cm: float) -> Optional[np.ndarray]:
    candidates = [
        f"Pose_{frame_idx:04d}.txt", f"pose_{frame_idx:04d}.txt",
        f"CameraPose_{frame_idx:04d}.txt", f"camera_pose_{frame_idx:04d}.txt",
        f"Transform_{frame_idx:04d}.txt", f"transform_{frame_idx:04d}.txt",
    ]
    for base in _pose_search_dirs(seq_dir):
        for name in candidates:
            p = base / name
            if not p.exists():
                continue
            vals = np.loadtxt(str(p), dtype=np.float32).reshape(-1)
            if vals.size == 16:
                T = _fix_matrix_layout(vals.reshape(4, 4), layout)
            elif vals.size == 12:
                T = np.eye(4, dtype=np.float32)
                T[:3, :4] = vals.reshape(3, 4)
                T = _fix_matrix_layout(T, layout)
            else:
                raise RuntimeError(f"Unsupported per-frame pose length {vals.size} in {p}")
            T[:3, 3] *= trans_scale_cm
            return T.astype(np.float32)
    return None


@dataclass
class _SeqIndex:
    name: str
    path: Path
    rgb_by_idx: Dict[int, Path]
    depth_by_idx: Dict[int, Path]
    poses: Optional[Dict[int, np.ndarray]]


class SimColDataset(Dataset):
    def __init__(self, root: str | Path, opt, split: str = "train"):
        self.root = Path(root)
        self.opt = opt
        self.split = split
        self.frame_interval = int(_opt_get(opt, "c3vd_frame_interval", _opt_get(opt, "frame_interval", 1)))
        self.use_triplets = _as_bool(_opt_get(opt, "use_triplet_pose", True), True) or _opt_get(opt, "ablation", "") == "pose_finetune"

        self.height = int(_opt_get(opt, "height", _opt_get(opt, "image_height", 518)))
        self.width = int(_opt_get(opt, "width", _opt_get(opt, "image_width", 518)))
        self.size_hw = (self.height, self.width)

        self.orig_h = int(_opt_get(opt, "simcol_orig_height", 475))
        self.orig_w = int(_opt_get(opt, "simcol_orig_width", 475))
        self.depth_scale_cm = float(_opt_get(opt, "simcol_depth_scale_cm", 20.0))
        self.pose_layout = str(_opt_get(opt, "simcol_pose_layout", "auto"))
        self.pose_convention = str(_opt_get(opt, "simcol_pose_convention", "c2w"))
        self.pose_scale_cm = _unit_scale_to_cm(str(_opt_get(opt, "simcol_pose_unit", "cm")))
        self.rotation_mode = str(_opt_get(opt, "simcol_rotation_mode", "euler_deg"))
        self.euler_order = str(_opt_get(opt, "simcol_euler_order", "xyz"))
        self.quat_format = str(_opt_get(opt, "simcol_quat_format", "xyzw"))

        self.K = self._make_K()
        self.seq_indices = self._find_sequences()
        self.samples: List[Tuple[_SeqIndex, int]] = []
        self._build_samples()

        if len(self.samples) == 0:
            raise RuntimeError(
                f"No SimCol samples found for split={split} under {self.root}. "
                "Check --simcol_root and file names FrameBuffer_XXXX / Depth_XXXX."
            )

        print(f"[SimColDataset] {split}: {len(self.samples)} samples from {self.root} "
              f"(interval={self.frame_interval}, triplets={self.use_triplets})")

    def _make_K(self) -> torch.Tensor:
        fx_n = float(_opt_get(self.opt, "simcol_fx_norm", 0.58))
        fy_n = float(_opt_get(self.opt, "simcol_fy_norm", 0.50))
        cx_n = float(_opt_get(self.opt, "simcol_cx_norm", 0.58))
        cy_n = float(_opt_get(self.opt, "simcol_cy_norm", 0.50))
        fx = fx_n * self.orig_w
        fy = fy_n * self.orig_h
        cx = cx_n * self.orig_w
        cy = cy_n * self.orig_h
        sx = self.width / float(self.orig_w)
        sy = self.height / float(self.orig_h)
        K = np.array([
            [fx * sx, 0.0, cx * sx],
            [0.0, fy * sy, cy * sy],
            [0.0, 0.0, 1.0],
        ], dtype=np.float32)
        return torch.from_numpy(K)

    def _candidate_sequence_dirs(self) -> Iterable[Path]:
        for d in self.root.rglob("*"):
            if not d.is_dir():
                continue
            if any(_FRAME_RE.match(p.name) for p in d.iterdir() if p.is_file()):
                yield d

    def _split_names(self) -> List[str]:
        split = self.split.lower()
        if split in SIMCOL_TMI2026_SPLITS:
            return SIMCOL_TMI2026_SPLITS[split]
        if split == "all":
            out: List[str] = []
            for v in SIMCOL_TMI2026_SPLITS.values():
                out.extend(v)
            return sorted(set(out))
        raise ValueError(f"Unknown SimCol split={self.split}")

    def _find_sequences(self) -> List[_SeqIndex]:
        allowed = set(self._split_names())
        seqs: List[_SeqIndex] = []
        for seq_dir in sorted(self._candidate_sequence_dirs()):
            key = _seq_key_from_dir(seq_dir)
            if key is None or key not in allowed:
                continue
            rgb_by_idx: Dict[int, Path] = {}
            depth_by_idx: Dict[int, Path] = {}
            for p in sorted(seq_dir.iterdir()):
                if not p.is_file():
                    continue
                m = _FRAME_RE.match(p.name)
                if m:
                    rgb_by_idx[int(m.group(1))] = p
                    continue
                m = _DEPTH_RE.match(p.name)
                if m:
                    depth_by_idx[int(m.group(1))] = p
            if not rgb_by_idx or not depth_by_idx:
                continue
            poses = _load_pose_table(
                seq_dir,
                self.pose_layout,
                self.pose_scale_cm,
                rotation_mode=self.rotation_mode,
                euler_order=self.euler_order,
                quat_format=self.quat_format,
            )
            seqs.append(_SeqIndex(name=key, path=seq_dir, rgb_by_idx=rgb_by_idx, depth_by_idx=depth_by_idx, poses=poses))
        return seqs

    def _has_pose(self, seq: _SeqIndex, idx: int) -> bool:
        if seq.poses is not None and idx in seq.poses:
            return True
        return _load_per_frame_pose(seq.path, idx, self.pose_layout, self.pose_scale_cm) is not None

    def _get_abs_pose(self, seq: _SeqIndex, idx: int) -> np.ndarray:
        if seq.poses is not None and idx in seq.poses:
            return seq.poses[idx].copy()
        T = _load_per_frame_pose(seq.path, idx, self.pose_layout, self.pose_scale_cm)
        if T is None:
            raise RuntimeError(f"No pose found for sequence={seq.name} frame={idx}")
        return T

    def _build_samples(self) -> None:
        k = self.frame_interval
        for seq in self.seq_indices:
            indices = sorted(set(seq.rgb_by_idx.keys()) & set(seq.depth_by_idx.keys()))
            idx_set = set(indices)
            for idx in indices:
                if idx + k not in idx_set:
                    continue
                if self.use_triplets and idx - k not in idx_set:
                    continue
                if not self._has_pose(seq, idx) or not self._has_pose(seq, idx + k):
                    continue
                if self.use_triplets and not self._has_pose(seq, idx - k):
                    continue
                self.samples.append((seq, idx))

    def __len__(self) -> int:
        return len(self.samples)

    def _load_rgb_pair(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        img = _resize_rgb(_pil_rgb(path), self.size_hw)
        img01 = _to_tensor_01(img)
        img_norm = _imagenet_normalize(img01)
        return img_norm, img01

    def _load_depth_cm(self, path: Path) -> torch.Tensor:
        depth_norm_or_raw = _read_depth_norm(path)
        if np.nanmax(depth_norm_or_raw) > 1.5 and _as_bool(_opt_get(self.opt, "simcol_depth_already_cm", False), False):
            depth_cm = depth_norm_or_raw.astype(np.float32)
        else:
            depth_cm = depth_norm_or_raw.astype(np.float32) * self.depth_scale_cm
        depth_cm = _resize_depth_np(depth_cm, self.size_hw)
        depth_cm = np.clip(depth_cm, 0.0, self.depth_scale_cm).astype(np.float32)
        return torch.from_numpy(depth_cm[None, ...])

    def __getitem__(self, i: int) -> Dict[str, torch.Tensor]:
        seq, idx = self.samples[i]
        k = self.frame_interval
        idx_next = idx + k
        idx_prev = idx - k if self.use_triplets else idx

        image, image01 = self._load_rgb_pair(seq.rgb_by_idx[idx])
        image_next, image_next01 = self._load_rgb_pair(seq.rgb_by_idx[idx_next])
        depth_gt = self._load_depth_cm(seq.depth_by_idx[idx])

        T_t = self._get_abs_pose(seq, idx)
        T_next = self._get_abs_pose(seq, idx_next)
        T_t_to_next = _relative_pose(T_t, T_next, self.pose_convention).astype(np.float32)

        out: Dict[str, torch.Tensor | str | int] = {
            "image": image,
            "image_01": image01,
            "image_next": image_next,
            "image_next_01": image_next01,
            "depth_gt": depth_gt,
            "has_depth": torch.tensor(True, dtype=torch.bool),
            "K": self.K.clone(),
            "pose_gt": torch.from_numpy(T_t_to_next),
            "has_pose": torch.tensor(True, dtype=torch.bool),
            "seq_name": seq.name,
            "frame_idx": int(idx),
            "frame_idx_next": int(idx_next),
            "dataset": "simcol",
        }

        if self.use_triplets:
            image_prev, image_prev01 = self._load_rgb_pair(seq.rgb_by_idx[idx_prev])
            T_prev = self._get_abs_pose(seq, idx_prev)
            T_t_to_prev = _relative_pose(T_t, T_prev, self.pose_convention).astype(np.float32)
            out.update({
                "image_prev": image_prev,
                "image_prev_01": image_prev01,
                "pose_gt_prev": torch.from_numpy(T_t_to_prev),
                "has_pose_prev": torch.tensor(True, dtype=torch.bool),
                "frame_idx_prev": int(idx_prev),
            })
        return out  # type: ignore[return-value]


def make_simcol_dataloaders(opt):
    from torch.utils.data import DataLoader
    root = _opt_get(opt, "simcol_root", None)
    if root is None:
        raise ValueError("--simcol_root is required for --dataset simcol")
    train_ds = SimColDataset(root, opt, split="train")
    val_ds = SimColDataset(root, opt, split="val")
    num_workers = int(_opt_get(opt, "num_workers", 4))
    batch_size = int(_opt_get(opt, "batch_size", 8))
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
    return train_loader, val_loader