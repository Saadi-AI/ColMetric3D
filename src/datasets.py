from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms.functional as TF

# Default C3VD / MetricCol-style undistorted camera matrix used by the user.
DEFAULT_C3VD_K = np.array(
    [[769.8074, -0.1412, 675.2264], [0.0, 769.7205, 548.9035], [0.0, 0.0, 1.0]], dtype=np.float32
)


def read_camera_txt(path: Optional[str]) -> np.ndarray:
    if path is None or str(path).lower() in {"", "none"}:
        return DEFAULT_C3VD_K.copy()
    p = Path(path)
    if not p.exists():
        print(f"[datasets] camera_txt not found: {p}. Using DEFAULT_C3VD_K.")
        return DEFAULT_C3VD_K.copy()
    vals: Dict[str, float] = {}
    with open(p, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(":", " ").split()
            if len(parts) >= 2:
                try:
                    vals[parts[0]] = float(parts[1])
                except ValueError:
                    pass
    if {"fx", "fy", "cx", "cy"}.issubset(vals):
        return np.array([[vals["fx"], vals.get("skew", 0.0), vals["cx"]], [0.0, vals["fy"], vals["cy"]], [0.0, 0.0, 1.0]], dtype=np.float32)
    print(f"[datasets] camera_txt missing fx/fy/cx/cy. Using DEFAULT_C3VD_K.")
    return DEFAULT_C3VD_K.copy()


def update_intrinsics_for_resize(K: np.ndarray, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
    old_h, old_w = old_size
    new_h, new_w = new_size
    sx = new_w / float(old_w)
    sy = new_h / float(old_h)
    K2 = K.copy().astype(np.float32)
    K2[0, 0] *= sx
    K2[0, 1] *= sx
    K2[0, 2] *= sx
    K2[1, 1] *= sy
    K2[1, 2] *= sy
    return K2


def safe_frame_stem(path: Path) -> str:
    stem = path.stem
    for suffix in ["_color", "_rgb", "_image", "_left", "_depth"]:
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
    return stem


def try_parse_frame_index(path: Path) -> int:
    stem = safe_frame_stem(path)
    digits = "".join(ch for ch in stem if ch.isdigit())
    if not digits:
        raise ValueError(f"Could not parse frame index from {path.name}")
    return int(digits)


def _fix_c3vd_pose_layout(T_raw: np.ndarray) -> np.ndarray:
    """C3VD pose.txt seen in practice often stores translation in the last row.

    Example line:
        r00,r01,r02,0, r10,r11,r12,0, r20,r21,r22,0, tx,ty,tz,1
    This is row-major text for a matrix that must be transposed to obtain the
    standard homogeneous transform with translation in the last column.
    """
    T = T_raw.astype(np.float32)
    last_col_zero = np.allclose(T[:3, 3], 0.0, atol=1e-7)
    last_row_has_translation = np.linalg.norm(T[3, :3]) > 1e-8
    if last_col_zero and last_row_has_translation:
        T = T.T.copy()
    return T


def load_pose_txt(pose_path: Path) -> Dict[int, np.ndarray]:
    """Load C3VD absolute camera-to-world poses with robust row/column fix."""
    pose_dict: Dict[int, np.ndarray] = {}
    if not pose_path.exists():
        return pose_dict
    with open(pose_path, "r") as f:
        for line_idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            vals = [float(v) for v in line.replace(",", " ").split()]
            if len(vals) == 17:
                frame_idx, pose_vals = int(vals[0]), vals[1:]
            elif len(vals) == 16:
                frame_idx, pose_vals = line_idx, vals
            else:
                continue
            T_raw = np.array(pose_vals, dtype=np.float32).reshape(4, 4)
            T = _fix_c3vd_pose_layout(T_raw)
            if abs(float(T[3, 3])) > 1e-8:
                pose_dict[frame_idx] = T
    return pose_dict


def convert_pose_translation_units(T: np.ndarray, source_unit: str = "mm") -> np.ndarray:
    T = T.copy().astype(np.float32)
    factors_to_cm = {"mm": 0.1, "cm": 1.0, "m": 100.0}
    if source_unit not in factors_to_cm:
        raise ValueError(f"Unsupported pose_source_unit={source_unit}; use mm, cm, or m")
    T[:3, 3] *= factors_to_cm[source_unit]
    return T


def relative_camera_transform(T_w_t: np.ndarray, T_w_s: np.ndarray) -> np.ndarray:
    """Transform 3D points from target camera t to source camera s.

    C3VD absolute poses are camera-to-world. Therefore point transform t->s is
    inv(T_w_s) @ T_w_t.
    """
    return np.linalg.inv(T_w_s) @ T_w_t


def load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


def load_depth_array(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr


def depth_to_cm(depth: np.ndarray, fmt: str = "auto") -> np.ndarray:
    d = depth.astype(np.float32)
    if fmt == "c3vd_16bit_100mm":
        return (d / 65535.0) * 10.0
    if fmt == "mm":
        return d / 10.0
    if fmt == "cm":
        return d
    if fmt == "m":
        return d * 100.0
    mx = float(np.nanmax(d)) if d.size else 0.0
    if np.issubdtype(depth.dtype, np.integer):
        if mx > 1000.0:
            return (d / 65535.0) * 10.0
        return d / 10.0
    if mx > 100.0:
        return d / 10.0
    if mx <= 2.0:
        return d * 100.0
    return d


class C3VDDataset(Dataset):
    def __init__(self, root: str, opt: Dict, split: str = "train", target_seq: Optional[str] = None):
        self.root = Path(root)
        self.opt = opt
        self.split = split
        self.target_seq = target_seq
        self.frame_interval = int(opt.get("c3vd_frame_interval", 1))
        self.use_triplets = bool(opt.get("use_triplet_pose", False))
        self.sample_step = int(opt.get("sample_step", 1 if split == "train" else opt.get("eval_step", 5)))
        self.image_size = (int(opt.get("image_height", 518)), int(opt.get("image_width", 518)))
        self.depth_format = opt.get("depth_format", "auto")
        self.pose_source_unit = opt.get("pose_source_unit", "mm")
        self.max_depth_cm = float(opt.get("max_depth_cm", 10.0))
        self.K_base = read_camera_txt(opt.get("camera_txt"))
        self.samples: List[Dict] = []
        self._index_dataset()
        if not self.samples:
            raise RuntimeError(f"No C3VD samples found in {self.root}")
        mode = "triplets" if self.use_triplets else "pairs"
        print(f"[C3VDDataset] {split}: {len(self.samples)} {mode} from {self.root}")

    def _seq_dirs(self) -> List[Path]:
        if self.target_seq:
            return [self.root / self.target_seq]
        if not self.root.exists():
            raise FileNotFoundError(f"Dataset root not found: {self.root}")
        if (self.root / "color").is_dir() and (self.root / "depth").is_dir():
            return [self.root]
        return sorted([d for d in self.root.iterdir() if d.is_dir()])

    @staticmethod
    def _find_color_dir(seq: Path) -> Optional[Path]:
        for name in ["color", "rgb", "images", "image"]:
            p = seq / name
            if p.is_dir():
                return p
        return None

    @staticmethod
    def _find_depth_file(depth_dir: Path, stem: str) -> Optional[Path]:
        if not depth_dir.is_dir():
            return None
        candidates = []
        for ext in [".tiff", ".tif", ".png", ".jpg", ".jpeg"]:
            candidates.extend([depth_dir / f"{stem}_depth{ext}", depth_dir / f"{stem}{ext}"])
        for c in candidates:
            if c.exists():
                return c
        matches = sorted(list(depth_dir.glob(f"{stem}*")))
        return matches[0] if matches else None

    def _index_dataset(self) -> None:
        for seq in self._seq_dirs():
            color_dir = self._find_color_dir(seq)
            depth_dir = seq / "depth"
            if color_dir is None:
                continue
            pose_dict = load_pose_txt(seq / "pose.txt")
            frames = sorted(list(color_dir.glob("*.png")) + list(color_dir.glob("*.jpg")) + list(color_dir.glob("*.jpeg")))
            if len(frames) <= self.frame_interval:
                continue

            if self.use_triplets:
                start = self.frame_interval
                end = len(frames) - self.frame_interval
                for i in range(start, end, self.sample_step):
                    prev_f = frames[i - self.frame_interval]
                    tgt_f = frames[i]
                    next_f = frames[i + self.frame_interval]
                    try:
                        prev_idx = try_parse_frame_index(prev_f)
                        tgt_idx = try_parse_frame_index(tgt_f)
                        next_idx = try_parse_frame_index(next_f)
                    except ValueError:
                        continue
                    stem = safe_frame_stem(tgt_f)
                    depth_p = self._find_depth_file(depth_dir, stem)
                    self.samples.append({
                        "seq_name": seq.name,
                        "tgt_path": tgt_f,
                        "src_path": next_f,
                        "prev_path": prev_f,
                        "depth_path": depth_p,
                        "tgt_idx": tgt_idx,
                        "src_idx": next_idx,
                        "prev_idx": prev_idx,
                        "pose_tgt": pose_dict.get(tgt_idx),
                        "pose_src": pose_dict.get(next_idx),
                        "pose_prev": pose_dict.get(prev_idx),
                    })
            else:
                for i in range(0, len(frames) - self.frame_interval, self.sample_step):
                    tgt = frames[i]
                    src = frames[i + self.frame_interval]
                    try:
                        t_idx = try_parse_frame_index(tgt)
                        s_idx = try_parse_frame_index(src)
                    except ValueError:
                        continue
                    stem = safe_frame_stem(tgt)
                    depth_p = self._find_depth_file(depth_dir, stem)
                    self.samples.append({
                        "seq_name": seq.name,
                        "tgt_path": tgt,
                        "src_path": src,
                        "prev_path": None,
                        "depth_path": depth_p,
                        "tgt_idx": t_idx,
                        "src_idx": s_idx,
                        "prev_idx": -1,
                        "pose_tgt": pose_dict.get(t_idx),
                        "pose_src": pose_dict.get(s_idx),
                        "pose_prev": None,
                    })

    def __len__(self) -> int:
        return len(self.samples)

    def _load_resized_rgb_pair(self, path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
        img = load_rgb(path)
        img_01 = TF.to_tensor(TF.resize(img, self.image_size, interpolation=TF.InterpolationMode.BILINEAR))
        mean = torch.tensor(self.opt.get("norm_mean", [0.485, 0.456, 0.406]), dtype=torch.float32).view(3, 1, 1)
        std = torch.tensor(self.opt.get("norm_std", [0.229, 0.224, 0.225]), dtype=torch.float32).view(3, 1, 1)
        return (img_01 - mean) / std, img_01

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        s = self.samples[idx]
        img_pil = load_rgb(s["tgt_path"])
        old_size = (img_pil.height, img_pil.width)
        new_h, new_w = self.image_size

        image, img_01 = self._load_resized_rgb_pair(s["tgt_path"])
        image_next, img_next_01 = self._load_resized_rgb_pair(s["src_path"])
        K = torch.from_numpy(update_intrinsics_for_resize(self.K_base, old_size, self.image_size)).float()

        out = {
            "image": image,
            "image_next": image_next,
            "image_01": img_01,
            "image_next_01": img_next_01,
            "K": K,
            "seq_name": s["seq_name"],
            "frame_idx": torch.tensor(s["tgt_idx"], dtype=torch.long),
            "src_idx": torch.tensor(s["src_idx"], dtype=torch.long),
        }

        if self.use_triplets and s.get("prev_path") is not None:
            image_prev, img_prev_01 = self._load_resized_rgb_pair(s["prev_path"])
            out["image_prev"] = image_prev
            out["image_prev_01"] = img_prev_01
            out["prev_idx"] = torch.tensor(s["prev_idx"], dtype=torch.long)

        if s["depth_path"] is not None:
            depth_raw = load_depth_array(s["depth_path"])
            depth_cm = depth_to_cm(depth_raw, self.depth_format)
            depth_img = Image.fromarray(depth_cm.astype(np.float32))
            depth_resized = np.array(TF.resize(depth_img, self.image_size, interpolation=TF.InterpolationMode.NEAREST), dtype=np.float32)
            depth_t = torch.from_numpy(depth_resized).unsqueeze(0).float().clamp(min=0.0, max=self.max_depth_cm)
            out["depth_gt"] = depth_t
            out["has_depth"] = torch.tensor(True)
        else:
            out["depth_gt"] = torch.zeros(1, new_h, new_w)
            out["has_depth"] = torch.tensor(False)

        if s["pose_tgt"] is not None and s["pose_src"] is not None:
            T_w_t = convert_pose_translation_units(s["pose_tgt"], self.pose_source_unit)
            T_w_s = convert_pose_translation_units(s["pose_src"], self.pose_source_unit)
            out["pose_gt"] = torch.from_numpy(relative_camera_transform(T_w_t, T_w_s).astype(np.float32))
            out["has_pose"] = torch.tensor(True)
        else:
            out["pose_gt"] = torch.eye(4, dtype=torch.float32)
            out["has_pose"] = torch.tensor(False)

        if self.use_triplets:
            if s["pose_tgt"] is not None and s["pose_prev"] is not None:
                T_w_t = convert_pose_translation_units(s["pose_tgt"], self.pose_source_unit)
                T_w_p = convert_pose_translation_units(s["pose_prev"], self.pose_source_unit)
                out["pose_gt_prev"] = torch.from_numpy(relative_camera_transform(T_w_t, T_w_p).astype(np.float32))
                out["has_pose_prev"] = torch.tensor(True)
            else:
                out["pose_gt_prev"] = torch.eye(4, dtype=torch.float32)
                out["has_pose_prev"] = torch.tensor(False)

        return out


def build_dataloaders(opt: Dict):
    dataset_name = str(opt.get("dataset", "c3vd")).lower()

    if dataset_name == "simcol":
        from datasets_simcol import make_simcol_dataloaders
        return make_simcol_dataloaders(opt)

    train_ds = C3VDDataset(opt["c3vd_root"], opt, split="train")
    val_ds = C3VDDataset(opt["c3vd_val_root"], opt, split="val")
    train_loader = DataLoader(train_ds, batch_size=int(opt.get("batch_size", 1)), shuffle=True,
                              num_workers=int(opt.get("num_workers", 4)), pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=1, shuffle=False, num_workers=int(opt.get("num_workers", 4)), pin_memory=True)
    return train_loader, val_loader