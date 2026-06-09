from __future__ import annotations

"""
Flexible zero-shot endoscopy depth datasets for EndoMetric3D.

Designed for:
  1) EndoSLAM / TMI-2026-style folders:
       Frames_O1/FrameBuffer_0001.png
       Frames_O1/Depth_0001.png              # optional
       split file lines: "Frames_O1 0001 l"

  2) SCARED-style nested folders:
       dataset_*/keyframe_*/data/left/*.png
       optional matching depth/disparity folders if present.

Returns model-compatible keys:
  image, image_01, K, depth_gt, has_depth, seq_name, frame_idx

Depth is always returned in centimeters.
If no GT depth is found, has_depth=False and depth_gt=zeros.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

try:
    from datasets import read_camera_txt, update_intrinsics_for_resize
except Exception:
    # fallback copy, in case imported outside src/
    DEFAULT_C3VD_K = np.array(
        [[769.8074, -0.1412, 675.2264], [0.0, 769.7205, 548.9035], [0.0, 0.0, 1.0]], dtype=np.float32
    )
    def read_camera_txt(path: Optional[str]) -> np.ndarray:
        if path is None or str(path).lower() in {"", "none"}:
            return DEFAULT_C3VD_K.copy()
        vals = {}
        for line in Path(path).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.replace(":", " ").split()
            if len(parts) >= 2:
                try:
                    vals[parts[0]] = float(parts[1])
                except Exception:
                    pass
        if {"fx", "fy", "cx", "cy"}.issubset(vals):
            return np.array([[vals["fx"], vals.get("skew", 0.0), vals["cx"]], [0.0, vals["fy"], vals["cy"]], [0.0, 0.0, 1.0]], dtype=np.float32)
        return DEFAULT_C3VD_K.copy()

    def update_intrinsics_for_resize(K: np.ndarray, old_size: Tuple[int, int], new_size: Tuple[int, int]) -> np.ndarray:
        old_h, old_w = old_size
        new_h, new_w = new_size
        sx, sy = new_w / float(old_w), new_h / float(old_h)
        K2 = K.copy().astype(np.float32)
        K2[0, 0] *= sx; K2[0, 1] *= sx; K2[0, 2] *= sx
        K2[1, 1] *= sy; K2[1, 2] *= sy
        return K2

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
DEPTH_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy", ".npz"}
BAD_IMAGE_PARTS = {
    "depth", "depths", "disp", "disparity", "disparities", "mask", "masks", "label", "labels",
    "seg", "segmentation", "pose", "poses", "calib", "calibration", "intrinsic", "intrinsics",
}
IMAGE_DIR_HINTS = {"left", "right", "rgb", "image", "images", "color", "frames", "frame", "video", "video_frames", "FrameBuffer"}
DEPTH_DIR_HINTS = ["depth", "depths", "groundtruth", "gt_depth", "gt", "scene_points", "disparity", "disp"]

_FRAMEBUFFER_RE = re.compile(r"FrameBuffer[_-]?(\d+)", re.IGNORECASE)
_DEPTH_RE = re.compile(r"Depth[_-]?(\d+)", re.IGNORECASE)
_DIGITS_RE = re.compile(r"(\d+)")

@dataclass
class EndoSample:
    image_path: Path
    depth_path: Optional[Path]
    seq_name: str
    frame_idx: int


def _pil_rgb(path: Path) -> Image.Image:
    with Image.open(path) as img:
        return img.convert("RGB")


def _to_tensor_01(img: Image.Image, size_hw: Tuple[int, int]) -> torch.Tensor:
    h, w = size_hw
    img_r = TF.resize(img, [h, w], interpolation=TF.InterpolationMode.BILINEAR)
    return TF.to_tensor(img_r).float()


def _imagenet_norm(img01: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    m = torch.tensor(mean, dtype=torch.float32).view(3, 1, 1)
    s = torch.tensor(std, dtype=torch.float32).view(3, 1, 1)
    return (img01 - m) / s


def _parse_frame_index(path: Path) -> int:
    for rx in [_FRAMEBUFFER_RE, _DEPTH_RE, _DIGITS_RE]:
        m = rx.search(path.stem)
        if m:
            return int(m.group(1))
    return -1


def _seq_name_from_path(image_path: Path, root: Path) -> str:
    try:
        rel = image_path.relative_to(root)
        parts = rel.parts
    except Exception:
        parts = image_path.parts
    # SCARED: dataset_x/keyframe_y/data/left/000.png -> dataset_x/keyframe_y
    if "keyframe_1" in parts or any(p.startswith("keyframe_") for p in parts):
        k = next(i for i, p in enumerate(parts) if p.startswith("keyframe_"))
        return "/".join(parts[max(0, k-1):k+1])
    # TMI / EndoSLAM: Frames_O1/FrameBuffer_0001.png -> Frames_O1
    for p in reversed(parts[:-1]):
        if p.lower() not in {"left", "right", "rgb", "images", "image", "color", "frames", "data"}:
            return p
    return image_path.parent.name


def _is_candidate_image(path: Path) -> bool:
    if path.suffix.lower() not in IMG_EXTS:
        return False
    parts_lower = {p.lower() for p in path.parts}
    if parts_lower & BAD_IMAGE_PARTS:
        # allow image dirs with exact names, reject depth/disp/etc.
        if not (parts_lower & {"left", "right", "rgb", "image", "images", "color", "frames"}):
            return False
        if parts_lower & {"depth", "depths", "disp", "disparity", "disparities", "scene_points"}:
            return False
    name = path.name.lower()
    if any(tok in name for tok in ["depth", "disp", "mask", "label", "seg"]):
        return False
    return True


def _read_depth_raw(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path).astype(np.float32)
    if path.suffix.lower() == ".npz":
        z = np.load(path)
        key = "depth" if "depth" in z.files else z.files[0]
        return z[key].astype(np.float32)
    with Image.open(path) as img:
        arr = np.asarray(img)
    if arr.ndim == 3:
        arr = arr[..., 0]
    return arr.astype(np.float32)


def depth_to_cm(depth: np.ndarray, unit: str = "auto", depth_scale_cm: float = 20.0,
                disparity_fx: Optional[float] = None, disparity_baseline_mm: Optional[float] = None) -> np.ndarray:
    """Convert a depth/disparity array to centimeters.

    unit options:
      auto       : heuristic
      cm         : already centimeters
      mm         : millimeters -> cm
      m          : meters -> cm
      norm20cm   : normalized [0,1] or uint depth used by TMI Endo/SimCol, scale to depth_scale_cm
      uint16_20cm: uint16 [0,65535] maps to [0,depth_scale_cm]
      disparity  : depth = fx * baseline / disparity. baseline in mm, output cm.
    """
    d = depth.astype(np.float32)
    unit = str(unit).lower()

    if unit == "disparity":
        if not disparity_fx or not disparity_baseline_mm:
            raise ValueError("For --depth_unit disparity, provide --disparity_fx and --disparity_baseline_mm")
        disp = np.maximum(d, 1e-6)
        depth_mm = float(disparity_fx) * float(disparity_baseline_mm) / disp
        return depth_mm / 10.0

    if unit == "cm":
        return d
    if unit == "mm":
        return d / 10.0
    if unit == "m":
        return d * 100.0
    if unit == "uint16_20cm":
        return d / 65535.0 * depth_scale_cm
    if unit == "norm20cm":
        if depth.dtype == np.uint16 or d.max() > 255:
            return d / 65535.0 * depth_scale_cm
        if depth.dtype == np.uint8 or d.max() > 2.0:
            return d / 255.0 * depth_scale_cm
        return d * depth_scale_cm

    # auto heuristic
    mx = float(np.nanmax(d)) if d.size else 0.0
    if depth.dtype == np.uint16:
        # common cases: depth in mm, or normalized depth in uint16. Use conservative rule.
        if mx <= 65535 and mx > 1000:
            # If values look like real mm, cm will be huge; endoscopic depths usually < 20 cm.
            cm_as_mm = d / 10.0
            if np.nanmedian(cm_as_mm) <= 50.0:
                return cm_as_mm
            return d / 65535.0 * depth_scale_cm
    if depth.dtype == np.uint8:
        return d / 255.0 * depth_scale_cm
    if mx <= 2.0:
        return d * 100.0  # likely meters
    if mx <= 50.0:
        return d          # likely cm
    return d / 10.0       # likely mm


def _candidate_depth_paths(image_path: Path, root: Path) -> List[Path]:
    stem = image_path.stem
    idx = _parse_frame_index(image_path)
    parent = image_path.parent
    seq_roots = [parent, parent.parent, parent.parent.parent if parent.parent else parent]
    seq_roots = [p for p in seq_roots if p is not None and p.exists()]

    names = [stem]
    if idx >= 0:
        names += [f"Depth_{idx:04d}", f"depth_{idx:04d}", f"{idx:04d}", f"{idx:06d}", f"{idx:010d}"]
    # if image is FrameBuffer_0001, try matching Depth_0001
    m = _FRAMEBUFFER_RE.search(stem)
    if m:
        n = int(m.group(1))
        names += [f"Depth_{n:04d}", f"Depth_{n:05d}", f"Depth_{n:06d}"]

    out: List[Path] = []
    for base in seq_roots:
        for dname in DEPTH_DIR_HINTS:
            ddir = base / dname
            if not ddir.exists():
                continue
            for nm in names:
                for ext in DEPTH_EXTS:
                    out.append(ddir / f"{nm}{ext}")
                    out.append(ddir / f"{nm}_depth{ext}")
        # TMI layout: depth files are in same folder as frames
        for nm in names:
            for ext in DEPTH_EXTS:
                out.append(base / f"{nm}{ext}")
                out.append(parent / f"{nm}{ext}")
    # Direct sibling replacement for SCARED: left -> disparity/depth
    for repl in ["depth", "disparity", "disp", "scene_points"]:
        if parent.name.lower() in {"left", "right", "rgb", "image", "images", "color"}:
            ddir = parent.parent / repl
            for nm in names:
                for ext in DEPTH_EXTS:
                    out.append(ddir / f"{nm}{ext}")
    # Deduplicate preserving order
    seen = set(); unique = []
    for p in out:
        if p not in seen:
            unique.append(p); seen.add(p)
    return unique


def _find_depth_for_image(image_path: Path, root: Path) -> Optional[Path]:
    for p in _candidate_depth_paths(image_path, root):
        if p.exists() and p.is_file() and p != image_path:
            return p
    return None


def _samples_from_split(root: Path, split_file: Path) -> List[EndoSample]:
    samples: List[EndoSample] = []
    for line in split_file.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        folder, frame = parts[0], int(parts[1])
        candidates = []
        for ext in IMG_EXTS:
            candidates += [root / folder / f"FrameBuffer_{frame:04d}{ext}", root / folder / f"{frame:04d}{ext}", root / folder / f"{frame:06d}{ext}"]
        image_path = next((p for p in candidates if p.exists()), None)
        if image_path is None:
            continue
        depth_path = _find_depth_for_image(image_path, root)
        samples.append(EndoSample(image_path=image_path, depth_path=depth_path, seq_name=folder, frame_idx=frame))
    return samples


def _scan_samples(root: Path, max_samples: int = 0) -> List[EndoSample]:
    images = [p for p in root.rglob("*") if p.is_file() and _is_candidate_image(p)]
    images = sorted(images)
    samples: List[EndoSample] = []
    for p in images:
        seq = _seq_name_from_path(p, root)
        idx = _parse_frame_index(p)
        depth_p = _find_depth_for_image(p, root)
        samples.append(EndoSample(p, depth_p, seq, idx))
        if max_samples and len(samples) >= max_samples:
            break
    return samples


class ZeroShotEndoscopyDepthDataset(Dataset):
    def __init__(self, root: str, opt: Dict, split_file: str = "", dataset_name: str = "endoscopy", max_samples: int = 0):
        self.root = Path(root)
        self.opt = opt
        self.dataset_name = dataset_name
        self.image_size = (int(opt.get("image_height", opt.get("height", 476))), int(opt.get("image_width", opt.get("width", 588))))
        self.K_base = read_camera_txt(opt.get("camera_txt", "none"))
        self.norm_mean = opt.get("norm_mean", [0.485, 0.456, 0.406])
        self.norm_std = opt.get("norm_std", [0.229, 0.224, 0.225])
        self.depth_unit = opt.get("zero_depth_unit", opt.get("depth_unit", "auto"))
        self.depth_scale_cm = float(opt.get("zero_depth_scale_cm", 20.0))
        self.disparity_fx = opt.get("disparity_fx", None)
        self.disparity_baseline_mm = opt.get("disparity_baseline_mm", None)
        sf = Path(split_file) if split_file else None
        if sf and sf.exists():
            self.samples = _samples_from_split(self.root, sf)
            print(f"[ZeroShotDataset] loaded split file {sf}: {len(self.samples)} samples")
        else:
            self.samples = _scan_samples(self.root, max_samples=max_samples)
            print(f"[ZeroShotDataset] scanned {self.root}: {len(self.samples)} samples")
        if not self.samples:
            raise RuntimeError(f"No image samples found under {self.root}. Check --zero_root and --split_file.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict:
        s = self.samples[idx]
        img = _pil_rgb(s.image_path)
        old_size = (img.height, img.width)
        img01 = _to_tensor_01(img, self.image_size)
        image = _imagenet_norm(img01, self.norm_mean, self.norm_std)
        K = torch.from_numpy(update_intrinsics_for_resize(self.K_base, old_size, self.image_size)).float()
        H, W = self.image_size
        out = {
            "image": image,
            "image_01": img01,
            "K": K,
            "seq_name": s.seq_name,
            "frame_idx": torch.tensor(s.frame_idx, dtype=torch.long),
            "image_path": str(s.image_path),
            "has_depth": torch.tensor(False),
            "depth_gt": torch.zeros(1, H, W, dtype=torch.float32),
        }
        if s.depth_path is not None:
            raw = _read_depth_raw(s.depth_path)
            depth_cm = depth_to_cm(raw, unit=self.depth_unit, depth_scale_cm=self.depth_scale_cm,
                                   disparity_fx=self.disparity_fx, disparity_baseline_mm=self.disparity_baseline_mm)
            # resize nearest
            depth_img = Image.fromarray(depth_cm.astype(np.float32), mode="F")
            depth_resized = np.asarray(depth_img.resize((W, H), resample=Image.NEAREST), dtype=np.float32)
            out["depth_gt"] = torch.from_numpy(depth_resized).unsqueeze(0).float()
            out["has_depth"] = torch.tensor(True)
            out["depth_path"] = str(s.depth_path)
        else:
            out["depth_path"] = ""
        return out
