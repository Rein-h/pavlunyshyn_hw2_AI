"""
hw2_detection.py — all logic for HW2 (object detection on drone videos, VisDrone2019-VID).

The notebook only calls functions from this file. Sections:

    0. Environment helpers (Colab / local, data preparation from a folder or a .zip, workers, GPU info)
    1. Constants and utilities (seeds, paths, box helpers)
    2. Data index (YOLO labels -> pixel boxes, ignore regions, test list)
    3. Splits and frame subsampling (split by clip, never by frame)
    4. EDA (tables and plots)
    5. Visualisation and conversion checks
    6. Dataset / DataLoader (ignore-region masking, augmentation)
    7. Models with one common interface:
         faster_rcnn (two-stage, anchor-based)   - torchvision
         retinanet   (one-stage, anchor-based)   - torchvision
         fcos        (one-stage, anchor-free)    - torchvision
         rtdetr      (query-based)               - Hugging Face transformers
    8. Training (same protocol for every model)
    9. Inference, speed, COCO evaluation (ignored regions are "don't care")
   10. Analysis helpers (anchors, FCOS assignment, RPN recall, failure cases, curves)
   11. Experiment runner (skips finished runs, resumes interrupted ones)
   12. Submission

Works locally (VS Code, Windows/Linux) and in Google Colab; see section 0 for environment helpers.

Class ids are 0..7 everywhere outside the models (same as YOLO labels and the submission).
Boxes are [x1, y1, x2, y2] in pixels of the original image everywhere outside the models.
"""
from __future__ import annotations

import contextlib
import copy
import gc
import io
import json
import math
import os
import pickle
import random
import re
import shutil
import sys
import time
import types
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:  # avoid thread oversubscription inside DataLoader workers
    import cv2
    cv2.setNumThreads(0)
except ImportError:
    pass


# =============================================================================
# 0. Environment helpers
# =============================================================================
def in_colab() -> bool:
    return "google.colab" in sys.modules


def default_num_workers() -> int:
    """DataLoader workers: Windows starts workers slowly and Colab has 2 CPUs -> 2; otherwise up to 4."""
    if os.name == "nt" or in_colab():
        return 2
    return max(0, min(4, (os.cpu_count() or 1) - 1))


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"gpu": None}
    p = torch.cuda.get_device_properties(0)
    return {"gpu": p.name, "vram_gb": round(p.total_memory / 1024 ** 3, 1), "torch": torch.__version__,
            "cuda": torch.version.cuda}


def extract_zip(zip_path, out_dir, every: int = 5000) -> Path:
    """Extract a zip with progress output; checks free disk space first. Skips if already extracted."""
    zip_path, out_dir = Path(zip_path), Path(out_dir)
    marker = out_dir / ".extracted_ok"
    if marker.exists():
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.infolist()
        need = sum(m.file_size for m in members)
        free = shutil.disk_usage(out_dir).free
        if free < need * 1.05:
            raise OSError(f"Not enough disk space in {out_dir}: need ~{need / 1e9:.1f} GB, free {free / 1e9:.1f} GB")
        print(f"Extracting {zip_path.name}: {len(members)} files, {need / 1e9:.1f} GB -> {out_dir}")
        t0 = time.time()
        for i, m in enumerate(members, 1):
            zf.extract(m, out_dir)
            if i % every == 0 or i == len(members):
                print(f"  {i}/{len(members)} files, {time.time() - t0:.0f} s", flush=True)
    marker.touch()
    return out_dir


def prepare_data(source, extract_dir=None) -> Path:
    """Return the data root (folder with train/, val/, test/, test_images.csv).

    `source` may be: the data root itself, a folder that contains it (searched a few levels down),
    a folder that contains the competition .zip, or the .zip file. A zip is extracted once into
    `extract_dir` (default: next to the zip, folder named like the zip).
    """
    source = Path(source)
    if not source.exists():
        raise FileNotFoundError(f"{source} does not exist")
    try:
        return find_data_root(source if source.is_dir() else source.parent)
    except FileNotFoundError:
        pass
    zips = [source] if source.suffix.lower() == ".zip" else sorted(source.glob("*.zip"))
    if not zips:
        raise FileNotFoundError(f"No data folder (train/, test_images.csv) and no .zip found in {source}")
    zip_path = max(zips, key=lambda p: p.stat().st_size)
    out = Path(extract_dir) if extract_dir else zip_path.with_suffix("")
    extract_zip(zip_path, out)
    return find_data_root(out)


def set_kaggle_credentials_from_colab() -> bool:
    """Copy Kaggle API credentials from Colab secrets (KAGGLE_USERNAME + KAGGLE_KEY, or KAGGLE_API_TOKEN)
    to environment variables, where kagglehub reads them."""
    try:
        from google.colab import userdata
    except ImportError:
        return False
    found = False
    for key in ("KAGGLE_USERNAME", "KAGGLE_KEY", "KAGGLE_API_TOKEN"):
        try:
            val = userdata.get(key)
        except Exception:
            val = None
        if val:
            os.environ[key] = val
            found = True
    return found


def download_competition(competition: str, cache_dir=None) -> Path:
    """Download (and unpack) the competition files with kagglehub; returns the data root.

    Needs Kaggle API credentials (~/.kaggle/kaggle.json, environment variables or Colab secrets)
    and accepted competition rules on the Kaggle website.
    """
    if cache_dir:
        os.environ["KAGGLEHUB_CACHE"] = str(cache_dir)
    if in_colab():
        set_kaggle_credentials_from_colab()
    import kagglehub
    path = kagglehub.competition_download(competition)
    return prepare_data(path)


# =============================================================================
# 1. Constants and utilities
# =============================================================================
CLASS_NAMES = ["person", "bicycle", "car", "van", "truck", "tricycle", "bus", "motor"]
NUM_CLASSES = len(CLASS_NAMES)

# Size bins by sqrt(box area) in pixels of the ORIGINAL image.
# COCO "small" is sqrt(area) < 32, so it contains both "tiny" and "small" below.
SIZE_BIN_EDGES = [0, 16, 32, 96, np.inf]
SIZE_BIN_LABELS = ["tiny (<16)", "small (16-32)", "medium (32-96)", "large (>=96)"]

# Area ranges used by COCO evaluation (area in px^2). "tiny" is an extra range.
AREA_RANGES = {
    "all": [0, 1e10],
    "small": [0, 32 ** 2],
    "medium": [32 ** 2, 96 ** 2],
    "large": [96 ** 2, 1e10],
    "tiny": [0, 16 ** 2],
}

FILL_VALUE = (124, 116, 104)  # ImageNet mean in 0..255, used to paint ignored regions
EMPTY_BOXES = np.zeros((0, 4), np.float32)
EMPTY_LABELS = np.zeros((0,), np.int64)


def seed_everything(seed: int = 42) -> None:
    """Fix all random seeds (Python, NumPy, PyTorch)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def find_data_root(start, max_depth: int = 3) -> Path:
    """Find the folder that contains train/ and test_images.csv (searches a few levels down)."""
    start = Path(start)

    def looks_like_root(p: Path) -> bool:
        return (p / "train").is_dir() and ((p / "test_images.csv").exists() or (p / "test").is_dir())

    frontier = [start]
    for _ in range(max_depth + 1):
        nxt = []
        for p in frontier:
            if looks_like_root(p):
                return p
            try:
                nxt += [c for c in sorted(p.iterdir())
                        if c.is_dir() and c.name not in ("images", "labels", "ignore")]
            except (PermissionError, FileNotFoundError, NotADirectoryError):
                pass
        frontier = nxt
    raise FileNotFoundError(f"Could not find train/ and test_images.csv under {start} (searched {max_depth} levels)")


def yolo_to_xyxy(arr: np.ndarray, width: int, height: int) -> np.ndarray:
    """Normalised [cx, cy, w, h] -> pixel [x1, y1, x2, y2]."""
    if len(arr) == 0:
        return EMPTY_BOXES.copy()
    cx, cy, w, h = arr[:, 0] * width, arr[:, 1] * height, arr[:, 2] * width, arr[:, 3] * height
    out = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1)
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, width)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, height)
    return out.astype(np.float32)


def box_area_np(b: np.ndarray) -> np.ndarray:
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def _intersection_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    wh = np.clip(rb - lt, 0, None)
    return wh[..., 0] * wh[..., 1]


def box_iou_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU matrix (len(a), len(b))."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    inter = _intersection_np(a, b)
    union = box_area_np(a)[:, None] + box_area_np(b)[None, :] - inter
    return inter / np.maximum(union, 1e-9)


def box_ioa_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Intersection over area of `a` (how much of each box in `a` is covered by each box in `b`)."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), np.float32)
    return _intersection_np(a, b) / np.maximum(box_area_np(a)[:, None], 1e-9)


def size_bin_of(sqrt_area) -> np.ndarray:
    return pd.cut(np.asarray(sqrt_area), SIZE_BIN_EDGES, labels=SIZE_BIN_LABELS, right=False)


def input_scale(h, w, min_size, max_size):
    """Resize factor used by torchvision detectors (shorter side -> min_size, longer side <= max_size)."""
    if isinstance(min_size, (list, tuple)):
        min_size = min_size[-1]
    return np.minimum(min_size / np.minimum(h, w), max_size / np.maximum(h, w))


# =============================================================================
# 2. Data index
# =============================================================================
def _read_txt(path: Path, ncols: int) -> np.ndarray:
    try:
        txt = path.read_text()
    except FileNotFoundError:
        return np.zeros((0, ncols), np.float32)
    vals = np.array(txt.split(), dtype=np.float32)
    return vals.reshape(-1, ncols) if vals.size else np.zeros((0, ncols), np.float32)


def _image_size(path: Path):
    from PIL import Image
    with Image.open(path) as im:
        return im.size  # (width, height)


def _index_clip(split_dir: Path, split: str, clip: str) -> list:
    img_dir, lab_dir, ign_dir = split_dir / "images" / clip, split_dir / "labels" / clip, split_dir / "ignore" / clip
    frames = sorted(f for f in os.listdir(img_dir) if f.lower().endswith((".jpg", ".jpeg", ".png")))
    if not frames:
        return []
    width, height = _image_size(img_dir / frames[0])
    size_ok = _image_size(img_dir / frames[-1]) == (width, height)
    ign_files = set(os.listdir(ign_dir)) if ign_dir.is_dir() else set()

    rows = []
    for i, fname in enumerate(frames):
        stem = os.path.splitext(fname)[0]
        lab = _read_txt(lab_dir / f"{stem}.txt", 5)
        boxes = yolo_to_xyxy(lab[:, 1:], width, height)
        labels = lab[:, 0].astype(np.int64)
        keep = ((boxes[:, 2] - boxes[:, 0]) >= 1) & ((boxes[:, 3] - boxes[:, 1]) >= 1)
        ign = (yolo_to_xyxy(_read_txt(ign_dir / f"{stem}.txt", 4), width, height)
               if f"{stem}.txt" in ign_files else EMPTY_BOXES.copy())
        rows.append(dict(
            image_id=f"{clip}_{stem}", split=split, clip=clip,
            frame=int(stem) if stem.isdigit() else i,
            path=str(img_dir / fname), width=width, height=height, size_consistent=size_ok,
            boxes=boxes[keep], labels=labels[keep], ignore=ign,
            n_obj=int(keep.sum()), n_degenerate=int((~keep).sum()), n_ignore=len(ign),
        ))
    return rows


def build_index(root, split: str, cache_dir=None, num_workers: int = 8, force: bool = False) -> pd.DataFrame:
    """One row per frame of train/ or val/: path, size, boxes (px, xyxy), labels (0..7), ignored regions.

    Image size is read once per clip (first frame) and checked on the last frame (`size_consistent`).
    """
    root = Path(root)
    # the index stores absolute image paths, so the cache file is tied to the data location
    import hashlib
    tag = hashlib.md5(str(root.resolve()).encode("utf-8")).hexdigest()[:8]
    cache = Path(cache_dir) / f"index_{split}_{tag}.pkl" if cache_dir else None
    if cache is not None and cache.exists() and not force:
        return pd.read_pickle(cache)

    split_dir = root / split
    clips = sorted(d for d in os.listdir(split_dir / "images") if (split_dir / "images" / d).is_dir())
    with ThreadPoolExecutor(num_workers) as ex:
        parts = list(ex.map(lambda c: _index_clip(split_dir, split, c), clips))
    df = pd.DataFrame([r for part in parts for r in part])
    df = df.sort_values(["clip", "frame"]).reset_index(drop=True)
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        df.to_pickle(cache)
    return df


def build_test_index(root) -> pd.DataFrame:
    """Rows of test_images.csv with absolute paths and empty boxes (same columns as the train index)."""
    root = Path(root)
    df = pd.read_csv(root / "test_images.csv")
    df["path"] = [str(root / f) for f in df["file_name"]]
    df["split"] = "test"
    n = len(df)
    df["boxes"] = [EMPTY_BOXES] * n
    df["labels"] = [EMPTY_LABELS] * n
    df["ignore"] = [EMPTY_BOXES] * n
    df["n_obj"] = 0
    return df.reset_index(drop=True)


# =============================================================================
# 3. Splits and subsampling
# =============================================================================
def subsample_frames(df: pd.DataFrame, step: int, offset: int = 0) -> pd.DataFrame:
    """Keep every `step`-th frame of every clip (by position inside the clip)."""
    if step <= 1:
        return df.reset_index(drop=True)
    df = df.sort_values(["clip", "frame"])
    rank = df.groupby("clip").cumcount()
    return df[((rank - offset) % step) == 0].reset_index(drop=True)


def check_clip_overlap(df_a: pd.DataFrame, df_b: pd.DataFrame) -> set:
    """Clips present in both tables (must be empty for a clean split)."""
    return set(df_a["clip"]) & set(df_b["clip"])


def make_clip_folds(df: pd.DataFrame, n_splits: int = 5, seed: int = 42) -> pd.DataFrame:
    """Group K-fold by clip: every clip goes to exactly one fold, folds have similar numbers of frames."""
    sizes = df.groupby("clip").size()
    clips = list(sizes.index)
    random.Random(seed).shuffle(clips)
    clips.sort(key=lambda c: -sizes[c])  # greedy: biggest clips first, each into the smallest fold
    load, assign = [0] * n_splits, {}
    for c in clips:
        k = int(np.argmin(load))
        assign[c] = k
        load[k] += sizes[c]
    out = df.copy()
    out["fold"] = out["clip"].map(assign)
    return out


# =============================================================================
# 4. EDA
# =============================================================================
def boxes_table(df: pd.DataFrame) -> pd.DataFrame:
    """One row per ground-truth object (long format) for EDA."""
    n = df["n_obj"].to_numpy()
    if n.sum() == 0:
        return pd.DataFrame()
    rep = np.repeat(np.arange(len(df)), n)
    b = np.concatenate([x for x in df["boxes"] if len(x)])
    lab = np.concatenate([x for x in df["labels"] if len(x)])
    w, h = b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]
    img_w, img_h = df["width"].to_numpy()[rep], df["height"].to_numpy()[rep]
    sq = np.sqrt(w * h)
    out = pd.DataFrame({
        "row": rep,
        "split": df["split"].to_numpy()[rep],
        "clip": df["clip"].to_numpy()[rep],
        "frame": df["frame"].to_numpy()[rep],
        "cls": lab,
        "class_name": np.array(CLASS_NAMES)[lab],
        "x1": b[:, 0], "y1": b[:, 1], "w": w, "h": h,
        "sqrt_area": sq,
        "aspect_hw": h / np.maximum(w, 1e-6),
        "img_w": img_w, "img_h": img_h,
        "rel_size": sq / np.sqrt(img_w * img_h),
    })
    out["size_bin"] = size_bin_of(sq)
    for col in ("split", "clip", "class_name"):
        out[col] = out[col].astype("category")
    return out


def dataset_overview(dfs: dict) -> pd.DataFrame:
    """High-level numbers per split. `dfs` = {"train": df, "val": df, ...}."""
    rows = []
    for name, df in dfs.items():
        res = df.groupby(["width", "height"]).size()
        rows.append({
            "split": name, "clips": df["clip"].nunique(), "frames": len(df),
            "objects": int(df["n_obj"].sum()),
            "obj/frame mean": round(df["n_obj"].mean(), 1),
            "obj/frame max": int(df["n_obj"].max()) if len(df) else 0,
            "frames with ignore %": round(100 * (df.get("n_ignore", pd.Series(0, index=df.index)) > 0).mean(), 1),
            "resolutions": ", ".join(f"{w}x{h} ({n})" for (w, h), n in res.items()),
        })
    return pd.DataFrame(rows).set_index("split")


def class_distribution(objs: pd.DataFrame) -> pd.DataFrame:
    """Object counts and shares per class and split."""
    tab = objs.groupby(["class_name", "split"], observed=True).size().unstack(fill_value=0)
    tab = tab.reindex(CLASS_NAMES).fillna(0).astype(int)
    tab.columns = tab.columns.astype(str)
    for c in list(tab.columns):
        tab[f"{c} %"] = (100 * tab[c] / max(tab[c].sum(), 1)).round(2)
    return tab


def size_table(objs: pd.DataFrame) -> pd.DataFrame:
    """Share (%) of each size bin per class, plus median box side in px."""
    tab = pd.crosstab(objs["class_name"], objs["size_bin"], normalize="index").mul(100).round(1)
    tab = tab.reindex(CLASS_NAMES)
    tab.columns = tab.columns.astype(str)
    tab["median sqrt(area) px"] = objs.groupby("class_name", observed=True)["sqrt_area"].median().reindex(CLASS_NAMES).round(1)
    tab["median aspect h/w"] = objs.groupby("class_name", observed=True)["aspect_hw"].median().reindex(CLASS_NAMES).round(2)
    return tab


def per_clip_stats(df: pd.DataFrame, objs: pd.DataFrame) -> pd.DataFrame:
    """How the data varies across sequences: density, object size, class mix, ignored regions."""
    g = df.groupby("clip")
    out = pd.DataFrame({
        "split": g["split"].first(),
        "frames": g.size(),
        "resolution": g["width"].first().astype(str) + "x" + g["height"].first().astype(str),
        "obj/frame mean": g["n_obj"].mean().round(1),
        "obj/frame max": g["n_obj"].max(),
        "frames with ignore %": (g["n_ignore"].apply(lambda s: (s > 0).mean()) * 100).round(1),
    })
    og = objs.groupby("clip", observed=True)
    out["median sqrt(area)"] = og["sqrt_area"].median().round(1)
    out["tiny %"] = (og["sqrt_area"].apply(lambda s: (s < 16).mean()) * 100).round(1)
    out["main class"] = og["class_name"].agg(lambda s: s.value_counts().index[0])
    out["main class %"] = (og["class_name"].agg(lambda s: s.value_counts(normalize=True).iloc[0]) * 100).round(1)
    return out.sort_values(["split", "obj/frame mean"], ascending=[True, False])


def ignore_stats(df: pd.DataFrame, objs: pd.DataFrame | None = None) -> pd.Series:
    """Ignored regions: how often, how big, and how many GT boxes touch them."""
    has = df["n_ignore"] > 0
    area_frac = [min(box_area_np(ig).sum() / (w * h), 1.0) if len(ig) else 0.0
                 for ig, w, h in zip(df["ignore"], df["width"], df["height"])]
    gt_touch = sum(int((box_ioa_np(b, ig).max(1) > 0.5).sum()) if len(b) and len(ig) else 0
                   for b, ig in zip(df["boxes"], df["ignore"]))
    return pd.Series({
        "frames with ignored regions %": round(100 * has.mean(), 1),
        "regions per frame (when present)": round(df.loc[has, "n_ignore"].mean(), 1) if has.any() else 0,
        "mean image area covered % (approx.)": round(100 * float(np.mean(area_frac)), 2),
        "GT boxes >50% inside ignore": gt_touch,
        "GT boxes total": int(df["n_obj"].sum()),
    })


def temporal_redundancy(df: pd.DataFrame, lags=(1, 2, 5, 10, 20, 40), n_clips: int = 8,
                        pairs_per_clip: int = 5, thumb_width: int = 160, seed: int = 0) -> pd.DataFrame:
    """How similar frame t and frame t+lag are. Used to justify the frame subsampling step.

    pixel_diff  : mean |gray_t - gray_{t+lag}| on small thumbnails (0..255)
    box_persist : share of GT boxes at t that still have a same-class box with IoU>0.5 at t+lag
    """
    import cv2
    rng = random.Random(seed)
    clips = sorted(df["clip"].unique())
    clips = rng.sample(clips, min(n_clips, len(clips)))
    cache = {}

    def thumb(path):
        if path not in cache:
            im = read_image_cv2(path, cv2.IMREAD_GRAYSCALE)
            h, w = im.shape
            cache[path] = cv2.resize(im, (thumb_width, int(h * thumb_width / w))).astype(np.float32)
        return cache[path]

    rows = []
    for clip in clips:
        sub = df[df["clip"] == clip].sort_values("frame").reset_index(drop=True)
        for lag in lags:
            if len(sub) <= lag:
                continue
            for t in rng.sample(range(len(sub) - lag), min(pairs_per_clip, len(sub) - lag)):
                a, b = sub.iloc[t], sub.iloc[t + lag]
                diff = float(np.abs(thumb(a["path"]) - thumb(b["path"])).mean())
                if len(a["boxes"]) and len(b["boxes"]):
                    iou = box_iou_np(a["boxes"], b["boxes"])
                    iou = iou * (a["labels"][:, None] == b["labels"][None, :])
                    persist = float((iou.max(1) > 0.5).mean())
                else:
                    persist = np.nan
                rows.append({"clip": clip, "lag": lag, "pixel_diff": diff, "box_persist": persist})
    res = pd.DataFrame(rows)
    return res.groupby("lag")[["pixel_diff", "box_persist"]].mean().round(3)


def attach_coco_attributes(df: pd.DataFrame, coco_json, iou_thr: float = 0.9) -> pd.DataFrame:
    """Add per-box `occlusion` and `truncation` arrays from annotations_coco.json (-1 if not matched).

    Boxes are matched to the YOLO-based index by class and IoU, so the order of `boxes` is kept.
    """
    with open(coco_json) as f:
        coco = json.load(f)
    key_of = _coco_image_keys(coco, df)
    by_img = {}
    for a in coco["annotations"]:
        if a.get("iscrowd", 0) == 0:
            by_img.setdefault(a["image_id"], []).append(a)
    occ, trunc = [], []
    for r in df.itertuples():
        anns = by_img.get(key_of.get((r.clip, r.frame)), [])
        o = np.full(len(r.boxes), -1, np.int64)
        t = np.full(len(r.boxes), -1, np.int64)
        if anns and len(r.boxes):
            cb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                           for a in anns], np.float32)
            cl = np.array([a["category_id"] - 1 for a in anns])
            iou = box_iou_np(r.boxes, cb) * (r.labels[:, None] == cl[None, :])
            best = iou.argmax(1)
            ok = iou.max(1) >= iou_thr
            o[ok] = [anns[j].get("occlusion", -1) for j in best[ok]]
            t[ok] = [anns[j].get("truncation", -1) for j in best[ok]]
        occ.append(o)
        trunc.append(t)
    out = df.copy()
    out["occlusion"], out["truncation"] = occ, trunc
    return out


def _coco_image_keys(coco: dict, df: pd.DataFrame) -> dict:
    """Map (clip, frame) -> COCO image id. Tries `video`/`frame_id` first, then the file name."""
    wanted = set(zip(df["clip"], df["frame"]))
    a = {(im.get("video"), int(im.get("frame_id", -1))): im["id"] for im in coco["images"]}
    b = {}
    for im in coco["images"]:
        p = Path(im.get("file_name", ""))
        if p.stem.isdigit():
            b[(p.parent.name, int(p.stem))] = im["id"]
    hits_a, hits_b = len(wanted & set(a)), len(wanted & set(b))
    return a if hits_a >= hits_b else b


# --- EDA plots ---------------------------------------------------------------
def plot_class_distribution(objs: pd.DataFrame):
    import matplotlib.pyplot as plt
    tab = objs.groupby(["class_name", "split"], observed=True).size().unstack(fill_value=0).reindex(CLASS_NAMES)
    fig, ax = plt.subplots(figsize=(9, 3.6))
    tab.plot.bar(ax=ax, logy=True, width=0.8)
    ax.set_ylabel("objects (log scale)")
    ax.set_xlabel("")
    ax.set_title("Class distribution")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout()
    return fig


def plot_size_distribution(objs: pd.DataFrame, min_size=800, max_size=1333):
    """sqrt(area) per class in original pixels and after the model's resize."""
    import matplotlib.pyplot as plt
    scaled = objs["sqrt_area"] * input_scale(objs["img_h"].to_numpy(), objs["img_w"].to_numpy(), min_size, max_size)
    bins = np.logspace(0, np.log10(max(objs["sqrt_area"].max(), 10) + 1), 50)
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.8))
    for name in CLASS_NAMES:
        s = objs.loc[objs["class_name"] == name, "sqrt_area"]
        if len(s):
            # share of the class per bin (density=True would inflate the narrow left bins of a log axis)
            axes[0].hist(s, bins=bins, histtype="step", label=name, weights=np.full(len(s), 100 / len(s)))
    axes[0].set_xscale("log")
    axes[0].set_xlabel("sqrt(box area), px (original image)")
    axes[0].set_ylabel("% of the class")
    axes[0].set_title("Object size per class")
    axes[0].legend(fontsize=7, ncol=2)
    axes[1].hist(objs["sqrt_area"], bins=bins, alpha=0.6, label="original")
    axes[1].hist(scaled, bins=bins, alpha=0.6, label=f"after resize (min_size={min_size})")
    for edge in (16, 32, 96):
        axes[1].axvline(edge, color="k", lw=0.8, ls="--")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("sqrt(box area), px")
    axes[1].set_title("All objects; dashed = 16 / 32 / 96 px")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    return fig


def plot_objects_per_frame(df: pd.DataFrame):
    """Histogram of objects per frame and per-clip box plot (scene density across sequences)."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw={"width_ratios": [1, 2.2]})
    for split, sub in df.groupby("split"):
        axes[0].hist(sub["n_obj"], bins=40, alpha=0.6, label=split, density=True)
    axes[0].set_xlabel("objects per frame")
    axes[0].set_title("Scene density")
    axes[0].legend()
    order = df.groupby("clip")["n_obj"].median().sort_values().index
    data = [df.loc[df["clip"] == c, "n_obj"].to_numpy() for c in order]
    colors = ["tab:orange" if df.loc[df["clip"] == c, "split"].iloc[0] == "val" else "tab:blue" for c in order]
    bp = axes[1].boxplot(data, showfliers=False, patch_artist=True)
    for patch, col in zip(bp["boxes"], colors):
        patch.set_facecolor(col)
        patch.set_alpha(0.5)
    axes[1].set_xticks([])
    axes[1].set_xlabel("clips sorted by median density (orange = val)")
    axes[1].set_ylabel("objects per frame")
    axes[1].set_title("Density per clip")
    fig.tight_layout()
    return fig


def plot_aspect_ratios(objs: pd.DataFrame):
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 3.4))
    bins = np.logspace(np.log10(0.2), np.log10(5), 40)
    for name in CLASS_NAMES:
        s = objs.loc[objs["class_name"] == name, "aspect_hw"]
        if len(s):
            ax.hist(s.clip(0.2, 5), bins=bins, histtype="step", label=name, weights=np.full(len(s), 100 / len(s)))
    ax.set_xscale("log")
    ax.set_xlabel("aspect ratio h/w (log scale)")
    ax.set_ylabel("% of the class")
    for r in (0.5, 1.0, 2.0):  # default anchor aspect ratios
        ax.axvline(r, color="k", lw=0.8, ls="--")
    ax.minorticks_off()
    ax.set_xticks([0.25, 0.5, 1, 2, 4])
    ax.set_xticklabels(["0.25", "0.5", "1", "2", "4"])
    ax.set_title("Box aspect ratios (anchor design)")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    return fig


# =============================================================================
# 5. Visualisation and conversion checks
# =============================================================================
CLASS_COLORS = ["#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4", "#42d4f4", "#f032e6", "#bfef45"]


def read_image_cv2(path: str, flags=None) -> np.ndarray:
    """cv2.imread that also works with non-ASCII paths on Windows (cv2.imread silently fails there)."""
    import cv2
    flags = cv2.IMREAD_COLOR if flags is None else flags
    img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), flags)
    if img is None:
        raise FileNotFoundError(f"Cannot read image {path}")
    return img


def load_image(path: str) -> np.ndarray:
    """RGB uint8 HxWx3."""
    import cv2
    return cv2.cvtColor(read_image_cv2(path), cv2.COLOR_BGR2RGB)


def draw_boxes(ax, boxes, labels=None, scores=None, color=None, lw=1.0, ls="-", text=False):
    import matplotlib.patches as patches
    for i, b in enumerate(boxes):
        c = color or (CLASS_COLORS[int(labels[i])] if labels is not None else "yellow")
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=c, lw=lw, ls=ls))
        if text and labels is not None:
            s = CLASS_NAMES[int(labels[i])] + (f" {scores[i]:.2f}" if scores is not None else "")
            ax.text(b[0], b[1] - 2, s, color=c, fontsize=6)


def draw_ignore(ax, ignore):
    import matplotlib.patches as patches
    for b in ignore:
        ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1],
                                       fill=True, fc=(0.5, 0.5, 0.5, 0.35), ec="white", lw=0.8, hatch="//"))


def show_gt_samples(df: pd.DataFrame, n: int = 4, seed: int = 0, cols: int = 2, figsize_per=(7, 4.2),
                    rows_idx=None):
    """Ground-truth boxes (class colours) and ignored regions (grey, hatched) on sample frames."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if rows_idx is None:
        rng = np.random.RandomState(seed)
        clips = rng.choice(df["clip"].unique(), size=min(n, df["clip"].nunique()), replace=False)
        rows_idx = [rng.choice(df.index[df["clip"] == c]) for c in clips]
    rws = int(math.ceil(len(rows_idx) / cols))
    fig, axes = plt.subplots(rws, cols, figsize=(figsize_per[0] * cols, figsize_per[1] * rws))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, rows_idx):
        r = df.loc[i]
        ax.imshow(load_image(r["path"]))
        draw_ignore(ax, r["ignore"])
        draw_boxes(ax, r["boxes"], r["labels"], lw=0.8)
        ax.set_title(f"{r['clip']} frame {r['frame']}: {r['n_obj']} objects, {len(r['ignore'])} ignored", fontsize=9)
        ax.axis("off")
    for ax in axes[len(rows_idx):]:
        ax.axis("off")
    handles = [Line2D([0], [0], color=c, lw=2, label=n) for c, n in zip(CLASS_COLORS, CLASS_NAMES)]
    fig.legend(handles=handles, loc="lower center", ncol=8, fontsize=8)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    return fig


def crosscheck_with_coco(df: pd.DataFrame, coco_json, n_frames: int = 100, seed: int = 0) -> pd.DataFrame:
    """Compare our YOLO-based boxes with annotations_coco.json on random frames (conversion check)."""
    with open(coco_json) as f:
        coco = json.load(f)
    key_of = _coco_image_keys(coco, df)
    anns_by_img = {}
    for a in coco["annotations"]:
        anns_by_img.setdefault(a["image_id"], []).append(a)
    sample = df.sample(min(n_frames, len(df)), random_state=seed)
    rows = []
    for r in sample.itertuples():
        anns = anns_by_img.get(key_of.get((r.clip, r.frame)), [])
        obj = [a for a in anns if a.get("iscrowd", 0) == 0]
        crowd = [a for a in anns if a.get("iscrowd", 0) == 1]
        cb = np.array([[a["bbox"][0], a["bbox"][1], a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                       for a in obj], np.float32).reshape(-1, 4)
        cl = np.array([a["category_id"] - 1 for a in obj], np.int64)
        iou = box_iou_np(r.boxes, cb) * (r.labels[:, None] == cl[None, :]) if len(cb) else np.zeros((len(r.boxes), 0))
        rows.append({
            "image_id": r.image_id, "yolo_boxes": len(r.boxes), "coco_boxes": len(cb),
            "yolo_ignore": len(r.ignore), "coco_crowd": len(crowd),
            "mean_best_iou": float(iou.max(1).mean()) if iou.size else np.nan,
        })
    return pd.DataFrame(rows)


# =============================================================================
# 6. Dataset / DataLoader
# =============================================================================
class DroneDetDataset(Dataset):
    """Frames + targets. Returns (uint8 tensor CxHxW, {"boxes": px xyxy, "labels": 0..7, "idx": i}).

    Training-time options:
      mask_ignore : paint ignored regions with the mean colour and drop GT boxes mostly inside them,
                    so the model is neither rewarded nor punished for these areas.
      hflip_p     : horizontal flip probability (aerial footage has no preferred left/right).
      color_jitter: random brightness/contrast strength (0 = off).
    Resizing and normalisation happen inside each model (same rule for all models).
    """

    def __init__(self, df: pd.DataFrame, train: bool = False, mask_ignore: bool = True, drop_ioa: float = 0.5,
                 hflip_p: float = 0.5, color_jitter: float = 0.0):
        self.records = df[["path", "boxes", "labels", "ignore"]].to_dict("records")
        self.train = train
        self.mask_ignore = mask_ignore
        self.drop_ioa = drop_ioa
        self.hflip_p = hflip_p
        self.color_jitter = color_jitter

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        r = self.records[i]
        img = load_image(r["path"])
        h, w = img.shape[:2]
        boxes, labels, ign = r["boxes"].copy(), r["labels"].copy(), r["ignore"]

        if self.mask_ignore and len(ign):
            for x1, y1, x2, y2 in np.round(ign).astype(int):
                img[max(y1, 0):max(y2, 0), max(x1, 0):max(x2, 0)] = FILL_VALUE
            if len(boxes):
                keep = box_ioa_np(boxes, ign).max(1) < self.drop_ioa
                boxes, labels = boxes[keep], labels[keep]

        if self.train:
            if random.random() < self.hflip_p:
                img = img[:, ::-1]
                boxes = boxes.copy()
                boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
            if self.color_jitter > 0:
                a = 1 + random.uniform(-self.color_jitter, self.color_jitter)
                b = 255 * random.uniform(-self.color_jitter, self.color_jitter) / 2
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)

        img_t = torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1)
        target = {"boxes": torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4),
                  "labels": torch.as_tensor(labels, dtype=torch.int64), "idx": i}
        return img_t, target


def collate_fn(batch):
    return tuple(zip(*batch))


def make_loader(ds: Dataset, batch_size: int, shuffle: bool, num_workers="auto", seed: int = 42) -> DataLoader:
    """num_workers="auto" -> default_num_workers(). On Windows the Dataset must come from this module
    (it does), because worker processes import it."""
    if num_workers in (None, "auto"):
        num_workers = default_num_workers()
    g = torch.Generator()
    g.manual_seed(seed)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=int(num_workers),
                      collate_fn=collate_fn, pin_memory=torch.cuda.is_available(), generator=g,
                      persistent_workers=int(num_workers) > 0)


def to_device(images, targets, device):
    images = [im.to(device, non_blocking=True).float().div_(255.0) for im in images]
    if targets is not None:
        targets = [{"boxes": t["boxes"].to(device), "labels": t["labels"].to(device)} for t in targets]
    return images, targets


def show_augmented(ds: DroneDetDataset, idx: int, n: int = 3, seed: int = 0):
    """Show the same training sample several times (masking + random augmentation)."""
    import matplotlib.pyplot as plt
    random.seed(seed)
    fig, axes = plt.subplots(1, n, figsize=(6 * n, 3.6))
    for ax in np.atleast_1d(axes):
        img, t = ds[idx]
        ax.imshow(img.permute(1, 2, 0).numpy())
        draw_boxes(ax, t["boxes"].numpy(), t["labels"].numpy(), lw=0.8)
        ax.axis("off")
    fig.suptitle("Training view: ignored regions painted, random flip / jitter", fontsize=10)
    fig.tight_layout()
    return fig


# =============================================================================
# 7. Models (common interface)
# =============================================================================
# Every wrapper takes a list of float images (0..1, CxHxW, original size) and
#   - in train mode with targets returns a dict of losses ("_total" = loss to optimise, if present),
#   - in eval mode returns [{"boxes": px xyxy in original image, "scores", "labels": 0..7}, ...].

# effective batch = batch_size * grad_accum (e.g. batch_size=2, grad_accum=2 on a GPU with less memory)
COMMON_CFG = dict(
    pretrained=True, seed=42,
    # same input resolution rule for all models (shorter side -> min_size, longer side <= max_size)
    min_size=800, max_size=1333,
    # data
    train_step=10, val_step=5, mask_ignore=True, drop_ioa=0.5, hflip_p=0.5, color_jitter=0.0,
    # optimisation
    epochs=8, batch_size=4, grad_accum=1, eval_batch_size=4, num_workers="auto",
    optimizer="sgd", lr=0.005, backbone_lr_mult=1.0, weight_decay=1e-4, momentum=0.9,
    scheduler="cosine", warmup_iters=300, warmup_factor=0.001, min_lr_ratio=0.05, milestones=(0.7, 0.9),
    grad_clip=10.0, amp=True, eval_every=1, max_minutes=None,
    trainable_backbone_layers=3, freeze_bn=True, max_det=500,
)

ARCH_DEFAULTS = {
    "faster_rcnn": dict(
        rpn_anchor_sizes=(32, 64, 128, 256, 512), rpn_aspect_ratios=(0.5, 1.0, 2.0),
        rpn_pre_nms_top_n_train=2000, rpn_pre_nms_top_n_test=1000,
        rpn_post_nms_top_n_train=2000, rpn_post_nms_top_n_test=1000,
        rpn_batch_size_per_image=256, rpn_positive_fraction=0.5,
        box_batch_size_per_image=512, box_positive_fraction=0.25,
        box_score_thresh=0.05, box_nms_thresh=0.5,
        box_loss="smooth_l1", box_loss_weight=1.0,
    ),
    "retinanet": dict(
        anchor_base_sizes=(32, 64, 128, 256, 512), anchor_scales_per_octave=3,
        anchor_aspect_ratios=(0.5, 1.0, 2.0),
        fg_iou_thresh=0.5, bg_iou_thresh=0.4,
        focal_alpha=0.25, focal_gamma=2.0, box_loss="giou",
        score_thresh=0.05, nms_thresh=0.5, topk_candidates=1000,
    ),
    "fcos": dict(
        center_sampling_radius=1.5, p2_level=False, box_loss="giou",
        score_thresh=0.05, nms_thresh=0.6, topk_candidates=1000,
    ),
    "rtdetr": dict(
        checkpoint="PekingU/rtdetr_r50vd",
        optimizer="adamw", lr=1e-4, backbone_lr_mult=0.1, weight_decay=1e-4, grad_clip=0.1,
        num_queries=300, num_denoising=100,
        matcher_class_cost=2.0, matcher_bbox_cost=5.0, matcher_giou_cost=2.0,
        weight_loss_vfl=1.0, weight_loss_bbox=5.0, weight_loss_giou=2.0,
        auxiliary_loss=True, freeze_bn=False, score_thresh=0.0,
    ),
}


def make_config(arch: str, **overrides) -> dict:
    """COMMON_CFG + architecture defaults + your overrides (unknown keys raise an error to catch typos)."""
    if arch not in ARCH_DEFAULTS:
        raise ValueError(f"arch must be one of {list(ARCH_DEFAULTS)}")
    cfg = copy.deepcopy(COMMON_CFG)
    cfg.update(copy.deepcopy(ARCH_DEFAULTS[arch]))
    unknown = set(overrides) - set(cfg)
    if unknown:
        raise KeyError(f"Unknown config keys: {sorted(unknown)}")
    cfg.update(overrides)
    cfg["arch"] = arch
    return cfg


def config_diff(cfg: dict, base: dict) -> dict:
    """Keys whose values differ from `base` (handy for tables of experiments)."""
    return {k: v for k, v in cfg.items() if base.get(k) != v}


def _as_tuple(v):
    return tuple(v) if isinstance(v, (list, tuple)) else v


def _new_conv(in_ch, out_ch, k, prior_bias=None):
    conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=1, padding=k // 2)
    nn.init.normal_(conv.weight, std=0.01)
    nn.init.constant_(conv.bias, -math.log((1 - prior_bias) / prior_bias) if prior_bias else 0.0)
    return conv


class TorchvisionDetector(nn.Module):
    """Wraps a torchvision detector: shifts labels (Faster R-CNN needs 0 = background) and can keep BN frozen."""

    def __init__(self, model: nn.Module, arch: str, label_offset: int, freeze_bn: bool = True):
        super().__init__()
        self.model = model
        self.arch = arch
        self.label_offset = label_offset
        self.freeze_bn = freeze_bn

    def train(self, mode: bool = True):
        super().train(mode)
        if mode and self.freeze_bn:  # batch of 4 images is too small to update BatchNorm statistics
            for m in self.model.backbone.body.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
        return self

    def forward(self, images, targets=None):
        if self.training:
            t = [{"boxes": x["boxes"], "labels": x["labels"] + self.label_offset} for x in targets]
            return self.model(images, t)
        out = self.model(images)
        return [{"boxes": o["boxes"], "scores": o["scores"], "labels": o["labels"] - self.label_offset} for o in out]


# --- Faster R-CNN ----------------------------------------------------------------
def _decode_unit_deltas(deltas: torch.Tensor, weights, clip=math.log(1000.0 / 16)) -> torch.Tensor:
    """Decode (dx, dy, dw, dh) against a unit reference box.

    IoU-type losses are invariant to scaling x and y independently, so decoding prediction and
    target against the same unit box gives exactly the same GIoU/DIoU/CIoU as against the real proposal.
    (DIoU/CIoU centre terms are only approximately invariant; GIoU is exact.)
    """
    wx, wy, ww, wh = weights
    dx, dy = deltas[:, 0] / wx, deltas[:, 1] / wy
    dw, dh = (deltas[:, 2] / ww).clamp(max=clip), (deltas[:, 3] / wh).clamp(max=clip)
    w, h = dw.exp(), dh.exp()
    return torch.stack([dx - w / 2, dy - h / 2, dx + w / 2, dy + h / 2], dim=1)


def make_fastrcnn_loss(box_loss: str = "smooth_l1", weight: float = 1.0, bbox_reg_weights=(10.0, 10.0, 5.0, 5.0)):
    """Replacement for torchvision's roi_heads.fastrcnn_loss with a choice of box regression loss."""
    from torchvision.ops import complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss
    iou_losses = {"giou": generalized_box_iou_loss, "diou": distance_box_iou_loss, "ciou": complete_box_iou_loss}
    if box_loss not in ("smooth_l1", "l1", *iou_losses):
        raise ValueError(box_loss)

    def fastrcnn_loss(class_logits, box_regression, labels, regression_targets):
        labels = torch.cat(labels, dim=0)
        regression_targets = torch.cat(regression_targets, dim=0)
        classification_loss = F.cross_entropy(class_logits, labels)
        pos = torch.where(labels > 0)[0]
        labels_pos = labels[pos]
        n = class_logits.shape[0]
        box_regression = box_regression.reshape(n, box_regression.size(-1) // 4, 4)
        pred, tgt = box_regression[pos, labels_pos], regression_targets[pos]
        if box_loss == "smooth_l1":
            loss = F.smooth_l1_loss(pred, tgt, beta=1 / 9, reduction="sum")
        elif box_loss == "l1":
            loss = F.l1_loss(pred, tgt, reduction="sum")
        else:
            loss = iou_losses[box_loss](_decode_unit_deltas(pred.float(), bbox_reg_weights),
                                        _decode_unit_deltas(tgt.float(), bbox_reg_weights), reduction="sum")
        return classification_loss, weight * loss / labels.numel()

    return fastrcnn_loss


def build_faster_rcnn(cfg: dict) -> TorchvisionDetector:
    import torchvision.models.detection.roi_heads as roi_heads_module
    from torchvision.models.detection import FasterRCNN_ResNet50_FPN_V2_Weights, fasterrcnn_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

    model = fasterrcnn_resnet50_fpn_v2(
        weights=FasterRCNN_ResNet50_FPN_V2_Weights.COCO_V1 if cfg["pretrained"] else None,
        weights_backbone=None,
        trainable_backbone_layers=cfg["trainable_backbone_layers"],
        min_size=_as_tuple(cfg["min_size"]), max_size=cfg["max_size"],
        rpn_pre_nms_top_n_train=cfg["rpn_pre_nms_top_n_train"], rpn_pre_nms_top_n_test=cfg["rpn_pre_nms_top_n_test"],
        rpn_post_nms_top_n_train=cfg["rpn_post_nms_top_n_train"], rpn_post_nms_top_n_test=cfg["rpn_post_nms_top_n_test"],
        rpn_batch_size_per_image=cfg["rpn_batch_size_per_image"], rpn_positive_fraction=cfg["rpn_positive_fraction"],
        box_batch_size_per_image=cfg["box_batch_size_per_image"], box_positive_fraction=cfg["box_positive_fraction"],
        box_score_thresh=cfg["box_score_thresh"], box_nms_thresh=cfg["box_nms_thresh"],
        box_detections_per_img=cfg["max_det"],
    )
    # RPN anchors: one size per FPN level (P2..P6), same aspect ratios on every level
    sizes = tuple(tuple(s) if isinstance(s, (list, tuple)) else (s,) for s in cfg["rpn_anchor_sizes"])
    ratios = (tuple(cfg["rpn_aspect_ratios"]),) * len(sizes)
    old_a = model.rpn.anchor_generator.num_anchors_per_location()[0]
    model.rpn.anchor_generator = AnchorGenerator(sizes, ratios)
    new_a = model.rpn.anchor_generator.num_anchors_per_location()[0]
    if new_a != old_a:  # number of anchors per location changed -> new RPN output layers
        in_ch = model.rpn.head.cls_logits.in_channels
        model.rpn.head.cls_logits = _new_conv(in_ch, new_a, 1)
        model.rpn.head.bbox_pred = _new_conv(in_ch, new_a * 4, 1)
    # new classifier for 8 classes + background
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, NUM_CLASSES + 1)
    # box regression loss of the second stage (patched at module level: affects all Faster R-CNN models
    # in this Python session, so it is reset every time a Faster R-CNN is built)
    roi_heads_module.fastrcnn_loss = make_fastrcnn_loss(cfg["box_loss"], cfg["box_loss_weight"],
                                                        model.roi_heads.box_coder.weights)
    return TorchvisionDetector(model, "faster_rcnn", label_offset=1, freeze_bn=cfg["freeze_bn"])


# --- RetinaNet ---------------------------------------------------------------
def retinanet_anchor_sizes(base_sizes, scales_per_octave: int = 3):
    """Per-level anchor sizes: base * 2^(i/n), i = 0..n-1 (torchvision RetinaNet convention)."""
    return tuple(tuple(int(b * 2 ** (i / scales_per_octave)) for i in range(scales_per_octave)) for b in base_sizes)


def _retina_cls_loss(self, targets, head_outputs, matched_idxs):
    """torchvision RetinaNet classification loss with configurable focal alpha / gamma."""
    from torchvision.ops import sigmoid_focal_loss
    losses = []
    cls_logits = head_outputs["cls_logits"]
    for t, logits, midx in zip(targets, cls_logits, matched_idxs):
        fg = midx >= 0
        num_fg = fg.sum()
        gt = torch.zeros_like(logits)
        gt[fg, t["labels"][midx[fg]]] = 1.0
        valid = midx != self.BETWEEN_THRESHOLDS
        losses.append(sigmoid_focal_loss(logits[valid], gt[valid], alpha=self.focal_alpha,
                                         gamma=self.focal_gamma, reduction="sum") / max(1, num_fg))
    return sum(losses) / len(targets)


def build_retinanet(cfg: dict) -> TorchvisionDetector:
    from torchvision.models.detection import RetinaNet_ResNet50_FPN_V2_Weights, retinanet_resnet50_fpn_v2
    from torchvision.models.detection.anchor_utils import AnchorGenerator

    model = retinanet_resnet50_fpn_v2(
        weights=RetinaNet_ResNet50_FPN_V2_Weights.COCO_V1 if cfg["pretrained"] else None,
        weights_backbone=None,
        trainable_backbone_layers=cfg["trainable_backbone_layers"],
        min_size=_as_tuple(cfg["min_size"]), max_size=cfg["max_size"],
        score_thresh=cfg["score_thresh"], nms_thresh=cfg["nms_thresh"],
        detections_per_img=cfg["max_det"], topk_candidates=cfg["topk_candidates"],
        fg_iou_thresh=cfg["fg_iou_thresh"], bg_iou_thresh=cfg["bg_iou_thresh"],
    )
    sizes = retinanet_anchor_sizes(cfg["anchor_base_sizes"], cfg["anchor_scales_per_octave"])
    model.anchor_generator = AnchorGenerator(sizes, (tuple(cfg["anchor_aspect_ratios"]),) * len(sizes))
    num_a = model.anchor_generator.num_anchors_per_location()[0]

    cls_head, reg_head = model.head.classification_head, model.head.regression_head
    in_ch = cls_head.cls_logits.in_channels
    cls_head.cls_logits = _new_conv(in_ch, num_a * NUM_CLASSES, 3, prior_bias=0.01)
    cls_head.num_classes, cls_head.num_anchors = NUM_CLASSES, num_a
    if reg_head.bbox_reg.out_channels != num_a * 4:
        reg_head.bbox_reg = _new_conv(in_ch, num_a * 4, 3)
    reg_head._loss_type = cfg["box_loss"]  # "l1", "smooth_l1", "giou", "diou", "ciou"
    cls_head.focal_alpha, cls_head.focal_gamma = cfg["focal_alpha"], cfg["focal_gamma"]
    cls_head.compute_loss = types.MethodType(_retina_cls_loss, cls_head)
    return TorchvisionDetector(model, "retinanet", label_offset=0, freeze_bn=cfg["freeze_bn"])


# --- FCOS --------------------------------------------------------------------
def _fcos_head_loss(self, targets, head_outputs, anchors, matched_idxs):
    """torchvision FCOSHead.compute_loss with a configurable box loss (self.box_loss_type)."""
    from torchvision.ops import (complete_box_iou_loss, distance_box_iou_loss, generalized_box_iou_loss,
                                 sigmoid_focal_loss)
    box_loss_fn = {"giou": generalized_box_iou_loss, "diou": distance_box_iou_loss,
                   "ciou": complete_box_iou_loss}[self.box_loss_type]
    cls_logits = head_outputs["cls_logits"]
    bbox_regression = head_outputs["bbox_regression"]
    bbox_ctrness = head_outputs["bbox_ctrness"]

    all_cls, all_box = [], []
    for t, midx in zip(targets, matched_idxs):
        if len(t["labels"]) == 0:
            cls_t = t["labels"].new_zeros((len(midx),))
            box_t = t["boxes"].new_zeros((len(midx), 4))
        else:
            cls_t = t["labels"][midx.clip(min=0)]
            box_t = t["boxes"][midx.clip(min=0)]
        cls_t[midx < 0] = -1
        all_cls.append(cls_t)
        all_box.append(box_t)
    all_box, all_cls, anchors = torch.stack(all_box), torch.stack(all_cls), torch.stack(anchors)

    fg = all_cls >= 0
    num_fg = max(1, int(fg.sum().item()))
    gt_cls = torch.zeros_like(cls_logits)
    gt_cls[fg, all_cls[fg]] = 1.0
    loss_cls = sigmoid_focal_loss(cls_logits, gt_cls, reduction="sum")

    pred_boxes = self.box_coder.decode(bbox_regression, anchors).float()
    loss_box = box_loss_fn(pred_boxes[fg], all_box[fg].float(), reduction="sum")

    reg_t = self.box_coder.encode(anchors, all_box)
    if len(reg_t) == 0:
        ctr_t = reg_t.new_zeros(reg_t.size()[:-1])
    else:
        lr_, tb_ = reg_t[:, :, [0, 2]], reg_t[:, :, [1, 3]]
        ctr_t = torch.sqrt((lr_.min(dim=-1)[0] / lr_.max(dim=-1)[0]) * (tb_.min(dim=-1)[0] / tb_.max(dim=-1)[0]))
    loss_ctr = F.binary_cross_entropy_with_logits(bbox_ctrness.squeeze(dim=2)[fg], ctr_t[fg], reduction="sum")
    return {"classification": loss_cls / num_fg, "bbox_regression": loss_box / num_fg,
            "bbox_ctrness": loss_ctr / num_fg}


def _shift_fpn_keys(state_dict: dict, shift: int) -> dict:
    """Renumber FPN blocks (needed when an extra, higher-resolution level P2 is added in front)."""
    pat = re.compile(r"^(backbone\.fpn\.(?:inner_blocks|layer_blocks))\.(\d+)\.")
    out = {}
    for k, v in state_dict.items():
        m = pat.match(k)
        if m:
            k = f"{m.group(1)}.{int(m.group(2)) + shift}." + k[m.end():]
        out[k] = v
    return out


def fcos_strides(p2_level: bool):
    return ([4] if p2_level else []) + [8, 16, 32, 64, 128]


def build_fcos(cfg: dict) -> TorchvisionDetector:
    """FCOS ResNet-50-FPN (P3-P7), optionally with an extra high-resolution level P2 (stride 4)."""
    from torchvision.models import resnet50
    from torchvision.models.detection import FCOS, FCOS_ResNet50_FPN_Weights
    from torchvision.models.detection.anchor_utils import AnchorGenerator
    from torchvision.models.detection.backbone_utils import _resnet_fpn_extractor
    from torchvision.ops.feature_pyramid_network import LastLevelP6P7
    from torchvision.ops.misc import FrozenBatchNorm2d

    p2 = cfg["p2_level"]
    body = resnet50(weights=None, norm_layer=FrozenBatchNorm2d)
    backbone = _resnet_fpn_extractor(body, cfg["trainable_backbone_layers"],
                                     returned_layers=[1, 2, 3, 4] if p2 else [2, 3, 4],
                                     extra_blocks=LastLevelP6P7(256, 256))
    strides = fcos_strides(p2)
    # in torchvision FCOS the "anchor" is a point; its size = stride and is used for scale assignment
    anchor_gen = AnchorGenerator(tuple((s,) for s in strides), ((1.0,),) * len(strides))
    model = FCOS(backbone, num_classes=91, anchor_generator=anchor_gen,
                 min_size=_as_tuple(cfg["min_size"]), max_size=cfg["max_size"],
                 center_sampling_radius=cfg["center_sampling_radius"], score_thresh=cfg["score_thresh"],
                 nms_thresh=cfg["nms_thresh"], detections_per_img=cfg["max_det"],
                 topk_candidates=cfg["topk_candidates"])
    if cfg["pretrained"]:
        sd = FCOS_ResNet50_FPN_Weights.COCO_V1.get_state_dict(progress=True)
        if p2:
            sd = _shift_fpn_keys(sd, 1)
            missing, unexpected = model.load_state_dict(sd, strict=False)
            print(f"FCOS-P2: new (randomly initialised) tensors: {len(missing)}, unused: {len(unexpected)}")
        else:
            model.load_state_dict(sd)

    cls_head = model.head.classification_head
    cls_head.cls_logits = _new_conv(cls_head.cls_logits.in_channels, NUM_CLASSES, 3, prior_bias=0.01)
    cls_head.num_classes = NUM_CLASSES
    model.head.box_loss_type = cfg["box_loss"]
    model.head.compute_loss = types.MethodType(_fcos_head_loss, model.head)
    return TorchvisionDetector(model, "fcos", label_offset=0, freeze_bn=False)


# --- RT-DETR -------------------------------------------------------------------
class RTDetrDetector(nn.Module):
    """Hugging Face RT-DETR with the same resize rule as the torchvision models.

    Images keep their aspect ratio (shorter side -> min_size, longer side <= max_size) and are padded to a
    multiple of 32. Inputs are in 0..1 without mean/std normalisation (as RTDetrImageProcessor does).
    """

    def __init__(self, hf_model, min_size=800, max_size=1333, max_det=500, score_thresh=0.0):
        super().__init__()
        self.model = hf_model
        self.arch = "rtdetr"
        self.min_size = _as_tuple(min_size)
        self.max_size = max_size
        self.max_det = max_det
        self.score_thresh = score_thresh

    def _resize_and_pad(self, images):
        if isinstance(self.min_size, tuple):
            size = random.choice(self.min_size) if self.training else self.min_size[-1]
        else:
            size = self.min_size
        resized, scales = [], []
        for im in images:
            h, w = im.shape[-2:]
            s = min(size / min(h, w), self.max_size / max(h, w))
            nh, nw = int(round(h * s)), int(round(w * s))
            resized.append(F.interpolate(im[None], size=(nh, nw), mode="bilinear", align_corners=False)[0])
            scales.append(s)
        ph = int(math.ceil(max(r.shape[-2] for r in resized) / 32) * 32)
        pw = int(math.ceil(max(r.shape[-1] for r in resized) / 32) * 32)
        batch = resized[0].new_zeros((len(resized), 3, ph, pw))
        for i, r in enumerate(resized):
            batch[i, :, :r.shape[-2], :r.shape[-1]] = r
        return batch, scales, (ph, pw)

    def forward(self, images, targets=None):
        batch, scales, (ph, pw) = self._resize_and_pad(images)
        if self.training:
            labels = []
            for t, s in zip(targets, scales):
                b = t["boxes"] * s
                cxcywh = torch.stack([(b[:, 0] + b[:, 2]) / 2 / pw, (b[:, 1] + b[:, 3]) / 2 / ph,
                                      (b[:, 2] - b[:, 0]) / pw, (b[:, 3] - b[:, 1]) / ph], dim=1)
                labels.append({"class_labels": t["labels"], "boxes": cxcywh})
            out = self.model(pixel_values=batch, labels=labels)
            losses = {"_total": out.loss}
            for k in ("loss_vfl", "loss_bbox", "loss_giou"):
                if out.loss_dict is not None and k in out.loss_dict:
                    losses[k] = out.loss_dict[k].detach()
            return losses
        out = self.model(pixel_values=batch)
        return self._postprocess(out.logits, out.pred_boxes, scales, (ph, pw), [im.shape[-2:] for im in images])

    def _postprocess(self, logits, pred_boxes, scales, pad_hw, orig_hw):
        ph, pw = pad_hw
        prob = logits.float().sigmoid()  # RT-DETR uses sigmoid (focal / VFL) scores
        b, q, c = prob.shape
        k = min(self.max_det, q * c)
        scores, idx = prob.flatten(1).topk(k, dim=1)
        labels, query = idx % c, idx // c
        boxes = pred_boxes.float().gather(1, query.unsqueeze(-1).expand(-1, -1, 4))
        cx, cy, w, h = boxes.unbind(-1)
        xyxy = torch.stack([(cx - w / 2) * pw, (cy - h / 2) * ph, (cx + w / 2) * pw, (cy + h / 2) * ph], dim=-1)
        results = []
        for i in range(b):
            bx = xyxy[i] / scales[i]
            hh, ww = orig_hw[i]
            bx[:, 0::2] = bx[:, 0::2].clamp(0, ww)
            bx[:, 1::2] = bx[:, 1::2].clamp(0, hh)
            keep = scores[i] > self.score_thresh
            results.append({"boxes": bx[keep], "scores": scores[i][keep], "labels": labels[i][keep]})
        return results


def _rtdetr_kwargs(cfg):
    return dict(
        num_labels=NUM_CLASSES,
        id2label={i: n for i, n in enumerate(CLASS_NAMES)}, label2id={n: i for i, n in enumerate(CLASS_NAMES)},
        num_queries=cfg["num_queries"], num_denoising=cfg["num_denoising"],
        matcher_class_cost=cfg["matcher_class_cost"], matcher_bbox_cost=cfg["matcher_bbox_cost"],
        matcher_giou_cost=cfg["matcher_giou_cost"], weight_loss_vfl=cfg["weight_loss_vfl"],
        weight_loss_bbox=cfg["weight_loss_bbox"], weight_loss_giou=cfg["weight_loss_giou"],
        auxiliary_loss=cfg["auxiliary_loss"],
        anchor_image_size=None,  # anchors are generated for the actual input size (not fixed 640x640)
    )


def build_rtdetr(cfg: dict, hf_config_path=None) -> RTDetrDetector:
    from transformers import RTDetrConfig, RTDetrForObjectDetection
    if hf_config_path is not None:  # rebuild exactly the architecture of a finished run
        hf = RTDetrForObjectDetection(RTDetrConfig.from_json_file(str(hf_config_path)))
    elif cfg["pretrained"]:
        hf = RTDetrForObjectDetection.from_pretrained(cfg["checkpoint"], ignore_mismatched_sizes=True,
                                                      **_rtdetr_kwargs(cfg))
    else:
        hf = RTDetrForObjectDetection(RTDetrConfig(**_rtdetr_kwargs(cfg)))
    return RTDetrDetector(hf, cfg["min_size"], cfg["max_size"], cfg["max_det"], cfg["score_thresh"])


def build_model(cfg: dict, hf_config_path=None) -> nn.Module:
    arch = cfg["arch"]
    if arch == "faster_rcnn":
        return build_faster_rcnn(cfg)
    if arch == "retinanet":
        return build_retinanet(cfg)
    if arch == "fcos":
        return build_fcos(cfg)
    if arch == "rtdetr":
        return build_rtdetr(cfg, hf_config_path)
    raise ValueError(arch)


def count_params(model: nn.Module) -> dict:
    total = sum(p.numel() for p in model.parameters())
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"params_M": round(total / 1e6, 2), "trainable_M": round(train / 1e6, 2)}


# =============================================================================
# 8. Training
# =============================================================================
def build_optimizer(model: nn.Module, cfg: dict):
    """Param groups: backbone (lr * backbone_lr_mult) vs rest; no weight decay for biases / norm layers."""
    groups = {}
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        is_backbone = "backbone" in name and "fpn" not in name
        lr = cfg["lr"] * (cfg["backbone_lr_mult"] if is_backbone else 1.0)
        wd = 0.0 if p.ndim <= 1 else cfg["weight_decay"]
        groups.setdefault((lr, wd), []).append(p)
    param_groups = [{"params": ps, "lr": lr, "weight_decay": wd} for (lr, wd), ps in groups.items()]
    if cfg["optimizer"] == "sgd":
        return torch.optim.SGD(param_groups, lr=cfg["lr"], momentum=cfg["momentum"])
    if cfg["optimizer"] == "adamw":
        return torch.optim.AdamW(param_groups, lr=cfg["lr"])
    raise ValueError(cfg["optimizer"])


def build_scheduler(optimizer, cfg: dict, total_iters: int):
    """Linear warm-up, then cosine / multistep / constant. Stepped every iteration."""
    warm = min(cfg["warmup_iters"], max(total_iters // 2, 1))
    milestones = [int(m * total_iters) for m in cfg["milestones"]]

    def factor(it):
        if it < warm:
            return cfg["warmup_factor"] + (1 - cfg["warmup_factor"]) * it / warm
        if cfg["scheduler"] == "cosine":
            p = (it - warm) / max(1, total_iters - warm)
            return cfg["min_lr_ratio"] + (1 - cfg["min_lr_ratio"]) * 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
        if cfg["scheduler"] == "multistep":
            return 0.1 ** sum(it >= m for m in milestones)
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, factor)


def _grad_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def _torch_load(path, device="cpu", weights_only=True):
    try:
        return torch.load(path, map_location=device, weights_only=weights_only)
    except TypeError:  # very old torch without weights_only
        return torch.load(path, map_location=device)


def _atomic_save(obj, path):
    """Write to a temporary file first, so an interrupted save never leaves a broken checkpoint."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def save_checkpoint(model, path):
    _atomic_save(model.state_dict(), path)


def load_checkpoint(model, path, device="cpu"):
    model.load_state_dict(_torch_load(path, device))
    return model


def train_detector(model, train_loader, val_loader, val_df, cfg: dict, run_dir, device, name: str = ""):
    """Same loop for every model.

    Saves in run_dir: best.pt (weights with the best val mAP), history.json, and resume.pt after every
    epoch (model + optimizer + scheduler state). If resume.pt exists, training continues after the
    last finished epoch, so a lost Colab session only costs the current epoch.
    Gradient accumulation: one optimizer step per `grad_accum` batches.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    model.to(device)
    optimizer = build_optimizer(model, cfg)
    accum = max(1, int(cfg.get("grad_accum", 1)))
    n_batches = len(train_loader)
    steps_per_epoch = math.ceil(n_batches / accum)
    scheduler = build_scheduler(optimizer, cfg, cfg["epochs"] * steps_per_epoch)
    use_amp = bool(cfg["amp"]) and device.type == "cuda"
    scaler = _grad_scaler(use_amp)

    history = {"epoch": [], "minutes": [], "lr": [], "train_loss": [], "loss_parts": [], "val": [],
               "skipped_iters": [], "best_epoch": None}
    best_map, start_epoch, elapsed_before = -1.0, 1, 0.0
    resume_path = run_dir / "resume.pt"
    if resume_path.exists():
        ck = _torch_load(resume_path, device, weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        history, best_map = ck["history"], ck["best_map"]
        start_epoch, elapsed_before = ck["epoch"] + 1, ck["elapsed_min"]
        print(f"[{name}] resuming after epoch {ck['epoch']} ({elapsed_before:.1f} min already trained)")

    t0 = time.time()

    def minutes():
        return elapsed_before + (time.time() - t0) / 60

    stop = False
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if getattr(train_loader, "generator", None) is not None:
            train_loader.generator.manual_seed(cfg["seed"] + epoch)  # same shuffling after a resume
        model.train()
        sums, n_ok, skipped, has_grad = {}, 0, 0, False
        optimizer.zero_grad(set_to_none=True)
        for it, (images, targets) in enumerate(train_loader):
            images, targets = to_device(images, targets, device)
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss_dict = model(images, targets)
            total = loss_dict.pop("_total") if "_total" in loss_dict else sum(loss_dict.values())
            if torch.isfinite(total):
                scaler.scale(total / accum).backward()
                has_grad = True
                n_ok += 1
                sums["total"] = sums.get("total", 0.0) + float(total.item())
                for k, v in loss_dict.items():
                    sums[k] = sums.get(k, 0.0) + float(v.item())
            else:
                skipped += 1

            if (it + 1) % accum == 0 or it + 1 == n_batches:  # end of an accumulation group
                if has_grad:
                    if cfg["grad_clip"]:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
                    scaler.step(optimizer)
                    scaler.update()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()
                has_grad = False
            if cfg["max_minutes"] and minutes() > cfg["max_minutes"]:
                stop = True
                optimizer.zero_grad(set_to_none=True)
                break

        parts = {k: v / max(n_ok, 1) for k, v in sums.items()}
        history["epoch"].append(epoch)
        history["minutes"].append(round(minutes(), 2))
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["train_loss"].append(parts.pop("total", float("nan")))
        history["loss_parts"].append(parts)
        history["skipped_iters"].append(skipped)

        msg = f"[{name}] epoch {epoch}/{cfg['epochs']} loss {history['train_loss'][-1]:.4f}"
        last_epoch = epoch == cfg["epochs"] or stop
        if val_loader is not None and (epoch % cfg["eval_every"] == 0 or last_epoch):
            preds = predict(model, val_loader, device, amp=cfg["amp"])
            m = evaluate_coco(val_df, preds, fast=True)
            history["val"].append({"epoch": epoch, "minutes": round(minutes(), 2),
                                   **{k: m[k] for k in ("mAP", "AP50", "AP75", "AP_small", "AR")}})
            msg += f" | val mAP {m['mAP']:.4f} AP50 {m['AP50']:.4f} AP_small {m['AP_small']:.4f}"
            if m["mAP"] > best_map:
                best_map = m["mAP"]
                history["best_epoch"] = epoch
                save_checkpoint(model, run_dir / "best.pt")
        if not (run_dir / "best.pt").exists():
            save_checkpoint(model, run_dir / "best.pt")
        msg += f" | {minutes():.1f} min" + (f" | skipped {skipped} non-finite iters" if skipped else "")
        print(msg, flush=True)
        with open(run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=1)
        if stop:
            print(f"[{name}] stopped by max_minutes={cfg['max_minutes']}")
            break
        _atomic_save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                      "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                      "history": history, "best_map": best_map, "epoch": epoch,
                      "elapsed_min": minutes()}, resume_path)
    return history


# =============================================================================
# 9. Inference, speed and evaluation
# =============================================================================
@torch.no_grad()
def predict(model, loader, device, amp: bool = True) -> list:
    """Predictions for every image of the loader, in dataset order (loader must not shuffle)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    out = []
    for images, _ in loader:
        images, _ = to_device(images, None, device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            dets = model(images)
        for d in dets:
            out.append({"boxes": d["boxes"].float().cpu().numpy().astype(np.float32),
                        "scores": d["scores"].float().cpu().numpy().astype(np.float32),
                        "labels": d["labels"].cpu().numpy().astype(np.int64)})
    return out


def predict_df(model, df, cfg, device) -> list:
    ds = DroneDetDataset(df, train=False, mask_ignore=False)
    return predict(model, make_loader(ds, cfg["eval_batch_size"], False, cfg["num_workers"]), device, cfg["amp"])


@torch.no_grad()
def measure_speed(model, df, device, n_images: int = 50, warmup: int = 5, amp: bool = True) -> dict:
    """Inference time per image (batch 1), including the model's own resize and post-processing."""
    model.eval()
    ds = DroneDetDataset(df, train=False, mask_ignore=False)
    use_amp = amp and device.type == "cuda"
    times = []
    for i in range(min(n_images + warmup, len(ds))):
        img = ds[i][0].to(device).float().div_(255.0)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            model([img])
        if device.type == "cuda":
            torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t)
    sec = float(np.mean(times)) if times else float("nan")
    return {"ms_per_img": round(1000 * sec, 1), "fps": round(1 / sec, 2) if times else float("nan")}


def filter_ignored(pred: dict, ignore: np.ndarray, ioa_thr: float = 0.5, gt_boxes=None) -> dict:
    """Drop detections lying mostly (IoA > thr) inside ignored regions (they are not scored).

    If `gt_boxes` is given, a detection that matches an annotated box (IoU >= 0.5) is kept even inside
    an ignored region, so a correct detection is never thrown away.
    """
    if len(ignore) == 0 or len(pred["boxes"]) == 0:
        return pred
    drop = box_ioa_np(pred["boxes"], ignore).max(1) > ioa_thr
    if gt_boxes is not None and len(gt_boxes):
        drop &= box_iou_np(pred["boxes"], gt_boxes).max(1) < 0.5
    return {k: v[~drop] for k, v in pred.items()}


def evaluate_coco(df: pd.DataFrame, preds: list, ignore_ioa: float = 0.5, max_dets=(1, 100, 500),
                  per_class: bool = True, fast: bool = False) -> dict:
    """COCO metrics with pycocotools. Detections inside ignored regions are removed first.

    Returns mAP (IoU .50:.95), AP50, AP75, AP_small/medium/large (COCO areas), AP_tiny (<16^2 px),
    AR (at max_dets[-1]) and per-class AP.
    fast=True (used during training) evaluates only the "all" and "small" areas at max_dets[-1],
    which is ~3x quicker; the final evaluation of every run uses the full version.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    assert len(df) == len(preds), "df and preds must be aligned"
    images, anns, dets = [], [], []
    for i, (r, p) in enumerate(zip(df.itertuples(), preds), start=1):
        images.append({"id": i, "width": int(r.width), "height": int(r.height)})
        for b, lab in zip(r.boxes, r.labels):
            w, h = float(b[2] - b[0]), float(b[3] - b[1])
            anns.append({"id": len(anns) + 1, "image_id": i, "category_id": int(lab) + 1,
                         "bbox": [float(b[0]), float(b[1]), w, h], "area": w * h, "iscrowd": 0})
        p = filter_ignored(p, r.ignore, ignore_ioa, gt_boxes=r.boxes)
        if len(p["boxes"]):
            b = p["boxes"]
            dets.append(np.column_stack([np.full(len(b), i), b[:, 0], b[:, 1], b[:, 2] - b[:, 0],
                                         b[:, 3] - b[:, 1], p["scores"], p["labels"] + 1]))
    empty = {k: 0.0 for k in ("mAP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AP_tiny",
                              "AR", "AR_small")}
    if not dets:
        return {**empty, "per_class": {}}

    gt = COCO()
    gt.dataset = {"images": images, "annotations": anns,
                  "categories": [{"id": k + 1, "name": n} for k, n in enumerate(CLASS_NAMES)]}
    with contextlib.redirect_stdout(io.StringIO()):
        gt.createIndex()
        dt = gt.loadRes(np.concatenate(dets).astype(np.float64))
        ev = COCOeval(gt, dt, "bbox")
        areas = ["all", "small"] if fast else list(AREA_RANGES)
        ev.params.maxDets = [max_dets[-1]] if fast else list(max_dets)
        ev.params.areaRng = [list(AREA_RANGES[a]) for a in areas]
        ev.params.areaRngLbl = areas
        ev.evaluate()
        ev.accumulate()
    return summarize_cocoeval(ev, per_class and not fast)


def summarize_cocoeval(ev, per_class: bool = True) -> dict:
    prec = ev.eval["precision"]  # [T iou, R recall, K class, A area, M maxDets]
    rec = ev.eval["recall"]      # [T, K, A, M]
    ious = ev.params.iouThrs
    area = {lbl: i for i, lbl in enumerate(ev.params.areaRngLbl)}
    m = len(ev.params.maxDets) - 1

    def mean_valid(x):
        x = x[x > -1]
        return float(x.mean()) if x.size else float("nan")

    def ap(iou=None, a="all", k=None):
        if a not in area:
            return float("nan")
        s = prec[:, :, :, area[a], m]
        if iou is not None:
            s = s[[int(np.argmin(np.abs(ious - iou)))]]
        if k is not None:
            s = s[:, :, [k]]
        return mean_valid(s)

    res = {
        "mAP": ap(), "AP50": ap(0.5), "AP75": ap(0.75),
        "AP_small": ap(a="small"), "AP_medium": ap(a="medium"), "AP_large": ap(a="large"), "AP_tiny": ap(a="tiny"),
        "AR": mean_valid(rec[:, :, area["all"], m]),
        "AR_small": mean_valid(rec[:, :, area["small"], m]) if "small" in area else float("nan"),
    }
    if per_class:
        res["per_class"] = {CLASS_NAMES[cat - 1]: {"AP": ap(k=k), "AP50": ap(0.5, k=k)}
                            for k, cat in enumerate(ev.params.catIds)}
    return res


def ap_by_density(df: pd.DataFrame, preds: list, bins=(0, 50, 100, 200, np.inf)) -> pd.DataFrame:
    """Metrics on groups of frames with different numbers of objects (dense vs sparse scenes)."""
    rows = []
    cut = pd.cut(df["n_obj"], bins, right=False)
    codes = cut.cat.codes.to_numpy()
    for code, interval in enumerate(cut.cat.categories):
        idx = np.where(codes == code)[0]
        if len(idx) == 0:
            continue
        m = evaluate_coco(df.iloc[idx], [preds[i] for i in idx], per_class=False)
        rows.append({"objects/frame": str(interval), "frames": len(idx), "mAP": m["mAP"], "AP50": m["AP50"],
                     "AP_small": m["AP_small"], "AR": m["AR"]})
    return pd.DataFrame(rows).set_index("objects/frame")


# =============================================================================
# 10. Analysis helpers
# =============================================================================
def suggest_anchor_stats(objs: pd.DataFrame, min_size=800, max_size=1333) -> dict:
    """Object sizes and aspect ratios after the model's resize -> grounding for anchor choice."""
    s = input_scale(objs["img_h"].to_numpy(), objs["img_w"].to_numpy(), min_size, max_size)
    side = objs["sqrt_area"].to_numpy() * s
    q = [0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    return {
        "scaled sqrt(area) quantiles": dict(zip([f"q{int(x * 100)}" for x in q], np.round(np.quantile(side, q), 1))),
        "aspect h/w quantiles": dict(zip(["q10", "q50", "q90"],
                                         np.round(np.quantile(objs["aspect_hw"], [0.1, 0.5, 0.9]), 2))),
        "share below 32 px after resize %": round(100 * float((side < 32).mean()), 1),
    }


def anchor_shapes(sizes, ratios) -> np.ndarray:
    """(w, h) of all anchor shapes; torchvision convention: ratio = h / w, area = size^2."""
    out = []
    for s in sizes:
        for r in ratios:
            hr = math.sqrt(r)
            out.append((s / hr, s * hr))
    return np.array(out, np.float32)


def anchor_coverage(objs: pd.DataFrame, anchor_sizes, aspect_ratios, min_size=800, max_size=1333,
                    iou_thr: float = 0.5) -> pd.DataFrame:
    """Best shape-IoU (centres aligned) between each GT box and any anchor, after resize.

    `anchor_sizes` is a flat list of all sizes over all levels. This is an upper bound on what the
    real assignment can give (real anchors also sit on a stride grid), but it shows directly whether
    an anchor set can reach the fg IoU threshold for small objects.
    """
    s = input_scale(objs["img_h"].to_numpy(), objs["img_w"].to_numpy(), min_size, max_size)
    w, h = objs["w"].to_numpy() * s, objs["h"].to_numpy() * s
    shapes = anchor_shapes(anchor_sizes, aspect_ratios)
    inter = np.minimum(w[:, None], shapes[None, :, 0]) * np.minimum(h[:, None], shapes[None, :, 1])
    union = (w * h)[:, None] + (shapes[:, 0] * shapes[:, 1])[None, :] - inter
    best = (inter / union).max(1)
    tab = pd.DataFrame({"size_bin": objs["size_bin"].to_numpy(), "best_iou": best, "covered": best >= iou_thr})
    out = tab.groupby("size_bin", observed=True).agg(objects=("best_iou", "size"), mean_best_iou=("best_iou", "mean"),
                                                     covered_pct=("covered", "mean"))
    out.index = out.index.astype(str)
    out.loc["all"] = [len(tab), tab["best_iou"].mean(), tab["covered"].mean()]
    out["covered_pct"] = (100 * out["covered_pct"]).round(1)
    out["mean_best_iou"] = out["mean_best_iou"].round(3)
    return out


def fcos_assignment_stats(df: pd.DataFrame, strides=(8, 16, 32, 64, 128), center_radius: float = 1.5,
                          min_size=800, max_size=1333, n_images: int = 200, seed: int = 0) -> tuple:
    """Simulate torchvision FCOS label assignment on GT boxes (no model needed).

    A feature location (k*stride) is positive for a box if it is inside the box, within
    center_radius*stride of its centre, and max(l, t, r, b) fits the level's range
    (stride*4, stride*8); first level has no lower bound, last no upper bound.
    If several boxes claim one location, the smallest box wins.
    Returns (per-size table, summary dict).
    """
    sample = df.sample(min(n_images, len(df)), random_state=seed)
    recs, amb_locs, pos_locs = [], 0, 0
    for r in sample.itertuples():
        if r.n_obj == 0:
            continue
        sc = float(input_scale(r.height, r.width, min_size, max_size))
        boxes = r.boxes * sc
        hp = math.ceil(r.height * sc / 32) * 32
        wp = math.ceil(r.width * sc / 32) * 32
        areas = box_area_np(boxes)
        loc_ids, gt_ids = [], []
        offset = 0
        for li, st in enumerate(strides):
            nx, ny = math.ceil(wp / st), math.ceil(hp / st)
            lo = 0 if li == 0 else st * 4
            hi = np.inf if li == len(strides) - 1 else st * 8
            for g, (x0, y0, x1, y1) in enumerate(boxes):
                cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
                ix = np.arange(max(0, math.floor(max(x0, cx - center_radius * st) / st)),
                               min(nx - 1, math.ceil(min(x1, cx + center_radius * st) / st)) + 1)
                iy = np.arange(max(0, math.floor(max(y0, cy - center_radius * st) / st)),
                               min(ny - 1, math.ceil(min(y1, cy + center_radius * st) / st)) + 1)
                if len(ix) == 0 or len(iy) == 0:
                    continue
                px, py = np.meshgrid(ix * st, iy * st)
                px, py = px.ravel(), py.ravel()
                l, t, rr, b = px - x0, py - y0, x1 - px, y1 - py
                m = np.max([l, t, rr, b], axis=0)
                ok = ((np.min([l, t, rr, b], axis=0) > 0) & (np.abs(px - cx) < center_radius * st)
                      & (np.abs(py - cy) < center_radius * st) & (m > lo) & (m < hi))
                if ok.any():
                    sel_x, sel_y = (px[ok] // st).astype(np.int64), (py[ok] // st).astype(np.int64)
                    loc_ids.append(offset + sel_y * nx + sel_x)
                    gt_ids.append(np.full(ok.sum(), g))
            offset += nx * ny
        n_pos = np.zeros(len(boxes), np.int64)
        n_cand = np.zeros(len(boxes), np.int64)
        if loc_ids:
            loc, gid = np.concatenate(loc_ids), np.concatenate(gt_ids)
            np.add.at(n_cand, gid, 1)
            order = np.lexsort((areas[gid], loc))  # by location, then smallest box first
            loc, gid = loc[order], gid[order]
            first = np.r_[True, loc[1:] != loc[:-1]]
            np.add.at(n_pos, gid[first], 1)
            _, counts = np.unique(loc, return_counts=True)
            amb_locs += int((counts > 1).sum())
            pos_locs += len(counts)
        recs.append(pd.DataFrame({"sqrt_area": np.sqrt(box_area_np(r.boxes)), "n_pos": n_pos,
                                  "lost": n_cand - n_pos, "density": r.n_obj}))
    tab = pd.concat(recs, ignore_index=True)
    tab["size_bin"] = size_bin_of(tab["sqrt_area"])
    out = tab.groupby("size_bin", observed=True).agg(
        objects=("n_pos", "size"), mean_pos_locations=("n_pos", "mean"),
        no_positive_pct=("n_pos", lambda s: 100 * (s == 0).mean()),
        lost_to_other_box_pct=("lost", lambda s: 100 * (s > 0).mean())).round(2)
    summary = {"strides": list(strides), "center_radius": center_radius,
               "GT without any positive location %": round(100 * float((tab["n_pos"] == 0).mean()), 2),
               "ambiguous locations (claimed by >1 box) %": round(100 * amb_locs / max(pos_locs, 1), 2)}
    return out, summary


@torch.no_grad()
def rpn_proposal_recall(det, df: pd.DataFrame, device, n_images: int = 100, top_ns=(100, 300, 1000),
                        iou_thr: float = 0.5, seed: int = 0) -> pd.DataFrame:
    """What each stage of Faster R-CNN contributes.

    For every GT box: is it covered (IoU >= thr) by the top-N RPN proposals? By the final detections
    (class-agnostic and with the correct class)? Also the mean best IoU (localisation quality)
    of proposals vs. final boxes. `det` is the TorchvisionDetector of a Faster R-CNN.
    """
    m = det.model
    det.eval()
    sample = df.sample(min(n_images, len(df)), random_state=seed).reset_index(drop=True)
    ds = DroneDetDataset(sample, train=False, mask_ignore=False)
    rows = []
    for i in range(len(ds)):
        img, t = ds[i]
        gt, gl = t["boxes"].numpy(), t["labels"].numpy()
        if len(gt) == 0:
            continue
        x = [img.to(device).float().div_(255.0)]
        il, _ = m.transform(x)
        feats = m.backbone(il.tensors)
        props, _ = m.rpn(il, feats)
        dets, _ = m.roi_heads(feats, props, il.image_sizes)
        dets = m.transform.postprocess(dets, il.image_sizes, [tuple(img.shape[-2:])])[0]
        rh, rw = il.image_sizes[0]
        p = props[0].cpu().numpy().copy()
        p[:, [0, 2]] *= img.shape[-1] / rw
        p[:, [1, 3]] *= img.shape[-2] / rh
        fb, fl = dets["boxes"].cpu().numpy(), dets["labels"].cpu().numpy() - det.label_offset
        rec = {"size_bin": size_bin_of(np.sqrt(box_area_np(gt)))}
        for n in top_ns:
            iou = box_iou_np(gt, p[:n])
            rec[f"proposals@{n}"] = iou.max(1) >= iou_thr if iou.size else np.zeros(len(gt), bool)
        iou_p = box_iou_np(gt, p)
        rec["best_iou_proposal"] = iou_p.max(1) if iou_p.size else np.zeros(len(gt))
        iou_f = box_iou_np(gt, fb)
        rec["final (any class)"] = iou_f.max(1) >= iou_thr if iou_f.size else np.zeros(len(gt), bool)
        iou_fc = iou_f * (gl[:, None] == fl[None, :]) if iou_f.size else iou_f
        rec["final (correct class)"] = iou_fc.max(1) >= iou_thr if iou_fc.size else np.zeros(len(gt), bool)
        rec["best_iou_final"] = iou_f.max(1) if iou_f.size else np.zeros(len(gt))
        rows.append(pd.DataFrame(rec))
    tab = pd.concat(rows, ignore_index=True)
    out = tab.groupby("size_bin", observed=True).mean(numeric_only=True)
    out.index = out.index.astype(str)
    out.loc["all"] = tab.drop(columns="size_bin").mean()
    out.insert(0, "objects", list(tab.groupby("size_bin", observed=True).size()) + [len(tab)])
    return out.round(3)


def _classify_frame(gt_b, gt_l, pr_b, pr_s, pr_l, iou_thr):
    """Greedy COCO-style matching of one frame. Returns per-pred types, matched-GT flags, confusions."""
    order = np.argsort(-pr_s)
    iou = box_iou_np(pr_b[order], gt_b)
    used = np.zeros(len(gt_b), bool)
    types, confusions = [], []
    for j, k in enumerate(order):
        if len(gt_b) == 0:
            types.append("background")
            continue
        same = gt_l == pr_l[k]
        cand = np.where(same & ~used, iou[j], 0)
        if cand.max() >= iou_thr:
            used[int(cand.argmax())] = True
            types.append("TP")
        elif np.where(same, iou[j], 0).max() >= iou_thr:
            types.append("duplicate")
        elif np.where(~same, iou[j], 0).max() >= iou_thr:
            types.append("wrong class")
            confusions.append((int(gt_l[int(np.where(~same, iou[j], 0).argmax())]), int(pr_l[k])))
        elif np.where(same, iou[j], 0).max() >= 0.1:
            types.append("poor localisation")
        else:
            types.append("background")
    pred_types = np.empty(len(order), dtype=object)
    pred_types[order] = types
    return pred_types, used, confusions


def analyze_failures(df: pd.DataFrame, preds: list, score_thr: float = 0.3, iou_thr: float = 0.5):
    """Error breakdown at a fixed score threshold (detections in ignored regions removed first).

    Returns (gt_table, pred_table, confusion_matrix):
      gt_table   - one row per GT box: detected or missed, size, frame density (+ occlusion if attached)
      pred_table - one row per kept detection: TP / duplicate / wrong class / poor localisation / background
    """
    gts, prs, conf = [], [], np.zeros((NUM_CLASSES, NUM_CLASSES), np.int64)
    has_occ = "occlusion" in df.columns
    for i, (r, p) in enumerate(zip(df.itertuples(), preds)):
        p = filter_ignored(p, r.ignore, gt_boxes=r.boxes)
        k = p["scores"] >= score_thr
        pb, ps, pl = p["boxes"][k], p["scores"][k], p["labels"][k]
        types, used, confusions = _classify_frame(r.boxes, r.labels, pb, ps, pl, iou_thr)
        for a, b in confusions:
            conf[a, b] += 1
        sq = np.sqrt(box_area_np(r.boxes))
        g = pd.DataFrame({"row": i, "image_id": r.image_id, "clip": r.clip, "cls": r.labels,
                          "sqrt_area": sq, "detected": used, "density": r.n_obj})
        if has_occ:
            g["occlusion"] = r.occlusion
        gts.append(g)
        prs.append(pd.DataFrame({"row": i, "image_id": r.image_id, "cls": pl, "score": ps, "type": types,
                                 "sqrt_area": np.sqrt(box_area_np(pb))}))
    gt_tab = pd.concat(gts, ignore_index=True)
    gt_tab["size_bin"] = size_bin_of(gt_tab["sqrt_area"])
    gt_tab["class_name"] = np.array(CLASS_NAMES)[gt_tab["cls"]]
    pr_tab = pd.concat(prs, ignore_index=True)
    if len(pr_tab):
        pr_tab["class_name"] = np.array(CLASS_NAMES)[pr_tab["cls"].astype(int)]
    conf_df = pd.DataFrame(conf, index=[f"GT {n}" for n in CLASS_NAMES], columns=[f"pred {n}" for n in CLASS_NAMES])
    return gt_tab, pr_tab, conf_df


def failure_summary(gt_tab: pd.DataFrame, pr_tab: pd.DataFrame) -> dict:
    """Recall by size / class / density and the share of each false-positive type."""
    out = {
        "recall by size": gt_tab.groupby("size_bin", observed=True)["detected"].agg(["size", "mean"]).round(3),
        "recall by class": gt_tab.groupby("class_name")["detected"].agg(["size", "mean"]).round(3),
        "recall by density": gt_tab.groupby(pd.cut(gt_tab["density"], [0, 50, 100, 200, np.inf], right=False),
                                            observed=True)["detected"].agg(["size", "mean"]).round(3),
        "prediction types %": (100 * pr_tab["type"].value_counts(normalize=True)).round(1) if len(pr_tab) else None,
    }
    if "occlusion" in gt_tab.columns:
        out["recall by occlusion"] = gt_tab.groupby("occlusion")["detected"].agg(["size", "mean"]).round(3)
    return out


def pick_failure_frames(gt_tab: pd.DataFrame, pr_tab: pd.DataFrame, n_each: int = 1) -> dict:
    """Frames with the most missed objects, the most background FPs and the most class confusions."""
    missed = gt_tab[~gt_tab["detected"]].groupby("row").size().sort_values(ascending=False)
    fp_bg = pr_tab[pr_tab["type"] == "background"].groupby("row").size().sort_values(ascending=False)
    wrong = pr_tab[pr_tab["type"] == "wrong class"].groupby("row").size().sort_values(ascending=False)
    return {"most missed objects": list(missed.index[:n_each]),
            "most background false positives": list(fp_bg.index[:n_each]),
            "most class confusions": list(wrong.index[:n_each])}


def plot_failure_frame(df: pd.DataFrame, preds: list, row: int, score_thr: float = 0.3, iou_thr: float = 0.5,
                       title: str = "", zoom: int = 360):
    """Full frame + zoom. Green = TP, red = false positive, yellow dashed = missed GT, grey = ignored."""
    import matplotlib.pyplot as plt
    r = df.iloc[row]
    p = filter_ignored(preds[row], r["ignore"], gt_boxes=r["boxes"])
    k = p["scores"] >= score_thr
    pb, ps, pl = p["boxes"][k], p["scores"][k], p["labels"][k]
    types, used, _ = _classify_frame(r["boxes"], r["labels"], pb, ps, pl, iou_thr)
    img = load_image(r["path"])
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.7, 1]})
    focus = r["boxes"][~used] if (~used).any() else (pb[types != "TP"] if (types != "TP").any() else r["boxes"])
    for ax, crop in zip(axes, (False, True)):
        ax.imshow(img)
        draw_ignore(ax, r["ignore"])
        draw_boxes(ax, pb[types == "TP"], color="lime", lw=1.0)
        draw_boxes(ax, pb[types != "TP"], pl[types != "TP"], ps[types != "TP"], color="red", lw=1.0, text=crop)
        draw_boxes(ax, r["boxes"][~used], r["labels"][~used], color="yellow", lw=1.2, ls="--", text=crop)
        ax.axis("off")
        if crop and len(focus):
            hh, ww = img.shape[:2]
            zw = min(zoom * 1.5, ww / 2)  # zoom window width, 3:2 aspect
            zh = zw / 1.5
            cx, cy = np.median((focus[:, 0] + focus[:, 2]) / 2), np.median((focus[:, 1] + focus[:, 3]) / 2)
            x0 = float(np.clip(cx - zw / 2, 0, ww - zw))
            y0 = float(np.clip(cy - zh / 2, 0, hh - zh))
            ax.set_xlim(x0, x0 + zw)
            ax.set_ylim(y0 + zh, y0)
    n_fp = int((types != "TP").sum())
    axes[0].set_title(f"{title}  {r['image_id']}: GT {len(r['boxes'])}, TP {int((types == 'TP').sum())}, "
                      f"FP {n_fp}, missed {int((~used).sum())} (score >= {score_thr})", fontsize=9)
    axes[1].set_title("zoom", fontsize=9)
    fig.tight_layout()
    return fig


def plot_predictions(df: pd.DataFrame, preds: list, rows, score_thr: float = 0.3, cols: int = 2):
    """Predictions (class colours) next to ground truth (white dashed)."""
    import matplotlib.pyplot as plt
    n = len(rows)
    fig, axes = plt.subplots(int(math.ceil(n / cols)), cols, figsize=(7 * cols, 4.2 * math.ceil(n / cols)))
    axes = np.atleast_1d(axes).ravel()
    for ax, row in zip(axes, rows):
        r = df.iloc[row]
        p = preds[row]
        k = p["scores"] >= score_thr
        ax.imshow(load_image(r["path"]))
        draw_ignore(ax, r["ignore"])
        draw_boxes(ax, r["boxes"], color="white", lw=0.6, ls="--")
        draw_boxes(ax, p["boxes"][k], p["labels"][k], lw=0.9)
        ax.set_title(f"{r['image_id']}  ({int(k.sum())} dets >= {score_thr})", fontsize=9)
        ax.axis("off")
    for ax in axes[n:]:
        ax.axis("off")
    fig.tight_layout()
    return fig


def plot_histories(histories: dict, metric: str = "mAP", x: str = "epoch", ax=None, title=None):
    """Validation metric over training for several runs. x = "epoch" or "minutes" (GPU time)."""
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        if h and h.get("val"):
            ax.plot([v[x] for v in h["val"]], [v[metric] for v in h["val"]], marker="o", ms=3, label=name)
    ax.set_xlabel("epoch" if x == "epoch" else "training time, min")
    ax.set_ylabel(f"val {metric}")
    ax.set_title(title or f"Validation {metric} during training")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


def plot_train_losses(histories: dict, ax=None):
    import matplotlib.pyplot as plt
    if ax is None:
        _, ax = plt.subplots(figsize=(7, 4))
    for name, h in histories.items():
        if h:
            ax.plot(h["epoch"], h["train_loss"], marker="o", ms=3, label=name)
    ax.set_xlabel("epoch")
    ax.set_ylabel("train loss")
    ax.set_title("Training loss (scales differ between architectures)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    return ax


# =============================================================================
# 11. Experiment runner
# =============================================================================
MAIN_METRICS = ["mAP", "AP50", "AP75", "AP_tiny", "AP_small", "AP_medium", "AP_large", "AR"]


def _find_finished_run(name: str, dirs) -> Path | None:
    for d in dirs:
        p = Path(d) / name
        if (p / "metrics.json").exists():
            return p
    return None


def load_run(run_dir) -> dict:
    run_dir = Path(run_dir)
    with open(run_dir / "metrics.json") as f:
        res = json.load(f)
    hist = run_dir / "history.json"
    if hist.exists():
        with open(hist) as f:
            res["history"] = json.load(f)
    return res


def load_val_preds(run_dir) -> list:
    with open(Path(run_dir) / "val_preds.pkl", "rb") as f:
        return pickle.load(f)


def load_model_from_run(run_dir, device, which: str = "best") -> nn.Module:
    """Rebuild the model of a finished run (no weight download) and load its checkpoint."""
    run_dir = Path(run_dir)
    with open(run_dir / "config.json") as f:
        cfg = json.load(f)
    cfg["pretrained"] = False
    hf_cfg = run_dir / "hf_config.json"
    model = build_model(cfg, hf_config_path=hf_cfg if hf_cfg.exists() else None)
    load_checkpoint(model, run_dir / f"{which}.pt", device)
    return model.to(device).eval()


def run_experiment(name: str, cfg: dict, train_df: pd.DataFrame, val_df: pd.DataFrame, results_dir,
                   extra_results_dirs=(), device=None, force: bool = False) -> dict:
    """Train + evaluate one configuration and save everything in results_dir/name:
    config.json, history.json, best.pt, val_preds.pkl, metrics.json.

    - finished run (metrics.json here or in one of `extra_results_dirs`) -> loaded, not retrained;
    - interrupted run (resume.pt left) -> continues after the last finished epoch;
    - force=True -> deletes the old folder and trains from scratch.
    """
    device = device or get_device()
    results_dir = Path(results_dir)
    run_dir = results_dir / name
    if force and run_dir.exists():
        shutil.rmtree(run_dir)
    if not force:
        found = _find_finished_run(name, [results_dir, *extra_results_dirs])
        if found is not None:
            if found.resolve() != run_dir.resolve():
                shutil.copytree(found, run_dir, dirs_exist_ok=True)
            res = load_run(run_dir)
            print(f"[{name}] already finished -> loaded (mAP {res['mAP']:.4f})")
            return res

    run_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = run_dir / "config.json"
    if (run_dir / "resume.pt").exists() and cfg_path.exists():
        with open(cfg_path) as f:
            saved = json.load(f)
        if json.loads(json.dumps(cfg)) != saved:
            raise ValueError(f"[{name}] an interrupted run exists with a different config. "
                             f"Use a new run name or force=True. Differences: "
                             f"{ {k: (saved.get(k), v) for k, v in json.loads(json.dumps(cfg)).items() if saved.get(k) != v} }")
    with open(cfg_path, "w") as f:
        json.dump(cfg, f, indent=1)
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True

    model = build_model(cfg)
    if cfg["arch"] == "rtdetr":
        model.model.config.to_json_file(str(run_dir / "hf_config.json"))
    train_ds = DroneDetDataset(train_df, train=True, mask_ignore=cfg["mask_ignore"], drop_ioa=cfg["drop_ioa"],
                               hflip_p=cfg["hflip_p"], color_jitter=cfg["color_jitter"])
    val_ds = DroneDetDataset(val_df, train=False, mask_ignore=False)
    train_loader = make_loader(train_ds, cfg["batch_size"], True, cfg["num_workers"], cfg["seed"])
    val_loader = make_loader(val_ds, cfg["eval_batch_size"], False, cfg["num_workers"], cfg["seed"])

    history = train_detector(model, train_loader, val_loader, val_df, cfg, run_dir, device, name)
    train_min = history["minutes"][-1] if history["minutes"] else 0.0

    load_checkpoint(model, run_dir / "best.pt", device)
    preds = predict(model, val_loader, device, cfg["amp"])
    metrics = evaluate_coco(val_df, preds)
    speed = measure_speed(model, val_df, device, amp=cfg["amp"])
    res = {"name": name, "arch": cfg["arch"], **metrics, **speed, **count_params(model),
           "train_minutes": round(train_min, 1), "best_epoch": history["best_epoch"],
           "epochs_done": len(history["epoch"])}
    with open(run_dir / "val_preds.pkl", "wb") as f:
        pickle.dump(preds, f)
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(res, f, indent=1)
    (run_dir / "resume.pt").unlink(missing_ok=True)  # finished: only best.pt is kept (saves disk / Drive space)
    res["history"] = history
    print(f"[{name}] done: mAP {res['mAP']:.4f} | AP50 {res['AP50']:.4f} | AP_small {res['AP_small']:.4f} "
          f"| {res['fps']} img/s | {train_min:.1f} min")

    del model, train_loader, val_loader
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return res


def results_table(results: list, cols=None) -> pd.DataFrame:
    """Comparison table of finished runs."""
    cols = cols or MAIN_METRICS + ["ms_per_img", "fps", "params_M", "train_minutes", "best_epoch"]
    rows = [{"run": r["name"], "arch": r["arch"], **{c: r.get(c) for c in cols}} for r in results]
    tab = pd.DataFrame(rows).set_index("run")
    num = [c for c in MAIN_METRICS if c in tab.columns]
    tab[num] = tab[num].astype(float).round(4)
    return tab


def per_class_table(results: list, metric: str = "AP") -> pd.DataFrame:
    return pd.DataFrame({r["name"]: {c: v[metric] for c, v in r.get("per_class", {}).items()}
                         for r in results}).reindex(CLASS_NAMES).round(4)


# =============================================================================
# 12. Submission
# =============================================================================
def make_submission(test_df: pd.DataFrame, preds: list, out_path, max_det: int = 500,
                    score_thr: float = 0.0) -> pd.DataFrame:
    """Write the CSV: `class score x y w h` groups, (x, y) = top-left corner in original pixels."""
    assert len(test_df) == len(preds)
    rows = []
    for r, p in zip(test_df.itertuples(), preds):
        b, s, lab = p["boxes"], p["scores"], p["labels"]
        order = np.argsort(-s)
        order = order[s[order] >= score_thr][:max_det]
        parts = []
        for j in order:
            x1, y1 = float(np.clip(b[j, 0], 0, r.width)), float(np.clip(b[j, 1], 0, r.height))
            x2, y2 = float(np.clip(b[j, 2], 0, r.width)), float(np.clip(b[j, 3], 0, r.height))
            if x2 - x1 <= 0 or y2 - y1 <= 0:
                continue
            parts.append(f"{int(lab[j])} {float(s[j]):.4f} {x1:.1f} {y1:.1f} {x2 - x1:.1f} {y2 - y1:.1f}")
        rows.append((r.image_id, " ".join(parts)))
    sub = pd.DataFrame(rows, columns=["image_id", "PredictionString"])
    sub.to_csv(out_path, index=False)
    return sub


def validate_submission(sub_path, sample_path, max_det: int = 500) -> dict:
    """Basic format checks against sample_submission.csv."""
    sub = pd.read_csv(sub_path, keep_default_na=False)
    sample = pd.read_csv(sample_path, keep_default_na=False)
    problems = []
    if list(sub.columns) != list(sample.columns):
        problems.append(f"columns {list(sub.columns)} != {list(sample.columns)}")
    if set(sub["image_id"]) != set(sample["image_id"]) or len(sub) != len(sample):
        problems.append("image_id set or row count differs from sample_submission")
    n_det = []
    for ps in sub["PredictionString"].astype(str):
        vals = ps.split()
        if len(vals) % 6:
            problems.append("a PredictionString length is not a multiple of 6")
            break
        arr = np.array(vals, dtype=np.float64).reshape(-1, 6) if vals else np.zeros((0, 6))
        if len(arr) and (not np.isin(arr[:, 0], range(NUM_CLASSES)).all() or (arr[:, 4:] <= 0).any()):
            problems.append("bad class id or non-positive width/height")
            break
        n_det.append(len(arr))
    n_det = np.array(n_det) if n_det else np.zeros(1)
    if n_det.max() > max_det:
        problems.append(f"more than {max_det} detections in an image")
    return {"ok": not problems, "problems": problems, "rows": len(sub),
            "mean detections/image": round(float(n_det.mean()), 1), "max detections/image": int(n_det.max())}
