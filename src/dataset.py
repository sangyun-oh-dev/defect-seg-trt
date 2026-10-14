"""MVTec AD를 다중 클래스 Segmentation 데이터셋으로 구성.

MVTec AD의 train 폴더는 정상 이미지만 있으므로, test 폴더의 결함 이미지와
ground_truth 마스크를 결함 유형별로 층화 분할해 지도학습 데이터로 사용한다.
실제 공정과 마찬가지로 결함 이미지가 유형별 수십 장 수준인 소량 데이터 조건이다.
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from common import normalize, preprocess

IMG_EXT = (".png", ".jpg", ".bmp")


@dataclass
class Sample:
    image: str
    mask: str | None  # None = 정상 이미지 (전부 배경)
    label: int        # 이미지 단위 정답 클래스


def scan_mvtec(cfg_data: dict, good_limit: int = 40) -> list[Sample]:
    root = Path(cfg_data["root"]) / cfg_data["category"]
    if not root.exists():
        raise FileNotFoundError(f"데이터 경로가 없습니다: {root} (README의 데이터 준비 참고)")
    samples: list[Sample] = []
    for defect, cls in cfg_data["class_map"].items():
        img_dir = root / "test" / defect
        for img in sorted(p for p in img_dir.iterdir() if p.suffix.lower() in IMG_EXT):
            mask = root / "ground_truth" / defect / f"{img.stem}_mask.png"
            if not mask.exists():
                raise FileNotFoundError(f"마스크 없음: {mask}")
            samples.append(Sample(str(img), str(mask), int(cls)))
    if cfg_data.get("include_good", True):
        goods = []
        for sub in ("train/good", "test/good"):
            d = root / sub
            if d.exists():
                goods += sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
        rng = random.Random(cfg_data["seed"])
        rng.shuffle(goods)
        samples += [Sample(str(p), None, 0) for p in goods[:good_limit]]
    return samples


def stratified_split(samples: list[Sample], ratios: list[float], seed: int) -> dict[str, list[Sample]]:
    rng = random.Random(seed)
    out = {"train": [], "val": [], "test": []}
    for label in sorted({s.label for s in samples}):
        group = [s for s in samples if s.label == label]
        rng.shuffle(group)
        n = len(group)
        n_tr = int(round(n * ratios[0]))
        n_va = int(round(n * ratios[1]))
        out["train"] += group[:n_tr]
        out["val"] += group[n_tr:n_tr + n_va]
        out["test"] += group[n_tr + n_va:]
    return out


def load_or_make_split(cfg: dict) -> dict[str, list[Sample]]:
    """분할 결과를 split.json으로 고정해 학습/평가/벤치마크가 같은 test셋을 쓰게 한다."""
    path = Path(cfg["output_dir"]) / "split.json"
    if path.exists():
        raw = json.loads(path.read_text(encoding="utf-8"))
        # Windows에서 만든 split.json(역슬래시 경로)을 Linux/Docker에서도 읽을 수 있게 정규화
        fix = lambda x: x.replace("\\", "/") if isinstance(x, str) else x
        return {k: [Sample(fix(s["image"]), fix(s["mask"]), s["label"]) for s in v] for k, v in raw.items()}
    split = stratified_split(scan_mvtec(cfg["data"]), cfg["data"]["split"], cfg["data"]["seed"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({k: [asdict(s) for s in v] for k, v in split.items()},
                               ensure_ascii=False, indent=1), encoding="utf-8")
    return split


def read_pair(s: Sample, cfg_pre: dict) -> tuple[np.ndarray, np.ndarray]:
    img = cv2.imread(s.image, cv2.IMREAD_COLOR)
    if img is None:
        raise IOError(f"이미지 읽기 실패: {s.image}")
    gray = preprocess(img, cfg_pre)
    mask = np.zeros(gray.shape, np.uint8)
    if s.mask is not None:
        m = cv2.imread(s.mask, cv2.IMREAD_GRAYSCALE)
        mask[m > 127] = s.label
    return gray, mask


class TrainPatches(Dataset):
    """학습용: 결함 위치 편향 random crop + 기하/밝기 증강."""

    def __init__(self, samples: list[Sample], cfg: dict, repeats: int = 4):
        self.samples = samples
        self.pre = cfg["preprocess"]
        self.p = cfg["train"]["patch_size"]
        self.defect_prob = cfg["train"]["defect_crop_prob"]
        self.repeats = repeats  # 소량 데이터이므로 epoch당 이미지별 여러 crop
        self.cache = [read_pair(s, self.pre) for s in samples]

    def __len__(self) -> int:
        return len(self.samples) * self.repeats

    def _crop(self, gray, mask):
        h, w = gray.shape
        p = self.p
        if h < p or w < p:
            gray = cv2.resize(gray, (max(w, p), max(h, p)))
            mask = cv2.resize(mask, (max(w, p), max(h, p)), interpolation=cv2.INTER_NEAREST)
            h, w = gray.shape
        ys, xs = np.nonzero(mask)
        if len(ys) and random.random() < self.defect_prob:
            i = random.randrange(len(ys))
            cy, cx = ys[i], xs[i]
            y0 = int(np.clip(cy - random.randint(p // 8, p - p // 8), 0, h - p))
            x0 = int(np.clip(cx - random.randint(p // 8, p - p // 8), 0, w - p))
        else:
            y0, x0 = random.randint(0, h - p), random.randint(0, w - p)
        return gray[y0:y0 + p, x0:x0 + p], mask[y0:y0 + p, x0:x0 + p]

    def __getitem__(self, idx):
        gray, mask = self.cache[idx % len(self.samples)]
        g, m = self._crop(gray, mask)
        k = random.randint(0, 3)
        g, m = np.rot90(g, k), np.rot90(m, k)
        if random.random() < 0.5:
            g, m = g[:, ::-1], m[:, ::-1]
        if random.random() < 0.5:  # 조명 변동 대응
            alpha = random.uniform(0.85, 1.15)
            beta = random.uniform(-15, 15)
            g = np.clip(g.astype(np.float32) * alpha + beta, 0, 255).astype(np.uint8)
        x = normalize(np.ascontiguousarray(g), self.pre)[None]
        return torch.from_numpy(x), torch.from_numpy(np.ascontiguousarray(m).astype(np.int64))

