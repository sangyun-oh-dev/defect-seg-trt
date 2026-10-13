"""학습/평가/변환/검증 스크립트가 공유하는 설정, 전처리, 판정 로직.

C++ DLL(cpp/src/seg_infer.cpp)도 같은 전처리와 판정 규칙을 구현한다.
값을 바꾸면 양쪽을 함께 바꿔야 Python과 DLL의 결과가 일치한다.
"""
from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def apply_clahe(gray: np.ndarray, clip: float, grid: int) -> np.ndarray:
    """미세 결함의 국소 대비를 올리는 전처리. DLL과 동일 파라미터 사용."""
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    return clahe.apply(gray)


def preprocess(img: np.ndarray, cfg_pre: dict) -> np.ndarray:
    """uint8 이미지 -> CLAHE 적용 uint8 gray (H, W)."""
    gray = to_gray(img)
    return apply_clahe(gray, cfg_pre["clahe_clip"], cfg_pre["clahe_grid"])


def normalize(gray_u8: np.ndarray, cfg_pre: dict) -> np.ndarray:
    """uint8 gray -> float32 정규화 (H, W). DLL의 CUDA 커널과 동일한 식."""
    return ((gray_u8.astype(np.float32) / 255.0) - cfg_pre["mean"]) / cfg_pre["std"]


def build_model(cfg_model: dict, pretrained: bool | None = None) -> torch.nn.Module:
    """U-Net 생성.

    decoder_norm: "none"(기본) | "batchnorm"
      작은 배치(GPU 메모리 4GB 등)에서는 BatchNorm 통계가 불안정해 학습/추론 결과가 크게 어긋난다.
      그래서 디코더는 정규화 없이 쓰고, 인코더 BN은 학습 중 고정(freeze_encoder_bn)한다.
    """
    import inspect

    import segmentation_models_pytorch as smp

    weights = cfg_model.get("encoder_weights")
    if pretrained is False:
        weights = None
    use_bn = str(cfg_model.get("decoder_norm", "none")).lower() == "batchnorm"
    kwargs = dict(encoder_name=cfg_model["encoder"], encoder_weights=weights,
                  in_channels=cfg_model["in_channels"], classes=cfg_model["num_classes"])
    params = inspect.signature(smp.Unet.__init__).parameters
    if "decoder_use_norm" in params:          # smp >= 0.5
        kwargs["decoder_use_norm"] = "batchnorm" if use_bn else False
    elif "decoder_use_batchnorm" in params:   # smp < 0.5
        kwargs["decoder_use_batchnorm"] = use_bn
    return smp.Unet(**kwargs)


def freeze_encoder_bn(model: torch.nn.Module) -> None:
    """인코더 BatchNorm을 ImageNet 통계로 고정 (작은 배치에서 통계가 흔들리는 문제 방지)."""
    for m in model.encoder.modules():
        if isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            m.eval()


def decide(mask: np.ndarray, num_classes: int, ng_class: int, min_area: int) -> tuple[int, np.ndarray]:
    """픽셀 마스크 -> 이미지 단위 판정.

    1) NG 클래스 면적이 min_area 이상이면 NG 클래스
    2) 아니면 나머지 결함 클래스 중 면적이 가장 큰 클래스 (min_area 이상)
    3) 모두 미만이면 0 (정상)
    DLL의 decide()와 동일한 규칙.
    """
    areas = np.bincount(mask.ravel(), minlength=num_classes)[:num_classes]
    if areas[ng_class] >= min_area:
        return ng_class, areas
    best, best_area = 0, 0
    for c in range(1, num_classes):
        if c != ng_class and areas[c] >= min_area and areas[c] > best_area:
            best, best_area = c, areas[c]
    return best, areas


def tile_starts(length: int, patch: int) -> list[int]:
    """patch 크기로 겹침 최소 타일링 시작 좌표 (끝은 경계에 맞춤)."""
    if length <= patch:
        return [0]
    n = int(np.ceil(length / patch))
    step = (length - patch) / (n - 1)
    return [int(round(i * step)) for i in range(n)]
