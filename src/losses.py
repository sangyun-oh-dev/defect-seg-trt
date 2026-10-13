"""CE + Dice + Focal 복합 손실.

- CE(클래스 가중치): 픽셀 단위 분류의 기본 항, 희소 클래스에 가중치
- Dice: 면적이 작은 결함 영역의 겹침을 직접 최적화 (배경 지배 완화)
- Focal: 쉬운 배경 픽셀의 기여를 낮추고 어려운 경계 픽셀에 집중
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def frequency_weights(masks: list[np.ndarray], num_classes: int, max_w: float = 10.0) -> torch.Tensor:
    """학습셋 픽셀 빈도 기반 클래스 가중치 (역빈도의 제곱근, 배경 = 1).

    배경 가중치를 지나치게 낮추면 모델이 결함을 과다 예측해 오검출이 폭증한다.
    그래서 배경을 1로 두고 결함 클래스만 sqrt(배경빈도/클래스빈도)로 올리되 max_w로 상한을 둔다.
    희소 클래스 대응의 나머지는 Dice / Focal 항이 맡는다.
    """
    counts = np.zeros(num_classes, np.float64)
    for m in masks:
        counts += np.bincount(m.ravel(), minlength=num_classes)[:num_classes]
    counts = np.maximum(counts, 1.0)
    w = np.sqrt(counts.max() / counts)
    return torch.tensor(np.clip(w, 1.0, max_w), dtype=torch.float32)


class DiceLoss(nn.Module):
    def __init__(self, num_classes: int, ignore_bg: bool = True, eps: float = 1.0):
        super().__init__()
        self.c, self.ignore_bg, self.eps = num_classes, ignore_bg, eps

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        prob = logits.softmax(1)
        onehot = F.one_hot(target, self.c).permute(0, 3, 1, 2).float()
        dims = (0, 2, 3)
        inter = (prob * onehot).sum(dims)
        denom = prob.sum(dims) + onehot.sum(dims)
        dice = (2 * inter + self.eps) / (denom + self.eps)
        if self.ignore_bg:
            dice = dice[1:]
        return 1.0 - dice.mean()


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, weight: torch.Tensor | None = None):
        super().__init__()
        self.gamma = gamma
        self.register_buffer("weight", weight if weight is not None else None)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logp = F.log_softmax(logits, 1).gather(1, target[:, None]).squeeze(1)
        p = logp.exp()
        loss = -((1 - p) ** self.gamma) * logp
        if self.weight is not None:
            loss = loss * self.weight[target]
        return loss.mean()


class ComboLoss(nn.Module):
    def __init__(self, num_classes: int, class_weights: torch.Tensor | None, cfg_loss: dict):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(weight=class_weights)
        self.dice = DiceLoss(num_classes)
        self.focal = FocalLoss(cfg_loss["focal_gamma"], class_weights)
        self.w = (cfg_loss["ce_weight"], cfg_loss["dice_weight"], cfg_loss["focal_weight"])

    def forward(self, logits, target):
        parts = (self.ce(logits, target), self.dice(logits, target), self.focal(logits, target))
        total = sum(w * p for w, p in zip(self.w, parts))
        return total, {"ce": parts[0].item(), "dice": parts[1].item(), "focal": parts[2].item()}
