"""test셋 예측 결과를 한 장의 그림으로 저장 (README용).

행 = 이미지 1장 (클래스별 대표 1장 + 오판 이미지 전부)
열 = 원본 | 정답 마스크 | 예측 마스크 | 결함 부위 확대(예측)

사용 예:
    python src/visualize.py --config configs/carpet.yaml --ckpt runs/carpet/best.pth
    -> docs/results.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch

from common import build_model, decide, load_config
from dataset import load_or_make_split, read_pair
from evaluate import predict_image

CELL = 300          # 한 칸 크기 (px)
CAPTION_H = 34      # 행 아래 설명 줄 높이
HEADER_H = 40
PAD = 8
# BGR. 0 배경은 칠하지 않음
COLORS = {1: (40, 40, 230), 2: (230, 170, 30)}
SHORT = {0: "good", 1: "metal (NG)", 2: "thread (OK)"}


def overlay(bgr: np.ndarray, mask: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    out = bgr.copy()
    for c, col in COLORS.items():
        m = mask == c
        if m.any():
            out[m] = (out[m] * (1 - alpha) + np.array(col) * alpha).astype(np.uint8)
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(out, cnts, -1, col, 3)
    return out


def fit(img: np.ndarray) -> np.ndarray:
    return cv2.resize(img, (CELL, CELL), interpolation=cv2.INTER_AREA)


def zoom_box(gt: np.ndarray, pred: np.ndarray, size: int = 256) -> tuple[int, int, int, int]:
    """결함이 있는 곳을 중심으로 size x size 영역 (정답 결함 우선, 없으면 예측 결함)."""
    h, w = gt.shape
    ys, xs = np.nonzero(gt > 0) if (gt > 0).any() else np.nonzero(pred > 0)
    cy, cx = (int(ys.mean()), int(xs.mean())) if len(ys) else (h // 2, w // 2)
    y0 = int(np.clip(cy - size // 2, 0, max(h - size, 0)))
    x0 = int(np.clip(cx - size // 2, 0, max(w - size, 0)))
    return y0, x0, min(size, h), min(size, w)


def text(img, s, org, scale=0.55, color=(30, 30, 30), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="docs/results.png")
    ap.add_argument("--max-rows", type=int, default=5)
    args = ap.parse_args()

    cfg = load_config(args.config)
    dec = cfg["decision"]
    c = cfg["model"]["num_classes"]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    model.eval()

    results = []
    for s in load_or_make_split(cfg)[args.split]:
        gray, gt = read_pair(s, cfg["preprocess"])
        pred = predict_image(model, gray, cfg, device)
        cls, _ = decide(pred, c, dec["ng_class"], dec["min_area"])
        results.append((s, gt, pred, cls))

    # 클래스별로 맞힌 대표 1장(NG 먼저) + 틀린 이미지 전부
    rows, used = [], set()
    for label in (1, 2, 0):
        for r in results:
            if r[0].label == label and r[3] == label:
                rows.append(r)
                used.add(r[0].image)
                break
    rows += [r for r in results if r[3] != r[0].label and r[0].image not in used]
    rows = rows[: args.max_rows]

    width = PAD + 4 * (CELL + PAD)
    height = HEADER_H + len(rows) * (CELL + CAPTION_H + PAD) + 40
    canvas = np.full((height, width, 3), 255, np.uint8)
    for i, h in enumerate(["Input", "Ground truth", "Prediction", "Zoom (prediction)"]):
        text(canvas, h, (PAD + i * (CELL + PAD) + 8, 27), 0.7, thick=2)

    for k, (s, gt, pred, cls) in enumerate(rows):
        src = cv2.imread(s.image, cv2.IMREAD_COLOR)
        y0, x0, zh, zw = zoom_box(gt, pred)
        zoom = overlay(src, pred)[y0:y0 + zh, x0:x0 + zw]
        cells = [src, overlay(src, gt), overlay(src, pred), zoom]
        top = HEADER_H + k * (CELL + CAPTION_H + PAD)
        for i, im in enumerate(cells):
            x = PAD + i * (CELL + PAD)
            canvas[top:top + CELL, x:x + CELL] = fit(im)
        # 확대 위치 표시
        sx = CELL / src.shape[1]
        cv2.rectangle(canvas, (PAD + int(x0 * sx), top + int(y0 * sx)),
                      (PAD + int((x0 + zw) * sx), top + int((y0 + zh) * sx)), (60, 60, 60), 1)
        ok = cls == s.label
        cap = f"{Path(s.image).parent.name}/{Path(s.image).name}   GT: {SHORT[s.label]}   ->   Pred: {SHORT[cls]}"
        text(canvas, cap, (PAD + 4, top + CELL + 23), 0.55)
        text(canvas, "correct" if ok else "wrong", (width - 90, top + CELL + 23), 0.6,
             (40, 140, 40) if ok else (40, 40, 210), 2)

    # 범례
    ly = height - 16
    lx = PAD + 4
    for cid, name in ((1, "metal_contamination = NG"), (2, "thread = OK")):
        cv2.rectangle(canvas, (lx, ly - 12), (lx + 14, ly + 2), COLORS[cid], -1)
        text(canvas, name, (lx + 22, ly), 0.55)
        lx += 290

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), canvas)
    print(f"{len(rows)} rows ({sum(r[3] != r[0].label for r in rows)} wrong) -> {out}")


if __name__ == "__main__":
    main()
