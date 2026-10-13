"""픽셀/이미지 단위 평가.

픽셀: 클래스별 IoU, F1(Dice)
이미지: 혼동행렬, 클래스별 Precision/Recall/F1, NG 미검출(Recall) 확인

사용 예:
    python src/evaluate.py --config configs/carpet.yaml --ckpt runs/carpet/best.pth --split test
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from common import build_model, decide, load_config, normalize, tile_starts
from dataset import Sample, load_or_make_split, read_pair


@torch.no_grad()
def predict_image(model, gray: np.ndarray, cfg: dict, device) -> np.ndarray:
    """이미지 전체를 타일(=chunk) 배치로 한 번에 추론하고 겹침 영역은 logit 평균."""
    p = cfg["train"]["patch_size"]
    pre = cfg["preprocess"]
    c = cfg["model"]["num_classes"]
    h, w = gray.shape
    coords = [(y, x) for y in tile_starts(h, p) for x in tile_starts(w, p)]
    batch = np.stack([normalize(gray[y:y + p, x:x + p], pre)[None] for y, x in coords])
    logits = model(torch.from_numpy(batch).to(device)).float().cpu().numpy()
    acc = np.zeros((c, h, w), np.float32)
    cnt = np.zeros((h, w), np.float32)
    for (y, x), lg in zip(coords, logits):
        acc[:, y:y + p, x:x + p] += lg
        cnt[y:y + p, x:x + p] += 1
    return (acc / cnt).argmax(0).astype(np.uint8)


def summarize(conf_pix: np.ndarray, conf_img: np.ndarray, names: list[str], ng_class: int) -> dict:
    eps = 1e-9
    tp = np.diag(conf_pix).astype(np.float64)
    fp = conf_pix.sum(0) - tp
    fn = conf_pix.sum(1) - tp
    iou = tp / (tp + fp + fn + eps)
    dice = 2 * tp / (2 * tp + fp + fn + eps)

    itp = np.diag(conf_img).astype(np.float64)
    prec = itp / (conf_img.sum(0) + eps)
    rec = itp / (conf_img.sum(1) + eps)
    f1 = 2 * prec * rec / (prec + rec + eps)
    present = conf_img.sum(1) > 0
    return {
        "pixel": {
            "iou": {n: round(float(v), 4) for n, v in zip(names, iou)},
            "dice": {n: round(float(v), 4) for n, v in zip(names, dice)},
            "mIoU_defect": round(float(iou[1:].mean()), 4),
        },
        "image": {
            "accuracy": round(float(itp.sum() / max(conf_img.sum(), 1)), 4),
            "precision": {n: round(float(v), 4) for n, v in zip(names, prec)},
            "recall": {n: round(float(v), 4) for n, v in zip(names, rec)},
            "f1": {n: round(float(v), 4) for n, v in zip(names, f1)},
            "macro_f1": round(float(f1[present].mean()), 4),
            "ng_recall": round(float(rec[ng_class]), 4),
            "ng_missed": int(conf_img[ng_class].sum() - conf_img[ng_class, ng_class]),
            "confusion_matrix(row=GT,col=Pred)": conf_img.tolist(),
        },
    }


def evaluate(model, samples: list[Sample], cfg: dict, device) -> dict:
    model.eval()
    c = cfg["model"]["num_classes"]
    dec = cfg["decision"]
    conf_pix = np.zeros((c, c), np.int64)
    conf_img = np.zeros((c, c), np.int64)
    for s in samples:
        gray, mask = read_pair(s, cfg["preprocess"])
        pred = predict_image(model, gray, cfg, device)
        conf_pix += np.bincount(mask.ravel().astype(np.int64) * c + pred.ravel(),
                                minlength=c * c).reshape(c, c)
        cls, _ = decide(pred, c, dec["ng_class"], dec["min_area"])
        conf_img[s.label, cls] += 1
    return summarize(conf_pix, conf_img, cfg["data"]["class_names"], dec["ng_class"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = ap.parse_args()

    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg["model"], pretrained=False).to(device)
    model.load_state_dict(torch.load(args.ckpt, map_location=device, weights_only=False)["model"])
    samples = load_or_make_split(cfg)[args.split]
    res = evaluate(model, samples, cfg, device)
    res["split"] = args.split
    res["num_images"] = len(samples)
    out = Path(cfg["output_dir"]) / f"eval_{args.split}.json"
    out.write_text(json.dumps(res, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(res, ensure_ascii=False, indent=2))
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
