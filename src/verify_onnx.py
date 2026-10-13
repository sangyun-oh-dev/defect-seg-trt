"""PyTorch vs ONNX Runtime 출력 정합성 검증 + ONNX Runtime 추론 속도 측정.

변환 과정에서 모델이 바뀌지 않았는지 확인하는 단계.
- logit 최대 오차
- argmax 마스크 픽셀 일치율
- test셋 이미지 단위 판정 일치 여부

사용 예:
    python src/verify_onnx.py --ckpt runs/carpet/best.pth --onnx runs/carpet/model.onnx
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch

from common import build_model, decide, normalize, tile_starts
from dataset import load_or_make_split, read_pair


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--split", default="test")
    ap.add_argument("--iters", type=int, default=50)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], pretrained=False).eval()
    model.load_state_dict(ckpt["model"])

    providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(args.onnx, providers=providers)
    print(f"[ort] providers: {sess.get_providers()}")

    p, pre, c, dec = cfg["train"]["patch_size"], cfg["preprocess"], cfg["model"]["num_classes"], cfg["decision"]
    samples = load_or_make_split(cfg)[args.split]
    max_diff, agree, total, img_match = 0.0, 0, 0, 0
    chunk = None
    for s in samples:
        gray, _ = read_pair(s, pre)
        h, w = gray.shape
        coords = [(y, x) for y in tile_starts(h, p) for x in tile_starts(w, p)]
        chunk = np.stack([normalize(gray[y:y + p, x:x + p], pre)[None] for y, x in coords]).astype(np.float32)
        with torch.no_grad():
            pt = model(torch.from_numpy(chunk)).numpy()
        ox = sess.run(["logits"], {"input": chunk})[0]
        max_diff = max(max_diff, float(np.abs(pt - ox).max()))
        a, b = pt.argmax(1), ox.argmax(1)
        agree += int((a == b).sum())
        total += a.size
        da = decide(a.astype(np.uint8), c, dec["ng_class"], dec["min_area"])[0]
        db = decide(b.astype(np.uint8), c, dec["ng_class"], dec["min_area"])[0]
        img_match += int(da == db)

    # 속도: 마지막 이미지의 chunk(타일 묶음)를 반복 추론
    for _ in range(5):
        sess.run(["logits"], {"input": chunk})
    t = []
    for _ in range(args.iters):
        t0 = time.perf_counter()
        sess.run(["logits"], {"input": chunk})
        t.append((time.perf_counter() - t0) * 1000)
    res = {
        "images": len(samples),
        "logit_max_abs_diff": round(max_diff, 6),
        "pixel_argmax_agreement": round(agree / max(total, 1), 6),
        "image_decision_match": f"{img_match}/{len(samples)}",
        "ort_provider": sess.get_providers()[0],
        "ort_chunk_size": int(chunk.shape[0]),
        "ort_ms_per_chunk_mean": round(float(np.mean(t)), 2),
        "ort_ms_per_chunk_p99": round(float(np.percentile(t, 99)), 2),
    }
    out = Path(cfg["output_dir"]) / "verify_onnx.json"
    out.write_text(json.dumps(res, indent=2), encoding="utf-8")
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
