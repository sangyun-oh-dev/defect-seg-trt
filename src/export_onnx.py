"""PyTorch 체크포인트 -> ONNX (동적 배치).

입력  input  : float32 [N, 1, P, P]  (CLAHE + 정규화 완료된 patch)
출력  logits : float32 [N, C, P, P]
argmax와 이미지 단위 판정은 C++ DLL의 CUDA 커널에서 처리한다.

사용 예:
    python src/export_onnx.py --ckpt runs/carpet/best.pth --out runs/carpet/model.onnx
"""
from __future__ import annotations

import argparse
from pathlib import Path

import onnx
import torch

from common import build_model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--opset", type=int, default=17)
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg["model"], pretrained=False).eval()
    model.load_state_dict(ckpt["model"])
    p = cfg["train"]["patch_size"]
    dummy = torch.randn(2, cfg["model"]["in_channels"], p, p)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model, dummy, args.out,
        input_names=["input"], output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=args.opset, do_constant_folding=True, dynamo=False,
    )
    m = onnx.load(args.out)
    onnx.checker.check_model(m)
    print(f"exported -> {args.out}  (opset {args.opset}, patch {p}, classes {cfg['model']['num_classes']})")


if __name__ == "__main__":
    main()
