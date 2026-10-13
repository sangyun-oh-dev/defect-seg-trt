"""MVTec AD와 같은 폴더 구조의 작은 합성 데이터셋 생성 (동작 확인용).

실제 데이터 없이 학습 -> ONNX -> DLL 파이프라인이 끝까지 도는지 확인할 때 사용.
    python tests/make_synthetic_mvtec.py --root data/synthetic --size 1024
"""
import argparse
from pathlib import Path

import cv2
import numpy as np


def texture(rng, size):
    base = rng.normal(128, 12, (size // 8, size // 8)).astype(np.float32)
    base = cv2.resize(base, (size, size), interpolation=cv2.INTER_CUBIC)
    noise = rng.normal(0, 6, (size, size)).astype(np.float32)
    return np.clip(base + noise, 0, 255).astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="data/synthetic")
    ap.add_argument("--size", type=int, default=1024)
    ap.add_argument("--n", type=int, default=10)
    a = ap.parse_args()
    rng = np.random.default_rng(0)
    cat = Path(a.root) / "carpet"
    for sub in ("train/good", "test/good", "test/metal_contamination", "test/thread",
                "ground_truth/metal_contamination", "ground_truth/thread"):
        (cat / sub).mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        cv2.imwrite(str(cat / "train/good" / f"{i:03d}.png"), cv2.cvtColor(texture(rng, a.size), cv2.COLOR_GRAY2BGR))
        img = texture(rng, a.size)
        m = np.zeros_like(img)
        c = rng.integers(100, a.size - 100, 2)
        cv2.circle(img, tuple(int(v) for v in c), int(rng.integers(15, 40)), 30, -1)       # 어두운 금속 이물
        cv2.circle(m, tuple(int(v) for v in c), int(rng.integers(15, 40)), 255, -1)
        cv2.imwrite(str(cat / "test/metal_contamination" / f"{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(cat / "ground_truth/metal_contamination" / f"{i:03d}_mask.png"), m)
        img = texture(rng, a.size)
        m = np.zeros_like(img)
        p1 = tuple(int(v) for v in rng.integers(100, a.size - 100, 2))
        p2 = (p1[0] + int(rng.integers(-150, 150)), p1[1] + int(rng.integers(-150, 150)))
        cv2.line(img, p1, p2, 220, 5)                                                       # 밝은 실 이물
        cv2.line(m, p1, p2, 255, 7)
        cv2.imwrite(str(cat / "test/thread" / f"{i:03d}.png"), cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        cv2.imwrite(str(cat / "ground_truth/thread" / f"{i:03d}_mask.png"), m)
    print(f"synthetic dataset -> {cat}")


if __name__ == "__main__":
    main()
