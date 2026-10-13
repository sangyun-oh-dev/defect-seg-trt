"""seg_infer.dll (TensorRT FP16)을 Python ctypes로 호출해 ONNX Runtime(FP32) 결과와 비교.

확인 항목
 - DLL 마스크 vs ONNX Runtime argmax 마스크 픽셀 일치율 (FP16 변환 영향)
 - 이미지 단위 판정 일치 수
 - 정답 대비 이미지 단위 정확도 / NG 미검출 수
 - C++ 벤치마크용 test 이미지 목록(test_images.txt) 저장

사용 예 (프로젝트 루트에서):
    python tests/test_dll.py --ckpt runs/carpet/best.pth --onnx runs/carpet/model.onnx ^
        --engine runs/carpet/model_fp16.engine --dll cpp/build/Release/seg_infer.dll
"""
from __future__ import annotations

import argparse
import ctypes as C
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from common import decide, normalize, tile_starts  # noqa: E402
from dataset import load_or_make_split, read_pair  # noqa: E402

MAX_CLASSES = 8


class SegConfig(C.Structure):
    _fields_ = [("patch_size", C.c_int), ("max_batch", C.c_int), ("num_classes", C.c_int),
                ("mean", C.c_float), ("std", C.c_float), ("ng_class", C.c_int), ("min_area", C.c_int)]


class SegResult(C.Structure):
    _fields_ = [("cls", C.c_int), ("area", C.c_int * MAX_CLASSES)]


def load_dll(path: str):
    if os.name == "nt":
        # Python 3.8+ 는 의존 DLL을 찾을 때 PATH를 보지 않는다.
        # CUDA/TensorRT 폴더와, PATH 중 nvinfer DLL이 있는 폴더를 직접 등록한다.
        dirs = []
        for env in ("CUDA_PATH", "TENSORRT_ROOT"):
            if os.environ.get(env):
                dirs += [Path(os.environ[env]) / "bin", Path(os.environ[env]) / "lib"]
        for p in os.environ.get("PATH", "").split(os.pathsep):
            if p and any(Path(p).glob("nvinfer*.dll")):
                dirs.append(Path(p))
        for d in dict.fromkeys(dirs):
            if d.exists():
                os.add_dll_directory(str(d))
    lib = C.CDLL(str(Path(path).resolve()))
    lib.seg_default_config.argtypes = [C.POINTER(SegConfig)]
    lib.seg_create.argtypes = [C.c_char_p, C.POINTER(SegConfig), C.POINTER(C.c_void_p)]
    lib.seg_infer.argtypes = [C.c_void_p, C.POINTER(C.c_uint8), C.c_int, C.POINTER(C.c_uint8),
                              C.POINTER(SegResult), C.POINTER(C.c_float)]
    lib.seg_destroy.argtypes = [C.c_void_p]
    lib.seg_last_error.argtypes = [C.c_void_p]
    lib.seg_last_error.restype = C.c_char_p
    lib.seg_version.restype = C.c_char_p
    return lib


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--dll", required=True)
    ap.add_argument("--onnx", help="지정하면 ONNX Runtime 결과와 픽셀 단위 비교")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-batch", type=int, default=4)
    args = ap.parse_args()

    import torch
    cfg = torch.load(args.ckpt, map_location="cpu", weights_only=False)["config"]
    p, pre, c, dec = cfg["train"]["patch_size"], cfg["preprocess"], cfg["model"]["num_classes"], cfg["decision"]
    samples = load_or_make_split(cfg)[args.split]

    out_dir = Path(cfg["output_dir"])
    (out_dir / "test_images.txt").write_text("\n".join(str(Path(s.image).resolve()) for s in samples),
                                             encoding="utf-8")

    lib = load_dll(args.dll)
    print(lib.seg_version().decode())
    sc = SegConfig()
    lib.seg_default_config(C.byref(sc))
    sc.patch_size, sc.max_batch, sc.num_classes = p, args.max_batch, c
    sc.mean, sc.std, sc.ng_class, sc.min_area = pre["mean"], pre["std"], dec["ng_class"], dec["min_area"]
    h = C.c_void_p()
    if lib.seg_create(args.engine.encode(), C.byref(sc), C.byref(h)) != 0:
        raise RuntimeError(lib.seg_last_error(None).decode())

    sess = None
    if args.onnx:
        import onnxruntime as ort
        prov = [x for x in ("CUDAExecutionProvider", "CPUExecutionProvider") if x in ort.get_available_providers()]
        sess = ort.InferenceSession(args.onnx, providers=prov)

    conf = np.zeros((c, c), np.int64)
    agree = total = dec_match = 0
    ms_list = []
    try:
        for s in samples:
            gray, _ = read_pair(s, pre)
            hh, ww = gray.shape
            coords = [(y, x) for y in tile_starts(hh, p) for x in tile_starts(ww, p)]
            n = len(coords)
            if n > args.max_batch:
                raise ValueError(f"patch 수 {n} > max_batch {args.max_batch}")
            patches = np.ascontiguousarray(np.stack([gray[y:y + p, x:x + p] for y, x in coords]), np.uint8)
            masks = np.zeros((n, p, p), np.uint8)
            res = (SegResult * n)()
            ms = C.c_float()
            rc = lib.seg_infer(h, patches.ctypes.data_as(C.POINTER(C.c_uint8)), n,
                               masks.ctypes.data_as(C.POINTER(C.c_uint8)), res, C.byref(ms))
            if rc != 0:
                raise RuntimeError(lib.seg_last_error(h).decode())
            ms_list.append(ms.value)
            area = np.sum([[r.area[k] for k in range(c)] for r in res], axis=0)
            cls_dll = int(lib.seg_decide((C.c_int * c)(*area.tolist()), c, dec["ng_class"], dec["min_area"]))
            conf[s.label, cls_dll] += 1
            if sess is not None:
                x = np.stack([normalize(t, pre)[None] for t in patches]).astype(np.float32)
                ref = sess.run(["logits"], {"input": x})[0].argmax(1).astype(np.uint8)
                agree += int((ref == masks).sum())
                total += ref.size
                dec_match += int(decide(ref, c, dec["ng_class"], dec["min_area"])[0] == cls_dll)
    finally:
        lib.seg_destroy(h)

    ng = dec["ng_class"]
    result = {
        "images": len(samples),
        "image_accuracy_vs_gt": round(float(np.trace(conf) / max(conf.sum(), 1)), 4),
        "ng_recall_vs_gt": round(float(conf[ng, ng] / max(conf[ng].sum(), 1)), 4),
        "ng_missed": int(conf[ng].sum() - conf[ng, ng]),
        "confusion_matrix(row=GT,col=Pred)": conf.tolist(),
        "dll_ms_per_chunk_mean": round(float(np.mean(ms_list)), 2),
    }
    if sess is not None:
        result["pixel_agreement_vs_onnx"] = round(agree / max(total, 1), 6)
        result["decision_match_vs_onnx"] = f"{dec_match}/{len(samples)}"
    (out_dir / "test_dll.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
