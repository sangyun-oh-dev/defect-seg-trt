#!/usr/bin/env bash
# 전체 파이프라인 한 번에 실행 (Docker 컨테이너 안에서 사용)
#   학습 -> 평가 -> ONNX 변환/검증 -> TensorRT 엔진 -> C++ 라이브러리 검증 -> 벤치마크 -> 결과 그림
#
# 환경 변수로 조절:
#   CONFIG=configs/carpet.yaml  BATCH=2  ACCUM=4  EPOCHS=(설정 파일 값)  SKIP_TRAIN=1 (best.pth 재사용)
set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-configs/carpet.yaml}
BATCH=${BATCH:-2}       # GPU 4GB 기준. 메모리가 크면 BATCH=8 ACCUM=1
ACCUM=${ACCUM:-4}
OUT=$(python -c "import yaml;print(yaml.safe_load(open('$CONFIG'))['output_dir'])")
LIB=cpp/build/libseg_infer.so
BENCH=cpp/build/seg_bench

step() { echo; echo "==================== $* ===================="; }

if [ ! -d "$(python -c "import yaml;c=yaml.safe_load(open('$CONFIG'))['data'];print(c['root']+'/'+c['category'])")" ]; then
  echo "데이터가 없습니다. 호스트의 data 폴더를 /workspace/data 로 마운트했는지 확인하세요 (README의 Docker 항목)." >&2
  exit 1
fi
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || {
  echo "GPU를 찾을 수 없습니다. docker run 에 --gpus all 을 붙였는지 확인하세요." >&2; exit 1; }

if [ "${SKIP_TRAIN:-0}" = "1" ] && [ -f "$OUT/best.pth" ]; then
  step "1. 학습 생략 ($OUT/best.pth 재사용)"
else
  step "1. 학습 (batch $BATCH x accum $ACCUM)"
  EXTRA=()
  [ -n "${EPOCHS:-}" ] && EXTRA+=(--epochs "$EPOCHS")
  python src/train.py --config "$CONFIG" --batch-size "$BATCH" --accum "$ACCUM" "${EXTRA[@]}"
fi

step "2. test셋 평가"
python src/evaluate.py --config "$CONFIG" --ckpt "$OUT/best.pth" --split test

step "3. ONNX 변환 + PyTorch 대비 검증"
python src/export_onnx.py --ckpt "$OUT/best.pth" --out "$OUT/model.onnx"
python src/verify_onnx.py --ckpt "$OUT/best.pth" --onnx "$OUT/model.onnx" --split test

step "4. TensorRT FP16 엔진 빌드"
bash scripts/build_engine.sh "$OUT/model.onnx" "$OUT/model_fp16.engine"

step "5. C++ 라이브러리 정합성 (ctypes, ONNX 대비)"
python tests/test_dll.py --ckpt "$OUT/best.pth" --onnx "$OUT/model.onnx" \
  --engine "$OUT/model_fp16.engine" --dll "$LIB"

step "6. 지연시간 벤치마크"
"$BENCH" --engine "$OUT/model_fp16.engine" --list "$OUT/test_images.txt" --out "$OUT/bench" --iters 500

step "7. 결과 그림"
python src/visualize.py --config "$CONFIG" --ckpt "$OUT/best.pth" --out "$OUT/results.png"

step "완료"
echo "결과 파일: $OUT/eval_test.json, verify_onnx.json, test_dll.json, bench/latency.json, results.png"
