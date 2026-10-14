#!/usr/bin/env bash
# ONNX -> TensorRT FP16 엔진 빌드 (Linux / Docker). Windows는 build_engine.ps1 사용
#   bash scripts/build_engine.sh runs/carpet/model.onnx runs/carpet/model_fp16.engine
# 엔진은 빌드한 GPU 아키텍처와 TensorRT 버전에서만 동작한다.
set -euo pipefail

ONNX=${1:?usage: build_engine.sh <model.onnx> <out.engine>}
ENGINE=${2:?usage: build_engine.sh <model.onnx> <out.engine>}
PATCH=${PATCH:-512}
OPT_BATCH=${OPT_BATCH:-4}   # 1024x1024 이미지 = 512 patch 4장
MAX_BATCH=${MAX_BATCH:-8}
LOG="${ENGINE%.*}.build.log"

shape="1x${PATCH}x${PATCH}"
trtexec \
  --onnx="$ONNX" --saveEngine="$ENGINE" \
  --minShapes=input:1x${shape} --optShapes=input:${OPT_BATCH}x${shape} --maxShapes=input:${MAX_BATCH}x${shape} \
  --memPoolSize=workspace:2048 --warmUp=500 --iterations=200 --useCudaGraph --separateProfileRun \
  --fp16 2>&1 | tee "$LOG"

echo
echo "엔진 저장: $ENGINE"
echo "빌드 로그: $LOG  (GPU Compute Time 항목이 순수 추론 시간)"
