# defect-seg-trt: 학습 -> ONNX -> TensorRT -> C++ 추론 라이브러리까지 한 컨테이너에서 재현
#
# 빌드 (저장소 루트에서):
#   docker build -f docker/Dockerfile -t defect-seg-trt .
# 실행:
#   docker run --rm --gpus all --ipc=host \
#     -v "$PWD/data:/workspace/data" -v "$PWD/runs_docker:/workspace/runs" \
#     defect-seg-trt bash scripts/run_all.sh
#
# 베이스: NVIDIA NGC TensorRT 24.08 (Ubuntu 22.04, CUDA 12.6, TensorRT 10.3, Python 3.10)
#  - CUDA, cuDNN, TensorRT, nvcc, trtexec 포함 -> 호스트에는 NVIDIA 드라이버만 있으면 됨
#  - 호스트 드라이버는 CUDA 12.6 이상을 지원해야 함 (R560 이상)
FROM nvcr.io/nvidia/tensorrt:24.08-py3

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1 \
    CUDACXX=/usr/local/cuda/bin/nvcc \
    PATH=/usr/src/tensorrt/bin:/usr/local/cuda/bin:${PATH}

# C++ 빌드 도구와 OpenCV (벤치마크 프로그램용)
RUN apt-get update \
 && apt-get install -y --no-install-recommends cmake libopencv-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python 패키지: 소스보다 먼저 설치해 코드 수정 시 이 층을 캐시로 재사용
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124 \
 && pip install -r /tmp/requirements.txt

COPY . .
# Windows에서 체크아웃한 경우 셸 스크립트의 CRLF 줄바꿈 제거
RUN sed -i 's/\r$//' scripts/*.sh

# C++ 추론 라이브러리(libseg_infer.so)와 벤치마크(seg_bench) 빌드
# docker build 중에는 GPU가 없으므로 아키텍처를 명시 (RTX 20/30/40, A100, L4 등)
RUN cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES="75;80;86;89" \
 && cmake --build cpp/build -j"$(nproc)"

CMD ["bash"]
