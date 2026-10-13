// seg_infer DLL 구현 (TensorRT 10.x API: setInputShape / setTensorAddress / enqueueV3)
#include "seg_infer.h"

#include <NvInfer.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "kernels.h"

namespace {

static_assert(SEG_MAX_CLASSES == seg::kMaxClasses, "class limit mismatch");

class SegError : public std::runtime_error {
public:
    SegError(int code, const std::string& msg) : std::runtime_error(msg), code(code) {}
    int code;
};

#define CUDA_CHECK(expr)                                                                       \
    do {                                                                                       \
        cudaError_t _e = (expr);                                                               \
        if (_e != cudaSuccess) {                                                               \
            std::ostringstream _os;                                                            \
            _os << #expr << " failed: " << cudaGetErrorString(_e) << " (" << __FILE__ << ":"   \
                << __LINE__ << ")";                                                            \
            throw SegError(SEG_ERR_CUDA, _os.str());                                           \
        }                                                                                      \
    } while (0)

class TrtLogger : public nvinfer1::ILogger {
public:
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) last = msg;
    }
    std::string last;
};

// RAII: cudaMalloc / cudaMallocHost 해제 보장
struct DeviceBuffer {
    void* ptr = nullptr;
    ~DeviceBuffer() { if (ptr) cudaFree(ptr); }
    void alloc(size_t bytes) { CUDA_CHECK(cudaMalloc(&ptr, bytes)); }
};
struct PinnedBuffer {
    void* ptr = nullptr;
    ~PinnedBuffer() { if (ptr) cudaFreeHost(ptr); }
    void alloc(size_t bytes) { CUDA_CHECK(cudaMallocHost(&ptr, bytes)); }
};

struct Engine {
    SegConfig cfg{};
    TrtLogger logger;
    std::unique_ptr<nvinfer1::IRuntime> runtime;
    std::unique_ptr<nvinfer1::ICudaEngine> engine;
    std::unique_ptr<nvinfer1::IExecutionContext> context;
    std::string input_name, output_name;
    cudaStream_t stream = nullptr;

    // 사전 할당 버퍼 (max_batch 기준)
    PinnedBuffer h_in, h_mask, h_counts;
    DeviceBuffer d_u8, d_input, d_logits, d_mask, d_counts;

    std::mutex mtx;        // 같은 핸들에 대한 동시 호출 직렬화
    std::string last_error;

    ~Engine() {
        context.reset();  // TensorRT 객체는 생성 역순으로 해제
        engine.reset();
        runtime.reset();
        if (stream) cudaStreamDestroy(stream);
    }
};

thread_local std::string g_create_error;

std::vector<char> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) throw SegError(SEG_ERR_IO, std::string("cannot open engine file: ") + path);
    const std::streamsize size = f.tellg();
    f.seekg(0, std::ios::beg);
    std::vector<char> buf(static_cast<size_t>(size));
    if (!f.read(buf.data(), size)) throw SegError(SEG_ERR_IO, std::string("cannot read engine file: ") + path);
    return buf;
}

void validate_config(const SegConfig& c) {
    if (c.patch_size <= 0 || c.max_batch <= 0 || c.num_classes < 2 || c.num_classes > SEG_MAX_CLASSES ||
        c.std <= 0.f || c.ng_class < 0 || c.ng_class >= c.num_classes || c.min_area < 0) {
        throw SegError(SEG_ERR_ARG, "invalid SegConfig");
    }
}

void bind_io(Engine& e) {
    const int n = e.engine->getNbIOTensors();
    for (int i = 0; i < n; ++i) {
        const char* name = e.engine->getIOTensorName(i);
        if (e.engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) e.input_name = name;
        else e.output_name = name;
        if (e.engine->getTensorDataType(name) != nvinfer1::DataType::kFLOAT) {
            throw SegError(SEG_ERR_TRT, std::string("expected float32 I/O tensor: ") + name);
        }
    }
    if (n != 2 || e.input_name.empty() || e.output_name.empty()) {
        throw SegError(SEG_ERR_TRT, "engine must have exactly one input and one output");
    }
    const auto maxd = e.engine->getProfileShape(e.input_name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
    if (maxd.nbDims != 4 || maxd.d[1] != 1 || maxd.d[2] != e.cfg.patch_size || maxd.d[3] != e.cfg.patch_size) {
        throw SegError(SEG_ERR_TRT, "engine input shape does not match patch_size (expected Nx1xPxP)");
    }
    if (maxd.d[0] < e.cfg.max_batch) {
        std::ostringstream os;
        os << "max_batch " << e.cfg.max_batch << " exceeds engine profile max " << maxd.d[0];
        throw SegError(SEG_ERR_ARG, os.str());
    }
}

void allocate(Engine& e) {
    const size_t px = static_cast<size_t>(e.cfg.patch_size) * e.cfg.patch_size;
    const size_t b = static_cast<size_t>(e.cfg.max_batch);
    const size_t c = static_cast<size_t>(e.cfg.num_classes);
    e.h_in.alloc(b * px);
    e.h_mask.alloc(b * px);
    e.h_counts.alloc(b * c * sizeof(int));
    e.d_u8.alloc(b * px);
    e.d_input.alloc(b * px * sizeof(float));
    e.d_logits.alloc(b * c * px * sizeof(float));
    e.d_mask.alloc(b * px);
    e.d_counts.alloc(b * c * sizeof(int));
    // 주소는 고정이므로 생성 시 1회만 바인딩
    if (!e.context->setTensorAddress(e.input_name.c_str(), e.d_input.ptr) ||
        !e.context->setTensorAddress(e.output_name.c_str(), e.d_logits.ptr)) {
        throw SegError(SEG_ERR_TRT, "setTensorAddress failed");
    }
}

int decide(const int* area, int classes, int ng_class, int min_area) {
    if (area[ng_class] >= min_area) return ng_class;
    int best = 0, best_area = 0;
    for (int c = 1; c < classes; ++c) {
        if (c != ng_class && area[c] >= min_area && area[c] > best_area) {
            best = c;
            best_area = area[c];
        }
    }
    return best;
}

void run(Engine& e, const uint8_t* patches, int batch, uint8_t* out_masks, SegResult* out_results) {
    if (!patches || batch < 1 || batch > e.cfg.max_batch) {
        std::ostringstream os;
        os << "invalid input (patches=" << static_cast<const void*>(patches) << ", batch=" << batch
           << ", max_batch=" << e.cfg.max_batch << ")";
        throw SegError(SEG_ERR_ARG, os.str());
    }
    const int p = e.cfg.patch_size;
    const int c = e.cfg.num_classes;
    const size_t px = static_cast<size_t>(p) * p;
    const size_t n = px * batch;

    std::memcpy(e.h_in.ptr, patches, n);  // pinned staging -> 비동기 전송 가능
    CUDA_CHECK(cudaMemcpyAsync(e.d_u8.ptr, e.h_in.ptr, n, cudaMemcpyHostToDevice, e.stream));
    CUDA_CHECK(seg::launch_normalize(static_cast<uint8_t*>(e.d_u8.ptr), static_cast<float*>(e.d_input.ptr), n,
                                     e.cfg.mean, e.cfg.std, e.stream));

    if (!e.context->setInputShape(e.input_name.c_str(), nvinfer1::Dims4{batch, 1, p, p})) {
        throw SegError(SEG_ERR_TRT, "setInputShape failed: " + e.logger.last);
    }
    if (!e.context->enqueueV3(e.stream)) {
        throw SegError(SEG_ERR_TRT, "enqueueV3 failed: " + e.logger.last);
    }

    CUDA_CHECK(seg::launch_argmax_count(static_cast<float*>(e.d_logits.ptr), static_cast<uint8_t*>(e.d_mask.ptr),
                                        static_cast<int*>(e.d_counts.ptr), batch, c, static_cast<int>(px), e.stream));
    CUDA_CHECK(cudaMemcpyAsync(e.h_counts.ptr, e.d_counts.ptr, sizeof(int) * batch * c, cudaMemcpyDeviceToHost,
                               e.stream));
    if (out_masks) {
        CUDA_CHECK(cudaMemcpyAsync(e.h_mask.ptr, e.d_mask.ptr, n, cudaMemcpyDeviceToHost, e.stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(e.stream));

    if (out_masks) std::memcpy(out_masks, e.h_mask.ptr, n);
    if (out_results) {
        const int* counts = static_cast<const int*>(e.h_counts.ptr);
        for (int b = 0; b < batch; ++b) {
            SegResult& r = out_results[b];
            std::memset(&r, 0, sizeof(r));
            for (int k = 0; k < c; ++k) r.area[k] = counts[b * c + k];
            r.cls = decide(r.area, c, e.cfg.ng_class, e.cfg.min_area);
        }
    }
}

// DLL 경계: 모든 예외를 상태 코드로 변환
template <typename F>
int guard(std::string& err_slot, F&& fn) {
    try {
        fn();
        err_slot.clear();
        return SEG_OK;
    } catch (const SegError& ex) {
        err_slot = ex.what();
        return ex.code;
    } catch (const std::bad_alloc&) {
        err_slot = "host out of memory";
        return SEG_ERR_UNKNOWN;
    } catch (const std::exception& ex) {
        err_slot = ex.what();
        return SEG_ERR_UNKNOWN;
    } catch (...) {
        err_slot = "unknown exception";
        return SEG_ERR_UNKNOWN;
    }
}

}  // namespace

extern "C" {

SEG_API void seg_default_config(SegConfig* cfg) {
    if (!cfg) return;
    cfg->patch_size = 512;
    cfg->max_batch = 4;
    cfg->num_classes = 3;
    cfg->mean = 0.449f;
    cfg->std = 0.226f;
    cfg->ng_class = 1;
    cfg->min_area = 30;
}

SEG_API int seg_create(const char* engine_path, const SegConfig* cfg, SegHandle* out) {
    return guard(g_create_error, [&] {
        if (!engine_path || !cfg || !out) throw SegError(SEG_ERR_ARG, "null argument");
        *out = nullptr;
        validate_config(*cfg);
        auto e = std::make_unique<Engine>();
        e->cfg = *cfg;

        const std::vector<char> blob = read_file(engine_path);
        e->runtime.reset(nvinfer1::createInferRuntime(e->logger));
        if (!e->runtime) throw SegError(SEG_ERR_TRT, "createInferRuntime failed");
        e->engine.reset(e->runtime->deserializeCudaEngine(blob.data(), blob.size()));
        if (!e->engine) {
            throw SegError(SEG_ERR_TRT, "deserializeCudaEngine failed (TensorRT/GPU version mismatch?): " +
                                            e->logger.last);
        }
        e->context.reset(e->engine->createExecutionContext());
        if (!e->context) throw SegError(SEG_ERR_TRT, "createExecutionContext failed");
        CUDA_CHECK(cudaStreamCreateWithFlags(&e->stream, cudaStreamNonBlocking));

        bind_io(*e);
        allocate(*e);
        *out = e.release();
    });
}

SEG_API int seg_infer(SegHandle h, const uint8_t* patches, int batch, uint8_t* out_masks, SegResult* out_results,
                      float* out_ms) {
    if (!h) return SEG_ERR_ARG;
    auto* e = static_cast<Engine*>(h);
    std::lock_guard<std::mutex> lock(e->mtx);
    const auto t0 = std::chrono::steady_clock::now();
    const int rc = guard(e->last_error, [&] { run(*e, patches, batch, out_masks, out_results); });
    if (out_ms) {
        out_ms[0] = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    }
    return rc;
}

SEG_API int seg_decide(const int* area, int num_classes, int ng_class, int min_area) {
    if (!area || num_classes < 2 || num_classes > SEG_MAX_CLASSES || ng_class < 0 || ng_class >= num_classes) {
        return SEG_ERR_ARG;
    }
    return decide(area, num_classes, ng_class, min_area);
}

SEG_API void seg_destroy(SegHandle h) {
    if (!h) return;
    auto* e = static_cast<Engine*>(h);
    {
        std::lock_guard<std::mutex> lock(e->mtx);  // 진행 중인 추론이 끝난 뒤 해제
        if (e->stream) cudaStreamSynchronize(e->stream);
    }
    delete e;
}

SEG_API const char* seg_last_error(SegHandle h) {
    if (!h) return g_create_error.c_str();
    return static_cast<Engine*>(h)->last_error.c_str();
}

SEG_API const char* seg_version(void) {
    static std::string v = [] {
        std::ostringstream os;
        os << "seg_infer 1.0 (TensorRT " << NV_TENSORRT_MAJOR << "." << NV_TENSORRT_MINOR << "."
           << NV_TENSORRT_PATCH << ")";
        return os.str();
    }();
    return v.c_str();
}

}  // extern "C"
