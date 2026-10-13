// seg_infer.dll 호출 예제 겸 벤치마크 (검사기 프로그램 역할)
//
//  - 이미지 읽기 -> gray -> CLAHE -> patch 타일링 (Python common.py와 동일 전처리)
//  - 이미지 1장의 patch 묶음을 1 chunk로 seg_infer 호출
//  - 이미지 단위 판정, 지연시간 통계(mean/p50/p99), 결과 CSV 저장
//
// 사용 예:
//   seg_bench.exe --engine runs\carpet\model_fp16.engine --list runs\carpet\test_images.txt --out runs\carpet\bench
#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "seg_infer.h"

namespace fs = std::filesystem;

struct Args {
    std::string engine, list, dir, out = "bench_out";
    int iters = 200, warmup = 20, max_batch = 4, patch = 512, min_area = 30;
    double clahe_clip = 2.0;
    int clahe_grid = 8;
    bool save_masks = false;
};

static void usage() {
    std::cout << "seg_bench --engine <file.engine> (--list <paths.txt> | --dir <image_dir>) [options]\n"
                 "  --out <dir>          결과 저장 폴더 (default: bench_out)\n"
                 "  --iters N            지연시간 측정 반복 횟수 (default: 200)\n"
                 "  --warmup N           워밍업 횟수 (default: 20)\n"
                 "  --max-batch N        사전 할당 patch 수 (default: 4)\n"
                 "  --patch N            patch 크기 (default: 512)\n"
                 "  --min-area N         판정 최소 면적 (default: 30)\n"
                 "  --save-masks         이미지별 예측 마스크 PNG 저장\n";
}

static bool parse(int argc, char** argv, Args& a) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        auto next = [&](void) -> const char* { return (i + 1 < argc) ? argv[++i] : ""; };
        if (k == "--engine") a.engine = next();
        else if (k == "--list") a.list = next();
        else if (k == "--dir") a.dir = next();
        else if (k == "--out") a.out = next();
        else if (k == "--iters") a.iters = std::atoi(next());
        else if (k == "--warmup") a.warmup = std::atoi(next());
        else if (k == "--max-batch") a.max_batch = std::atoi(next());
        else if (k == "--patch") a.patch = std::atoi(next());
        else if (k == "--min-area") a.min_area = std::atoi(next());
        else if (k == "--save-masks") a.save_masks = true;
        else { usage(); return false; }
    }
    if (a.engine.empty() || (a.list.empty() && a.dir.empty())) { usage(); return false; }
    return true;
}

static std::vector<int> tile_starts(int length, int patch) {
    if (length <= patch) return {0};
    const int n = static_cast<int>(std::ceil(static_cast<double>(length) / patch));
    const double step = static_cast<double>(length - patch) / (n - 1);
    std::vector<int> s;
    for (int i = 0; i < n; ++i) s.push_back(static_cast<int>(std::lround(i * step)));
    return s;
}

struct Chunk {
    std::string path;
    int h = 0, w = 0;
    std::vector<cv::Point> origins;
    std::vector<uint8_t> data;  // [n][patch][patch]
};

static Chunk make_chunk(const std::string& path, const Args& a, const cv::Ptr<cv::CLAHE>& clahe) {
    Chunk c;
    c.path = path;
    cv::Mat color = cv::imread(path, cv::IMREAD_COLOR);  // Python과 동일하게 BGR 로드 후 gray 변환
    if (color.empty()) throw std::runtime_error("cannot read image: " + path);
    cv::Mat img;
    cv::cvtColor(color, img, cv::COLOR_BGR2GRAY);
    if (img.rows < a.patch || img.cols < a.patch) {
        cv::resize(img, img, cv::Size(std::max(img.cols, a.patch), std::max(img.rows, a.patch)));
    }
    cv::Mat eq;
    clahe->apply(img, eq);  // 전체 이미지에 CLAHE 후 타일링 (학습과 동일 순서)
    c.h = eq.rows;
    c.w = eq.cols;
    for (int y : tile_starts(eq.rows, a.patch))
        for (int x : tile_starts(eq.cols, a.patch)) c.origins.emplace_back(x, y);
    const size_t px = static_cast<size_t>(a.patch) * a.patch;
    c.data.resize(px * c.origins.size());
    for (size_t i = 0; i < c.origins.size(); ++i) {
        cv::Mat roi = eq(cv::Rect(c.origins[i].x, c.origins[i].y, a.patch, a.patch)).clone();
        std::memcpy(c.data.data() + i * px, roi.data, px);
    }
    return c;
}

static double percentile(std::vector<float> v, double q) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    const double idx = q / 100.0 * (v.size() - 1);
    const size_t lo = static_cast<size_t>(std::floor(idx)), hi = static_cast<size_t>(std::ceil(idx));
    return v[lo] + (v[hi] - v[lo]) * (idx - lo);
}

int main(int argc, char** argv) {
    Args a;
    if (!parse(argc, argv, a)) return 2;

    std::vector<std::string> paths;
    if (!a.list.empty()) {
        std::ifstream f(a.list);
        for (std::string line; std::getline(f, line);) {
            while (!line.empty() && (line.back() == '\r' || line.back() == ' ')) line.pop_back();
            if (!line.empty()) paths.push_back(line);
        }
    } else {
        for (auto& e : fs::directory_iterator(a.dir)) {
            auto ext = e.path().extension().string();
            if (ext == ".png" || ext == ".jpg" || ext == ".bmp") paths.push_back(e.path().string());
        }
        std::sort(paths.begin(), paths.end());
    }
    if (paths.empty()) { std::cerr << "no images\n"; return 2; }
    fs::create_directories(a.out);

    SegConfig cfg;
    seg_default_config(&cfg);
    cfg.patch_size = a.patch;
    cfg.max_batch = a.max_batch;
    cfg.min_area = a.min_area;

    SegHandle h = nullptr;
    if (seg_create(a.engine.c_str(), &cfg, &h) != SEG_OK) {
        std::cerr << "seg_create failed: " << seg_last_error(nullptr) << "\n";
        return 1;
    }
    std::cout << seg_version() << "\n";

    auto clahe = cv::createCLAHE(a.clahe_clip, cv::Size(a.clahe_grid, a.clahe_grid));
    std::vector<Chunk> chunks;
    try {
        for (auto& p : paths) chunks.push_back(make_chunk(p, a, clahe));
    } catch (const std::exception& ex) {
        std::cerr << ex.what() << "\n";
        seg_destroy(h);
        return 1;
    }

    const size_t px = static_cast<size_t>(a.patch) * a.patch;
    std::vector<uint8_t> masks(px * a.max_batch);
    std::vector<SegResult> res(a.max_batch);

    // 1) 이미지별 판정 + CSV
    std::ofstream csv(fs::path(a.out) / "predictions.csv");
    csv << "image,pred_class";
    for (int c = 0; c < cfg.num_classes; ++c) csv << ",area_" << c;
    csv << ",ms\n";
    for (auto& ch : chunks) {
        const int n = static_cast<int>(ch.origins.size());
        if (n > a.max_batch) {
            std::cerr << "patches per image (" << n << ") > max_batch; increase --max-batch\n";
            seg_destroy(h);
            return 1;
        }
        float ms = 0.f;
        if (seg_infer(h, ch.data.data(), n, a.save_masks ? masks.data() : nullptr, res.data(), &ms) != SEG_OK) {
            std::cerr << "seg_infer failed: " << seg_last_error(h) << "\n";
            seg_destroy(h);
            return 1;
        }
        int area[SEG_MAX_CLASSES] = {0};
        for (int i = 0; i < n; ++i)
            for (int c = 0; c < cfg.num_classes; ++c) area[c] += res[i].area[c];
        const int cls = seg_decide(area, cfg.num_classes, cfg.ng_class, cfg.min_area);
        csv << '"' << ch.path << '"' << ',' << cls;
        for (int c = 0; c < cfg.num_classes; ++c) csv << ',' << area[c];
        csv << ',' << ms << '\n';
        if (a.save_masks) {
            cv::Mat full(ch.h, ch.w, CV_8UC1, cv::Scalar(0));
            for (int i = 0; i < n; ++i) {
                cv::Mat m(a.patch, a.patch, CV_8UC1, masks.data() + i * px);
                m.copyTo(full(cv::Rect(ch.origins[i].x, ch.origins[i].y, a.patch, a.patch)));
            }
            cv::imwrite((fs::path(a.out) / (fs::path(ch.path).stem().string() + "_pred.png")).string(),
                        full * (255 / std::max(1, cfg.num_classes - 1)));
        }
    }

    // 2) 지연시간 측정: 첫 이미지 chunk를 반복 (마스크 복사 포함 = 실제 검사기 호출과 동일 조건)
    const Chunk& ch = chunks.front();
    const int n = static_cast<int>(ch.origins.size());
    for (int i = 0; i < a.warmup; ++i) seg_infer(h, ch.data.data(), n, masks.data(), res.data(), nullptr);
    std::vector<float> lat;
    lat.reserve(a.iters);
    for (int i = 0; i < a.iters; ++i) {
        float ms = 0.f;
        if (seg_infer(h, ch.data.data(), n, masks.data(), res.data(), &ms) != SEG_OK) {
            std::cerr << "seg_infer failed: " << seg_last_error(h) << "\n";
            break;
        }
        lat.push_back(ms);
    }
    const double mean = lat.empty() ? 0.0 : std::accumulate(lat.begin(), lat.end(), 0.0) / lat.size();

    std::ofstream js(fs::path(a.out) / "latency.json");
    js << "{\n  \"version\": \"" << seg_version() << "\",\n"
       << "  \"images\": " << chunks.size() << ",\n"
       << "  \"patches_per_chunk\": " << n << ",\n"
       << "  \"patch_size\": " << a.patch << ",\n"
       << "  \"iters\": " << lat.size() << ",\n"
       << "  \"ms_per_chunk_mean\": " << mean << ",\n"
       << "  \"ms_per_chunk_p50\": " << percentile(lat, 50) << ",\n"
       << "  \"ms_per_chunk_p99\": " << percentile(lat, 99) << ",\n"
       << "  \"ms_per_patch_mean\": " << (n ? mean / n : 0.0) << "\n}\n";
    std::printf("chunk(%d x %dx%d): mean %.2f ms | p50 %.2f ms | p99 %.2f ms | per patch %.2f ms\n", n, a.patch,
                a.patch, mean, percentile(lat, 50), percentile(lat, 99), n ? mean / n : 0.0);
    std::cout << "saved -> " << a.out << "\n";

    seg_destroy(h);
    return 0;
}
