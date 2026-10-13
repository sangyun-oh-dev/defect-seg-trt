/*
 * seg_infer: TensorRT 기반 결함 Segmentation 추론 DLL (C ABI)
 *
 * 설계 원칙
 *  - 검사기 프로그램과의 경계는 extern "C" 함수만 노출 (C#, C++, Python ctypes 어디서든 호출)
 *  - 전처리(CLAHE, 타일링)는 호출 측 담당, DLL은 정규화 -> 추론 -> argmax -> 면적 집계 -> 판정만 수행
 *  - 모든 GPU/호스트 버퍼는 seg_create 시점에 max_batch 기준으로 1회 사전 할당
 *  - 하나의 핸들에 대한 동시 호출은 내부 mutex로 직렬화 (GPU 컨텍스트 경합 방지)
 *  - C++ 예외는 DLL 경계를 넘지 않음: 모든 함수는 상태 코드를 반환하고 메시지는 seg_last_error로 조회
 */
#ifndef SEG_INFER_H
#define SEG_INFER_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(SEG_INFER_EXPORTS)
#    define SEG_API __declspec(dllexport)
#  else
#    define SEG_API __declspec(dllimport)
#  endif
#else
#  define SEG_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SEG_MAX_CLASSES 8

typedef void* SegHandle;

typedef enum SegStatus {
    SEG_OK = 0,
    SEG_ERR_ARG = -1,      /* 잘못된 인자 (null, batch 범위 초과 등) */
    SEG_ERR_IO = -2,       /* 엔진 파일 읽기 실패 */
    SEG_ERR_TRT = -3,      /* TensorRT 역직렬화/실행 실패 */
    SEG_ERR_CUDA = -4,     /* CUDA API 실패 */
    SEG_ERR_UNKNOWN = -99
} SegStatus;

typedef struct SegConfig {
    int patch_size;    /* 정사각 patch 한 변 (학습 설정과 동일, 기본 512) */
    int max_batch;     /* 한 번에 처리할 최대 patch 수 = 사전 할당 크기 (엔진 max profile 이하) */
    int num_classes;   /* 출력 클래스 수 (배경 포함) */
    float mean;        /* 정규화: (x/255 - mean) / std  -- Python common.normalize 와 동일 */
    float std;
    int ng_class;      /* 판정 규칙: 이 클래스 면적 >= min_area 이면 NG */
    int min_area;      /* 노이즈 제거용 최소 픽셀 수 */
} SegConfig;

typedef struct SegResult {
    int cls;                          /* 판정 클래스 (0 = 정상) */
    int area[SEG_MAX_CLASSES];        /* 클래스별 픽셀 수 */
} SegResult;

/* 기본 설정값 채우기 (configs/carpet.yaml 과 동일) */
SEG_API void seg_default_config(SegConfig* cfg);

/* 엔진 로드 + 버퍼 사전 할당. 성공 시 *out 에 핸들 */
SEG_API int seg_create(const char* engine_path, const SegConfig* cfg, SegHandle* out);

/*
 * patch 묶음(chunk) 추론
 *  patches     : uint8 gray, [batch][patch][patch] 연속 메모리 (CLAHE 적용 완료)
 *  batch       : 1 .. max_batch
 *  out_masks   : (선택, NULL 가능) uint8 [batch][patch][patch] 클래스 마스크
 *  out_results : (선택, NULL 가능) SegResult[batch], patch별 면적과 판정
 *  out_ms      : (선택, NULL 가능) 이번 호출 소요 시간 (H2D ~ D2H 포함)
 */
SEG_API int seg_infer(SegHandle h, const uint8_t* patches, int batch,
                      uint8_t* out_masks, SegResult* out_results, float* out_ms);

/* 여러 patch의 면적 합으로 이미지 단위 판정 (Python common.decide 와 동일 규칙) */
SEG_API int seg_decide(const int* area, int num_classes, int ng_class, int min_area);

SEG_API void seg_destroy(SegHandle h);

/* 마지막 오류 메시지. h == NULL 이면 seg_create 실패 메시지 (스레드별) */
SEG_API const char* seg_last_error(SegHandle h);

SEG_API const char* seg_version(void);

#ifdef __cplusplus
}
#endif

#endif /* SEG_INFER_H */
