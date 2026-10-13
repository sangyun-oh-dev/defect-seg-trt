# ONNX -> TensorRT FP16 엔진 빌드 (Windows PowerShell)
#
# 사용 예:
#   .\scripts\build_engine.ps1 -Onnx runs\carpet\model.onnx -Engine runs\carpet\model_fp16.engine
#
# 주의: TensorRT 엔진은 빌드한 GPU 아키텍처 / TensorRT 버전에서만 동작한다.
#       다른 PC나 다른 GPU에서는 반드시 다시 빌드할 것.
param(
    [Parameter(Mandatory = $true)][string]$Onnx,
    [Parameter(Mandatory = $true)][string]$Engine,
    [int]$Patch = 512,
    [int]$OptBatch = 4,     # 1024x1024 이미지 = 512 patch 4장 = 1 chunk
    [int]$MaxBatch = 8,
    [switch]$Fp32,
    [string]$Trtexec = "trtexec.exe"
)

# trtexec는 경고를 stderr로 출력한다. Windows PowerShell 5.1에서 "Stop"이면 경고만으로 중단되므로 Continue 사용
$ErrorActionPreference = "Continue"

if (-not (Get-Command $Trtexec -ErrorAction SilentlyContinue)) {
    throw "trtexec를 찾을 수 없습니다. TensorRT의 bin 폴더를 PATH에 추가하거나 -Trtexec 로 경로를 지정하세요."
}
if (-not (Test-Path $Onnx)) { throw "ONNX 파일이 없습니다: $Onnx" }

$shape = "1x${Patch}x${Patch}"
$trtArgs = @(
    "--onnx=$Onnx",
    "--saveEngine=$Engine",
    "--minShapes=input:1x$shape",
    "--optShapes=input:${OptBatch}x$shape",
    "--maxShapes=input:${MaxBatch}x$shape",
    "--memPoolSize=workspace:2048",
    "--warmUp=500",
    "--iterations=200",
    "--useCudaGraph",
    "--separateProfileRun"
)
if (-not $Fp32) { $trtArgs += "--fp16" }

$log = [System.IO.Path]::ChangeExtension($Engine, ".build.log")
Write-Host "[trtexec] $($trtArgs -join ' ')"
& $Trtexec @trtArgs 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log
if ($LASTEXITCODE -ne 0) { throw "엔진 빌드 실패 (로그: $log)" }

Write-Host ""
Write-Host "엔진 저장: $Engine"
Write-Host "빌드 로그: $log  (GPU Compute Time 항목이 순수 추론 시간)"
