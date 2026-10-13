# 실행 환경 점검 (Windows PowerShell, 64비트 창에서 실행)
#   .\scripts\check_env.ps1
# MSVC(cl)은 MISS여도 됨: CMake가 Visual Studio를 직접 찾아 빌드한다.

function Check($name, [scriptblock]$cmd) {
    $global:LASTEXITCODE = 0
    try {
        $out = @(& $cmd 2>&1)
        if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" }
        $first = ($out | Where-Object { "$_".Trim() } | Select-Object -First 1)
        if (-not $first) { throw "empty" }
        Write-Host ("[OK]   {0,-14} {1}" -f $name, "$first".Trim())
    } catch {
        Write-Host ("[MISS] {0,-14} {1}" -f $name, "찾을 수 없음 - README 0. 준비물 참고") -ForegroundColor Yellow
    }
}

if (-not [Environment]::Is64BitProcess) {
    Write-Host "경고: 32비트 PowerShell입니다. 시작 메뉴의 'Windows PowerShell'(64비트)에서 다시 실행하세요." -ForegroundColor Red
}

Check "GPU/Driver"  { nvidia-smi --query-gpu="name,driver_version,compute_cap" --format=csv,noheader }
Check "CUDA(nvcc)"  { (nvcc --version | Select-String "release").ToString().Trim() }
Check "TensorRT"    { (Get-Command trtexec.exe -ErrorAction Stop).Source }
Check "CMake"       { cmake --version }
Check "MSVC(cl)"    { (Get-Command cl.exe -ErrorAction Stop).Source }
Check "Python"      { python --version }
Check "PyTorch"     { python -c "import torch;print(torch.__version__,'cuda' if torch.cuda.is_available() else 'NO CUDA', torch.version.cuda)" }
Check "ONNXRuntime" { python -c "import onnxruntime as o;print(o.__version__,o.get_available_providers())" }
Check "OpenCV_DIR"  { if ($env:OpenCV_DIR -and (Test-Path $env:OpenCV_DIR)) { $env:OpenCV_DIR } else { throw "unset" } }

Write-Host ""
Write-Host "MSVC(cl)은 MISS여도 괜찮습니다. 빌드할 때 CMake가 Visual Studio 2022를 직접 찾습니다."
