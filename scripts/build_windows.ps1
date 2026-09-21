param([Parameter(Mandatory=$true)][string]$BuildOutput)
$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$build = [System.IO.Path]::GetFullPath($BuildOutput)
if (Test-Path -LiteralPath $build) { throw "Choose a new empty build directory: $build" }
New-Item -ItemType Directory -Path $build | Out-Null
$python = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $python)) { throw "Prepare the locked Python environment first." }
$buildTools = Join-Path $build "tools"
& $python -m pip install --target $buildTools "pyinstaller==6.22.3" "pyinstaller-hooks-contrib==2026.7"
if ($LASTEXITCODE -ne 0) { throw "Build dependency installation failed" }
$savedPythonPath = $env:PYTHONPATH
try {
    $env:PYTHONPATH = $buildTools
    & $python -m PyInstaller --noconfirm --onedir --windowed --name Neftekod --distpath (Join-Path $build "dist") --workpath (Join-Path $build "work") --specpath $build --paths $root --collect-all lightgbm --hidden-import sklearn.ensemble._hist_gradient_boosting.predictor --hidden-import sklearn.linear_model._ridge (Join-Path $root "source\desktop.py")
    if ($LASTEXITCODE -ne 0) { throw "Freezing failed" }
} finally { $env:PYTHONPATH = $savedPythonPath }
Write-Output "Pass -BinaryDirectory '$build\dist\Neftekod' to scripts/build_release.ps1."
