param(
    [string]$Output = (Join-Path (Split-Path -Parent $PSScriptRoot) "..\output\neftekod-release"),
    [switch]$Archive,
    [string]$BinaryDirectory,
    [Parameter(Mandatory=$true)][string]$JuryDirectory
)

$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$destination = [System.IO.Path]::GetFullPath($Output)
if (Test-Path -LiteralPath $destination) {
    throw "Output already exists; choose a new path: $destination"
}

$manifestPath = Join-Path $root "config\release_manifest.json"
$manifest = Get-Content -Raw -LiteralPath $manifestPath | ConvertFrom-Json
$required = @(
    "README.md", "requirements.txt", "requirements.lock.txt",
    "pyproject.toml", "source", "config", "global_tests", "materials",
    "scripts", "docs", "third_party"
)
$required += @(Get-ChildItem -LiteralPath $root -Filter "*.md" -File |
    Where-Object { $_.Name -notin @("README.md", "AGENTS.md") } |
    ForEach-Object { $_.Name })
$selected = @(
    [string]$manifest.prepared_dataset,
    [string]$manifest.forecast_artifact,
    [string]$manifest.v2_artifact
)
$paths = $required + $selected
foreach ($relative in $paths) {
    $source = Join-Path $root $relative
    if (-not (Test-Path -LiteralPath $source)) {
        throw "Required release input is missing: $relative"
    }
}

$juryRoot = (Resolve-Path -LiteralPath $JuryDirectory).Path
$juryFiles = @("Neftekod_Presentation.pptx", "Neftekod_Jury.docx")
foreach ($name in $juryFiles) {
    if (-not (Test-Path -LiteralPath (Join-Path $juryRoot $name) -PathType Leaf)) {
        throw "Missing jury document: $name"
    }
}

$excludedDirectoryNames = @(".git", "__pycache__", ".pytest_cache", ".ruff_cache", ".mypy_cache", ".test_tmp")
function Copy-ReleaseItem {
    param(
        [string]$Source,
        [string]$Target
    )
    if (Test-Path -LiteralPath $Source -PathType Container) {
        New-Item -ItemType Directory -Path $Target -Force | Out-Null
        foreach ($child in Get-ChildItem -LiteralPath $Source -Force) {
            if ($child.PSIsContainer -and $excludedDirectoryNames -contains $child.Name) {
                continue
            }
            Copy-ReleaseItem $child.FullName (Join-Path $Target $child.Name)
        }
        return
    }
    New-Item -ItemType Directory -Path (Split-Path -Parent $Target) -Force | Out-Null
    Copy-Item -LiteralPath $Source -Destination $Target -Force
}

New-Item -ItemType Directory -Path $destination -Force | Out-Null
foreach ($relative in $required) {
    $source = Join-Path $root $relative
    $target = Join-Path $destination $relative
    Copy-ReleaseItem $source $target
}
foreach ($relative in $selected) {
    $target = Join-Path $destination $relative
    Copy-ReleaseItem (Join-Path $root $relative) $target
}
foreach ($name in $juryFiles) {
    Copy-ReleaseItem (Join-Path $juryRoot $name) (Join-Path $destination "docs\jury\$name")
}
if ($BinaryDirectory) {
    $binaryRoot = (Resolve-Path -LiteralPath $BinaryDirectory).Path
    foreach ($name in @("Neftekod.exe", "_internal")) {
        $inputPath = Join-Path $binaryRoot $name
        if (-not (Test-Path -LiteralPath $inputPath)) { throw "Missing portable runtime: $inputPath" }
        Copy-ReleaseItem $inputPath (Join-Path $destination $name)
    }
}

$releaseReports = Join-Path $destination "reports\release-acceptance"
$python = Join-Path $root ".venv\Scripts\python.exe"
if (Test-Path -LiteralPath $python) {
    Push-Location $destination
    try {
        & $python -B -m source.main acceptance --output $releaseReports | Out-Null
        if ($LASTEXITCODE -ne 0) {
            throw "Release acceptance failed with exit code $LASTEXITCODE"
        }
    } finally {
        Pop-Location
    }
} else {
    throw "A prepared Python environment is required to verify the copied release."
}

if ($BinaryDirectory) {
    $smokeReport = Join-Path $destination "reports\portable-smoke.json"
    $savedPath = $env:PATH
    $savedPythonPath = $env:PYTHONPATH
    try {
        $env:PATH = "$env:SystemRoot\System32;$env:SystemRoot"
        $env:PYTHONPATH = $null
        $process = Start-Process -FilePath (Join-Path $destination "Neftekod.exe") -ArgumentList @("--smoke-report", "`"$smokeReport`"") -WorkingDirectory $destination -WindowStyle Hidden -Wait -PassThru
        if ($process.ExitCode -ne 0 -or -not (Test-Path -LiteralPath $smokeReport)) { throw "Portable EXE verification failed" }
        $smoke = Get-Content -Raw -LiteralPath $smokeReport | ConvertFrom-Json
        if ($smoke.error -or $smoke.tk -ne "ok" -or $smoke.history.points -ne 2 -or
            -not $smoke.history_chart.rendered -or $smoke.what_if.editor -ne "ok") {
            throw "Portable smoke report is incomplete"
        }
    } finally {
        $env:PATH = $savedPath
        $env:PYTHONPATH = $savedPythonPath
    }
}

$commit = git -c "safe.directory=$($root -replace '\\', '/')" -C $root rev-parse HEAD
if ($LASTEXITCODE -ne 0) { throw "Cannot read source commit" }
$commit = $commit.Trim()
$status = git -c "safe.directory=$($root -replace '\\', '/')" -C $root status --porcelain
if ($LASTEXITCODE -ne 0) { throw "Cannot read source status" }
$dirty = [bool]$status
$releaseMetadata = [ordered]@{
    schema_version = "1.0"
    release_id = $manifest.release_id
    source_commit = $commit
    source_tree_dirty = $dirty
    generated_at_utc = [DateTime]::UtcNow.ToString("o")
    pinned_dataset = $manifest.prepared_dataset
    pinned_forecast = $manifest.forecast_artifact
    pinned_v2 = $manifest.v2_artifact
    supports_actions = $false
    offline_scope = $(if ($BinaryDirectory) { "Windows x64 portable runtime included; no Python installation or network required for execution" } else { "works without external network after dependencies are installed" })
}
$releaseMetadata | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath (Join-Path $destination "RELEASE_MANIFEST.json") -Encoding UTF8

$hashes = Get-ChildItem -LiteralPath $destination -Recurse -File |
    Where-Object { $_.Name -notin @("SHA256SUMS.txt") } |
    Sort-Object FullName |
    ForEach-Object {
        $relative = $_.FullName.Substring($destination.Length).TrimStart("\", "/") -replace "\\", "/"
        "{0}  {1}" -f (Get-FileHash -Algorithm SHA256 -LiteralPath $_.FullName).Hash.ToLowerInvariant(), $relative
    }
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllLines((Join-Path $destination "SHA256SUMS.txt"), [string[]]$hashes, $utf8NoBom)

if ($Archive) {
    $archivePath = "$destination.zip"
    Compress-Archive -Path (Join-Path $destination "*") -DestinationPath $archivePath -CompressionLevel Optimal
    Write-Output $archivePath
} else {
    Write-Output $destination
}
