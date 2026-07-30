# Copy core + shared api/jobs into YOLO_DOCKER (Docker image: app/, api/, config/ only).
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Docker = Join-Path $Root "YOLO_DOCKER"

$files = @(
    "api\jobs.py",
    "app\core\cross_check.py",
    "app\core\gpu_cleanup.py",
    "app\core\video_processor.py",
    "app\core\detect_engine.py",
    "app\core\progress_hook.py",
    "app\core\video_encode.py",
    "app\core\pipeline.py",
    "app\config\settings.py",
    "app\config\ui_fast_profile.json"
)

Write-Host "Sync root -> YOLO_DOCKER"
foreach ($rel in $files) {
    $src = Join-Path $Root $rel
    $dst = Join-Path $Docker $rel
    if (-not (Test-Path -LiteralPath $src)) {
        Write-Host "SKIP missing: $rel"
        continue
    }
    $dir = Split-Path -Parent $dst
    if (-not (Test-Path $dir)) { New-Item -ItemType Directory -Path $dir -Force | Out-Null }
    Copy-Item -LiteralPath $src -Destination $dst -Force
    Write-Host "  $rel"
}

$profileSrc = Join-Path $Root "app\config\ui_fast_profile.json"
$profileDst = Join-Path $Docker "config\ui_fast_profile.json"
Copy-Item -LiteralPath $profileSrc -Destination $profileDst -Force
Write-Host "  config\ui_fast_profile.json"

& (Join-Path $PSScriptRoot "verify_sync.ps1")
