# Compare SHA256 of critical root vs YOLO_DOCKER copies. Exit 1 on mismatch.
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..\\..")).Path
$Docker = Join-Path $Root "YOLO_DOCKER"

$pairs = @(
    @{ Rel = "api\jobs.py" },
    @{ Rel = "app\core\cross_check.py" },
    @{ Rel = "app\core\gpu_cleanup.py" },
    @{ Rel = "app\core\video_processor.py" },
    @{ Rel = "app\core\progress_hook.py" },
    @{ Rel = "app\core\detect_engine.py" },
    @{ Rel = "app\config\ui_fast_profile.json" },
    @{ Rel = "config\ui_fast_profile.json"; RootOnly = $false; DockerRel = "config\ui_fast_profile.json"; SrcRel = "app\config\ui_fast_profile.json" },
    @{ Rel = "app\core\video_encode.py" },
    @{ Rel = "app\core\ffmpeg_utils.py" },
    @{ Rel = "app\core\pipeline.py" },
    @{ Rel = "app\config\settings.py" }
)

function Get-FileSha256([string]$Path) {
    if (-not (Test-Path -LiteralPath $Path)) {
        return $null
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash
}

$failed = $false
Write-Host "Root:   $Root"
Write-Host "Docker: $Docker"
Write-Host ""

foreach ($p in $pairs) {
    $rel = $p.Rel
    $srcRel = if ($p.SrcRel) { $p.SrcRel } else { $rel }
    $dstRel = if ($p.DockerRel) { $p.DockerRel } else { $rel }
    $src = Join-Path $Root $srcRel
    $dst = Join-Path $Docker $dstRel
    $srcHash = Get-FileSha256 $src
    $dstHash = Get-FileSha256 $dst

    if ($null -eq $srcHash) {
        Write-Host "MISSING root: $rel" -ForegroundColor Red
        $failed = $true
        continue
    }
    if ($null -eq $dstHash) {
        Write-Host "MISSING docker: $rel" -ForegroundColor Red
        $failed = $true
        continue
    }

    if ($srcHash -eq $dstHash) {
        Write-Host "OK  $dstRel"
    } else {
        Write-Host "MISMATCH $dstRel" -ForegroundColor Red
        Write-Host "  root:   $srcHash"
        Write-Host "  docker: $dstHash"
        $failed = $true
    }
}

if ($failed) {
    Write-Host ""
    Write-Host "verify_sync: FAILED" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "verify_sync: OK (all pairs match)"
exit 0
