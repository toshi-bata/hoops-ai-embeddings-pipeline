# Shared helper, dot-sourced by run_benchmark.ps1 / preflight.ps1 /
# run_local_sweep.ps1 right after their own param block, e.g.:
#
#   . (Join-Path $PSScriptRoot "resolve_local_env.ps1")
#
# Resolves $CpuPy / $GpuPy / $Ckpt in this order of precedence:
#   1. an explicit -CpuPy / -GpuPy / -Ckpt argument (already set by the
#      caller's own param block -- this script only fills in what's blank)
#   2. CPU_PY / GPU_PY / HOOPS_AI_CKPT in .env
#   3. neither -> print what to do and exit, rather than silently guessing
#      a folder name that may not exist on this machine.
#
# .env is the SAME file bench_common.py reads for HOOPS_AI_LICENSE (Python's
# load_dotenv() picks up CPU_PY / GPU_PY / HOOPS_AI_CKPT from it too, since
# it loads every KEY=VALUE pair, not just the license) -- one file, read by
# both the PowerShell orchestrators and every bench_step*.py script,
# whichever you invoke directly. Search order matches load_dotenv() exactly:
# ./.env, ../.env, ../CPU1.1/.env, ../GPU1.1/.env (first one found wins).
#
# Example .env entries (alongside HOOPS_AI_LICENSE):
#   CPU_PY='C:\SDK\HOOPS_AI\V1.1\.venv\Scripts\python.exe'
#   GPU_PY='C:\SDK\HOOPS_AI\GPU1.1\.venv\Scripts\python.exe'
#   HOOPS_AI_CKPT='C:\SDK\HOOPS_AI\V1.1\packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt'

function Read-DotEnv([string]$RepoRoot) {
    $candidates = @(
        (Join-Path $RepoRoot ".env"),
        (Join-Path (Split-Path -Parent $RepoRoot) ".env"),
        (Join-Path (Split-Path -Parent $RepoRoot) "CPU1.1\.env"),
        (Join-Path (Split-Path -Parent $RepoRoot) "GPU1.1\.env")
    )
    foreach ($path in $candidates) {
        if (-not (Test-Path $path)) { continue }
        $vals = @{}
        foreach ($raw in Get-Content -Path $path -Encoding UTF8) {
            $line = $raw.Trim()
            if (-not $line -or $line.StartsWith("#") -or -not $line.Contains("=")) { continue }
            $key, $val = $line -split "=", 2
            $key = $key.Trim()
            $val = $val.Trim().Trim("'").Trim('"')
            if ($val -match " #") { $val = ($val -split " #", 2)[0].Trim() }
            $vals[$key] = $val
        }
        return $vals
    }
    return @{}
}

$dotenv = Read-DotEnv -RepoRoot $PSScriptRoot
if (-not $CpuPy -and $dotenv.ContainsKey("CPU_PY")) { $CpuPy = $dotenv["CPU_PY"] }
if (-not $GpuPy -and $dotenv.ContainsKey("GPU_PY")) { $GpuPy = $dotenv["GPU_PY"] }
if (-not $Ckpt -and $dotenv.ContainsKey("HOOPS_AI_CKPT")) { $Ckpt = $dotenv["HOOPS_AI_CKPT"] }

if (-not $CpuPy -or -not $GpuPy) {
    Write-Host "CpuPy and/or GpuPy are not set." -ForegroundColor Red
    Write-Host "HOOPS AI SDK install folder names vary by machine (e.g. 'V1.1' vs 'CPU1.1')," -ForegroundColor Red
    Write-Host "so this repo does not guess them. Either:" -ForegroundColor Red
    Write-Host "  - pass -CpuPy <path> -GpuPy <path> explicitly, or" -ForegroundColor Red
    Write-Host "  - add CPU_PY=<path> / GPU_PY=<path> to your .env file" -ForegroundColor Red
    Write-Host "    (the same file HOOPS_AI_LICENSE lives in)." -ForegroundColor Red
    exit 1
}
if (-not (Test-Path $CpuPy)) { Write-Host "WARNING: CpuPy not found: $CpuPy" -ForegroundColor Yellow }
if (-not (Test-Path $GpuPy)) { Write-Host "WARNING: GpuPy not found: $GpuPy" -ForegroundColor Yellow }
