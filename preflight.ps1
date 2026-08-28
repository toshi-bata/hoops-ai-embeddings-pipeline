<#
.SYNOPSIS
  Two-minute preflight: prove that steps 2 and 3 can actually reach hoops_ai
  before committing to a multi-hour run.

.DESCRIPTION
  The previous overnight session lost phases 3 and 4 - 25 measurements - to a
  single missing hoops_ai.set_license() call. The smoke test in phase 0 only
  exercised step 1, which happened to activate the license as a side effect of
  importing bench_tasks, so the fault was invisible until the real run.

  This script checks each runner along the path that actually failed:
    * license activation in a fresh interpreter (both venvs)
    * the encoded dataset that step 2 depends on
    * the checkpoint that steps 2 and 3 depend on
    * one real training epoch and one real 5-file embedding run

  Anything that would have killed the overnight run shows up here in minutes.
#>
[CmdletBinding()]
param(
    # HOOPS AI SDK install folder names vary by machine (e.g. "V1.1" vs
    # "CPU1.1"). Leave these blank to resolve from CPU_PY/GPU_PY in .env
    # (copy .env.example and edit it once), or pass them explicitly here.
    [string]$CpuPy = "",
    [string]$GpuPy = ""
)

$ErrorActionPreference = "Continue"
$Bench = $PSScriptRoot
. (Join-Path $Bench "resolve_local_env.ps1")
$Res   = Join-Path $Bench "results"
$Fail  = 0

function Say([string]$m, [string]$c = "Gray") { Write-Host $m -ForegroundColor $c }
function Check([string]$name, [bool]$ok, [string]$detail = "") {
    if ($ok) { Say ("  PASS  " + $name) "Green" }
    else { Say ("  FAIL  " + $name) "Red"; if ($detail) { Say ("        " + $detail) "Red" }; $script:Fail++ }
}

Say ""
Say "PREFLIGHT - checks the failure path from the last session" "Cyan"
Say ("=" * 70) "DarkCyan"

# ---------------------------------------------------------------- 1. license
Say ""
Say "1. hoops_ai license activation (the bug that cost the last run)" "White"
$licTest = @'
import sys, os
sys.path.insert(0, r"__BENCH__")
from bench_common import apply_license
try:
    apply_license()
except SystemExit as e:
    print("LICFAIL " + str(e)); sys.exit(1)
except Exception as e:
    print("LICFAIL " + repr(e)); sys.exit(1)
# Now touch a real API the way step 2 and step 3 do.
try:
    from hoops_ai.dataset import DatasetLoader          # noqa: F401
    from hoops_ai.ml.embeddings import HOOPSEmbeddings  # noqa: F401
    from hoops_ai.ml import CADSearch                   # noqa: F401
    print("LICOK")
except Exception as e:
    print("LICFAIL " + repr(e)[:300]); sys.exit(1)
'@
$licTest = $licTest.Replace("__BENCH__", $Bench)
$tmp = Join-Path $env:TEMP "bench_preflight_lic.py"
Set-Content -Path $tmp -Value $licTest -Encoding UTF8

foreach ($pair in @(@("gpu1.1", $GpuPy), @("cpu1.1", $CpuPy))) {
    $tag = $pair[0]; $py = $pair[1]
    if (-not (Test-Path $py)) { Check "$tag interpreter present" $false $py; continue }
    $out = & $py $tmp 2>&1
    $ok = ($out | Select-String -Pattern 'LICOK' -Quiet)
    $err = ($out | Select-String -Pattern 'LICFAIL' | Select-Object -First 1)
    Check "$tag can activate the license and import hoops_ai APIs" $ok `
        ($(if ($err) { $err.Line } else { ($out | Select-Object -Last 3) -join ' | ' }))
}
Remove-Item $tmp -ErrorAction SilentlyContinue

# ---------------------------------------------------------------- 2. artifacts
Say ""
Say "2. Artifacts that phases 3 and 4 depend on" "White"
foreach ($tag in @("gpu1.1", "cpu1.1")) {
    $ptr = Join-Path $Res "dataset_$tag.json"
    if (-not (Test-Path $ptr)) {
        Check "$tag encoded dataset pointer" $false "missing $ptr - run phase 2"
        continue
    }
    $d = Get-Content $ptr -Raw | ConvertFrom-Json
    $haveDs = Test-Path $d.dataset
    $haveIs = Test-Path $d.infoset
    Check "$tag .dataset exists ($($d.n_files) files)" $haveDs $d.dataset
    Check "$tag .infoset exists" $haveIs $d.infoset
}

$ckptFound = $false
$ckptCandidates = @()
if ($Ckpt) { $ckptCandidates += $Ckpt }
# venv python.exe is <sdk-dir>\.venv\Scripts\python.exe -> <sdk-dir>\packages\...
foreach ($py in @($CpuPy, $GpuPy)) {
    if (-not $py) { continue }
    $sdkDir = Split-Path (Split-Path (Split-Path $py))
    $ckptCandidates += Join-Path $sdkDir "packages\trained_ml_models\ts3d_2M_hoops_embeddings_SIGNAL-preview.ckpt"
}
foreach ($p in $ckptCandidates) {
    if (Test-Path $p) { $ckptFound = $true; Say ("        using " + $p) "DarkGray"; break }
}
Check "SIGNAL checkpoint resolvable" $ckptFound "searched: $($ckptCandidates -join ', ')"

# ---------------------------------------------------------------- 3. step 2
Say ""
Say "3. One real training epoch (this is what phase 3 does 15 times)" "White"
Push-Location $Bench
$log2 = Join-Path $Bench "logs\preflight-step2.log"
& $GpuPy "bench_step2_training.py" --env-tag gpu1.1 --accelerator gpu `
    --batch-size 8 --num-workers 0 --new-epochs 1 --phase preflight `
    --note "preflight - excluded from analysis" *> $log2
$ok2 = (Select-String -Path $log2 -Pattern '\[bench\] recorded:.*\(OK\)' -Quiet -ErrorAction SilentlyContinue)
Pop-Location
$tail2 = if (Test-Path $log2) { (Get-Content $log2 -Tail 4) -join ' | ' } else { "no log" }
Check "step 2 completes 1 epoch on the GPU" $ok2 $tail2

# ---------------------------------------------------------------- 4. step 3
Say ""
Say "4. One real embedding run (this is what phase 4 does 10 times)" "White"
Push-Location $Bench
$smoke = Join-Path $Bench "filelists\smoke5.txt"
if (-not (Test-Path $smoke)) {
    $first5 = Get-Content (Join-Path $Bench "filelists\clean485.txt") -TotalCount 5
    [System.IO.File]::WriteAllLines($smoke, $first5)
}
$log3 = Join-Path $Bench "logs\preflight-step3.log"
& $GpuPy "bench_step3_indexing.py" --env-tag gpu1.1 --num-workers 2 `
    --filelist "filelists/smoke5.txt" --phase preflight `
    --note "preflight - excluded from analysis" *> $log3
$ok3 = (Select-String -Path $log3 -Pattern '\[bench\] recorded:.*\(OK\)' -Quiet -ErrorAction SilentlyContinue)
Pop-Location
$tail3 = if (Test-Path $log3) { (Get-Content $log3 -Tail 4) -join ' | ' } else { "no log" }
Check "step 3 embeds 5 files and builds a FAISS index" $ok3 $tail3

# ---------------------------------------------------------------- verdict
Say ""
Say ("=" * 70) "DarkCyan"
if ($Fail -eq 0) {
    Say "ALL CHECKS PASSED - phases 3 and 4 will reach hoops_ai this time." "Green"
    Say ""
    Say "Start the run:" "Cyan"
    Say "  .\run_benchmark.ps1 -Phases 1,3,4,5 -Unattended" "White"
    Say ""
    Say "(Phase 2 can be skipped: the 485-file datasets from the last run are" "DarkGray"
    Say " still on disk and valid. Include 2 if you want it re-measured with" "DarkGray"
    Say " the warm-up fix.)" "DarkGray"
    exit 0
} else {
    Say "$Fail CHECK(S) FAILED - fix these before starting a long run." "Red"
    Say "Logs: $($Bench)\logs\preflight-step2.log, preflight-step3.log" "Yellow"
    exit 1
}
