<#
.SYNOPSIS
  Local (this-PC) benchmark sweep for HOOPS AI 1.1 steps 1/2/3.

.DESCRIPTION
  Redesigned, heavy-assembly-free benchmark tuned for this workstation
  (i9-13900H, 14 physical / 20 logical cores, ~32 GB RAM, RTX 2000 Ada).

  All three steps use the SAME failure-free 500-part sample
  (filelists/parts500.txt, the in-repo bench_parts dataset) and a per-file
  time_limit of 120 s. Every run is checked for status=OK and n_failed=0.

  Step 1  encoding    : max_workers sweep on CPU -> pick best -> confirm on GPU.
                        (Step 1 is CPU-bound B-rep encoding; GPU is expected to
                        match CPU. num_workers=1 is intentionally EXCLUDED
                        because that path behaves differently, e.g. the per-file
                        time limit is not enforced.)
  Step 2  training    : the current batch_size x num_workers x accel matrix.
  Step 3  indexing    : num_workers sweep, CPU and GPU (GPU is VRAM-limited).

.NOTES
  Edit the CONFIG block below to change venvs, sweep values, or the sample.
  Use -DryRun to print the plan without running anything.
#>
[CmdletBinding()]
param(
    # HOOPS AI SDK install folder names vary by machine (e.g. "V1.1" vs
    # "CPU1.1"). Leave these blank to resolve from CPU_PY/GPU_PY in .env
    # (copy .env.example and edit it once), or pass them explicitly here.
    [string]  $CpuPy    = "",
    [string]  $GpuPy    = "",
    [string]  $Ckpt     = "",
    [string]  $FileList = "filelists\parts500.txt",
    [double]  $TimeLimit = 120,
    # Steps to run, any subset of 1,2,3.
    [int[]]   $Steps    = @(1, 2, 3),
    # Step 1 max_workers sweep (NO 1, NO 2 - low counts add little; base = 4).
    [int[]]   $Step1Workers = @(4, 6, 8, 10, 12, 14, 16, 18),
    [int[]]   $Step1WorkersGpu = @(8, 10, 12, 14, 16),
    # Step 3 num_workers sweeps (GPU is capped by ~2 GB/worker model on 8 GB VRAM).
    [int[]]   $Step3WorkersCpu = @(8, 10, 12, 14, 16, 18),
    [int[]]   $Step3WorkersGpu = @(8, 10, 12, 14, 16, 18),
    [switch]  $DryRun,
    [switch]  $SaveIndex,
    # Step 1 only: run the CPU max_workers sweep and stop (skip KEEP + GPU
    # confirm). Use this to observe the peak/plateau before deciding Step 3.
    [switch]  $Step1SweepOnly
)

$ErrorActionPreference = "Continue"
$Bench   = Split-Path -Parent $MyInvocation.MyCommand.Path
. (Join-Path $Bench "resolve_local_env.ps1")
$CsvPath = Join-Path $Bench "results\results.csv"
$LogDir  = Join-Path $Bench "logs"
$Stamp   = Get-Date -Format "yyyyMMdd_HHmmss"
New-Item -ItemType Directory -Force -Path $LogDir, (Split-Path $CsvPath) | Out-Null

$CpuTag = "cpu1.1"
$GpuTag = "gpu1.1"
$Global:Warnings = @()

# --------------------------------------------------------------------------
function Say([string]$msg, [string]$color = "Gray") { Write-Host $msg -ForegroundColor $color }

function Section([string]$title) {
    Say ""
    Say ("=" * 78) "Cyan"
    Say "  $title" "Cyan"
    Say ("=" * 78) "Cyan"
}

function Count-Rows {
    if (Test-Path $CsvPath) { @(Import-Csv $CsvPath).Count } else { 0 }
}

# Run one python measurement, tee to a log, then verify the row(s) it appended.
function Invoke-Run {
    param(
        [string]$Label,
        [string]$Python,
        [string]$Script,
        [string[]]$PyArgs,
        [hashtable]$EnvVars = @{}
    )
    $log = Join-Path $LogDir "$Stamp-$($Label -replace '[^\w.-]','_').log"
    Say ""
    Say ">> $Label" "White"
    Say "   $Python $Script $($PyArgs -join ' ')" "DarkGray"
    if ($DryRun) { Say "   [dry-run] not executed" "DarkGray"; return }
    if (-not (Test-Path $Python)) { Say "   [SKIP] python missing: $Python" "Yellow"; $Global:Warnings += "$Label : python missing"; return }

    $before = Count-Rows
    $applied = @{}
    foreach ($k in $EnvVars.Keys) { $applied[$k] = [Environment]::GetEnvironmentVariable($k); Set-Item "env:$k" $EnvVars[$k] }
    try {
        & $Python $Script @PyArgs 2>&1 | Tee-Object -FilePath $log
        $code = $LASTEXITCODE
    } finally {
        foreach ($k in $EnvVars.Keys) {
            if ($null -eq $applied[$k]) { Remove-Item "env:$k" -ErrorAction SilentlyContinue }
            else { Set-Item "env:$k" $applied[$k] }
        }
    }

    if ($code -ne 0) { Say "   [FAIL] exit=$code (log: $log)" "Red"; $Global:Warnings += "$Label : exit=$code"; return }

    # Inspect appended row(s).
    $after = Count-Rows
    if ($after -le $before) { Say "   [WARN] no results row appended" "Yellow"; $Global:Warnings += "$Label : no row"; return }
    $new = @(Import-Csv $CsvPath)[$before..($after - 1)]
    foreach ($r in $new) {
        $nf = 0; [void][int]::TryParse("$($r.n_failed)", [ref]$nf)
        if ($r.status -ne "OK") { Say "   [WARN] status=$($r.status)" "Yellow"; $Global:Warnings += "$Label : status=$($r.status)" }
        elseif ($nf -gt 0)      { Say "   [WARN] n_failed=$nf" "Yellow";        $Global:Warnings += "$Label : n_failed=$nf" }
        else                    { Say "   [OK] wall=$($r.wall_s)s n_ok=$($r.n_ok) n_failed=$nf" "Green" }
    }
}

# Pick the fastest Step-1 CPU sweep point that had zero failures.
function Get-Step1Best([int]$fallback = 8) {
    if (-not (Test-Path $CsvPath)) { return $fallback }
    $rows = @(Import-Csv $CsvPath) | Where-Object {
        $_.step -eq "dataprep" -and $_.phase -eq "S1" -and $_.env_tag -eq $CpuTag -and
        $_.status -eq "OK" -and ("$($_.n_failed)" -eq "0" -or "$($_.n_failed)" -eq "")
    }
    if (-not $rows) { return $fallback }
    $scored = foreach ($r in $rows) {
        $enc = $null
        try { $enc = ($r.sub_timings | ConvertFrom-Json).EncodingTask } catch { }
        if (-not $enc) { $enc = [double]$r.wall_s }
        [pscustomobject]@{ mw = [int]$r.param_value; enc = [double]$enc }
    }
    ($scored | Sort-Object enc | Select-Object -First 1).mw
}

# --------------------------------------------------------------------------
Section "HOOPS AI 1.1 local sweep  (parts500, time_limit=$TimeLimit s)"
Say "bench dir : $Bench"
Say "results   : $CsvPath"
Say "logs      : $LogDir"
Say "filelist  : $FileList"
Say "ckpt      : $(if ($Ckpt) { $Ckpt } else { '(auto-resolve)' })  $(if ($Ckpt -and (Test-Path $Ckpt)) {'[ok]'} elseif (-not $Ckpt) {''} else {'[MISSING]'})"
Say "CPU venv  : $CpuPy  $(if (Test-Path $CpuPy) {'[ok]'} else {'[MISSING]'})"
Say "GPU venv  : $GpuPy  $(if (Test-Path $GpuPy) {'[ok]'} else {'[MISSING]'})"
Say "steps     : $($Steps -join ', ')"
if ($DryRun) { Say "MODE      : DRY RUN (nothing is executed)" "Yellow" }

# ==========================================================================
# STEP 1  -  encoding: CPU max_workers sweep -> best -> GPU confirm + KEEP
# ==========================================================================
if ($Steps -contains 1) {
    Section "STEP 1  encoding - CPU max_workers sweep $($Step1Workers -join ',')"
    foreach ($mw in $Step1Workers) {
        Invoke-Run -Label "s1-cpu-mw$mw" -Python $CpuPy -Script "bench_step1_dataprep.py" -PyArgs @(
            "--env-tag", $CpuTag, "--max-workers", "$mw", "--filelist", $FileList,
            "--phase", "S1", "--time-limit-s", "$TimeLimit", "--note", "local-sweep")
    }

    Section "STEP 1  encoding - GPU max_workers sweep $($Step1WorkersGpu -join ',')"
    foreach ($mw in $Step1WorkersGpu) {
        Invoke-Run -Label "s1-gpu-mw$mw" -Python $GpuPy -Script "bench_step1_dataprep.py" -PyArgs @(
            "--env-tag", $GpuTag, "--max-workers", "$mw", "--filelist", $FileList,
            "--phase", "S1", "--time-limit-s", "$TimeLimit", "--note", "gpu-sweep")
    }

    $best = if ($DryRun) { 8 } else { Get-Step1Best 8 }
    Say ""
    Say "STEP 1 best CPU max_workers = $best (fastest EncodingTask, 0 failures)" "Green"

    if ($Step1SweepOnly) {
        Say ""
        Say "STEP 1 sweep-only: skipping KEEP (rerun without -Step1SweepOnly to materialise datasets for Step 2)." "Yellow"
    } else {
        # KEEP the encoded dataset for Step 2 (one per env-tag) at the best worker
        # count. Both KEEP runs are tagged S1K so the report drops them; the S1
        # sweeps above already carry the plotted CPU and GPU points.
        Invoke-Run -Label "s1-cpu-mw$best-KEEP" -Python $CpuPy -Script "bench_step1_dataprep.py" -PyArgs @(
            "--env-tag", $CpuTag, "--max-workers", "$best", "--filelist", $FileList,
            "--phase", "S1K", "--time-limit-s", "$TimeLimit", "--keep-output", "--note", "keep-best")

        Invoke-Run -Label "s1-gpu-mw$best-KEEP" -Python $GpuPy -Script "bench_step1_dataprep.py" -PyArgs @(
            "--env-tag", $GpuTag, "--max-workers", "$best", "--filelist", $FileList,
            "--phase", "S1K", "--time-limit-s", "$TimeLimit", "--keep-output", "--note", "keep-best")
    }
}

# ==========================================================================
# STEP 2  -  training: current batch_size x num_workers x accel matrix
# ==========================================================================
if ($Steps -contains 2) {
    Section "STEP 2  training - batch matrix"
    $needCpu = Join-Path $Bench "results\dataset_$CpuTag.json"
    $needGpu = Join-Path $Bench "results\dataset_$GpuTag.json"
    if (-not $DryRun) {
        if (-not (Test-Path $needCpu)) { Say "  [SKIP CPU rows] missing $needCpu (run Step 1 first)" "Yellow" }
        if (-not (Test-Path $needGpu)) { Say "  [SKIP GPU rows] missing $needGpu (run Step 1 first)" "Yellow" }
    }

    # Core question: same-condition CPU vs GPU throughput, and the batch_size
    # that maximises each. So this is a batch_size SWEEP (like the Step1/Step3
    # worker sweeps), run once on the real CPU install (cpu1.1, torch+cpu) and
    # once on the real GPU install (gpu1.1, torch+cu130). Everything else is
    # pinned: num_workers=0 (best on Windows), matmul=high (the notebook's
    # default), from scratch, 10 epochs, early_stopping off.
    # A few secondary rows probe TF32, the DataLoader-worker penalty and epoch
    # linearity; they are kept out of the batch-sweep tables.
    # env-tag, accelerator, batch_size, num_workers, epochs, matmul, extra
    $matrix = @(
        @($CpuTag, "cpu",   8,  0, 10, "high",    ""),   # --- CPU batch sweep ---
        @($CpuTag, "cpu",  16,  0, 10, "high",    ""),
        @($CpuTag, "cpu",  32,  0, 10, "high",    ""),
        @($CpuTag, "cpu",  64,  0, 10, "high",    ""),
        @($CpuTag, "cpu", 128,  0, 10, "high",    ""),
        @($CpuTag, "cpu", 256,  0, 10, "high",    ""),   # full-batch (165 train samples)
        @($GpuTag, "gpu",   8,  0, 10, "high",    ""),   # --- GPU batch sweep ---
        @($GpuTag, "gpu",  16,  0, 10, "high",    ""),
        @($GpuTag, "gpu",  32,  0, 10, "high",    ""),
        @($GpuTag, "gpu",  64,  0, 10, "high",    ""),
        @($GpuTag, "gpu", 128,  0, 10, "high",    ""),
        @($GpuTag, "gpu", 256,  0, 10, "high",    ""),   # full-batch (165 train samples)
        @($GpuTag, "gpu",  64,  0, 10, "highest", ""),   # TF32 off (control)
        @($GpuTag, "gpu",  64,  4, 10, "high",    ""),   # DataLoader worker penalty
        @($GpuTag, "gpu",  64,  0,  5, "high",    "")    # epoch-linearity 2nd point
    )
    foreach ($m in $matrix) {
        $tag = $m[0]; $accel = $m[1]; $bs = $m[2]; $nw = $m[3]; $ep = $m[4]; $mm = $m[5]; $extra = $m[6]
        $py = if ($tag -eq $GpuTag) { $GpuPy } else { $CpuPy }
        $ptr = Join-Path $Bench "results\dataset_$tag.json"
        if (-not $DryRun -and -not (Test-Path $ptr)) { continue }
        $a = @("--env-tag", $tag, "--accelerator", $accel, "--batch-size", "$bs",
               "--num-workers", "$nw", "--no-warm-start", "--max-epochs", "$ep",
               "--matmul-precision", $mm, "--phase", "S2")
        if ($extra) { $a += $extra }
        $lbl = "s2-$tag-$accel-bs$bs-nw$nw-ep$ep-$mm$(if($extra){'-test'})"
        Invoke-Run -Label $lbl -Python $py -Script "bench_step2_training.py" -PyArgs $a
    }
}

# ==========================================================================
# STEP 3  -  indexing: num_workers sweep, CPU and GPU
# ==========================================================================
if ($Steps -contains 3) {
    Section "STEP 3  indexing - num_workers sweep (CPU $($Step3WorkersCpu -join ','); GPU $($Step3WorkersGpu -join ','))"
    $saveFlag = if ($SaveIndex) { @("--save-index") } else { @() }
    # Only pass --ckpt if we actually have one; otherwise let bench_step3
    # auto-resolve it (searches */packages/trained_ml_models/ itself).
    $ckptFlag = if ($Ckpt) { @("--ckpt", $Ckpt) } else { @() }

    foreach ($nw in $Step3WorkersCpu) {
        Invoke-Run -Label "s3-cpu-nw$nw" -Python $CpuPy -Script "bench_step3_indexing.py" -EnvVars @{ BENCH_NO_CUDA_MASK = "1" } -PyArgs (@(
            "--env-tag", $CpuTag, "--num-workers", "$nw", "--filelist", $FileList,
            "--time-limit", "$TimeLimit", "--phase", "S3", "--note", "cpu-sweep") + $ckptFlag + $saveFlag)
    }
    foreach ($nw in $Step3WorkersGpu) {
        Invoke-Run -Label "s3-gpu-nw$nw" -Python $GpuPy -Script "bench_step3_indexing.py" -PyArgs (@(
            "--env-tag", $GpuTag, "--num-workers", "$nw", "--filelist", $FileList,
            "--time-limit", "$TimeLimit", "--phase", "S3", "--note", "gpu-sweep") + $ckptFlag + $saveFlag)
    }
}

# ==========================================================================
Section "DONE"
if ($DryRun) {
    Say "Dry run complete - no measurements were taken." "Yellow"
} elseif ($Global:Warnings.Count -eq 0) {
    Say "All runs completed with status=OK and n_failed=0." "Green"
} else {
    Say "Completed with $($Global:Warnings.Count) warning(s):" "Yellow"
    $Global:Warnings | ForEach-Object { Say "  - $_" "Yellow" }
}
Say "Results CSV : $CsvPath"
Say "Report      : run  $CpuPy make_report.py  to regenerate results\REPORT.md/html" "DarkGray"
