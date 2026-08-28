<#
.SYNOPSIS
  HOOPS AI 1.1 benchmark: CPU1.1 vs GPU1.1 across the 3-step embeddings pipeline.

.DESCRIPTION
  Runs the whole measurement matrix unattended and appends every result to
  benchmark\results\results.csv. Each configuration runs in a fresh Python
  process so CUDA contexts, worker pools and allocator state never leak between
  measurements.

  A global wall-clock budget (default 10 h) and per-phase caps are enforced.
  When a phase runs out of time the remaining configurations are logged as
  SKIPPED and the harness moves on, so the run always terminates and always
  produces a report.

.PARAMETER Phases
  Comma-separated phases to run. Default "0,1,2,3,4,5".
    0 probe + file lists + smoke test
    1 step 1 encoding: max_workers sweep (100-file subset)
    2 step 1 encoding: full 485 files, both envs (produces datasets for 3 and 4)
    3 step 2 training: accelerator x batch_size x num_workers
    4 step 3 indexing: num_workers sweep + full index build
    5 report

.PARAMETER BudgetHours
  Hard global wall-clock budget. Default 10.

.PARAMETER DryRun
  Print the plan with time estimates and exit without running anything.

.PARAMETER Force
  Skip the "this deletes previous benchmark output" confirmation.

.PARAMETER Unattended
  Overnight mode. Implies -Force (never prompts) and suppresses Windows sleep
  for the duration of the run.

.PARAMETER RunTimeoutFactor
  Per-run hard timeout as a multiple of the estimate (default 3.0, floor
  estimate+15min). A run exceeding it has its process tree killed and is
  recorded as incomplete. This is what stops one hung measurement from eating
  the entire night, since the global budget is only checked between runs.

.PARAMETER AllowSleep
  Do not suppress Windows sleep.

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -DryRun
  powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -Unattended
  powershell -ExecutionPolicy Bypass -File .\run_benchmark.ps1 -Phases 3,4,5 -Unattended
#>
[CmdletBinding()]
param(
    [string]$Phases = "0,1,2,3,4,5",
    [double]$BudgetHours = 10,
    [switch]$DryRun,
    [switch]$Force,
    [switch]$Unattended,
    [double]$RunTimeoutFactor = 4.0,
    [switch]$AllowSleep,
    [int]$SubsetSize = 100,
    # HOOPS AI SDK install folder names vary by machine (e.g. "V1.1" vs
    # "CPU1.1"). Leave these blank to resolve from CPU_PY/GPU_PY in .env
    # (copy .env.example and edit it once), or pass them explicitly here.
    [string]$CpuPy = "",
    [string]$GpuPy = ""
)

$ErrorActionPreference = "Continue"
$Bench = $PSScriptRoot
. (Join-Path $Bench "resolve_local_env.ps1")

# -Unattended is the overnight mode: no prompts, and keep Windows awake.
if ($Unattended) { $Force = $true }
$LogDir  = Join-Path $Bench "logs"
$ResDir  = Join-Path $Bench "results"
$CsvPath = Join-Path $ResDir "results.csv"
$Stamp   = Get-Date -Format "yyyyMMdd-HHmmss"

$PhaseList = $Phases.Split(",") | ForEach-Object { $_.Trim() }
$Global:Deadline  = (Get-Date).AddHours($BudgetHours)
$Global:PhaseCaps = @{ "1" = 90; "2" = 60; "3" = 210; "4" = 150 }   # minutes
$Global:Skipped   = @()
$Global:Ran       = 0

New-Item -ItemType Directory -Force -Path $LogDir, $ResDir | Out-Null

function Say([string]$msg, [string]$color = "Gray") {
    Write-Host $msg -ForegroundColor $color
}

function Section([string]$title) {
    Say ""
    Say ("=" * 78) "DarkCyan"
    Say $title "Cyan"
    Say ("=" * 78) "DarkCyan"
}

function Fmt([double]$sec) {
    if ($sec -lt 90) { return ("{0:N0}s" -f $sec) }
    return ("{0:N1}m" -f ($sec / 60))
}

function RemainingSeconds { return ($Global:Deadline - (Get-Date)).TotalSeconds }

# --------------------------------------------------------------------------
# One measurement = one python process.
# --------------------------------------------------------------------------
function Invoke-Bench {
    param(
        [string]$Label,
        [string]$Python,
        [string]$Script,
        [string[]]$PyArgs,          # deliberately not named Args (automatic variable)
        [double]$EstimateSec,
        [string]$Phase,
        [datetime]$PhaseDeadline
    )

    $rem      = RemainingSeconds
    $phaseRem = ($PhaseDeadline - (Get-Date)).TotalSeconds

    if ($DryRun) {
        Say ("  PLAN  {0,-52} est {1,7}" -f $Label, (Fmt $EstimateSec))
        return
    }
    if ($EstimateSec -gt $rem) {
        Say ("  SKIP  {0,-52} est {1} > global remaining {2}" -f $Label, (Fmt $EstimateSec), (Fmt $rem)) "Yellow"
        $Global:Skipped += "P${Phase} $Label (global budget)"
        return
    }
    if ($EstimateSec -gt $phaseRem) {
        Say ("  SKIP  {0,-52} est {1} > phase remaining {2}" -f $Label, (Fmt $EstimateSec), (Fmt $phaseRem)) "Yellow"
        $Global:Skipped += "P${Phase} $Label (phase cap)"
        return
    }
    if (-not (Test-Path $Python)) {
        Say ("  FAIL  {0,-52} python not found: {1}" -f $Label, $Python) "Red"
        $Global:Skipped += "P${Phase} $Label (missing interpreter)"
        return
    }

    $safe = ($Label -replace '[^\w\.\-]', '_')
    $log  = Join-Path $LogDir "$Stamp-P$Phase-$safe.log"
    $errLog = "$log.err"

    # Hard per-run timeout. The global budget is only checked BETWEEN runs, so
    # without this a single hung measurement would silently consume the whole
    # night. Generous by design: 4x the estimate, at least estimate+30min, and
    # never longer than the remaining global budget.
    #
    # The floor was raised from 15 to 30 minutes because max_workers=16 was
    # killed at 1179 s in the previous session. Heavy oversubscription being
    # genuinely slow is a RESULT worth capturing, not a hang worth killing.
    $timeoutSec = [Math]::Max($EstimateSec * $RunTimeoutFactor, $EstimateSec + 1800)
    $timeoutSec = [Math]::Min($timeoutSec, [Math]::Max(120, $rem))

    Say ("  RUN   {0,-52} est {1}  kill after {2}" -f $Label, (Fmt $EstimateSec), (Fmt $timeoutSec)) "White"
    $t0 = Get-Date

    # Start-Process rather than the call operator: we need a handle we can wait
    # on with a timeout and then kill, which a pipeline into Tee-Object cannot
    # give us. Output goes to files; the heartbeat below keeps the console alive.
    #
    # -ArgumentList joins the array with spaces and does NOT quote, so any
    # argument containing a space (e.g. --note "env parity check") has to be
    # quoted here or the child process sees it as several arguments.
    # No argument we pass ever contains a double quote, so wrapping the ones
    # with whitespace is enough. [char]34 avoids nesting quote characters.
    $dq = [char]34
    $quoted = $PyArgs | ForEach-Object {
        if ($_ -match '\s') { $dq + $_ + $dq } else { $_ }
    }
    $proc = Start-Process -FilePath $Python `
        -ArgumentList (@($Script) + $quoted) `
        -WorkingDirectory $Bench -NoNewWindow -PassThru `
        -RedirectStandardOutput $log -RedirectStandardError $errLog

    # Touch .Handle so that .NET keeps the process handle open. Without this,
    # Start-Process -PassThru often yields $null for .ExitCode once the process
    # has ended, which previously made successful runs look like failures.
    try { $null = $proc.Handle } catch { }

    $killed = $false
    $nextBeat = 300
    while (-not $proc.HasExited) {
        Start-Sleep -Seconds 5
        $el = ((Get-Date) - $t0).TotalSeconds
        if ($el -ge $timeoutSec) {
            Say ("        TIMEOUT after {0} - killing process tree {1}" -f (Fmt $el), $proc.Id) "Red"
            # /T so the HOOPS AI worker processes die with the parent.
            & taskkill /PID $proc.Id /T /F 2>$null | Out-Null
            Start-Sleep -Seconds 5
            $killed = $true
            break
        }
        if ($el -ge $nextBeat) {
            Say ("        ... running {0} / limit {1}" -f (Fmt $el), (Fmt $timeoutSec)) "DarkGray"
            $nextBeat += 300
        }
    }

    # WaitForExit() before reading ExitCode: with -PassThru the property can be
    # unpopulated if we only polled HasExited.
    $code = $null
    if (-not $killed) {
        try { $proc.WaitForExit() } catch { }
        try { $code = $proc.ExitCode } catch { $code = $null }
    }
    $elapsed = ((Get-Date) - $t0).TotalSeconds
    $Global:Ran++

    # Fold stderr into the main log so one file per run is enough to debug.
    if ((Test-Path $errLog) -and (Get-Item $errLog).Length -gt 0) {
        Add-Content -Path $log -Value "`n----- stderr -----"
        Get-Content $errLog | Add-Content -Path $log
    }
    if (Test-Path $errLog) { Remove-Item $errLog -ErrorAction SilentlyContinue }

    # Independent success signal: the runners print "[bench] recorded:" after
    # they have written their CSV row. Exit-code plumbing through Start-Process
    # is not fully reliable, so trust our own marker when the code is missing.
    $recorded = $false
    if (Test-Path $log) {
        if (Select-String -Path $log -Pattern '\[bench\] recorded:' -Quiet -ErrorAction SilentlyContinue) {
            $recorded = $true
        }
    }

    if ($killed) {
        $Global:Skipped += "P${Phase} $Label (KILLED after $([int]$elapsed)s)"
        Say ("        killed - recorded as incomplete, moving on. log: {0}" -f (Split-Path $log -Leaf)) "Red"
    } elseif ($code -eq 0 -or ($null -eq $code -and $recorded)) {
        $suffix = if ($null -eq $code) { " (exit code unavailable, CSV row written)" } else { "" }
        Say ("        done in {0}  (est {1})  log: {2}{3}" -f (Fmt $elapsed), (Fmt $EstimateSec), (Split-Path $log -Leaf), $suffix) "Green"
    } elseif ($recorded) {
        Say ("        exit {0} but a CSV row was written - treating as done. log: {1}" -f $code, (Split-Path $log -Leaf)) "Yellow"
    } else {
        $shown = if ($null -eq $code) { "unknown" } else { "$code" }
        Say ("        FAILED (exit {0}) after {1}  -> see {2}" -f $shown, (Fmt $elapsed), $log) "Red"
    }
    Say ("        budget left: {0}" -f (Fmt (RemainingSeconds))) "DarkGray"
}

# --------------------------------------------------------------------------
# Read helpers
# --------------------------------------------------------------------------
function Get-EnvJson([string]$tag) {
    $p = Join-Path $ResDir "env_$tag.json"
    if (Test-Path $p) { return Get-Content $p -Raw | ConvertFrom-Json }
    return $null
}

function Get-BestMaxWorkers([int]$fallback = 12) {
    if (-not (Test-Path $CsvPath)) { return $fallback }
    $rows = Import-Csv $CsvPath | Where-Object {
        $_.step -eq "dataprep" -and $_.phase -eq "1" -and $_.status -eq "OK" -and $_.wall_s
    }
    if (-not $rows) { return $fallback }
    $best = $rows | Sort-Object { [double]$_.wall_s } | Select-Object -First 1
    return [int]$best.param_value
}

function Get-SecPerEpoch([string]$envTag, [string]$accel) {
    if (-not (Test-Path $CsvPath)) { return $null }
    $rows = Import-Csv $CsvPath | Where-Object {
        $_.step -eq "training" -and $_.env_tag -eq $envTag -and
        $_.accelerator -eq $accel -and $_.status -eq "OK" -and $_.sub_timings
    }
    if (-not $rows) { return $null }
    $vals = foreach ($r in $rows) {
        try { ($r.sub_timings | ConvertFrom-Json).s_per_epoch } catch { }
    }
    $vals = $vals | Where-Object { $_ -gt 0 }
    if (-not $vals) { return $null }
    return ($vals | Measure-Object -Minimum).Minimum
}

# --------------------------------------------------------------------------
# Banner + destructive-action confirmation
# --------------------------------------------------------------------------
Section "HOOPS AI 1.1 benchmark harness"
Say "bench dir     : $Bench"
Say "budget        : $BudgetHours h  (deadline $($Global:Deadline.ToString('yyyy-MM-dd HH:mm')))"
Say "phases        : $($PhaseList -join ', ')"
Say "results csv   : $CsvPath"
Say "logs          : $LogDir"
Say "CPU venv      : $CpuPy  $(if (Test-Path $CpuPy) {'[ok]'} else {'[MISSING]'})"
Say "GPU venv      : $GpuPy  $(if (Test-Path $GpuPy) {'[ok]'} else {'[MISSING]'})"

if (-not $DryRun -and -not $Force) {
    Say ""
    Say "WARNING - this run DELETES DATA. Please read before continuing:" "Yellow"
    Say "  * Each encoding measurement calls flow.process(clean_ouput_dir=True), which" "Yellow"
    Say "    WIPES its own flow directory under:" "Yellow"
    Say "      $Bench\out\<env>\flows\<flow_name>" "Yellow"
    Say "  * Repeated phases overwrite earlier benchmark artifacts in $Bench\out." "Yellow"
    Say "  * Nothing outside $Bench\out is deleted. Your CPU/GPU SDK install directories" "Yellow"
    Say "    are NOT touched, and the source CAD files (default: ..\screw next to this" "Yellow"
    Say "    repo; override with bench_step1_dataprep.py --source-dir) are only ever read." "Yellow"
    Say "  * Peak disk use of $Bench\out is roughly 3-6 GB." "Yellow"
    Say ""
    Say "  For an unattended overnight run use -Unattended, which accepts the above" "DarkGray"
    Say "  without prompting and keeps Windows from sleeping." "DarkGray"
    Say ""
    $ans = Read-Host "Type YES to proceed (anything else aborts)"
    if ($ans -ne "YES") { Say "Aborted - nothing was run or deleted." "Red"; exit 1 }
}

$RunStart = Get-Date

# --------------------------------------------------------------------------
# Keep Windows awake for the duration. Without this an overnight run stops
# the moment the machine sleeps, and you lose the night with no result.
# ES_CONTINUOUS | ES_SYSTEM_REQUIRED: system stays up, display may still blank.
# --------------------------------------------------------------------------
$Global:PowerHandle  = $null
$Global:PowerRestore = $null
if (-not $DryRun -and -not $AllowSleep) {
    # Attempt 1: SetThreadExecutionState. Cleanest, no side effects, but needs
    # Add-Type to compile C#, which is not available on every machine.
    try {
        if (-not ('Bench.PowerUtil' -as [type])) {
            $csharp = @"
using System;
using System.Runtime.InteropServices;
public class PowerUtil {
    [DllImport("kernel32.dll", SetLastError = true)]
    public static extern uint SetThreadExecutionState(uint esFlags);
}
"@
            Add-Type -TypeDefinition $csharp -ErrorAction Stop
        }
        # PowerShell parses 0x80000000 as Int32 (= -2147483648), so
        # "0x80000000 -bor 1" yields -2147483647 and cannot be cast to UInt32.
        # Use decimal literals, which parse as Int64 and cast cleanly.
        #   ES_CONTINUOUS       = 0x80000000 = 2147483648
        #   ES_SYSTEM_REQUIRED  = 0x00000001
        $r = [PowerUtil]::SetThreadExecutionState([uint32]2147483649)
        if ($r -eq 0) { throw "SetThreadExecutionState returned 0" }
        $Global:PowerHandle = $true
        Say "sleep         : suppressed via SetThreadExecutionState" "DarkGray"
    } catch {
        Say ("sleep         : SetThreadExecutionState unavailable ({0})" -f $_.Exception.Message) "DarkGray"
    }

    # Attempt 2 (fallback): change the active power scheme's sleep timeouts to
    # never, remembering the old values so they can be put back at the end.
    if (-not $Global:PowerHandle) {
        try {
            # Do NOT match on English labels: powercfg is localised, so
            # "Current AC Power Setting" does not exist on a Japanese Windows and
            # the original value silently came back $null.
            #
            # Match on structure instead. The STANDBYIDLE block lists, in order:
            #   min / max / increment / units, then current AC index, then current
            #   DC index - all as 0xXXXXXXXX. So AC is the second-to-last value.
            # (Taking the FIRST match would give the min, which is 0x00000000.)
            $out = (& powercfg /query SCHEME_CURRENT SUB_SLEEP STANDBYIDLE 2>$null) -join "`n"
            $hex = [regex]::Matches($out, '0x([0-9a-fA-F]{8})')
            $old = $null
            if ($hex.Count -ge 2) {
                $old = [Convert]::ToInt64($hex[$hex.Count - 2].Groups[1].Value, 16)
            }
            # 0xffffffff is not a real timeout; anything above a day is bogus too.
            if ($null -ne $old -and ($old -lt 0 -or $old -gt 86400)) { $old = $null }

            if ($null -eq $old) {
                throw "could not read the current AC standby timeout"
            }
            & powercfg /change standby-timeout-ac 0 2>$null | Out-Null
            if ($LASTEXITCODE -ne 0) { throw "powercfg exit $LASTEXITCODE" }
            $Global:PowerRestore = $old
            Say ("sleep         : suppressed via powercfg (AC standby was {0}s, restored on exit)" -f $old) "DarkGray"
        } catch {
            Say "sleep         : COULD NOT SUPPRESS SLEEP - set it manually before leaving:" "Yellow"
            Say "                powercfg /change standby-timeout-ac 0" "Yellow"
            Say "                (or Settings > System > Power > Screen and sleep > Never)" "Yellow"
        }
    }
}

# ==========================================================================
# PHASE 0 - probe, file lists, smoke test
# ==========================================================================
if ($PhaseList -contains "0") {
    Section "PHASE 0  environment probe / file lists / smoke test   (~15 min)"
    $pd = (Get-Date).AddMinutes(20)

    if (-not $DryRun) {
        foreach ($pair in @(@("gpu1.1", $GpuPy), @("cpu1.1", $CpuPy))) {
            $tag = $pair[0]; $py = $pair[1]
            if (-not (Test-Path $py)) { Say "  probe ${tag}: interpreter missing, skipped" "Yellow"; continue }
            Say "  probe $tag" "White"
            Push-Location $Bench
            $env:BENCH_ENV_TAG = $tag
            & $py "bench_common.py" *>&1 | Tee-Object -FilePath (Join-Path $LogDir "$Stamp-P0-probe-$tag.log")
            Pop-Location
        }
        Push-Location $Bench
        & $GpuPy "make_filelist.py" "--subset-size" "$SubsetSize" *>&1 |
            Tee-Object -FilePath (Join-Path $LogDir "$Stamp-P0-filelists.log")
        Pop-Location
    } else {
        Say "  PLAN  probe both venvs, build filelists, 5-file smoke test    est 15m"
    }

    # Smoke test: 5 files through step 1 so a broken setup fails in 2 minutes,
    # not 4 hours in.
    Push-Location $Bench
    if (-not $DryRun) {
        $smoke = Join-Path $Bench "filelists\smoke5.txt"
        $first5 = Get-Content (Join-Path $Bench "filelists\clean485.txt") -TotalCount 5
        # WriteAllLines, not Set-Content: PowerShell 5.1's "-Encoding UTF8" emits a
        # BOM that would corrupt the first path in the list.
        [System.IO.File]::WriteAllLines($smoke, $first5)
    }
    Pop-Location
    Invoke-Bench -Label "smoke step1 gpu mw4 n5" -Python $GpuPy `
        -Script "bench_step1_dataprep.py" -Phase "0" -PhaseDeadline $pd -EstimateSec 240 `
        -PyArgs @("--env-tag", "gpu1.1", "--max-workers", "4",
                "--filelist", "filelists/smoke5.txt", "--phase", "0", "--note", "smoke")
}

# ==========================================================================
# PHASE 1 - step 1 encoding, max_workers sweep on the 100-file subset
# ==========================================================================
if ($PhaseList -contains "1") {
    Section "PHASE 1  step 1 encoding - max_workers sweep (n=$SubsetSize)   cap 90 min"
    $pd = (Get-Date).AddMinutes($Global:PhaseCaps["1"])

    $envGpu = Get-EnvJson "gpu1.1"
    $phys = 0; $logi = 0
    if ($envGpu) {
        if ($envGpu.physical_cores) { $phys = [int]$envGpu.physical_cores }
        if ($envGpu.logical_cores)  { $logi = [int]$envGpu.logical_cores }
        Say "  detected cores: physical=$phys logical=$logi" "DarkGray"
    }

    # Ordered by information value: the shipped default first, then the shoulder
    # of the curve, then the expensive low-worker anchors last so that a budget
    # overrun costs us the least interesting points.
    $sweep = @(12, 8, 16, 4, 2)
    foreach ($c in @($phys, $logi)) {
        if ($c -gt 0 -and $sweep -notcontains $c) { $sweep += $c }
    }

    # 8 core-seconds/file, measured on this machine in phase 2
    # (485 files, max_workers=8, EncodingTask 467.6 s).
    # The earlier figure of 34 came from the shipped 500-file baseline, which was
    # inflated ~4x by nine files stalling a worker for 120 s each. Those files are
    # excluded here, so the real per-file cost is far lower.
    $corePerFile = 8.0
    $overhead = 60.0

    # WARM-UP: the first run of the phase pays for a cold OS file cache, cold
    # venv DLLs and a first Defender scan of the STEP files. In the previous
    # session that made the first datapoint (max_workers=12) come out at 559 s
    # while the same configuration measured 158 s later in the phase - a 3.5x
    # error that would have been read as "12 workers are catastrophically slow".
    # This throwaway run absorbs that cost; its result is not recorded as a
    # sweep point.
    $warmEst = ($SubsetSize * $corePerFile / 8) + $overhead
    Invoke-Bench -Label "warm-up (discarded) gpu1.1 mw=8 n=$SubsetSize" -Python $GpuPy `
        -Script "bench_step1_dataprep.py" -Phase "1" -PhaseDeadline $pd -EstimateSec $warmEst `
        -PyArgs @("--env-tag", "gpu1.1", "--max-workers", "8",
                "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "1w",
                "--note", "warm-up - excluded from analysis")

    foreach ($mw in $sweep) {
        $est = ($SubsetSize * $corePerFile / $mw) + $overhead
        Invoke-Bench -Label "step1 gpu1.1 mw=$mw n=$SubsetSize" -Python $GpuPy `
            -Script "bench_step1_dataprep.py" -Phase "1" -PhaseDeadline $pd -EstimateSec $est `
            -PyArgs @("--env-tag", "gpu1.1", "--max-workers", "$mw",
                    "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "1")
    }

    # Parity check: encoding is pure CPU work (HOOPS Exchange + numpy), so the
    # two venvs should agree. One run confirms that rather than assuming it.
    $est = ($SubsetSize * $corePerFile / 12) + $overhead
    Invoke-Bench -Label "step1 cpu1.1 mw=12 n=$SubsetSize (parity)" -Python $CpuPy `
        -Script "bench_step1_dataprep.py" -Phase "1" -PhaseDeadline $pd -EstimateSec $est `
        -PyArgs @("--env-tag", "cpu1.1", "--max-workers", "12",
                "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "1",
                "--note", "env parity check")
}

# ==========================================================================
# PHASE 2 - step 1 encoding, full 485 files, both envs (keeps the datasets)
# ==========================================================================
if ($PhaseList -contains "2") {
    Section "PHASE 2  step 1 encoding - full 485 files, both envs   cap 60 min"
    $pd = (Get-Date).AddMinutes($Global:PhaseCaps["2"])

    $bestMw = Get-BestMaxWorkers 12
    Say "  using max_workers=$bestMw (fastest in phase 1)" "DarkGray"
    $est = (485 * 8.0 / $bestMw) + 120   # 8 core-s/file, measured (see phase 1)

    Invoke-Bench -Label "step1 gpu1.1 mw=$bestMw n=485 KEEP" -Python $GpuPy `
        -Script "bench_step1_dataprep.py" -Phase "2" -PhaseDeadline $pd -EstimateSec $est `
        -PyArgs @("--env-tag", "gpu1.1", "--max-workers", "$bestMw",
                "--filelist", "filelists/clean485.txt", "--phase", "2", "--keep-output")

    Invoke-Bench -Label "step1 cpu1.1 mw=$bestMw n=485 KEEP" -Python $CpuPy `
        -Script "bench_step1_dataprep.py" -Phase "2" -PhaseDeadline $pd -EstimateSec $est `
        -PyArgs @("--env-tag", "cpu1.1", "--max-workers", "$bestMw",
                "--filelist", "filelists/clean485.txt", "--phase", "2", "--keep-output")
}

# ==========================================================================
# PHASE 3 - step 2 training
# ==========================================================================
if ($PhaseList -contains "3") {
    Section "PHASE 3  step 2 training - accelerator x batch_size x num_workers   cap 210 min"
    $pd = (Get-Date).AddMinutes($Global:PhaseCaps["3"])

    # Precondition: phase 2 must have produced an encoded dataset per env.
    # Without this guard all 15 configurations would fail one after another on
    # the same missing file, filling the CSV with noise.
    $missingData = @()
    foreach ($tag in @("gpu1.1", "cpu1.1")) {
        if (-not (Test-Path (Join-Path $ResDir "dataset_$tag.json"))) { $missingData += $tag }
    }
    if ($DryRun) {
        # In a dry run phase 2 has not created anything yet, so pretend the
        # datasets are there and show the full plan.
        $missingData = @()
    } elseif ($missingData.Count -gt 0) {
        Say ("  dataset pointer missing for: {0}" -f ($missingData -join ', ')) "Yellow"
        Say "  Run phase 2 first:  .\run_benchmark.ps1 -Phases 2 -Force" "Yellow"
        if ($missingData.Count -eq 2) {
            Say "  Skipping phase 3 entirely." "Yellow"
            $Global:Skipped += "P3 whole phase (no encoded dataset - run phase 2)"
        }
    }

    # Is the GPU actually usable? torch.cuda.is_available() returns True even when
    # the wheel contains no kernels for the card's compute capability, so phase 0
    # ran a real matmul and recorded the verdict.
    $gpuUsable = $true
    $envGpu3 = Get-EnvJson "gpu1.1"
    if ($envGpu3 -and ($null -eq $envGpu3.cuda_usable)) {
        Say "  env_gpu1.1.json predates the CUDA usability probe - re-run phase 0:" "Yellow"
        Say "    .\run_benchmark.ps1 -Phases 0 -Unattended" "Yellow"
        Say "  Continuing and assuming the GPU works; GPU rows will fail if it does not." "Yellow"
    }
    if ($envGpu3 -and ($null -ne $envGpu3.cuda_usable) -and (-not $envGpu3.cuda_usable)) {
        $gpuUsable = $false
        Say "  GPU present but NOT usable by this torch build:" "Red"
        Say ("    device {0}, wheel built for {1}" -f $envGpu3.gpu_capability,
             ($envGpu3.torch_arch_list -join ' ')) "Red"
        if ($envGpu3.cuda_error) { Say ("    {0}" -f $envGpu3.cuda_error) "Red" }
        Say "  -> dropping all accelerator=gpu rows; CPU rows still run." "Yellow"
        Say "  -> see README section 'GPU が使えない場合' to fix the torch build." "Yellow"
        $Global:Skipped += "P3 all accelerator=gpu rows (wheel lacks kernels for $($envGpu3.gpu_capability))"
    }

    # --- calibration: 1 epoch each on GPU and CPU, so every later estimate is
    #     grounded in this machine rather than a guess.
    # Epoch counts below are NEW epochs past the resumed checkpoint. The shipped
    # SIGNAL checkpoint is at epoch 7, so Lightning requires max_epochs > 7;
    # bench_step2_training derives max_epochs = 7 + new-epochs itself.
    if (($missingData -notcontains "gpu1.1") -and $gpuUsable) {
        Invoke-Bench -Label "cal train gpu new_ep=1" -Python $GpuPy `
            -Script "bench_step2_training.py" -Phase "3c" -PhaseDeadline $pd -EstimateSec 300 `
            -PyArgs @("--env-tag", "gpu1.1", "--accelerator", "gpu", "--batch-size", "8",
                    "--num-workers", "0", "--new-epochs", "1", "--phase", "3c", "--note", "calibration")
    }

    if ($missingData -notcontains "cpu1.1") {
        Invoke-Bench -Label "cal train cpu new_ep=1" -Python $CpuPy `
            -Script "bench_step2_training.py" -Phase "3c" -PhaseDeadline $pd -EstimateSec 900 `
            -PyArgs @("--env-tag", "cpu1.1", "--accelerator", "cpu", "--batch-size", "8",
                    "--num-workers", "0", "--new-epochs", "1", "--phase", "3c", "--note", "calibration")
    }

    $gpuEp = Get-SecPerEpoch "gpu1.1" "gpu"; if (-not $gpuEp) { $gpuEp = 30.0 }
    $cpuEp = Get-SecPerEpoch "cpu1.1" "cpu"; if (-not $cpuEp) { $cpuEp = 300.0 }
    $startup = 90.0   # CUDA context + checkpoint restore + Dask cluster
    Say ("  calibrated: gpu {0:N1} s/epoch, cpu {1:N1} s/epoch" -f $gpuEp, $cpuEp) "DarkGray"

    # env, accelerator, bs, nw, NEW epochs, matmul, extraFlag, s/epoch basis
    $matrix = @(
        @("gpu1.1", "gpu",  8,  0,  3, "high",    "",            $gpuEp),
        @("gpu1.1", "gpu", 16,  0,  3, "high",    "",            $gpuEp),
        @("gpu1.1", "gpu", 32,  0,  3, "high",    "",            $gpuEp),
        @("gpu1.1", "gpu",  4,  0,  3, "high",    "",            $gpuEp),
        @("gpu1.1", "gpu",  8,  4,  3, "high",    "",            $gpuEp),
        @("gpu1.1", "gpu",  8,  8,  3, "high",    "",            $gpuEp),
        @("gpu1.1", "gpu",  8,  0,  3, "highest", "",            $gpuEp),   # TF32 off
        @("gpu1.1", "cpu",  8,  0,  3, "high",    "",            $cpuEp),   # same wheel, CPU device
        @("cpu1.1", "cpu",  8,  0,  3, "high",    "",            $cpuEp),
        @("cpu1.1", "cpu", 32,  0,  3, "high",    "",            $cpuEp),
        @("cpu1.1", "cpu",  8,  8,  3, "high",    "",            $cpuEp),
        @("gpu1.1", "gpu",  8,  0,  6, "high",    "",            $gpuEp),   # epoch linearity
        @("gpu1.1", "gpu",  8,  0,  3, "high",    "--run-test",  $gpuEp)
    )

    # If the GPU cannot run kernels, every batch_size / num_workers row above is
    # dropped and the batch_size axis disappears from the report entirely. Put it
    # back on the CPU at reduced epochs so the axis is still answered, just more
    # cheaply. 2 epochs is enough for a stable s/epoch once start-up is excluded.
    if (-not $gpuUsable) {
        Say "  substituting a CPU batch_size sweep so that axis is not lost" "Yellow"
        $matrix = @(
            @("cpu1.1", "cpu",  8,  0,  2, "high", "", $cpuEp),
            @("cpu1.1", "cpu", 32,  0,  2, "high", "", $cpuEp),
            @("cpu1.1", "cpu",  4,  0,  1, "high", "", $cpuEp),
            @("cpu1.1", "cpu", 16,  0,  1, "high", "", $cpuEp),
            @("cpu1.1", "cpu",  8,  4,  1, "high", "", $cpuEp),
            @("cpu1.1", "cpu",  8,  8,  1, "high", "", $cpuEp),
            @("gpu1.1", "cpu",  8,  0,  2, "high", "", $cpuEp),   # wheel comparison
            @("cpu1.1", "cpu",  8,  0,  4, "high", "", $cpuEp),   # epoch linearity
            @("cpu1.1", "cpu",  8,  0,  1, "high", "--run-test", $cpuEp)
        )
    }

    foreach ($m in $matrix) {
        $envTag = $m[0]; $accel = $m[1]; $bs = $m[2]; $nw = $m[3]
        $ep = $m[4]; $mm = $m[5]; $extra = $m[6]; $basis = [double]$m[7]
        if ($missingData -contains $envTag) { continue }
        if ($accel -eq "gpu" -and -not $gpuUsable) { continue }
        $py = if ($envTag -eq "gpu1.1") { $GpuPy } else { $CpuPy }
        $est = ($ep * $basis) + $startup
        if ($extra -eq "--run-test") { $est += $basis }

        $a = @("--env-tag", $envTag, "--accelerator", $accel, "--batch-size", "$bs",
               "--num-workers", "$nw", "--new-epochs", "$ep",
               "--matmul-precision", $mm, "--phase", "3")
        if ($extra) { $a += $extra }

        $label = "step2 $envTag/$accel bs=$bs nw=$nw newep=$ep mm=$mm$(if($extra){' +test'})"
        Invoke-Bench -Label $label -Python $py -Script "bench_step2_training.py" `
            -Phase "3" -PhaseDeadline $pd -EstimateSec $est -PyArgs $a
    }
}

# ==========================================================================
# PHASE 4 - step 3 indexing
# ==========================================================================
if ($PhaseList -contains "4") {
    Section "PHASE 4  step 3 indexing - num_workers sweep + full index   cap 150 min"
    $pd = (Get-Date).AddMinutes($Global:PhaseCaps["4"])

    $envGpu = Get-EnvJson "gpu1.1"
    $phys = 0; if ($envGpu -and $envGpu.physical_cores) { $phys = [int]$envGpu.physical_cores }

    # nw=2 costs ~30 min on the 100-file subset for very little extra insight,
    # and phase 4 is the phase most likely to hit its cap. Dropped; nw=4 is the
    # low anchor.
    $sweep = @(12, 8, 4, 16)
    if ($phys -gt 0 -and $sweep -notcontains $phys) { $sweep += $phys }

    # Embedding does the same B-rep encoding as step 1 plus a model forward pass,
    # so start from the measured 8 core-s/file and allow some headroom.
    $corePerFile = 10.0
    $overhead = 90.0   # model load + FAISS + process start

    # Warm-up, same reasoning as phase 1: absorb cold-cache and model-load cost
    # so the first recorded sweep point is not inflated.
    $warmEst = ($SubsetSize * $corePerFile / 8) + $overhead
    Invoke-Bench -Label "warm-up (discarded) step3 gpu1.1 nw=8 n=$SubsetSize" -Python $GpuPy `
        -Script "bench_step3_indexing.py" -Phase "4w" -PhaseDeadline $pd -EstimateSec $warmEst `
        -PyArgs @("--env-tag", "gpu1.1", "--num-workers", "8",
                "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "4w",
                "--note", "warm-up - excluded from analysis")

    # Ordered so that the two most decision-relevant items - the sweep around the
    # default, and the full 485-file production build - happen before the
    # optional extras. Anything the cap drops is then genuinely optional.
    foreach ($nw in $sweep) {
        $est = ($SubsetSize * $corePerFile / $nw) + $overhead
        Invoke-Bench -Label "step3 gpu1.1 nw=$nw n=$SubsetSize" -Python $GpuPy `
            -Script "bench_step3_indexing.py" -Phase "4" -PhaseDeadline $pd -EstimateSec $est `
            -PyArgs @("--env-tag", "gpu1.1", "--num-workers", "$nw",
                    "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "4")
    }

    # Full production index, both envs. Moved ahead of the extras: this is the
    # number anyone actually quotes.
    foreach ($pair in @(@("gpu1.1", $GpuPy), @("cpu1.1", $CpuPy))) {
        $tag = $pair[0]; $py = $pair[1]
        $est = (485 * $corePerFile / 12) + $overhead + 60
        Invoke-Bench -Label "step3 $tag nw=12 n=485 +save" -Python $py `
            -Script "bench_step3_indexing.py" -Phase "4" -PhaseDeadline $pd -EstimateSec $est `
            -PyArgs @("--env-tag", $tag, "--num-workers", "12",
                    "--filelist", "filelists/clean485.txt", "--phase", "4", "--save-index")
    }

    foreach ($nw in @(12, 4)) {
        $est = ($SubsetSize * $corePerFile / $nw) + $overhead
        Invoke-Bench -Label "step3 cpu1.1 nw=$nw n=$SubsetSize" -Python $CpuPy `
            -Script "bench_step3_indexing.py" -Phase "4" -PhaseDeadline $pd -EstimateSec $est `
            -PyArgs @("--env-tag", "cpu1.1", "--num-workers", "$nw",
                    "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "4")
    }

    # Cost of generate_images=True, isolated on the same subset. Last: useful,
    # but nobody's decision hinges on it.
    $est = ($SubsetSize * $corePerFile / 12) + $overhead + (2.0 * $SubsetSize)
    Invoke-Bench -Label "step3 gpu1.1 nw=12 n=$SubsetSize +images" -Python $GpuPy `
        -Script "bench_step3_indexing.py" -Phase "4" -PhaseDeadline $pd -EstimateSec $est `
        -PyArgs @("--env-tag", "gpu1.1", "--num-workers", "12",
                "--filelist", "filelists/sub$SubsetSize.txt", "--phase", "4",
                "--gen-images", "--note", "image generation overhead")
}

# ==========================================================================
# PHASE 5 - report
# ==========================================================================
if ($PhaseList -contains "5") {
    Section "PHASE 5  report"
    if (-not $DryRun) {
        Push-Location $Bench
        & $GpuPy "make_report.py" *>&1 | Tee-Object -FilePath (Join-Path $LogDir "$Stamp-P5-report.log")
        Pop-Location
    } else {
        Say "  PLAN  make_report.py -> results\\REPORT.md + REPORT.html          est 1m"
    }
}

# ==========================================================================
Section "SUMMARY"
$total = ((Get-Date) - $RunStart).TotalSeconds
Say ("elapsed        : {0}" -f (Fmt $total))
Say ("measurements   : {0}" -f $Global:Ran)
Say ("budget left    : {0}" -f (Fmt (RemainingSeconds)))
if ($Global:Skipped.Count -gt 0) {
    Say ""
    Say "skipped for budget:" "Yellow"
    $Global:Skipped | ForEach-Object { Say "  - $_" "Yellow" }
    $Global:Skipped | Set-Content -Path (Join-Path $ResDir "skipped.txt") -Encoding UTF8
}
# Release the sleep suppression (ES_CONTINUOUS alone clears the request).
if ($Global:PowerHandle) {
    # ES_CONTINUOUS on its own clears the previous request.
    try { [PowerUtil]::SetThreadExecutionState([uint32]2147483648) | Out-Null } catch { }
}
if ($null -ne $Global:PowerRestore) {
    try {
        & powercfg /change standby-timeout-ac $Global:PowerRestore 2>$null | Out-Null
        Say ("sleep          : AC standby timeout restored to {0}s" -f $Global:PowerRestore) "DarkGray"
    } catch {
        Say ("sleep          : COULD NOT RESTORE the AC standby timeout. Run manually:" -f "") "Yellow"
        Say ("                 powercfg /change standby-timeout-ac {0}" -f $Global:PowerRestore) "Yellow"
    }
}

if (-not $DryRun) {
    Say ""
    Say "results : $CsvPath" "Green"
    Say "report  : $(Join-Path $ResDir 'REPORT.md')" "Green"
    Say ""
    Say "This run never waited for input. If it stopped early, check the tail of" "DarkGray"
    Say "the newest file in $LogDir and results\skipped.txt." "DarkGray"
}
