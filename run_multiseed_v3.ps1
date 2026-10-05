# run_multiseed_v3.ps1 - fix UnicodeEncodeError
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"

$seeds = @(42, 43, 44, 45, 46)
$models = @(
    @{Name="M1"; Script="M1.py"},
    @{Name="M2"; Script="M2.py"},
    @{Name="M3"; Script="M3.py"},
    @{Name="M4"; Script="M4.py"},
    @{Name="M5"; Script="M5.py"},
    @{Name="M6_v3_Markov"; Script="M6_v3_Markov.py"}
)

$startTime = Get-Date
$logFile = "multiseed_log_$($startTime.ToString('yyyyMMdd_HHmmss')).txt"
Write-Host "=== Multi-seed run started: $startTime ===" -ForegroundColor Green
Write-Host "Log file: $logFile" -ForegroundColor Cyan
Write-Host "PYTHONIOENCODING = utf-8 (fix Vietnamese chars)" -ForegroundColor Cyan

$totalRuns = $seeds.Count * $models.Count
$currentRun = 0

foreach ($seed in $seeds) {
    Write-Host "`n########## SEED = $seed ##########" -ForegroundColor Yellow
    "########## SEED = $seed ##########" | Add-Content $logFile

    $env:DGA_SEED = $seed

    if (Test-Path "benign_markov.npz") {
        Remove-Item "benign_markov.npz" -Force
    }

    foreach ($m in $models) {
        $currentRun++
        $pct = [math]::Round(100 * $currentRun / $totalRuns, 1)
        $modelStart = Get-Date

        $outDir = "out_$($m.Name)_seed$seed"

        if (Test-Path "$outDir\summary.txt") {
            $fileSize = (Get-Item "$outDir\summary.txt").Length
            if ($fileSize -gt 100) {
                Write-Host "  [SKIP] $($m.Name) seed=$seed - exists ($fileSize bytes)" -ForegroundColor DarkGray
                continue
            }
        }

        Write-Host "  [RUN $currentRun/$totalRuns $pct%] $($m.Name) seed=$seed -> $outDir" -ForegroundColor Cyan
        "  [RUN $currentRun/$totalRuns $pct%] $($m.Name) seed=$seed -> $outDir - start $modelStart" | Add-Content $logFile

        $output = python $m.Script 2>&1
        $exitCode = $LASTEXITCODE
        $output | Add-Content $logFile

        if ($exitCode -ne 0) {
            Write-Host "    [FAIL] Exit code $exitCode" -ForegroundColor Red
            Write-Host "    --- Last 20 lines of output ---" -ForegroundColor Red
            $output | Select-Object -Last 20 | ForEach-Object {
                Write-Host "    $_" -ForegroundColor DarkYellow
            }
            Write-Host "    --- End ---" -ForegroundColor Red
        } else {
            $elapsed = (Get-Date) - $modelStart
            $elapsedStr = "{0:hh\:mm\:ss}" -f $elapsed
            Write-Host "    [OK] completed in $elapsedStr" -ForegroundColor Green
        }
    }
}

Write-Host "`n=== Done ===" -ForegroundColor Green
