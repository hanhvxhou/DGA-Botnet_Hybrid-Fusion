# run_fullseed.ps1 - Huong A: huan luyen lai backbone theo tung seed
#   Seed 43-46: selectAlgorithm (chi BiLSTM, DistilBERT, BERT) -> out_select_seed<N>/
#               roi M6_v3_Markov doc backbone cung seed -> out_M6_v3_Markov_fullseed<N>/
#   Seed 42: dung lai out_select/ va out_M6_v3_Markov_seed42/ da co (backbone seed 42)
# Uoc tinh: ~1.8 gio/seed x 4 = ~7.5 gio tren RTX 5070 Ti, can ~5 GB dia trong.
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$env:DGA_CANDIDATES = "bilstm,distilbert,bert"
$seeds = @(43, 44, 45, 46)

$startTime = Get-Date
$logFile = "fullseed_log_$($startTime.ToString('yyyyMMdd_HHmmss')).txt"
Write-Host "=== Full-seed run started: $startTime ===" -ForegroundColor Green
Write-Host "Log: $logFile   Candidates: $env:DGA_CANDIDATES" -ForegroundColor Cyan

function Run-Step($label, $script) {
    $t0 = Get-Date
    Write-Host "  [RUN] $label" -ForegroundColor Cyan
    "  [RUN] $label - start $t0" | Add-Content $logFile
    $out = python $script 2>&1
    $code = $LASTEXITCODE
    $out | Add-Content $logFile
    if ($code -ne 0) {
        Write-Host "    [FAIL] exit $code - 20 dong cuoi:" -ForegroundColor Red
        $out | Select-Object -Last 20 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
        return $false
    }
    $el = "{0:hh\:mm\:ss}" -f ((Get-Date) - $t0)
    Write-Host "    [OK] $el" -ForegroundColor Green
    "    [OK] $el" | Add-Content $logFile
    return $true
}

foreach ($seed in $seeds) {
    Write-Host "`n########## SEED = $seed ##########" -ForegroundColor Yellow
    "########## SEED = $seed ##########" | Add-Content $logFile
    $env:DGA_SEED = $seed
    $sel = "out_select_seed$seed"
    $env:DGA_SELECT_ROOT = $sel

    # 1) Backbone
    $need = @("bilstm","distilbert","bert") | Where-Object { -not (Test-Path "$sel\$_\summary.txt") }
    if ($need.Count -eq 0) {
        Write-Host "  [SKIP] selectAlgorithm seed=$seed - da du 3 backbone" -ForegroundColor DarkGray
    } else {
        $env:DGA_CANDIDATES = ($need -join ",")
        $ok = Run-Step "selectAlgorithm seed=$seed ($($env:DGA_CANDIDATES)) -> $sel" "selectAlgorithm.py"
        $env:DGA_CANDIDATES = "bilstm,distilbert,bert"
        if (-not $ok) { Write-Host "  Bo qua M6 seed=$seed vi backbone loi" -ForegroundColor Red; continue }
    }

    # 2) M6 voi backbone cung seed
    $m6 = "out_M6_v3_Markov_fullseed$seed"
    if (Test-Path "$m6\summary.txt") {
        Write-Host "  [SKIP] M6 seed=$seed - da co $m6" -ForegroundColor DarkGray
    } else {
        $env:DGA_OUTPUT_DIR = $m6
        Run-Step "M6_v3_Markov seed=$seed -> $m6" "M6_v3_Markov.py" | Out-Null
        Remove-Item Env:DGA_OUTPUT_DIR
    }
}

Write-Host "`n=== Xong: $((Get-Date)) - tong $("{0:hh\:mm\:ss}" -f ((Get-Date) - $startTime)) ===" -ForegroundColor Green
foreach ($seed in $seeds) {
    $a = (Test-Path "out_select_seed$seed\bert\summary.txt") -and (Test-Path "out_select_seed$seed\distilbert\summary.txt") -and (Test-Path "out_select_seed$seed\bilstm\summary.txt")
    $b = Test-Path "out_M6_v3_Markov_fullseed$seed\summary.txt"
    Write-Host ("  seed {0}: backbone={1}  M6={2}" -f $seed, $a, $b)
}
