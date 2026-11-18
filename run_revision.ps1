# run_revision.ps1 - IJICS revision: additional experiments (DGA-HLF)
# Run from the project root (folder with M1.py ... M6_v3_Markov.py, DataNew, test_external).
# Steps (resume-friendly, each step is skipped when its output exists):
#   1) download Tranco + Cisco Umbrella top-1M lists (independent benign sets)
#   2) prep   : evaluation sets, handcraft caches (orig + BOS/EOS Markov), tokenisation stats
#   3) dump   : seeds 42-46, all models x all sets, early fusion, corrected-Markov XGB
#   4) random-init DistilBERT (same architecture/tokenizer, no pretraining), seeds 42-46
#   5) dump again (adds the random-init model only)
#   6) latency benchmark (GPU + CPU)
#   7) collect -> rev_upload.zip
# Estimated time on RTX 5070 Ti: ~1.5-2 h (steps 2-3) + ~3 h (step 4) + ~0.5 h (5-6)
$ErrorActionPreference = "Continue"
$env:PYTHONIOENCODING = "utf-8"
$env:PYTHONUTF8 = "1"
$seeds = @(42, 43, 44, 45, 46)
$start = Get-Date
$logFile = "revision_log_$($start.ToString('yyyyMMdd_HHmmss')).txt"
Write-Host "=== Revision experiments started: $start ===" -ForegroundColor Green

function Run-Py($label, [string[]]$pyArgs) {
    $t0 = Get-Date
    Write-Host "  [RUN] $label" -ForegroundColor Cyan
    "  [RUN] $label - $t0" | Add-Content $logFile
    $out = & python @pyArgs 2>&1
    $code = $LASTEXITCODE
    $out | Add-Content $logFile
    if ($code -ne 0) {
        Write-Host "    [FAIL] exit $code - last 25 lines:" -ForegroundColor Red
        $out | Select-Object -Last 25 | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
        return $false
    }
    $out | Select-String -Pattern "\[FAIL\]|\[WARN\]|checks:|errors:" | ForEach-Object { Write-Host "    $_" -ForegroundColor DarkYellow }
    Write-Host ("    [OK] {0:hh\:mm\:ss}" -f ((Get-Date) - $t0)) -ForegroundColor Green
    return $true
}

foreach ($f in @("revision_experiments.py", "selectAlgorithm.py", "M6_v3_Markov.py", "DataNew\train.csv")) {
    if (-not (Test-Path $f)) { Write-Host "Missing $f - copy it to the project root first." -ForegroundColor Red; exit 1 }
}

# 0) optional packages (NVML energy, public-suffix parsing)
& python -m pip install -q nvidia-ml-py tldextract 2>&1 | Out-Null

# 1) benign lists
$tranco = "tranco_top-1m.csv.zip"
$umbrella = "umbrella_top-1m.csv.zip"
if (-not (Test-Path $tranco)) {
    try { Invoke-WebRequest -Uri "https://tranco-list.eu/top-1m.csv.zip" -OutFile $tranco -UseBasicParsing
          "Tranco downloaded $(Get-Date -Format s)" | Add-Content $logFile }
    catch { Write-Host "  [WARN] Tranco download failed: $_" -ForegroundColor Yellow }
}
if (-not (Test-Path $umbrella)) {
    try { Invoke-WebRequest -Uri "http://s3-us-west-1.amazonaws.com/umbrella-static/top-1m.csv.zip" -OutFile $umbrella -UseBasicParsing
          "Umbrella downloaded $(Get-Date -Format s)" | Add-Content $logFile }
    catch { Write-Host "  [WARN] Umbrella download failed: $_" -ForegroundColor Yellow }
}

# 2) prep
if (-not (Test-Path "rev_out\cache\hc_eos_test.npy")) {
    $ok = Run-Py "prep" @("revision_experiments.py", "prep", "--tranco", $tranco, "--umbrella", $umbrella)
    if (-not $ok) { Write-Host "prep failed - stop." -ForegroundColor Red; exit 1 }
} else { Write-Host "  [SKIP] prep (cache exists)" -ForegroundColor DarkGray }

# 3) dump per seed
foreach ($s in $seeds) { Run-Py "dump seed=$s" @("revision_experiments.py", "dump", "--seed", "$s") | Out-Null }

# 4) random-init DistilBERT
foreach ($s in $seeds) {
    $root = "out_select_rand_seed$s"
    if (Test-Path "$root\distilbert\summary.txt") { Write-Host "  [SKIP] random-init seed=$s" -ForegroundColor DarkGray; continue }
    $env:DGA_SEED = "$s"; $env:DGA_SELECT_ROOT = $root
    $env:DGA_CANDIDATES = "distilbert"; $env:DGA_BERT_RANDOM_INIT = "1"
    Run-Py "random-init DistilBERT seed=$s -> $root" @("selectAlgorithm.py") | Out-Null
    Remove-Item Env:DGA_BERT_RANDOM_INIT, Env:DGA_CANDIDATES, Env:DGA_SELECT_ROOT, Env:DGA_SEED -ErrorAction SilentlyContinue
}

# 5) dump again (only the new model is computed)
foreach ($s in $seeds) { Run-Py "dump (random-init) seed=$s" @("revision_experiments.py", "dump", "--seed", "$s") | Out-Null }

# 6) latency
if (-not (Test-Path "rev_out\latency.csv")) { Run-Py "latency" @("revision_experiments.py", "latency", "--seed", "42", "--cpu") | Out-Null }

# 7) collect
Run-Py "collect" @("revision_experiments.py", "collect", "--with-data") | Out-Null

Write-Host ("`n=== Done: {0} (total {1:hh\:mm\:ss}) ===" -f (Get-Date), ((Get-Date) - $start)) -ForegroundColor Green
Write-Host "Upload rev_upload.zip (and $logFile)." -ForegroundColor Magenta
