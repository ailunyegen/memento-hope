param(
    [string]$Scenario = "data/sample_joint_operation.json",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$BaseOutputDir = "result/multiseed_optimized",
    [int]$Iterations = 10,
    [int]$SimRuns = 50,
    [float]$HopeTheta = 0.9,
    [float]$HopeEpsilon = 0.005,
    [int[]]$Seeds = @(7, 42, 123),
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

$base = Resolve-Path "." | ForEach-Object { $_.Path }
$suiteDir = Join-Path $base $BaseOutputDir
New-Item -ItemType Directory -Force -Path $suiteDir | Out-Null

Write-Host "=== Optimized-Parameter Multi-Seed Experiment ===" -ForegroundColor Cyan
Write-Host "HOPE theta = $HopeTheta, epsilon = $HopeEpsilon"
Write-Host "Seeds: $($Seeds -join ', ')"
Write-Host "Iterations: $Iterations, MC runs: $SimRuns"
Write-Host "Output base: $suiteDir"
Write-Host ""

$totalStart = Get-Date

foreach ($seed in $Seeds) {
    $seedDir = Join-Path $suiteDir "seed_$seed"
    $fullDir = Join-Path $seedDir "ablation_full_optimized"

    Write-Host "--- Seed $seed : Full (optimized) ---" -ForegroundColor Yellow
    Write-Host "Output: $fullDir"

    if (Test-Path $fullDir) {
        Remove-Item -LiteralPath $fullDir -Recurse -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 200
    }

    $seedStart = Get-Date
    & $PythonExe -X utf8 -m military_research.cli `
        --scenario $Scenario `
        --case-bank $CaseBank `
        --iterations $Iterations `
        --sim-runs $SimRuns `
        --seed $seed `
        --hope-theta $HopeTheta `
        --hope-epsilon $HopeEpsilon `
        --disable-writeback `
        --output-dir $fullDir

    if ($LASTEXITCODE -ne 0) {
        Write-Host "ERROR: Seed $seed failed with exit code $LASTEXITCODE" -ForegroundColor Red
        continue
    }

    $seedElapsed = [math]::Round(((Get-Date) - $seedStart).TotalMinutes, 1)
    Write-Host "Seed $seed completed in ${seedElapsed} min." -ForegroundColor Green
}

$totalElapsed = [math]::Round(((Get-Date) - $totalStart).TotalMinutes, 1)
Write-Host ""
Write-Host "=== Optimized multi-seed suite finished (${totalElapsed} min) ===" -ForegroundColor Green
Write-Host "Results saved to: $suiteDir"
Write-Host ""
Write-Host "To summarize: python -m military_research.summarize_multiseed_ablation --suite-dir $suiteDir --output $suiteDir/summary.md"
