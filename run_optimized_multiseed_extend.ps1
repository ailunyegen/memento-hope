param(
    [int[]]$Seeds = @(999, 2024),
    [string]$Scenario = "data/sample_joint_operation.json",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$BaseOutputDir = "result/multiseed_optimized",
    [int]$Iterations = 10,
    [int]$SimRuns = 50,
    [float]$Theta = 0.9,
    [float]$Epsilon = 0.005,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

foreach ($seed in $Seeds) {
    $outputDir = "$BaseOutputDir/seed_$seed"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Optimized Full: seed=$seed (theta=$Theta, epsilon=$Epsilon)" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    if (Test-Path $outputDir) {
        Remove-Item -LiteralPath $outputDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    & $PythonExe -X utf8 -m military_research.cli `
        --scenario $Scenario `
        --case-bank $CaseBank `
        --iterations $Iterations `
        --sim-runs $SimRuns `
        --seed $seed `
        --hope-theta $Theta `
        --hope-epsilon $Epsilon `
        --disable-writeback `
        --output-dir $outputDir
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: Optimized Full run failed for seed=$seed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Optimized Full multi-seed runs completed." -ForegroundColor Green
Write-Host "Results in: $BaseOutputDir"
