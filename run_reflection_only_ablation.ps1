param(
    [string]$Scenario = "data/sample_joint_operation.json",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$OutputDir = "result/ablation_reflection_only_seed7",
    [int]$Iterations = 10,
    [int]$SimRuns = 50,
    [int]$Seed = 7,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

if (Test-Path $OutputDir) {
    Remove-Item -LiteralPath $OutputDir -Recurse -Force -ErrorAction SilentlyContinue
    Start-Sleep -Milliseconds 200
}

& $PythonExe -X utf8 -m military_research.cli `
    --scenario $Scenario `
    --case-bank $CaseBank `
    --iterations $Iterations `
    --sim-runs $SimRuns `
    --seed $Seed `
    --disable-memory `
    --disable-hope `
    --disable-writeback `
    --output-dir $OutputDir

if ($LASTEXITCODE -ne 0) {
    throw "Reflection-only ablation run failed."
}

Write-Host ""
Write-Host "Reflection-only ablation completed." -ForegroundColor Green
Write-Host "Output directory: $OutputDir"
