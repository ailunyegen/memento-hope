param(
    [int[]]$Seeds = @(42, 123, 999),
    [string]$ScenarioDir = "data/generalization_scenarios",
    [string]$BaseOutputDir = "result/reflection_only_multiseed",
    [int]$Iterations = 20,
    [int]$SimRuns = 50,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"
$manifest = Get-Content "$ScenarioDir/manifest.json" | ConvertFrom-Json

foreach ($seed in $Seeds) {
    $seedOutputDir = "$BaseOutputDir/seed_$seed"
    Write-Host ""
    Write-Host "========================================" -ForegroundColor Cyan
    Write-Host "  Reflection-only: seed=$seed" -ForegroundColor Cyan
    Write-Host "========================================" -ForegroundColor Cyan

    foreach ($scene in $manifest.scenarios) {
        $sceneFile = "$ScenarioDir/$scene"
        $sceneData = Get-Content $sceneFile | ConvertFrom-Json
        $sceneName = $sceneData.name -replace '[^a-zA-Z0-9_-]', '_'
        
        # Build scene-out case bank
        $sceneOutDir = "data/generalization_case_banks_v2_sceneout"
        $caseBank = "$sceneOutDir/$sceneName.jsonl"
        if (-not (Test-Path $caseBank)) {
            Write-Host "  WARNING: Case bank not found: $caseBank, using seed bank" -ForegroundColor Yellow
            $caseBank = "data/military_case_bank_frozen_seed.jsonl"
        }
        
        $sceneOutputDir = "$seedOutputDir/$sceneName"
        
        Write-Host "  Running: $sceneName (seed=$seed)" -ForegroundColor Green
        
        if (Test-Path $sceneOutputDir) {
            Remove-Item -LiteralPath $sceneOutputDir -Recurse -Force -ErrorAction SilentlyContinue
        }
        
        & $PythonExe -X utf8 -m military_research.cli `
            --scenario $sceneFile `
            --case-bank $caseBank `
            --iterations $Iterations `
            --sim-runs $SimRuns `
            --seed $seed `
            --disable-memory `
            --disable-hope `
            --disable-writeback `
            --output-dir $sceneOutputDir
        
        if ($LASTEXITCODE -ne 0) {
            Write-Host "  ERROR: Reflection-only run failed for $sceneName (seed=$seed)" -ForegroundColor Red
        }
    }
    
    # Summarize this seed's results
    Write-Host ""
    Write-Host "  Summarizing seed $seed results..." -ForegroundColor Green
    & $PythonExe -X utf8 -m military_research.summarize_generalization `
        --suite-dir $seedOutputDir `
        --output "$seedOutputDir/generalization_summary.md" `
        --style chapter
}

Write-Host ""
Write-Host "All Reflection-only multi-seed runs completed." -ForegroundColor Green
Write-Host "Results in: $BaseOutputDir"
