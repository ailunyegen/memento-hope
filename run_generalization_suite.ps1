param(
    [string]$ScenarioDir = "data/generalization_scenarios",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$CaseBankDir = "",
    [string]$BaseOutputDir = "result/generalization_suite_seed7",
    [int]$Iterations = 6,
    [int]$SimRuns = 30,
    [int]$Seed = 7,
    [ValidateSet("report", "thesis", "chapter")]
    [string]$SummaryStyle = "chapter",
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

function Invoke-Step {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Label,
        [Parameter(Mandatory = $true)]
        [scriptblock]$Action
    )

    Write-Host ""
    Write-Host "=== $Label ===" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        throw "Step failed: $Label"
    }
}

function Resolve-OutputPath {
    param(
        [Parameter(Mandatory = $true)]
        [string]$BasePath,
        [Parameter(Mandatory = $true)]
        [string]$CandidatePath
    )

    if ([System.IO.Path]::IsPathRooted($CandidatePath)) {
        return $CandidatePath
    }
    return (Join-Path $BasePath $CandidatePath)
}

$base = Resolve-Path "." | ForEach-Object { $_.Path }
$suiteDir = Resolve-OutputPath -BasePath $base -CandidatePath $BaseOutputDir
New-Item -ItemType Directory -Force -Path $suiteDir | Out-Null

$scenarioFiles = Get-ChildItem -Path (Join-Path $base $ScenarioDir) -Filter *.json |
    Where-Object { $_.Name -ne "manifest.json" } |
    Sort-Object Name

if (-not $scenarioFiles) {
    throw "No scenario JSON files found under $ScenarioDir"
}

foreach ($scenarioFile in $scenarioFiles) {
    $sceneName = [System.IO.Path]::GetFileNameWithoutExtension($scenarioFile.Name)
    $sceneDir = Join-Path $suiteDir $sceneName
    $sceneCaseBank = $CaseBank

    if ($CaseBankDir) {
        $caseBankBaseDir = Resolve-OutputPath -BasePath $base -CandidatePath $CaseBankDir
        $candidateCaseBank = Join-Path $caseBankBaseDir "$sceneName.jsonl"
        if (Test-Path $candidateCaseBank) {
            $sceneCaseBank = $candidateCaseBank
        }
    }

    Write-Host "Using case bank: $sceneCaseBank" -ForegroundColor DarkGray
    Invoke-Step -Label "Scenario $sceneName" -Action {
        & (Join-Path $base "run_ablation_suite.ps1") `
            -Scenario $scenarioFile.FullName `
            -CaseBank $sceneCaseBank `
            -BaseOutputDir $sceneDir `
            -Iterations $Iterations `
            -SimRuns $SimRuns `
            -Seed $Seed `
            -SummaryStyle $SummaryStyle `
            -PythonExe $PythonExe
    }
}

$summaryPath = Join-Path $suiteDir "generalization_summary.md"
Invoke-Step -Label "Summarize Generalization" -Action {
    & $PythonExe -X utf8 -m military_research.summarize_generalization `
        --suite-dir $suiteDir `
        --output $summaryPath
}

Write-Host ""
Write-Host "Generalization suite completed." -ForegroundColor Green
Write-Host "Suite directory: $suiteDir"
Write-Host "Summary file: $summaryPath"
