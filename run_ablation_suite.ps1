param(
    [string]$Scenario = "data/sample_joint_operation.json",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$BaseOutputDir = "result/ablation_suite_seed7",
    [int]$Iterations = 3,
    [int]$SimRuns = 50,
    [int]$Seed = 7,
    [ValidateSet("report", "thesis", "chapter")]
    [string]$SummaryStyle = "report",
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
$pureDir = Join-Path $suiteDir "ablation_pure_llm"
$mementoDir = Join-Path $suiteDir "ablation_memento"
$hopeDir = Join-Path $suiteDir "ablation_hope"
$fullDir = Join-Path $suiteDir "ablation_full"
$summaryPath = Join-Path $suiteDir "ablation_summary_$SummaryStyle.md"

New-Item -ItemType Directory -Force -Path $suiteDir | Out-Null

function Reset-OutputDir {
    param(
        [Parameter(Mandatory = $true)]
        [string]$TargetPath
    )

    if (Test-Path $TargetPath) {
        Remove-Item -LiteralPath $TargetPath -Recurse -Force -ErrorAction SilentlyContinue
        Start-Sleep -Milliseconds 200
    }
}

Invoke-Step -Label "Pure LLM" -Action {
    Reset-OutputDir -TargetPath $pureDir
    & $PythonExe -X utf8 -m military_research.cli `
        --scenario $Scenario `
        --case-bank $CaseBank `
        --iterations $Iterations `
        --sim-runs $SimRuns `
        --seed $Seed `
        --disable-memory `
        --disable-hope `
        --disable-reflection `
        --disable-writeback `
        --output-dir $pureDir
}

Invoke-Step -Label "LLM + Memento" -Action {
    Reset-OutputDir -TargetPath $mementoDir
    & $PythonExe -X utf8 -m military_research.cli `
        --scenario $Scenario `
        --case-bank $CaseBank `
        --iterations $Iterations `
        --sim-runs $SimRuns `
        --seed $Seed `
        --disable-hope `
        --disable-reflection `
        --disable-writeback `
        --output-dir $mementoDir
}

Invoke-Step -Label "LLM + Hope" -Action {
    Reset-OutputDir -TargetPath $hopeDir
    & $PythonExe -X utf8 -m military_research.cli `
        --scenario $Scenario `
        --case-bank $CaseBank `
        --iterations $Iterations `
        --sim-runs $SimRuns `
        --seed $Seed `
        --disable-memory `
        --disable-reflection `
        --disable-writeback `
        --output-dir $hopeDir
}

Invoke-Step -Label "Full System" -Action {
    Reset-OutputDir -TargetPath $fullDir
    & $PythonExe -X utf8 -m military_research.cli `
        --scenario $Scenario `
        --case-bank $CaseBank `
        --iterations $Iterations `
        --sim-runs $SimRuns `
        --seed $Seed `
        --disable-writeback `
        --output-dir $fullDir
}

Invoke-Step -Label "Summarize Ablations" -Action {
    & $PythonExe -X utf8 -m military_research.summarize_ablations `
        --pure-llm (Join-Path $pureDir "full_result.json") `
        --memento (Join-Path $mementoDir "full_result.json") `
        --hope (Join-Path $hopeDir "full_result.json") `
        --full (Join-Path $fullDir "full_result.json") `
        --style $SummaryStyle `
        --output $summaryPath
}

Write-Host ""
Write-Host "Ablation suite completed." -ForegroundColor Green
Write-Host "Suite directory: $suiteDir"
Write-Host "Summary file: $summaryPath"
