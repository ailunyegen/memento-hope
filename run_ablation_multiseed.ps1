param(
    [string]$Scenario = "data/sample_joint_operation.json",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$BaseOutputDir = "result/ablation_multiseed",
    [int]$Iterations = 10,
    [int]$SimRuns = 50,
    [int[]]$Seeds = @(3, 7, 11, 19, 23),
    [ValidateSet("report", "thesis", "chapter")]
    [string]$SummaryStyle = "report",
    [string]$PythonExe = "python",
    [switch]$IncludeReflectionOnly
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

$base = Resolve-Path "." | ForEach-Object { $_.Path }
$suiteDir = if ([System.IO.Path]::IsPathRooted($BaseOutputDir)) { $BaseOutputDir } else { Join-Path $base $BaseOutputDir }
New-Item -ItemType Directory -Force -Path $suiteDir | Out-Null

$manifest = [ordered]@{
    scenario = $Scenario
    case_bank = $CaseBank
    iterations = $Iterations
    sim_runs = $SimRuns
    seeds = @()
    include_reflection_only = [bool]$IncludeReflectionOnly
}

foreach ($seed in $Seeds) {
    $seedDir = Join-Path $suiteDir ("seed_" + $seed)
    Invoke-Step -Label "Ablation suite seed $seed" -Action {
        & (Join-Path $base "run_ablation_suite.ps1") `
            -Scenario $Scenario `
            -CaseBank $CaseBank `
            -BaseOutputDir $seedDir `
            -Iterations $Iterations `
            -SimRuns $SimRuns `
            -Seed $seed `
            -SummaryStyle $SummaryStyle `
            -PythonExe $PythonExe
    }

    if ($IncludeReflectionOnly) {
        Invoke-Step -Label "Reflection-only seed $seed" -Action {
            & (Join-Path $base "run_reflection_only_ablation.ps1") `
                -Scenario $Scenario `
                -CaseBank $CaseBank `
                -OutputDir (Join-Path $seedDir "ablation_reflection_only") `
                -Iterations $Iterations `
                -SimRuns $SimRuns `
                -Seed $seed `
                -PythonExe $PythonExe
        }
    }

    $manifest.seeds += [ordered]@{
        seed = $seed
        output_dir = $seedDir
    }
}

$manifestPath = Join-Path $suiteDir "multiseed_manifest.json"
$manifest | ConvertTo-Json -Depth 5 | Set-Content -Path $manifestPath -Encoding utf8

Write-Host ""
Write-Host "Multi-seed ablation suite completed." -ForegroundColor Green
Write-Host "Suite directory: $suiteDir"
Write-Host "Manifest: $manifestPath"
