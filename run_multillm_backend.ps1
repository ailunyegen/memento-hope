param(
    [string]$BackendName = "deepseek-cloud",
    [string]$BaseURL = "https://api.deepseek.com/v1",
    [string]$ApiKey = "your_deepseek_api_key_here",
    [string]$Model = "deepseek-chat",
    [string]$Scenario = "data/sample_joint_operation.json",
    [string]$CaseBank = "data/military_case_bank_frozen_seed.jsonl",
    [string]$BaseOutputDir = "result/multillm_backend",
    [int]$Iterations = 10,
    [int]$SimRuns = 50,
    [int]$Seed = 7,
    [string]$PythonExe = "python"
)

$ErrorActionPreference = "Stop"

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Multi-LLM Backend Comparison" -ForegroundColor Cyan
Write-Host "  Backend: $BackendName ($Model)" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan

$env:LOCAL_LLM_BASE_URL = $BaseURL
$env:LOCAL_LLM_API_KEY = $ApiKey
$env:LOCAL_LLM_MODEL = $Model

foreach ($setting in @("pure_llm", "full_default", "full_optimized")) {
    $outputDir = "$BaseOutputDir/$BackendName/$setting"
    Write-Host "  Running: $setting" -ForegroundColor Green
    
    if (Test-Path $outputDir) {
        Remove-Item -LiteralPath $outputDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    
    $args = @(
        "-X", "utf8", "-m", "military_research.cli",
        "--scenario", $Scenario,
        "--case-bank", $CaseBank,
        "--iterations", $Iterations,
        "--sim-runs", $SimRuns,
        "--seed", $Seed,
        "--disable-writeback",
        "--output-dir", $outputDir
    )
    
    if ($setting -eq "pure_llm") {
        $args += "--disable-memory"
        $args += "--disable-hope"
        $args += "--disable-reflection"
    } elseif ($setting -eq "full_default") {
        $args += "--hope-theta"; $args += "0.5"
        $args += "--hope-epsilon"; $args += "0.02"
    } elseif ($setting -eq "full_optimized") {
        $args += "--hope-theta"; $args += "0.9"
        $args += "--hope-epsilon"; $args += "0.005"
    }
    
    & $PythonExe $args
    
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  ERROR: $setting run failed" -ForegroundColor Red
    }
}

Write-Host ""
Write-Host "Multi-LLM backend comparison completed." -ForegroundColor Green
Write-Host "Results in: $BaseOutputDir/$BackendName"
