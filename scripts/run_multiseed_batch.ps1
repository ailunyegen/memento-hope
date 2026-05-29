param(
    [int]$StartSeed = 42,
    [string[]]$Variants = @("pure_llm", "memento", "hope", "full")
)

$projectRoot = "C:\Users\Alcoholic\Desktop\Memento-military"
$venvPython = "$projectRoot\.venv\Scripts\python.exe"
$scenario = "$projectRoot\data\sample_joint_operation.json"
$caseBank = "$projectRoot\data\military_case_bank_frozen_seed.jsonl"

# Seed 42 already done: pure_llm
$done = @{
    "42_pure_llm" = $true
}

foreach ($seed in $StartSeed, 123, 999, 2024) {
    foreach ($variant in $Variants) {
        $key = "${seed}_${variant}"
        if ($done[$key]) {
            Write-Host "SKIP $key (already done)" -ForegroundColor Gray
            continue
        }

        $outDir = "$projectRoot\result\multiseed\seed_${seed}\ablation_${variant}"
        $args = @(
            "--scenario", $scenario,
            "--case-bank", $caseBank,
            "--output-dir", $outDir,
            "--iterations", "10",
            "--seed", $seed,
            "--sim-runs", "50"
        )

        switch ($variant) {
            "pure_llm" {
                $args += "--disable-memory", "--disable-hope", "--disable-reflection"
            }
            "memento" {
                $args += "--disable-hope", "--disable-reflection"
            }
            "hope" {
                $args += "--disable-memory", "--disable-reflection"
            }
            "full" {
                # No disables
            }
        }

        Write-Host "RUN seed=${seed} variant=${variant}" -ForegroundColor Green
        Write-Host "  Output: $outDir" -ForegroundColor Cyan

        $proc = Start-Process -FilePath $venvPython -ArgumentList @("-m", "military_research.cli") + $args -NoNewWindow -Wait -PassThru

        if ($proc.ExitCode -ne 0) {
            Write-Host "FAILED seed=${seed} variant=${variant} (exit $($proc.ExitCode))" -ForegroundColor Red
        } else {
            Write-Host "DONE seed=${seed} variant=${variant}" -ForegroundColor Green
        }
    }
}
