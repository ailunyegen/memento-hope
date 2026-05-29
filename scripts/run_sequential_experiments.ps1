# Sequential experiment runner - avoids LM Studio concurrency issues
$ErrorActionPreference = "Stop"
$Python = "D:\Programs\Python312\python.exe"
$Repo = "C:\Users\Alcoholic\Desktop\Memento-military"

Write-Host "=== STEP 1: Full ablation with optimized hyperparameters ===" -ForegroundColor Cyan
& $Python -u -m military_research.cli `
    --scenario "$Repo\data\sample_joint_operation.json" `
    --case-bank "$Repo\data\military_case_bank_frozen_seed.jsonl" `
    --output-dir "$Repo\result\ablation_suite_seed7_v3\ablation_full_optimized" `
    --iterations 10 --seed 7 --sim-runs 50 `
    --hope-theta 0.9 --hope-epsilon 0.005
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Full optimized failed" -ForegroundColor Red }

Write-Host "`n=== STEP 2: Urban hub defense Reflection ===" -ForegroundColor Cyan
& $Python -u -m military_research.cli `
    --scenario "$Repo\data\generalization_scenarios\urban_hub_defense.json" `
    --case-bank "$Repo\data\generalization_case_banks_v2_sceneout\urban_hub_defense.jsonl" `
    --output-dir "$Repo\result\generalization_suite_seed7_v2_sceneout_iter20\urban_hub_defense\ablation_reflection_only" `
    --iterations 20 --seed 7 --sim-runs 50 `
    --disable-memory --disable-hope
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Urban Reflection failed" -ForegroundColor Red }

Write-Host "`n=== STEP 3: Mountain corridor recon Reflection ===" -ForegroundColor Cyan
& $Python -u -m military_research.cli `
    --scenario "$Repo\data\generalization_scenarios\mountain_corridor_recon.json" `
    --case-bank "$Repo\data\generalization_case_banks_v2_sceneout\mountain_corridor_recon.jsonl" `
    --output-dir "$Repo\result\generalization_suite_seed7_v2_sceneout_iter20\mountain_corridor_recon\ablation_reflection_only" `
    --iterations 20 --seed 7 --sim-runs 50 `
    --disable-memory --disable-hope
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Mountain Reflection failed" -ForegroundColor Red }

Write-Host "`n=== STEP 4: River crossing breakthrough Reflection ===" -ForegroundColor Cyan
& $Python -u -m military_research.cli `
    --scenario "$Repo\data\generalization_scenarios\river_crossing_breakthrough.json" `
    --case-bank "$Repo\data\generalization_case_banks_v2_sceneout\river_crossing_breakthrough.jsonl" `
    --output-dir "$Repo\result\generalization_suite_seed7_v2_sceneout_iter20\river_crossing_breakthrough\ablation_reflection_only" `
    --iterations 20 --seed 7 --sim-runs 50 `
    --disable-memory --disable-hope
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: River Reflection failed" -ForegroundColor Red }

Write-Host "`n=== STEP 5: Island resupply corridor Reflection ===" -ForegroundColor Cyan
& $Python -u -m military_research.cli `
    --scenario "$Repo\data\generalization_scenarios\island_resupply_corridor.json" `
    --case-bank "$Repo\data\generalization_case_banks_v2_sceneout\island_resupply_corridor.jsonl" `
    --output-dir "$Repo\result\generalization_suite_seed7_v2_sceneout_iter20\island_resupply_corridor\ablation_reflection_only" `
    --iterations 20 --seed 7 --sim-runs 50 `
    --disable-memory --disable-hope
if ($LASTEXITCODE -ne 0) { Write-Host "ERROR: Island Reflection failed" -ForegroundColor Red }

Write-Host "`n=== ALL EXPERIMENTS COMPLETE ===" -ForegroundColor Green
