# 超参数敏感性分析批量脚本
# θ (sde_theta): [0.1, 0.3, 0.5, 0.7, 0.9]  — 固定 ε=0.02
# ε (sde_epsilon): [0.005, 0.01, 0.02, 0.05, 0.1]  — 固定 θ=0.5
# 使用 Hope-only 变体 (--disable-memory --disable-reflection) 直接测量 HOPE 敏感性

$python = "D:\Programs\Python312\python.exe"
$projectRoot = "C:\Users\Alcoholic\Desktop\Memento-military"
$scenario = "$projectRoot\data\sample_joint_operation.json"
$caseBank = "$projectRoot\data\military_case_bank_frozen_seed.jsonl"

# θ 敏感性 (epsilon fixed at 0.02)
foreach ($theta in 0.1, 0.3, 0.7, 0.9) {
    $outDir = "$projectRoot\result\hyperparam\theta_${theta}"
    Write-Host "=== theta=$theta epsilon=0.02 ===" -ForegroundColor Green
    & $python -m military_research.cli `
        --scenario $scenario --case-bank $caseBank `
        --output-dir $outDir --iterations 10 --seed 7 --sim-runs 50 `
        --disable-memory --disable-reflection `
        --hope-theta $theta --hope-epsilon 0.02
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED theta=$theta" -ForegroundColor Red }
}

# ε 敏感性 (theta fixed at 0.5)
foreach ($epsilon in 0.005, 0.01, 0.05, 0.1) {
    $outDir = "$projectRoot\result\hyperparam\epsilon_$epsilon"
    Write-Host "=== theta=0.5 epsilon=$epsilon ===" -ForegroundColor Green
    & $python -m military_research.cli `
        --scenario $scenario --case-bank $caseBank `
        --output-dir $outDir --iterations 10 --seed 7 --sim-runs 50 `
        --disable-memory --disable-reflection `
        --hope-theta 0.5 --hope-epsilon $epsilon
    if ($LASTEXITCODE -ne 0) { Write-Host "FAILED epsilon=$epsilon" -ForegroundColor Red }
}
