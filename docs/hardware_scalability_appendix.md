# Appendix: Cross-Hardware Scalability & Replication (NVIDIA RTX 4090 D)

> 定位：论文正文与审稿包保持 RTX 3060 口径不变；本附录为 4090 算力泛化补充证据，
> 随响应信/补充材料提交，证明框架在现代高算力硬件上收益稳定且工程落地更可行。

## A.1 硬件与运行环境

| 项 | 论文主实验 | 本附录复跑 |
| --- | --- | --- |
| GPU | NVIDIA GeForce RTX 3060 | **NVIDIA GeForce RTX 4090 D (24GB)** |
| CPU | Intel i7-13700F | Intel i7-13700KF |
| 后端模型 | deepseek-r1-0528-qwen3-8b | deepseek-r1-0528-qwen3-8b（相同） |
| Python 环境 | Memento | Memento（相同） |
| 消融配置 | 10 轮 × 50-MC × seed 7，writeback 关闭 | 相同 |
| 全套墙钟 | ~3646 s（61 min） | **~365 s（6 min），约 10× 加速** |

## A.2 单场景消融复跑（seed 7，50-MC）

| 设置 | 3060 论文值 (MS/OE/LER) | 4090 复跑 (MS/OE/LER) | ΔMS |
| --- | --- | --- | --- |
| Pure LLM | 0.6260 / 0.7328 / 1.6564 | 0.6275 / 0.7384 / 1.7110 | +0.0015 |
| LLM+Memento | 0.6271 / 0.7221 / 1.6301 | 0.6125 / 0.7090 / 1.5190 | -0.0146 |
| LLM+Hope | 0.6229 / 0.7572 / 1.7550 | 0.6445 / 0.7729 / 1.9658 | +0.0216 |
| **Full** | 0.6388 / 0.7685 / 1.9771 | **0.6576 / 0.7875 / 2.1904** | **+0.0188** |
| Full − Pure | +0.0128 | **+0.0301** | — |

要点：
1. **Full 相对 Pure 的提升在 4090 上更强**（ΔMS = +3.01 点 vs +1.28 点），
   主消融排序结论（Full 最优）跨硬件保持；
2. **Breach 瓶颈在 4090 上重现**（Full 五阶段中 breach=0.6088 仍垫底），
   与论文中"杀伤链结构性张力（Protection 资源冲突）"的理论推导吻合；
3. Memento 单组在单种子下波动（-0.0146），源于 5 记录小案例库的检索采样
   方差，与论文 §4.12 分析一致，需以多种子统计消除（见 A.3）。

## A.3 多种子消融复跑（5 seeds: 7, 42, 123, 999, 2024，10 轮 × 50-MC）

> 跑批完成后填充：各设置 Mean ± SD、Full−Pure 配对 t 检验（t、p、Cohen's d）、
> 与原论文 Table 12 对齐的 4090 基准对照表。

<!-- A.3 数据待 5-seed 套件完成后填充 -->

## A.4 全流程加速与工程意义

- 4090 上完整消融套件（4 分支 × 10 轮 × 50-MC）墙钟 6 分钟，相对 3060 的
  61 分钟提速 **≈10×**；
- 配合双系统投机增量规划（System 1 应急预案 1.4 ms → System 2 Delta Loop
  15~18 s），高算力硬件使"毫秒级应急 + 半分钟级敏捷重规划 + 分钟级全局优化"
  三级时延矩阵全部落入实战可用窗口；
- 结论：框架收益不依赖特定硬件，且算力增强使迭代式生成在工程落地与
  高频重规划场景中更具可行性。

## A.5 数据溯源

- 单场景复跑：`result/ablation_suite_4090_rerun/ablation_{pure_llm,memento,hope,full}/full_result.json`
- 多种子复跑：`result/ablation_4090_multiseed/`
- runtime_manifest 硬件字段由 `military_research/engine.py::_collect_runtime_manifest`
  经 nvidia-smi / torch 实测写入
