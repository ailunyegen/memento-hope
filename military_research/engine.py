from __future__ import annotations

import json
import math
import os
import platform
import random
import re
import subprocess
import time
import uuid
from collections import defaultdict
from datetime import datetime
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .case_memory import CaseBank, CaseRecord
from .domain import (
    ACTION_TYPES,
    CAPABILITY_KEYS,
    ActionItem,
    CombatPlan,
    PhaseSimulation,
    PlanPhase,
    Scenario,
    SimulationResult,
    clamp_value,
    infer_action_type,
)
from .exporters import build_c2sim_xml, build_ov5b, build_ov6c


SLOW_WEIGHT_LIBRARY: Dict[str, Dict[str, float]] = {
    "balanced_joint": {
        "fires": 0.72,
        "mobility": 0.70,
        "protection": 0.70,
        "awareness": 0.78,
        "ew": 0.60,
        "sustainment": 0.66,
        "c2": 0.80,
    },
    "offensive_breakthrough": {
        "fires": 0.82,
        "mobility": 0.80,
        "protection": 0.64,
        "awareness": 0.76,
        "ew": 0.66,
        "sustainment": 0.62,
        "c2": 0.76,
    },
    "defense_in_depth": {
        "fires": 0.74,
        "mobility": 0.52,
        "protection": 0.84,
        "awareness": 0.78,
        "ew": 0.62,
        "sustainment": 0.74,
        "c2": 0.80,
    },
}


class BottleneckDetector:
    """Dynamically identifies bottleneck stages from simulation feedback and computes
    targeted capability boosts.  Replaces the fixed ``bottleneck_boost`` vector with a
    history-aware diagnosis that answers the reviewer observation that *breach* is the
    persistent weak point while giving the controller a principled way to re-allocate
    emphasis toward the current bottleneck."""

    STAGE_KEYS = ["detect", "disrupt", "breach", "control", "sustain"]

    # stage → capability-dimension mapping (same indices as CAPABILITY_KEYS)
    STAGE_TO_CAP: Dict[str, torch.Tensor] = {
        "detect":   torch.tensor([0.08, 0.06, 0.04, 0.42, 0.18, 0.02, 0.20]),
        "disrupt":  torch.tensor([0.42, 0.06, 0.04, 0.08, 0.20, 0.04, 0.16]),
        "breach":   torch.tensor([0.26, 0.30, 0.22, 0.06, 0.04, 0.02, 0.10]),
        "control":  torch.tensor([0.08, 0.18, 0.24, 0.10, 0.06, 0.06, 0.28]),
        "sustain":  torch.tensor([0.06, 0.12, 0.12, 0.06, 0.04, 0.40, 0.20]),
    }

    def __init__(self, window_size: int = 6) -> None:
        self.window_size = window_size
        self.stage_history: List[Dict[str, float]] = []
        self.bottleneck_history: List[Dict[str, Any]] = []
        self.bottleneck_counts: Dict[str, int] = {k: 0 for k in self.STAGE_KEYS}
        self.capability_boost_history: List[Dict[str, float]] = []

    # ------------------------------------------------------------------
    def update(self, stage_scores: Dict[str, float]) -> None:
        """Record one iteration's stage scores."""
        snap = {k: stage_scores.get(k, 0.5) for k in self.STAGE_KEYS}
        self.stage_history.append(snap)
        if len(self.stage_history) > self.window_size * 2:
            self.stage_history = self.stage_history[-self.window_size :]

    # ------------------------------------------------------------------
    def identify(self) -> Tuple[str, float, Dict[str, float]]:
        """Return ``(bottleneck_stage, severity, deficits)`` for the current iteration.

        *severity* = how much the weakest stage lags behind the mean of the other
        four stages.  Larger values signal a sharper, more isolated bottleneck.
        """
        if not self.stage_history:
            return ("breach", 0.30, {k: 0.5 for k in self.STAGE_KEYS})

        latest = self.stage_history[-1]
        deficits = {k: max(0.0, 1.0 - latest[k]) for k in self.STAGE_KEYS}

        ranked = sorted(deficits.items(), key=lambda x: x[1], reverse=True)
        bottleneck_stage, bottleneck_deficit = ranked[0]
        others = [d for _, d in ranked[1:]]
        mean_other = sum(others) / max(1, len(others))

        severity = max(0.0, bottleneck_deficit - mean_other)  # [0, ~0.5]
        severity = min(severity, 0.50)

        self.bottleneck_counts[bottleneck_stage] += 1

        info = {
            "bottleneck_stage": bottleneck_stage,
            "severity": round(severity, 4),
            "deficits": {k: round(v, 4) for k, v in deficits.items()},
        }
        self.bottleneck_history.append(info)
        return bottleneck_stage, severity, deficits

    # ------------------------------------------------------------------
    def compute_targeted_boost(
        self, bottleneck_stage: str, severity: float
    ) -> torch.Tensor:
        """Capability boost vector focused on *bottleneck_stage*, scaled by severity."""
        base = self.STAGE_TO_CAP.get(bottleneck_stage, torch.zeros(7))
        scale = 0.40 + 1.6 * severity  # 0.40 → 1.20 depending on severity
        return base * scale

    # ------------------------------------------------------------------
    def summary(self) -> Dict[str, Any]:
        total = max(1, sum(self.bottleneck_counts.values()))
        return {
            "bottleneck_distribution": {
                k: round(v / total, 4) for k, v in self.bottleneck_counts.items()
            },
            "most_common": max(
                self.bottleneck_counts, key=lambda k: self.bottleneck_counts[k]
            ),
            "counts": dict(self.bottleneck_counts),
        }

    # ------------------------------------------------------------------
    def snapshot(self) -> Dict[str, Any]:
        return {
            "bottleneck_history": list(self.bottleneck_history),
            "summary": self.summary(),
        }


class HOPEAdapter(nn.Module):
    hidden_state: torch.Tensor
    t_current: torch.Tensor
    t_create: torch.Tensor

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入门与候选状态使用独立线性层，增强门控表达能力
        self.input_gate_proj = nn.Linear(input_dim, hidden_dim)
        self.candidate_proj = nn.Linear(input_dim, hidden_dim)
        self.i2o = nn.Linear(input_dim, output_dim)
        
        # 价值估计器 V(c_t) (替代原本的粗糙遗忘门)
        self.value_estimator = nn.Sequential(
            nn.Linear(input_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
        # 信息论遗忘门控参数
        self.forget_gamma = nn.Parameter(torch.ones(output_dim))      # 价值系数
        self.forget_delta = nn.Parameter(torch.full((output_dim,), -0.1)) # 时间衰减系数 (负数)
        self.forget_beta = nn.Parameter(torch.zeros(output_dim))      # 偏置
        
        # 隐藏状态仅在显式反馈后刷新，避免推理阶段漂移
        self.register_buffer("hidden_state", torch.zeros(hidden_dim))
        
        # 时间追踪器
        self.register_buffer('t_current', torch.tensor(1.0))
        self.register_buffer('t_create', torch.ones(output_dim))
        
        # 隐藏到输出
        self.h2o = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor, update_state: bool = False, refresh_rate: float = 0.0) -> torch.Tensor:
        # x: [batch, input_dim]
        h = self.hidden_state.unsqueeze(0).expand(x.size(0), -1)  # [batch, hidden_dim]
        
        # 1. 记忆价值估计 V(c_t) = I / H 的神经网络逼近
        val_input = torch.cat([x, h], dim=1)  # [batch, input_dim + hidden_dim]
        v = self.value_estimator(val_input)   # [batch, output_dim]
        
        # 2. 年龄计算 log(t_current / t_create)
        time_floor = torch.ones((), dtype=self.t_create.dtype, device=self.t_create.device)
        current_time = torch.clamp(self.t_current, min=time_floor)
        create_time = torch.clamp(self.t_create, min=time_floor)
        age_ratio = torch.clamp(current_time / create_time, min=time_floor)  # [output_dim]
        log_age = torch.log(age_ratio).unsqueeze(0).expand(x.size(0), -1)  # [batch, output_dim]
        
        # 3. 双维度遗忘门控
        # 保证 delta 始终为负，以确保随时间衰减的物理意义
        delta_neg = -F.relu(-self.forget_delta) 
        forget_logits = self.forget_gamma * v + delta_neg * log_age + self.forget_beta
        forget = torch.sigmoid(forget_logits)  # [batch, output_dim]
        
        # 输入门与候选状态
        input_gate = torch.sigmoid(self.input_gate_proj(x))  # [batch, hidden_dim]
        candidate = torch.tanh(self.candidate_proj(x))       # [batch, hidden_dim]
        
        # 仅在反馈阶段更新隐藏状态
        new_h = (1 - input_gate) * h + input_gate * candidate  # [batch, hidden_dim]
        if update_state:
            bounded_refresh = float(max(0.0, min(1.0, refresh_rate)))
            self.t_current.add_(1.0)
            self.hidden_state.copy_(new_h.mean(dim=0).detach())
            updated_t_create = (1.0 - bounded_refresh) * self.t_create + bounded_refresh * self.t_current
            self.t_create.copy_(updated_t_create)
        
        # 输出
        output = self.i2o(x) + self.h2o(h)  # [batch, output_dim]
        
        # 应用基于价值和时间的遗忘
        output = forget * output
        
        return output


class DualMemoryController(nn.Module):
    doctrine_prior: torch.Tensor
    past_slow_weights: torch.Tensor
    current_fast: torch.Tensor
    previous_reward: torch.Tensor
    current_features: torch.Tensor | None
    torch_generator: torch.Generator

    def __init__(self, doctrine_profile: str, seed: int | None = None,
                  sde_theta: float = 0.5, sde_epsilon: float = 0.02):
        super().__init__()
        self.doctrine_profile = doctrine_profile
        self.current_scenario: Scenario | None = None
        if seed is not None:
            torch.manual_seed(seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(seed)
        self.torch_generator = torch.Generator()
        if seed is not None:
            self.torch_generator.manual_seed(seed)
        self.bottleneck_detector = BottleneckDetector(window_size=6)
        
        # 初始化slow_weights为可学习参数
        initial_slow = SLOW_WEIGHT_LIBRARY.get(doctrine_profile, SLOW_WEIGHT_LIBRARY["balanced_joint"])
        self.slow_weights = nn.Parameter(torch.tensor(list(initial_slow.values()), dtype=torch.float32))
        self.register_buffer("doctrine_prior", torch.tensor(list(initial_slow.values()), dtype=torch.float32))
        
        # 时滞动量追踪
        self.register_buffer('past_slow_weights', torch.tensor(list(initial_slow.values()), dtype=torch.float32))
        
        # 定义terrain和weather的词汇表
        self.terrain_vocab = ["urban", "mountain", "coastal", "littoral", "desert", "other"]
        self.weather_vocab = ["rain", "storm", "fog", "clear", "other"]
        
        # 输入维度：terrain_onehot (6) + weather_onehot (5) + ew_threat + threat_level + time_pressure = 6+5+3=14
        input_dim = len(self.terrain_vocab) + len(self.weather_vocab) + 3
        hidden_dim = 32
        output_dim = len(CAPABILITY_KEYS)
        
        self.adapter = HOPEAdapter(input_dim, hidden_dim, output_dim)
        
        # SDE参数
        self.sde_theta = sde_theta      # 回复系数 (遗忘率)
        self.sde_epsilon = sde_epsilon   # 噪声强度
        self.sde_dt = 1.0         # 离散化时间步长
        self.register_buffer('current_fast', torch.zeros(output_dim))
        self.register_buffer("previous_reward", torch.tensor(0.0))
        self.current_features = None
        
        # 优化器 (仅用于Adapter网络)
        self.adapter_optimizer = torch.optim.Adam(self.adapter.parameters(), lr=0.01)
        
    def _encode_scenario(self, scenario: Scenario) -> torch.Tensor:
        # One-hot编码terrain
        terrain_lower = scenario.terrain.lower()
        terrain_onehot = torch.zeros(len(self.terrain_vocab))
        for i, t in enumerate(self.terrain_vocab):
            if t in terrain_lower:
                terrain_onehot[i] = 1.0
        if terrain_onehot.sum() == 0:
            terrain_onehot[-1] = 1.0  # other
        
        # One-hot编码weather
        weather_lower = scenario.weather.lower()
        weather_onehot = torch.zeros(len(self.weather_vocab))
        for i, w in enumerate(self.weather_vocab):
            if w in weather_lower:
                weather_onehot[i] = 1.0
        if weather_onehot.sum() == 0:
            weather_onehot[-1] = 1.0  # other
        
        # 数值特征
        numeric = torch.tensor([
            scenario.ew_threat,
            scenario.threat_level,
            scenario.time_pressure
        ], dtype=torch.float32)
        
        # 拼接
        features = torch.cat([terrain_onehot, weather_onehot, numeric])
        return features.unsqueeze(0)  # [1, input_dim]
    
    def fast_weights(
        self,
        scenario: Scenario,
        memory_metrics: List[Dict[str, float]] | None = None,
    ) -> Dict[str, float]:
        self.current_scenario = scenario
        self.current_features = self._encode_scenario(scenario)
        self.adapter.eval()
        with torch.no_grad():
            target_fast = self.adapter(self.current_features, update_state=False).squeeze(0)  # [output_dim]

        # Memory-conditioned correction: retrieved case metrics bias fast-weight
        # initialisation toward capabilities that were weak in similar past cases.
        if memory_metrics:
            mem_correction = self._compute_memory_correction(memory_metrics)
            target_fast = target_fast + 0.30 * mem_correction

        # SDE Euler-Maruyama 离散化步进:
        # dW_fast = theta * (F(W_slow, Phi) - W_fast) * dt + epsilon * sqrt(dt) * noise
        noise = torch.empty_like(self.current_fast).normal_(generator=self.torch_generator)
        with torch.no_grad():
            dW = self.sde_theta * (target_fast - self.current_fast) * self.sde_dt + \
                 self.sde_epsilon * math.sqrt(self.sde_dt) * noise
            self.current_fast += dW
            lower_bounds, upper_bounds = self._fast_weight_bounds(scenario)
            self.current_fast.copy_(torch.max(torch.min(self.current_fast, upper_bounds), lower_bounds))

        weights = {key: float(adj) for key, adj in zip(CAPABILITY_KEYS, self.current_fast)}
        return {key: round(value, 4) for key, value in weights.items()}

    # ------------------------------------------------------------------
    def _compute_memory_correction(
        self, memory_metrics: List[Dict[str, float]]
    ) -> torch.Tensor:
        """Translate retrieved case metrics into a fast-weight correction vector.

        The intuition: if a retrieved case had a low score in a particular kill-chain
        stage, the controller nudges fast weights toward the capabilities that feed
        that stage, so the planner is biased to compensate for known weaknesses.
        """
        correction = torch.zeros(len(CAPABILITY_KEYS))
        for metrics in memory_metrics:
            for stage in BottleneckDetector.STAGE_KEYS:
                score = float(metrics.get(stage, 0.5))
                if score < 0.55:
                    deficit = 1.0 - score
                    correction += (
                        BottleneckDetector.STAGE_TO_CAP[stage] * deficit * 0.18
                    )
        return torch.clamp(correction, -0.18, 0.28)
    
    def fuse(self, fast_weights: Dict[str, float]) -> Dict[str, float]:
        # ???????????????????????
        fast_tensor = torch.tensor(
            list(fast_weights.values()),
            dtype=self.slow_weights.dtype,
            device=self.slow_weights.device,
        )
        fused_tensor = torch.clamp(self.slow_weights + 0.72 * fast_tensor, 0.24, 1.0)
        if self.current_scenario is not None:
            scenario = self.current_scenario
            if scenario.mission_type in {"assault", "defense"}:
                fused_tensor[0] = torch.maximum(fused_tensor[0], fused_tensor.new_tensor(max(0.62, float(self.doctrine_prior[0]) - 0.02)))
                fused_tensor[1] = torch.maximum(fused_tensor[1], fused_tensor.new_tensor(max(0.58, float(self.doctrine_prior[1]) - 0.03)))
                fused_tensor[2] = torch.maximum(fused_tensor[2], fused_tensor.new_tensor(max(0.56, float(self.doctrine_prior[2]) - 0.02)))
            if scenario.mission_type in {"recon", "sustain"}:
                fused_tensor[3] = torch.maximum(fused_tensor[3], fused_tensor.new_tensor(max(0.66, float(self.doctrine_prior[3]) - 0.02)))
                fused_tensor[5] = torch.maximum(fused_tensor[5], fused_tensor.new_tensor(max(0.58, float(self.doctrine_prior[5]) - 0.02)))
                fused_tensor[6] = torch.maximum(fused_tensor[6], fused_tensor.new_tensor(max(0.68, float(self.doctrine_prior[6]) - 0.02)))
        if self.current_scenario is not None and self.current_scenario.ew_threat > 0.55:
            awareness_floor = max(0.68, float(self.doctrine_prior[3]) - 0.03)
            c2_floor = max(0.70, float(self.doctrine_prior[6]) - 0.02)
            fused_tensor[3] = torch.maximum(fused_tensor[3], fused_tensor.new_tensor(awareness_floor))
            fused_tensor[6] = torch.maximum(fused_tensor[6], fused_tensor.new_tensor(c2_floor))
        fused = {key: round(float(val), 4) for key, val in zip(CAPABILITY_KEYS, fused_tensor)}
        return fused

    def _fast_weight_bounds(self, scenario: Scenario) -> Tuple[torch.Tensor, torch.Tensor]:
        lower = torch.tensor([-0.08, -0.05, -0.05, -0.03, -0.08, -0.06, -0.02], dtype=torch.float32)
        upper = torch.tensor([0.28, 0.24, 0.22, 0.12, 0.28, 0.16, 0.14], dtype=torch.float32)
        if scenario.ew_threat > 0.55:
            lower[3] = -0.02
            lower[6] = 0.00
            upper[4] = 0.32
            upper[6] = 0.18
        if scenario.mission_type in {"assault", "defense"}:
            lower[0] = max(lower[0], -0.04)
            lower[1] = max(lower[1], -0.03)
            lower[2] = max(lower[2], -0.03)
        if scenario.mission_type in {"recon", "sustain"}:
            lower[3] = max(lower[3], -0.02)
            lower[5] = max(lower[5], -0.02)
            lower[6] = max(lower[6], -0.01)
        if scenario.threat_level > 0.70:
            upper[0] += 0.04
            upper[1] += 0.03
            upper[2] += 0.03
        return lower, upper
    def integrate_feedback(
        self,
        current_plan: CombatPlan,
        result: SimulationResult,
        best_plan: CombatPlan | None = None,
        best_result: SimulationResult | None = None,
    ) -> None:
        stage_scores = result.stage_scores or {}
        deficits = {
            "detect": 1.0 - stage_scores.get("detect", 0.5),
            "disrupt": 1.0 - stage_scores.get("disrupt", 0.5),
            "breach": 1.0 - stage_scores.get("breach", 0.5),
            "control": 1.0 - stage_scores.get("control", 0.5),
            "sustain": 1.0 - stage_scores.get("sustain", 0.5),
        }
        stage_priority = {
            "detect": 1.00,
            "disrupt": 1.30,
            "breach": 1.45,
            "control": 0.95,
            "sustain": 0.90,
        }
        stage_to_capability = {
            "detect": torch.tensor([0.08, 0.06, 0.04, 0.42, 0.18, 0.02, 0.20]),
            "disrupt": torch.tensor([0.42, 0.06, 0.04, 0.08, 0.20, 0.04, 0.16]),
            "breach": torch.tensor([0.26, 0.30, 0.22, 0.06, 0.04, 0.02, 0.10]),
            "control": torch.tensor([0.08, 0.18, 0.24, 0.10, 0.06, 0.06, 0.28]),
            "sustain": torch.tensor([0.06, 0.12, 0.12, 0.06, 0.04, 0.40, 0.20]),
        }

        # ── Dynamic bottleneck detection ──
        self.bottleneck_detector.update(stage_scores)
        bn_stage, bn_severity, _ = self.bottleneck_detector.identify()

        deficit_vector = torch.zeros_like(self.slow_weights)
        for stage_name, deficit in deficits.items():
            deficit_vector += stage_to_capability[stage_name] * float(deficit) * stage_priority[stage_name]

        # Dynamic targeted boost replaces the old fixed bottleneck_boost
        dynamic_boost = self.bottleneck_detector.compute_targeted_boost(
            bn_stage, bn_severity
        )

        reward = 0.7 * float(result.mission_success) + 0.3 * float(result.overall_effectiveness)
        if best_result is not None:
            reward += 0.05 * max(0.0, float(best_result.mission_success) - float(result.mission_success))
        reward_delta = reward - float(self.previous_reward.item())
        regularization = 0.05 * (self.slow_weights.detach() - self.doctrine_prior)
        utility_gradient = deficit_vector + dynamic_boost - regularization

        if self.current_features is not None:
            ew_pressure = float(self.current_features[0, -3].item())
            support_vector = torch.tensor(
                [
                    0.05 * deficits["disrupt"],
                    0.05 * deficits["breach"],
                    0.04 * deficits["breach"],
                    0.12 * deficits["detect"] + 0.05 * ew_pressure,
                    0.08 * deficits["disrupt"] + 0.04 * ew_pressure,
                    0.04 * deficits["sustain"],
                    0.12 * deficits["detect"] + 0.08 * ew_pressure,
                ],
                dtype=torch.float32,
            )
            self.adapter.train()
            target_fast = self.adapter(self.current_features, update_state=False).squeeze(0)
            desired_fast = 0.42 * utility_gradient + 0.18 * dynamic_boost + support_vector
            if self.current_scenario is not None:
                lower_bounds, upper_bounds = self._fast_weight_bounds(self.current_scenario)
                desired_fast = torch.max(torch.min(desired_fast, upper_bounds), lower_bounds)
            else:
                desired_fast = torch.clamp(desired_fast, -0.20, 0.32)
            loss_adapter = F.smooth_l1_loss(target_fast, desired_fast)
            self.adapter_optimizer.zero_grad()
            loss_adapter.backward()
            self.adapter_optimizer.step()
            with torch.no_grad():
                refresh_rate = max(0.05, min(0.95, reward))
                self.adapter(self.current_features, update_state=True, refresh_rate=refresh_rate)

        with torch.no_grad():
            W_t = self.slow_weights.data.clone()
            delay_diff = W_t - self.past_slow_weights
            g_final = utility_gradient + 0.08 * delay_diff
            self.past_slow_weights.data.copy_(W_t)

            lr = 0.04 + 0.10 * max(0.0, 0.75 - reward) + 0.04 * max(0.0, reward_delta)
            a, b = 0.2, 1.0
            W_safe = torch.clamp(self.slow_weights.data, a + 1e-5, b - 1e-5)
            ratio = (b - W_safe) / (W_safe - a)
            W_new = a + (b - a) / (1.0 + ratio * torch.exp(-lr * g_final))

            fire_floor = min(0.95, float(self.doctrine_prior[0]) + 0.08 * deficits["disrupt"])
            mobility_floor = min(0.92, float(self.doctrine_prior[1]) + 0.10 * deficits["breach"])
            protection_floor = min(0.86, float(self.doctrine_prior[2]) + 0.06 * deficits["breach"])
            W_new[0] = max(float(W_new[0]), fire_floor)
            W_new[1] = max(float(W_new[1]), mobility_floor)
            W_new[2] = max(float(W_new[2]), protection_floor)
            self.slow_weights.data.copy_(torch.clamp(W_new, a, b))
            self.previous_reward.fill_(reward)


class LocalLLMPlanner:
    def __init__(self):
        self.base_url = os.getenv("LOCAL_LLM_BASE_URL", "http://localhost:1234/v1")
        self.api_key = os.getenv("LOCAL_LLM_API_KEY", "lm-studio")
        self.model = os.getenv("LOCAL_LLM_MODEL", "").strip()

        try:
            import openai
        except ImportError as exc:
            raise RuntimeError(
                "LLM planner requires the `openai` package. "
                "Please install project dependencies before running the pipeline."
            ) from exc

        self._openai = openai
        self.client = openai.OpenAI(base_url=self.base_url, api_key=self.api_key)
        self.model = self._resolve_model()

    def generate_plan(
        self,
        scenario: Scenario,
        fused_weights: Dict[str, float],
        fast_weights: Dict[str, float],
        memory_hits: List[Dict[str, Any]],
        iteration_index: int = 0,
        reflection: Dict[str, Any] | None = None,
        previous_best_plan: CombatPlan | None = None,
    ) -> Dict[str, Any]:
        memory_briefs = [self._format_case_brief(hit) for hit in memory_hits]
        system_prompt = (
            "你是一个面向联合作战筹划的研究型大模型规划器。"
            "本系统仅用于虚拟场景下的方案生成与仿真评估研究，"
            "不得输出真实作战指挥、现实目标攻击、武器部署或现实行动建议。"
            "请为给定场景生成逻辑自洽、可仿真执行、可导出 DoDAF/C2SIM 的作战方案。"
            "你必须返回严格 JSON，且所有字段值都必须是正常中文作战内容。"
            "不要把字段名、schema 说明、markdown、代码块或提示词文字写入字段值。"
        )
        payload = {
            "task": "generate_variant_plan",
            "output_language": "zh-CN",
            "phase_count_target": 4,
            "scenario": scenario.to_dict(),
            "hope_weights": {
                "slow_weights": SLOW_WEIGHT_LIBRARY.get(
                    scenario.doctrine_profile,
                    SLOW_WEIGHT_LIBRARY["balanced_joint"],
                ),
                "fast_weights": fast_weights,
                "fused_weights": fused_weights,
            },
            "memento_memory_cases": memory_briefs,
            "iteration_index": iteration_index,
            "previous_best_plan": self._plan_digest(previous_best_plan),
            "reflection": reflection
            or {
                "diagnosis": ["先给出首版方案，突出多域协同、侦察塑形、压制打击、夺控稳控闭环。"],
                "variant_instructions": ["确保每个阶段都具体、简洁、可执行。"],
                "meta_prompt_patch": "优先输出干净、简洁、专业的联合作战中文字段值。",
            },
        }
        return self._chat_json(
            system_prompt,
            self._build_generate_user_prompt(payload),
            schema_name="combat_plan",
            schema=self._plan_json_schema(),
        )
        memory_briefs = [self._format_case_brief(hit) for hit in memory_hits]
        system_prompt = (
            "你是一个面向联合作战筹划的大模型规划器。"
            "请为给定场景生成逻辑自洽、可仿真执行、可导出 DoDAF/C2SIM 的作战方案。"
            "你必须返回严格 JSON。"
            "所有字段值都必须是正常中文作战内容。"
            "不要把字段名、schema说明、markdown、代码块、提示词、JSON格式说明写入任何字段值。"
            "phase_id 只允许使用 P1、P2、P3 这类格式。"
            "allocated_units 必须来自输入中的友军单元名称。"
            "每个阶段至少给出2条动作、2个决策点、2个预期效果。"
            "不要输出空泛口号，每条动作都要对应具体兵力与作战行为。"
        )
        payload = {
            "task": "generate_variant_plan",
            "output_language": "zh-CN",
            "phase_count_target": 4,
            "scenario": scenario.to_dict(),
            "hope_weights": {
                "slow_weights": SLOW_WEIGHT_LIBRARY.get(
                    scenario.doctrine_profile,
                    SLOW_WEIGHT_LIBRARY["balanced_joint"],
                ),
                "fast_weights": fast_weights,
                "fused_weights": fused_weights,
            },
            "memento_memory_cases": memory_briefs,
            "iteration_index": iteration_index,
            "previous_best_plan": self._plan_digest(previous_best_plan),
            "reflection": reflection or {
                "diagnosis": ["先给出首版方案，突出多域协同、塑形-打击-夺控-稳控闭环。"],
                "variant_instructions": ["确保每个阶段都具体、简洁、可执行。"],
                "meta_prompt_patch": "优先输出干净、简洁、专业的联合作战中文字段值。",
            },
        }
        return self._chat_json(
            system_prompt,
            self._build_generate_user_prompt(payload),
            schema_name="combat_plan",
            schema=self._plan_json_schema(),
        )

    def reflect(
        self,
        scenario: Scenario,
        best_plan: CombatPlan,
        best_result: SimulationResult,
    ) -> Dict[str, Any]:
        system_prompt = (
            "你是作战方案优化器。"
            "本系统仅用于虚拟场景下的方案生成与仿真评估研究，"
            "不得输出真实作战指挥、现实目标攻击、武器部署或现实行动建议。"
            "你需要根据仿真失败教训，给出下一轮 Prompt 的反思增量。"
            "只输出严格 JSON。"
        )
        payload = {
            "task": "reflect_and_prepare_next_prompt",
            "scenario": scenario.to_dict(),
            "best_plan_summary": self._plan_digest(best_plan),
            "simulation_result": best_result.to_dict(),
            "required_schema": {
                "diagnosis": ["string"],
                "variant_instructions": ["string"],
                "meta_prompt_patch": "string",
            },
        }
        reflection = self._chat_json(
            system_prompt,
            self._build_reflection_user_prompt(payload),
            schema_name="reflection_patch",
            schema=self._reflection_json_schema(),
        )
        reflection.setdefault("diagnosis", [])
        reflection.setdefault("variant_instructions", [])
        reflection.setdefault("meta_prompt_patch", "")
        return reflection
        system_prompt = (
            "你是作战方案优化器。"
            "你需要根据仿真失败教训，给出下一轮 Prompt 的反思增量。"
            "只输出严格 JSON。"
            "不要复述 schema，不要输出乱码，不要写与任务无关的格式说明。"
        )
        payload = {
            "task": "reflect_and_prepare_next_prompt",
            "scenario": scenario.to_dict(),
            "best_plan_summary": self._plan_digest(best_plan),
            "simulation_result": best_result.to_dict(),
            "required_schema": {
                "diagnosis": ["string"],
                "variant_instructions": ["string"],
                "meta_prompt_patch": "string",
            },
        }
        reflection = self._chat_json(
            system_prompt,
            self._build_reflection_user_prompt(payload),
            schema_name="reflection_patch",
            schema=self._reflection_json_schema(),
        )
        reflection.setdefault("diagnosis", [])
        reflection.setdefault("variant_instructions", [])
        reflection.setdefault("meta_prompt_patch", "")
        return reflection

    def _chat_json(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
    ) -> Dict[str, Any]:
        schema_errors: List[str] = []
        for attempt in range(1, 4):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={
                        "type": "json_schema",
                        "json_schema": {"name": schema_name, "schema": schema},
                    },
                )
                content = response.choices[0].message.content or ""
                return self._parse_json_with_repair(system_prompt, user_prompt, schema_name, schema, content)
            except Exception as exc:
                schema_errors.append(f"json_schema attempt {attempt}: {exc}")

        fallback_errors: List[str] = []
        for attempt in range(1, 3):
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    temperature=0.1,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {
                            "role": "user",
                            "content": user_prompt
                            + "\n\n请严格只返回一个合法 JSON 对象，不要包含 markdown、解释、代码块或额外文本。",
                        },
                    ],
                )
                content = response.choices[0].message.content or ""
                return self._parse_json_with_repair(system_prompt, user_prompt, schema_name, schema, content)
            except Exception as exc:
                fallback_errors.append(f"fallback attempt {attempt}: {exc}")

        raise RuntimeError(
            "Local LLM failed after retry/fallback. "
            f"base_url={self.base_url}, model={self.model}, schema={schema_name}, "
            f"json_schema_errors={schema_errors}, fallback_errors={fallback_errors}"
        )
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                temperature=0.1,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema_name,
                        "schema": schema,
                    },
                },
            )
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to local LLM at {self.base_url}. "
                "Please ensure LM Studio or another OpenAI-compatible server is running."
            ) from exc

        content = response.choices[0].message.content or ""
        cleaned = self._extract_json_block(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"Local LLM returned invalid JSON: {content}") from exc

    def _resolve_model(self) -> str:
        if self.model:
            return self.model
        try:
            models = self.client.models.list()
            if getattr(models, "data", None):
                return models.data[0].id
        except Exception:
            pass
        return "local-model"

    def _parse_json_with_repair(
        self,
        system_prompt: str,
        user_prompt: str,
        schema_name: str,
        schema: Dict[str, Any],
        content: str,
    ) -> Dict[str, Any]:
        cleaned = self._extract_json_block(content)
        try:
            return json.loads(cleaned)
        except json.JSONDecodeError as first_error:
            repair_prompt = (
                "上一个回答不是合法 JSON。"
                "请仅依据原任务，重新输出一个完全合法的 JSON 对象。"
                "不要包含 markdown、解释、代码块或额外文本。"
            )
            last_exc: Exception = first_error
            for _ in range(2):
                try:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        temperature=0.0,
                        messages=[
                            {"role": "system", "content": system_prompt},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": content},
                            {"role": "user", "content": repair_prompt},
                        ],
                        response_format={
                            "type": "json_schema",
                            "json_schema": {"name": schema_name, "schema": schema},
                        },
                    )
                    repaired = response.choices[0].message.content or ""
                    return json.loads(self._extract_json_block(repaired))
                except Exception as exc:  # noqa: PERF203
                    last_exc = exc
            raise RuntimeError(
                f"Local LLM returned invalid JSON after repair attempts. schema={schema_name}, content={content}"
            ) from last_exc

    def _extract_json_block(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[^\n]*\n", "", stripped)
            stripped = re.sub(r"\n```$", "", stripped).strip()
        match = re.search(r"\{[\s\S]*\}", stripped)
        return match.group(0) if match else stripped

    def _format_case_brief(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        record: CaseRecord = hit["record"]
        usage_mode = "imitate" if record.outcome == "positive" else "avoid"
        return {
            "case_id": record.case_id,
            "similarity": hit["score"],
            "confidence": hit["confidence"],
            "quality": hit.get("quality"),
            "usage_mode": usage_mode,
            "mission_type": record.mission_type,
            "terrain": record.terrain,
            "objective": record.objective,
            "friendly_roles": record.friendly_roles,
            "enemy_roles": record.enemy_roles,
            "key_actions": record.key_actions,
            "lessons": record.lessons,
            "outcome": record.outcome,
            "metrics": record.metrics,
        }
        record: CaseRecord = hit["record"]
        return {
            "case_id": record.case_id,
            "similarity": hit["score"],
            "confidence": hit["confidence"],
            "quality": hit.get("quality"),
            "mission_type": record.mission_type,
            "terrain": record.terrain,
            "objective": record.objective,
            "friendly_roles": record.friendly_roles,
            "enemy_roles": record.enemy_roles,
            "key_actions": record.key_actions,
            "lessons": record.lessons,
            "outcome": record.outcome,
            "metrics": record.metrics,
        }

    def _plan_digest(self, plan: CombatPlan | None) -> Dict[str, Any] | None:
        if plan is None:
            return None
        return {
            "title": plan.title,
            "commander_intent": plan.commander_intent,
            "theory_of_victory": plan.theory_of_victory,
            "risk_controls": plan.risk_controls,
            "phase_names": [phase.name for phase in plan.phases],
            "phase_actions": {
                phase.phase_id: [action.to_dict() for action in phase.actions]
                for phase in plan.phases
            },
        }

    def _build_generate_user_prompt(self, payload: Dict[str, Any]) -> str:
        scenario = payload["scenario"]
        memory_cases = payload["memento_memory_cases"]
        reflection = payload["reflection"]
        previous_best_plan = payload["previous_best_plan"]
        hope = payload["hope_weights"]

        lines = [
            "请基于以下信息生成联合作战方案。",
            "",
            "安全边界：",
            "- 仅用于虚拟场景下的方案生成与仿真评估研究。",
            "- 不得输出真实作战指挥、现实目标攻击、武器部署或现实行动建议。",
            "",
            "场景信息：",
            f"- 名称：{scenario['name']}",
            f"- 任务类型：{scenario['mission_type']}",
            f"- 作战目标：{scenario['objective']}",
            f"- 地形：{scenario['terrain']}",
            f"- 天气：{scenario['weather']}",
            f"- 威胁等级：{scenario['threat_level']}",
            f"- 电磁威胁：{scenario['ew_threat']}",
            f"- 平民密度：{scenario['civilian_presence']}",
            f"- 时间压力：{scenario['time_pressure']}",
            f"- 期望终态：{scenario['desired_end_state']}",
            f"- 约束：{'；'.join(scenario['constraints']) if scenario['constraints'] else '无'}",
            "",
            "友军单元：",
            f"- {'；'.join(unit['name'] for unit in scenario['friendly_forces'])}",
            "",
            "敌军单元：",
            f"- {'；'.join(unit['name'] for unit in scenario['enemy_forces'])}",
            "",
            "Hope 权重：",
            f"- 慢权重：{json.dumps(hope['slow_weights'], ensure_ascii=False)}",
            f"- 快权重：{json.dumps(hope['fast_weights'], ensure_ascii=False)}",
            f"- 融合权重：{json.dumps(hope['fused_weights'], ensure_ascii=False)}",
            "",
            "Memento 相关案例：",
        ]
        for case in memory_cases:
            lines.append(
                f"- {case['case_id']} | 用法={case['usage_mode']} | 相似度={case['similarity']:.2f} | "
                f"关键动作：{'；'.join(case['key_actions'][:4])} | 教训：{'；'.join(case['lessons'][:2])}"
            )
        if memory_cases:
            lines.extend(
                [
                    "- positive / imitate：仅复用高质量正样本的关键动作、阶段顺序和兵力协同。",
                    "- negative / avoid：把负样本作为规避经验，不得模仿其失败链路。",
                ]
            )
        if previous_best_plan:
            lines.extend(
                [
                    "",
                    "上一版较优方案摘要：",
                    f"- 标题：{previous_best_plan['title']}",
                    f"- 指挥意图：{previous_best_plan['commander_intent']}",
                    f"- 阶段：{'；'.join(previous_best_plan['phase_names'])}",
                ]
            )
        lines.extend(
            [
                "",
                "上一轮反思：",
                f"- 诊断：{'；'.join(reflection['diagnosis'])}",
                f"- 变种指令：{'；'.join(reflection['variant_instructions'])}",
                f"- Prompt 补丁：{reflection['meta_prompt_patch']}",
                "",
                "输出要求：",
                "- 固定输出 4 个阶段。",
                "- phase_id 必须是 P1 到 P4。",
                "- 每个阶段至少 2 个 allocated_units，且全方案必须覆盖全部友军单元。",
                "- 每个阶段至少 2 条 actions、2 条 decision_points、2 条 expected_effects。",
                "- actions 必须输出为对象数组，每项都包含 action_type 和 description。",
                f"- action_type 只能从 {', '.join(ACTION_TYPES)} 中选择。",
                "- description、decision_points、expected_effects 都必须是简洁中文。",
                "",
                "反思约束：",
                "- 保留当前方案中得分较高阶段的有效动作，不要整案推翻重写。",
                "- 下一轮只重点修正最弱的1到2个阶段，并保持其余阶段结构稳定。",
                "- 如果侦察发现链或指挥韧性偏低，优先补强 awareness、c2 与抗干扰协同。",
            ]
        )
        return "\n".join(lines)

    def _build_markdown_report_preview(self, result: Dict[str, Any]) -> str:
        best_plan = result["best_plan"]
        sim = result["best_simulation"]
        experiment = result.get("experiment_config", {})
        monte = sim.get("monte_carlo_stats", {})

        def render_action(action: Any) -> str:
            if isinstance(action, dict):
                action_type = action.get("action_type", "other")
                description = action.get("description", "")
                return f"{action_type}: {description}".strip(": ")
            if isinstance(action, ActionItem):
                return f"{action.action_type}: {action.description}"
            return str(action)

        def format_toggle(flag_name: str) -> str:
            return "??" if experiment.get(flag_name) else "??"

        lines = [
            "# ??????????",
            "",
            "## ????",
            "- ????????????????????????",
            "- ?????????????????????????????",
            "",
            "## 1. ????",
            f"- ?????{experiment.get('seed', 'N/A')}",
            f"- ???????{experiment.get('iterations', 'N/A')}",
            f"- Memory?{format_toggle('disable_memory')}",
            f"- Hope?{format_toggle('disable_hope')}",
            f"- Reflection?{format_toggle('disable_reflection')}",
            f"- Writeback?{'??' if experiment.get('allow_writeback') is False else '??'}",
            f"- ?????{experiment.get('sim_runs', 'N/A')}",
            f"- ?????{experiment.get('scenario_path', 'N/A')}",
            f"- ??????{experiment.get('case_bank_path', 'N/A')}",
            f"- ?????{experiment.get('output_dir', 'N/A')}",
            "",
            "## 2. ??????",
            f"- ?????{best_plan['title']}",
            f"- ?????{best_plan['commander_intent']}",
            f"- ?????{best_plan['theory_of_victory']}",
            f"- Memento ??????{len(best_plan.get('memory_insights', []))}",
            "",
            "## 3. ????",
            f"- ??????{sim['mission_success']:.2%}",
            f"- LER?{sim['ler']:.2f}",
            f"- ?????{sim['completion_time_hours']:.2f} ??",
            f"- ????{sim['survivability']:.2%}",
            f"- ?????{sim['command_resilience']:.2%}",
            f"- ?????{sim['overall_effectiveness']:.2%}",
            "",
            "## 4. ??????",
        ]
        for stage_name, score in sim.get("stage_scores", {}).items():
            lines.append(f"- {stage_name}: {score:.2%}")

        if monte:
            lines.extend(["", "## 5. Monte Carlo ??"])
            for metric_name, stats in monte.items():
                lines.append(
                    f"- {metric_name}: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}, "
                    f"95% CI=[{stats.get('ci95_low', 'N/A')}, {stats.get('ci95_high', 'N/A')}], "
                    f"min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}, n={stats.get('n', 'N/A')}"
                )

        lines.extend(["", "## 6. ???????"])
        for item in sim.get("notes", []):
            lines.append(f"- {item}")

        lines.extend(["", "## 7. ????"])
        for item in sim.get("recommendations", []):
            lines.append(f"- {item}")

        lines.extend(["", "## 8. ????", ""])
        for phase in best_plan.get("phases", []):
            lines.append(f"### {phase['phase_id']} {phase['name']}")
            lines.append(f"- ???{phase['intent']}")
            lines.append(f"- ???{', '.join(phase.get('allocated_units', []))}")
            action_texts = [render_action(action) for action in phase.get("actions", [])]
            lines.append(f"- ???{'?'.join(action_texts)}")
            lines.append("")
        return "\n".join(lines)

    def _build_reflection_user_prompt(self, payload: Dict[str, Any]) -> str:
        scenario = payload["scenario"]
        best_plan = payload["best_plan_summary"]
        result = payload["simulation_result"]
        lines = [
            "请根据以下仿真结果给出下一轮 Prompt 反思增量。",
            "",
            "安全边界：",
            "- 仅用于虚拟场景下的方案生成与仿真评估研究。",
            "- 不得输出真实作战指挥、现实目标攻击、武器部署或现实行动建议。",
            "",
            "场景：",
            f"- 名称：{scenario['name']}",
            f"- 目标：{scenario['objective']}",
            "",
            "当前较优方案：",
            f"- 标题：{best_plan['title']}",
            f"- 指挥意图：{best_plan['commander_intent']}",
            f"- 阶段：{'；'.join(best_plan['phase_names'])}",
            "",
            "仿真结果：",
            f"- 任务成功度：{result['mission_success']}",
            f"- LER：{result['ler']}",
            f"- 完成时间：{result['completion_time_hours']}",
            f"- 指挥韧性：{result['command_resilience']}",
            f"- 五阶段评分：{json.dumps(result.get('stage_scores', {}), ensure_ascii=False)}",
            f"- 失败教训：{'；'.join(result['notes'])}",
            "",
            "输出要求：",
            "- diagnosis 聚焦失败根因。",
            "- variant_instructions 聚焦下一轮可执行改进方向。",
            "- meta_prompt_patch 写成一句简短的提示增强语。",
        ]
        return "\n".join(lines)
        scenario = payload["scenario"]
        best_plan = payload["best_plan_summary"]
        result = payload["simulation_result"]
        lines = [
            "请根据以下仿真结果给出下一轮 Prompt 反思增量。",
            "",
            "场景：",
            f"- 名称：{scenario['name']}",
            f"- 目标：{scenario['objective']}",
            "",
            "当前较优方案：",
            f"- 标题：{best_plan['title']}",
            f"- 指挥意图：{best_plan['commander_intent']}",
            f"- 阶段：{'；'.join(best_plan['phase_names'])}",
            "",
            "仿真结果：",
            f"- 任务成功度：{result['mission_success']}",
            f"- LER：{result['ler']}",
            f"- 完成时间：{result['completion_time_hours']}",
            f"- 指挥韧性：{result['command_resilience']}",
            f"- 五阶段评分：{json.dumps(result.get('stage_scores', {}), ensure_ascii=False)}",
            f"- 失败教训：{'；'.join(result['notes'])}",
            "",
            "输出要求：",
            "- diagnosis 聚焦失败根因。",
            "- variant_instructions 聚焦下一轮可执行改进方向。",
            "- meta_prompt_patch 写成一句简短的提示增强语。",
        ]
        return "\n".join(lines)

    def _plan_json_schema(self) -> Dict[str, Any]:
        action_schema = {
            "type": "object",
            "properties": {
                "action_type": {"type": "string", "enum": ACTION_TYPES},
                "description": {"type": "string"},
            },
            "required": ["action_type", "description"],
            "additionalProperties": False,
        }
        phase_schema = {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string"},
                "name": {"type": "string"},
                "intent": {"type": "string"},
                "actions": {"type": "array", "items": action_schema},
                "allocated_units": {"type": "array", "items": {"type": "string"}},
                "decision_points": {"type": "array", "items": {"type": "string"}},
                "expected_effects": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "phase_id",
                "name",
                "intent",
                "actions",
                "allocated_units",
                "decision_points",
                "expected_effects",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "commander_intent": {"type": "string"},
                "theory_of_victory": {"type": "string"},
                "risk_controls": {"type": "array", "items": {"type": "string"}},
                "assessment_metrics": {"type": "array", "items": {"type": "string"}},
                "phases": {"type": "array", "items": phase_schema, "minItems": 4},
            },
            "required": [
                "title",
                "commander_intent",
                "theory_of_victory",
                "risk_controls",
                "assessment_metrics",
                "phases",
            ],
            "additionalProperties": False,
        }
        phase_schema = {
            "type": "object",
            "properties": {
                "phase_id": {"type": "string"},
                "name": {"type": "string"},
                "intent": {"type": "string"},
                "actions": {"type": "array", "items": {"type": "string"}},
                "allocated_units": {"type": "array", "items": {"type": "string"}},
                "decision_points": {"type": "array", "items": {"type": "string"}},
                "expected_effects": {"type": "array", "items": {"type": "string"}},
            },
            "required": [
                "phase_id",
                "name",
                "intent",
                "actions",
                "allocated_units",
                "decision_points",
                "expected_effects",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "commander_intent": {"type": "string"},
                "theory_of_victory": {"type": "string"},
                "risk_controls": {"type": "array", "items": {"type": "string"}},
                "assessment_metrics": {"type": "array", "items": {"type": "string"}},
                "phases": {"type": "array", "items": phase_schema, "minItems": 4},
            },
            "required": [
                "title",
                "commander_intent",
                "theory_of_victory",
                "risk_controls",
                "assessment_metrics",
                "phases",
            ],
            "additionalProperties": False,
        }

    def _reflection_json_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "diagnosis": {"type": "array", "items": {"type": "string"}},
                "variant_instructions": {"type": "array", "items": {"type": "string"}},
                "meta_prompt_patch": {"type": "string"},
            },
            "required": ["diagnosis", "variant_instructions", "meta_prompt_patch"],
            "additionalProperties": False,
        }


class PlanGenerator:
    def __init__(self):
        self.llm = LocalLLMPlanner()

    def generate(
        self,
        scenario: Scenario,
        fused_weights: Dict[str, float],
        fast_weights: Dict[str, float],
        memory_hits: List[Dict[str, Any]],
        iteration_index: int = 0,
        reflection: Dict[str, Any] | None = None,
        previous_best_plan: CombatPlan | None = None,
    ) -> CombatPlan:
        payload = self.llm.generate_plan(
            scenario=scenario,
            fused_weights=fused_weights,
            fast_weights=fast_weights,
            memory_hits=memory_hits,
            iteration_index=iteration_index,
            reflection=reflection,
            previous_best_plan=previous_best_plan,
        )
        phases = [self._build_phase(item, idx + 1) for idx, item in enumerate(payload.get("phases", []))]
        if not phases:
            raise RuntimeError("Local LLM did not return any plan phases.")
        phases = self._repair_plan_structure(phases, scenario)
        phases = self._apply_memory_action_guidance(phases, scenario, memory_hits)

        commander_intent = self._sanitize_text(
            payload.get("commander_intent"),
            fallback=(
                f"在{scenario.time_pressure:.0%}时间压力与{scenario.threat_level:.0%}威胁强度下，"
                f"围绕{scenario.objective}组织多域协同、精准打击与稳控扩展。"
            ),
        )
        theory_of_victory = self._sanitize_text(
            payload.get("theory_of_victory"),
            fallback=(
                f"以侦察塑形、联合火力、机动夺控和持续投送形成闭环优势，"
                f"实现{scenario.objective}。"
            ),
        )
        title = self._sanitize_text(payload.get("title"), fallback=f"{scenario.name}作战方案")

        return CombatPlan(
            title=title,
            commander_intent=commander_intent,
            theory_of_victory=theory_of_victory,
            doctrine_weights={
                key: round(value, 4)
                for key, value in SLOW_WEIGHT_LIBRARY.get(
                    scenario.doctrine_profile,
                    SLOW_WEIGHT_LIBRARY["balanced_joint"],
                ).items()
            },
            fast_weights=fast_weights,
            fused_weights=fused_weights,
            memory_insights=self._memory_insights(memory_hits),
            risk_controls=self._ensure_list(payload.get("risk_controls"), fallback=[
                "采用受限任务分解与模板化输出，降低自然语言歧义",
                "使用红蓝损失阈值和任务时限双门限触发重规划",
                "将阶段结果写入案例库并标记正负样本用于检索修正",
            ]),
            assessment_metrics=self._ensure_list(payload.get("assessment_metrics"), fallback=[
                "战损交换比(LER)",
                "任务完成时间",
                "关键节点压制率",
                "指挥链路完整度",
                "后续行动可持续性",
            ]),
            phases=phases,
        )

    def _phase_fallback_units(self, scenario: Scenario) -> Dict[str, List[str]]:
        units = list(scenario.friendly_forces)

        def top_units(metric: str, count: int = 2) -> List[str]:
            ranked = sorted(units, key=lambda unit: getattr(unit, metric), reverse=True)
            return [unit.name for unit in ranked[:count]]

        return {
            "detect": list(dict.fromkeys(top_units("awareness", 1) + top_units("c2", 1) + top_units("ew", 1)))[:3],
            "disrupt": list(dict.fromkeys(top_units("fires", 1) + top_units("ew", 1) + top_units("c2", 1)))[:3],
            "breach": list(dict.fromkeys(top_units("mobility", 1) + top_units("fires", 1) + top_units("protection", 1)))[:3],
            "control": list(dict.fromkeys(top_units("mobility", 1) + top_units("c2", 1) + top_units("protection", 1)))[:3],
            "sustain": list(dict.fromkeys(top_units("sustainment", 1) + top_units("c2", 1) + top_units("mobility", 1)))[:3],
        }

    def _normalize_units(
        self,
        allocated_units: List[str],
        friendly_units: List[str],
        fallback_units: List[str],
    ) -> List[str]:
        valid_units = [unit for unit in allocated_units if unit in friendly_units]
        for unit in fallback_units:
            if unit in friendly_units and unit not in valid_units:
                valid_units.append(unit)
            if len(valid_units) >= 2:
                break
        if len(valid_units) < 2:
            for unit in friendly_units:
                if unit not in valid_units:
                    valid_units.append(unit)
                if len(valid_units) >= 2:
                    break
        return valid_units[: max(2, len(valid_units))]

    def _infer_stage_name(self, index: int) -> str:
        return ["detect", "disrupt", "breach", "control"][min(index, 3)] if index < 4 else "sustain"

    def _default_actions_legacy(self, stage_name: str, phase: PlanPhase, scenario: Scenario) -> List[str]:
        objective = scenario.objective
        templates = {
            "detect": [
                f"{phase.allocated_units[0]}前出侦察并回传敌关键节点位置，形成{objective}的初始目标图谱。",
                f"{phase.allocated_units[1]}同步建立抗干扰信息链路，完成侦察结果与火力单元的实时共享。",
            ],
            "disrupt": [
                f"{phase.allocated_units[0]}对敌防空、岸防或通信节点实施先制压制，削弱其反制能力。",
                f"{phase.allocated_units[1]}对主突击轴实施电子干扰或指挥协同，压缩敌方响应窗口。",
            ],
            "breach": [
                f"{phase.allocated_units[0]}沿主突击轴快速机动，依托压制窗口突入关键地域。",
                f"{phase.allocated_units[1]}对突破口两翼实施火力与防护协同，保障突击群连续推进。",
            ],
            "control": [
                f"{phase.allocated_units[0]}完成关键地域夺控后迅速展开稳控部署，封控敌方反扑通道。",
                f"{phase.allocated_units[1]}接续指挥与态势更新，维持区域控制和兵力协同。",
            ],
            "sustain": [
                f"{phase.allocated_units[0]}组织补给、伤员后送和战损恢复，维持后续持续作战能力。",
                f"{phase.allocated_units[1]}准备再打击与增援接替，防止既得控制成果回吐。",
            ],
        }
        return templates.get(stage_name, [f"{phase.name}补充动作一。", f"{phase.name}补充动作二。"])

    def _default_decision_points(self, stage_name: str, scenario: Scenario) -> List[str]:
        templates = {
            "detect": [
                "若敌关键节点识别不足，则延长侦察窗口并增加多源校核。",
                "若电磁压制超出预期，则切换备用链路并调整侦察航线。",
            ],
            "disrupt": [
                "若敌防空或岸防节点未被有效压制，则追加火力波次并延后主攻发起。",
                "若平民密度上升或附带损伤风险增加，则收束打击范围并转为精确压制。",
            ],
            "breach": [
                "若突破口受阻，则调整主攻轴并集中火力再开辟通路。",
                "若敌机动反扑提前出现，则优先封控侧翼并申请纵深火力支援。",
            ],
            "control": [
                "若夺控区域出现指挥脱节，则立即切换接续指挥节点。",
                "若敌残余兵力持续渗透，则加强区域封控并投入预备队稳控要点。",
            ],
            "sustain": [
                "若补给投送延迟，则压缩再打击节奏并优先保障前沿关键单元。",
                "若敌远程火力重新活跃，则调整投送路线并实施二次压制。",
            ],
        }
        return templates.get(stage_name, [f"围绕{scenario.objective}补充决策点一。", f"围绕{scenario.objective}补充决策点二。"])

    def _default_expected_effects(self, stage_name: str, scenario: Scenario) -> List[str]:
        templates = {
            "detect": [
                "形成稳定、可共享的目标态势图，为后续火力与机动行动提供可信输入。",
                "压缩敌方预警时间，降低首波突击暴露风险。",
            ],
            "disrupt": [
                "削弱敌关键防御节点的感知、指挥和反击能力。",
                "为后续突破行动创造可利用的时间窗和火力窗口。",
            ],
            "breach": [
                "形成突破口并完成主攻力量向目标地域的有效突入。",
                "降低突击阶段停滞和局部高损耗风险。",
            ],
            "control": [
                "稳定控制关键地域并维持战场秩序与指挥连续性。",
                "为后续持续投送和再打击保留可用支撑条件。",
            ],
            "sustain": [
                "维持补给、通信和战损恢复能力，支撑连续作战。",
                "巩固既得控制成果并为下一轮行动保留机动余度。",
            ],
        }
        return templates.get(stage_name, [f"支撑{scenario.objective}的效果一。", f"支撑{scenario.objective}的效果二。"])

    def _preferred_stage_for_unit(self, unit_name: str, scenario: Scenario) -> int:
        for unit in scenario.friendly_forces:
            if unit.name != unit_name:
                continue
            scores = [unit.awareness + unit.c2 + unit.ew, unit.fires + unit.ew, unit.mobility + unit.protection, unit.sustainment + unit.c2]
            return max(range(len(scores)), key=lambda idx: scores[idx])
        return 0

    def reflect(
        self,
        scenario: Scenario,
        best_plan: CombatPlan,
        best_result: SimulationResult,
    ) -> Dict[str, Any]:
        return self.llm.reflect(scenario, best_plan, best_result)

    def _memory_insights(self, memory_hits: List[Dict[str, Any]]) -> List[str]:
        if not memory_hits:
            return ["未检索到高置信案例，本轮方案主要依赖模型推理与任务先验。"]
        insights: List[str] = []
        for hit in memory_hits:
            record: CaseRecord = hit["record"]
            lessons = "；".join(record.lessons[:2])
            usage_mode = hit.get("usage_mode") or ("imitate" if record.outcome == "positive" else "avoid")
            if usage_mode == "avoid":
                insights.append(f"负样本 {record.case_id}（相似度 {hit['score']:.2f}）提示规避：{lessons}")
            else:
                insights.append(f"正样本 {record.case_id}（相似度 {hit['score']:.2f}）提示复用：{lessons}")
        return insights
        if not memory_hits:
            return ["未检索到高置信案例，本轮方案主要依赖模型推理与任务先验。"]
        insights: List[str] = []
        for hit in memory_hits:
            record: CaseRecord = hit["record"]
            lessons = "；".join(record.lessons[:2])
            insights.append(f"案例{record.case_id}（相似度 {hit['score']:.2f}）提示：{lessons}")
        return insights

    def _build_phase_legacy(self, payload: Dict[str, Any], default_index: int) -> PlanPhase:
        phase_id = str(payload.get("phase_id") or "").strip() or f"P{default_index}"
        phase_name = str(payload.get("name") or "").strip() or f"Phase {default_index}"
        phase_actions: List[ActionItem] = []
        raw_actions = payload.get("actions")
        if isinstance(raw_actions, list):
            for item in raw_actions:
                try:
                    phase_actions.append(ActionItem.from_any(item))
                except ValueError:
                    continue
        if not phase_actions:
            phase_actions = [
                ActionItem(action_type="other", description="Placeholder action one"),
                ActionItem(action_type="other", description="Placeholder action two"),
            ]
        return PlanPhase(
            phase_id=phase_id,
            name=phase_name,
            intent=str(payload.get("intent") or "").strip() or f"Execute tasks around {phase_id}.",
            actions=phase_actions,
            allocated_units=self._ensure_list(payload.get("allocated_units"), fallback=["Pending unit assignment"]),
            decision_points=self._ensure_list(payload.get("decision_points"), fallback=["Pending decision point"]),
            expected_effects=self._ensure_list(payload.get("expected_effects"), fallback=["Pending expected effect"]),
        )

    def _ensure_list_legacy(self, value: Any, fallback: List[str]) -> List[str]:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                return cleaned
        return list(fallback)

    def _sanitize_text(self, value: Any, fallback: str) -> str:
        text = str(value or "").strip()
        if not text:
            return fallback
        bad_markers = ["<|", "analysis", "final<|message|>", "commentary", "json_schema"]
        if any(marker in text.lower() for marker in bad_markers):
            return fallback
        if text.count("…") > 20 or text.count(".") > 30:
            return fallback
        return text

    def _build_phase(self, payload: Dict[str, Any], default_index: int) -> PlanPhase:
        phase_id = str(payload.get("phase_id") or "").strip() or f"P{default_index}"
        phase_name = str(payload.get("name") or "").strip() or f"Phase {default_index}"
        return PlanPhase(
            phase_id=phase_id,
            name=phase_name,
            intent=self._sanitize_text(payload.get("intent"), fallback=f"Execute tasks around {phase_id}."),
            actions=self._ensure_actions(payload.get("actions"), fallback=["Placeholder action one", "Placeholder action two"]),
            allocated_units=self._ensure_list(payload.get("allocated_units"), fallback=["Pending unit assignment"]),
            decision_points=self._ensure_list(payload.get("decision_points"), fallback=["Pending decision point"]),
            expected_effects=self._ensure_list(payload.get("expected_effects"), fallback=["Pending expected effect"]),
        )

    def _ensure_list(self, value: Any, fallback: List[str]) -> List[str]:
        if isinstance(value, list):
            cleaned = [str(item).strip() for item in value if str(item).strip()]
            if cleaned:
                return cleaned
        return list(fallback)

    def _ensure_actions(self, value: Any, fallback: List[str]) -> List[ActionItem]:
        actions: List[ActionItem] = []
        if isinstance(value, list):
            for item in value:
                try:
                    actions.append(ActionItem.from_any(item))
                except ValueError:
                    continue
        if actions:
            return actions
        return [ActionItem(action_type=infer_action_type(text), description=text) for text in fallback]

    def _repair_plan_structure(self, phases: List[PlanPhase], scenario: Scenario) -> List[PlanPhase]:
        friendly_units = [unit.name for unit in scenario.friendly_forces]
        if not phases:
            return phases

        fallback_units = self._phase_fallback_units(scenario)
        for idx, phase in enumerate(phases):
            stage_name = self._infer_stage_name(idx)
            phase.allocated_units = self._normalize_units(
                phase.allocated_units,
                friendly_units,
                fallback_units[stage_name],
            )
            phase.actions = self._ensure_minimum_actions(
                phase.actions,
                self._default_actions(stage_name, phase, scenario),
                minimum=2,
            )
            phase.decision_points = self._ensure_minimum_items(
                phase.decision_points,
                self._default_decision_points(stage_name, scenario),
                minimum=2,
            )
            phase.expected_effects = self._ensure_minimum_items(
                phase.expected_effects,
                self._default_expected_effects(stage_name, scenario),
                minimum=2,
            )

        assigned = {unit for phase in phases for unit in phase.allocated_units if unit in friendly_units}
        missing_units = [unit for unit in friendly_units if unit not in assigned]
        for unit_name in missing_units:
            target_stage = min(self._preferred_stage_for_unit(unit_name, scenario), len(phases) - 1)
            phases[target_stage].allocated_units.append(unit_name)

        for idx, phase in enumerate(phases):
            stage_name = self._infer_stage_name(idx)
            phase.allocated_units = self._normalize_units(
                phase.allocated_units,
                friendly_units,
                fallback_units[stage_name],
            )
        return phases

    def _ensure_minimum_items(self, values: List[str], fallback_values: List[str], minimum: int) -> List[str]:
        cleaned = [item.strip() for item in values if item and item.strip()]
        for candidate in fallback_values:
            candidate = candidate.strip()
            if candidate and candidate not in cleaned:
                cleaned.append(candidate)
            if len(cleaned) >= minimum:
                break
        return cleaned[: max(minimum, len(cleaned))]

    def _ensure_minimum_actions(
        self,
        values: List[ActionItem],
        fallback_values: List[ActionItem],
        minimum: int,
    ) -> List[ActionItem]:
        cleaned: List[ActionItem] = []
        seen = set()
        for action in values + fallback_values:
            key = (action.action_type, action.description)
            if key in seen:
                continue
            seen.add(key)
            cleaned.append(action)
            if len(cleaned) >= minimum:
                break
        return cleaned[: max(minimum, len(cleaned))]

    def _default_actions(self, stage_name: str, phase: PlanPhase, scenario: Scenario) -> List[ActionItem]:
        objective = scenario.objective
        templates = {
            "detect": [
                ActionItem("recon", f"{phase.allocated_units[0]}前出侦察并回传敌关键节点位置，形成{objective}的初始目标图谱。"),
                ActionItem("c2", f"{phase.allocated_units[1]}建立抗干扰信息链路，完成侦察结果与火力单元的实时共享。"),
            ],
            "disrupt": [
                ActionItem("strike", f"{phase.allocated_units[0]}对敌防空、岸防或通信节点实施先制压制，削弱其反制能力。"),
                ActionItem("ew", f"{phase.allocated_units[1]}对主突击轴实施电子干扰或指挥协同，压缩敌方响应窗口。"),
            ],
            "breach": [
                ActionItem("mobility", f"{phase.allocated_units[0]}沿主突击轴快速机动，依托压制窗口突入关键地域。"),
                ActionItem("strike", f"{phase.allocated_units[1]}对突破口两翼实施火力与防护协同，保障突击群连续推进。"),
            ],
            "control": [
                ActionItem("control", f"{phase.allocated_units[0]}完成关键地域夺控后迅速展开稳控部署，封控敌方反扑通道。"),
                ActionItem("c2", f"{phase.allocated_units[1]}接续指挥与态势更新，维持区域控制和兵力协同。"),
            ],
            "sustain": [
                ActionItem("sustain", f"{phase.allocated_units[0]}组织补给、伤员后送和战损恢复，维持后续持续作战能力。"),
                ActionItem("sustain", f"{phase.allocated_units[1]}准备再打击与增援接替，防止既得控制成果回吐。"),
            ],
        }
        return templates.get(
            stage_name,
            [ActionItem("other", f"{phase.name}补充动作一。"), ActionItem("other", f"{phase.name}补充动作二。")],
        )


    def _apply_memory_action_guidance(
        self,
        phases: List[PlanPhase],
        scenario: Scenario,
        memory_hits: List[Dict[str, Any]],
    ) -> List[PlanPhase]:
        if not phases or not memory_hits:
            return phases

        terrain_value = scenario.terrain.lower()
        mission_type = scenario.mission_type.lower()

        if mission_type == "assault" and ("coastal" in terrain_value or "littoral" in terrain_value):
            if len(phases) >= 2:
                self._ensure_phase_action(phases[1], "strike", "集中压制敌防空与岸防关键节点，扩大压制窗口。")
                self._ensure_phase_action(phases[1], "ew", "同步实施电子对抗，压缩敌方发现与响应时间。")
            if len(phases) >= 3:
                self._ensure_phase_action(phases[2], "mobility", "沿主突击轴快速机动穿透突破口，避免停滞暴露。")
                self._ensure_phase_action(phases[2], "strike", "对突破口两翼实施伴随火力支援，维持突破连续性。")
        elif mission_type == "defense" and "urban" in terrain_value:
            if len(phases) >= 2:
                self._ensure_phase_action(phases[1], "strike", "依托街区火力点压制敌装甲或突击单元。")
                self._ensure_phase_action(phases[1], "c2", "保持主备指挥链在线，缩短局部反击决策时间。")
            if len(phases) >= 3:
                self._ensure_phase_action(phases[2], "mobility", "预备队沿侧街实施短距机动，封堵敌主攻轴扩张。")
            if len(phases) >= 4:
                self._ensure_phase_action(phases[3], "control", "快速恢复关键街区控制与封控秩序。")
                self._ensure_phase_action(phases[3], "c2", "切换备用通信节点，维持控制区态势同步。")

        return phases

    def _ensure_phase_action(self, phase: PlanPhase, action_type: str, description: str) -> None:
        if any(action.action_type == action_type for action in phase.actions):
            return
        phase.actions.append(ActionItem(action_type, description))


class PlanSimulator:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def _action_descriptions(self, phase: PlanPhase) -> List[str]:
        return [action.description for action in phase.actions]

    def _action_types(self, phase: PlanPhase) -> List[str]:
        return [action.action_type for action in phase.actions]

    def _phase_joined_text(self, phase: PlanPhase) -> str:
        return " ".join([phase.name, phase.intent, *self._action_types(phase), *self._action_descriptions(phase)])

    def run(self, scenario: Scenario, plan: CombatPlan, stochastic: bool = False) -> SimulationResult:
        blue_caps = self._aggregate_capabilities(scenario.friendly_forces)
        red_caps = self._aggregate_capabilities(scenario.enemy_forces)
        phases: List[PhaseSimulation] = []
        total_friendly_loss = 0.0
        total_enemy_loss = 0.0
        total_time = 0.0
        stage_accumulator: Dict[str, List[float]] = defaultdict(list)

        for phase in plan.phases:
            requirements = self._phase_requirements(phase)
            phase_blue_caps = self._phase_blue_capabilities(phase, scenario, blue_caps)
            stage_scores = self._evaluate_kill_chain(
                phase,
                scenario,
                plan,
                phase_blue_caps,
                blue_caps,
                red_caps,
                requirements,
            )
            if stochastic:
                stage_scores = {
                    name: round(clamp_value(score * self.rng.uniform(0.97, 1.03), 0.12, 0.96), 4)
                    for name, score in stage_scores.items()
                }
            emphasis = self._phase_stage_emphasis(phase)
            success = sum(stage_scores[name] * weight for name, weight in emphasis.items())
            blue_power = self._blue_effectiveness(phase_blue_caps, plan.fused_weights, requirements)
            red_power = self._red_effectiveness(red_caps, scenario, requirements)
            allocation_fit = self._allocation_fit(phase_blue_caps, blue_caps, requirements)
            if stochastic:
                blue_power *= self.rng.uniform(0.96, 1.04)
                red_power *= self.rng.uniform(0.96, 1.04)

            duration = max(
                1.0,
                len(phase.actions) * 1.05
                + 2.8 * scenario.time_pressure
                - 1.8 * plan.fused_weights["mobility"]
                + 0.4 * max(0.0, 0.65 - allocation_fit),
            )
            if stochastic:
                duration *= self.rng.uniform(0.97, 1.05)
            friendly_loss = max(0.005, red_power * (1 - success) * 0.006 * (1.10 - 0.18 * allocation_fit))
            enemy_loss = max(0.008, blue_power * success * 0.008 * (0.82 + 0.24 * allocation_fit))
            phases.append(
                PhaseSimulation(
                    phase_id=phase.phase_id,
                    phase_name=phase.name,
                    success_score=round(success, 4),
                    duration_hours=round(duration, 2),
                    friendly_loss=round(friendly_loss, 3),
                    enemy_loss=round(enemy_loss, 3),
                    notes=[
                        f"阶段成功评分 {success:.0%}",
                        f"蓝方能力指数 {blue_power:.2f}",
                        f"红方压力指数 {red_power:.2f}",
                        f"编组匹配度 {allocation_fit:.2f}",
                        (
                            f"杀伤链评分 detect={stage_scores['detect']:.2f}, "
                            f"disrupt={stage_scores['disrupt']:.2f}, "
                            f"breach={stage_scores['breach']:.2f}, "
                            f"control={stage_scores['control']:.2f}, "
                            f"sustain={stage_scores['sustain']:.2f}"
                        ),
                    ],
                )
            )
            total_friendly_loss += friendly_loss
            total_enemy_loss += enemy_loss
            total_time += duration
            for stage_name, score in stage_scores.items():
                stage_accumulator[stage_name].append(score)

        aggregated_stage_scores = {
            stage_name: round(sum(values) / len(values), 4)
            for stage_name, values in stage_accumulator.items()
            if values
        }
        stage_based_success = (
            0.22 * aggregated_stage_scores.get("detect", 0.5)
            + 0.24 * aggregated_stage_scores.get("disrupt", 0.5)
            + 0.24 * aggregated_stage_scores.get("breach", 0.5)
            + 0.18 * aggregated_stage_scores.get("control", 0.5)
            + 0.12 * aggregated_stage_scores.get("sustain", 0.5)
        )
        phase_based_success = sum(phase.success_score for phase in phases) / len(phases)
        mission_success = 0.55 * stage_based_success + 0.45 * phase_based_success
        if plan.fused_weights["c2"] < 0.65:
            mission_success *= 0.95
        if scenario.time_pressure > 0.8:
            mission_success *= 0.97

        ler = total_enemy_loss / max(total_friendly_loss, 0.05)
        survivability = clamp_value(1.0 - total_friendly_loss / max(len(scenario.friendly_forces) * 1.6, 1.0))
        command_resilience = clamp_value(
            0.42 * plan.fused_weights["c2"] + 0.28 * plan.fused_weights["ew"] + 0.30 * plan.fused_weights["awareness"]
            - 0.18 * scenario.ew_threat
        )
        overall = clamp_value(
            0.42 * mission_success
            + 0.18 * clamp_value(ler / 2.0)
            + 0.18 * survivability
            + 0.12 * command_resilience
            + 0.10 * clamp_value(1 - total_time / 36.0)
        )
        recommendations = self._recommend(
            plan,
            scenario,
            aggregated_stage_scores,
            mission_success,
            ler,
            total_time,
            command_resilience,
        )
        notes = self._build_failure_notes(
            scenario,
            phases,
            aggregated_stage_scores,
            mission_success,
            ler,
            total_time,
            command_resilience,
        )
        return SimulationResult(
            mission_success=round(mission_success, 4),
            ler=round(ler, 4),
            completion_time_hours=round(total_time, 2),
            survivability=round(survivability, 4),
            command_resilience=round(command_resilience, 4),
            overall_effectiveness=round(overall, 4),
            stage_scores=aggregated_stage_scores,
            notes=notes,
            phases=phases,
            recommendations=recommendations,
            monte_carlo_stats={
                "mission_success": {"mean": round(mission_success, 4), "std": 0.0, "min": round(mission_success, 4), "max": round(mission_success, 4), "ci95_low": round(mission_success, 4), "ci95_high": round(mission_success, 4), "n": 1},
                "ler": {"mean": round(ler, 4), "std": 0.0, "min": round(ler, 4), "max": round(ler, 4), "ci95_low": round(ler, 4), "ci95_high": round(ler, 4), "n": 1},
                "overall_effectiveness": {"mean": round(overall, 4), "std": 0.0, "min": round(overall, 4), "max": round(overall, 4), "ci95_low": round(overall, 4), "ci95_high": round(overall, 4), "n": 1},
                "completion_time_hours": {"mean": round(total_time, 2), "std": 0.0, "min": round(total_time, 2), "max": round(total_time, 2), "ci95_low": round(total_time, 2), "ci95_high": round(total_time, 2), "n": 1},
                "command_resilience": {"mean": round(command_resilience, 4), "std": 0.0, "min": round(command_resilience, 4), "max": round(command_resilience, 4), "ci95_low": round(command_resilience, 4), "ci95_high": round(command_resilience, 4), "n": 1},
            },
        )

    def _blue_effectiveness(
        self,
        blue_caps: Dict[str, float],
        fused_weights: Dict[str, float],
        requirements: Dict[str, float],
    ) -> float:
        return sum(
            blue_caps[key] * fused_weights[key] * requirements[key]
            for key in CAPABILITY_KEYS
        )

    def _red_effectiveness(
        self,
        red_caps: Dict[str, float],
        scenario: Scenario,
        requirements: Dict[str, float],
    ) -> float:
        return sum(
            red_caps[key] * (0.65 + 0.55 * scenario.threat_level) * requirements[key]
            for key in CAPABILITY_KEYS
        )

    def _stage_signal_from_actions(self, phase: PlanPhase) -> Dict[str, float]:
        stage_signal = {
            "detect": 0.0,
            "disrupt": 0.0,
            "breach": 0.0,
            "control": 0.0,
            "sustain": 0.0,
        }
        action_map = {
            "recon": {"detect": 1.0, "disrupt": 0.15},
            "c2": {"detect": 0.35, "disrupt": 0.25, "control": 0.40, "sustain": 0.25},
            "ew": {"detect": 0.20, "disrupt": 1.0},
            "strike": {"disrupt": 1.0, "breach": 0.35},
            "mobility": {"breach": 1.0, "control": 0.20},
            "protection": {"breach": 0.35, "control": 0.55, "sustain": 0.15},
            "control": {"control": 1.0, "sustain": 0.20},
            "sustain": {"sustain": 1.0, "control": 0.15},
            "other": {},
        }
        fallback_map = {
            "detect": {"recon", "c2"},
            "disrupt": {"strike", "ew"},
            "breach": {"mobility", "strike", "protection"},
            "control": {"control", "c2", "protection"},
            "sustain": {"sustain", "c2"},
        }

        effective_action_count = 0
        for action in phase.actions:
            action_type = action.action_type
            if action_type == "other":
                action_type = infer_action_type(action.description)
            contributions = action_map.get(action_type, {})
            if not contributions:
                continue
            effective_action_count += 1
            for stage_name, value in contributions.items():
                stage_signal[stage_name] += value

        if effective_action_count == 0:
            inferred_types = {infer_action_type(text) for text in self._action_descriptions(phase)}
            for stage_name, types in fallback_map.items():
                overlap = len(types & inferred_types)
                if overlap:
                    stage_signal[stage_name] = max(stage_signal[stage_name], 0.45 * overlap)

        normalizer = max(1.0, float(max(effective_action_count, len(phase.actions))))
        return {
            stage_name: clamp_value(value / normalizer, 0.0, 1.0)
            for stage_name, value in stage_signal.items()
        }

    def _evaluate_kill_chain(
        self,
        phase: PlanPhase,
        scenario: Scenario,
        plan: CombatPlan,
        phase_blue_caps: Dict[str, float],
        full_blue_caps: Dict[str, float],
        red_caps: Dict[str, float],
        requirements: Dict[str, float],
    ) -> Dict[str, float]:
        terrain_penalty = 0.06 if any(token in scenario.terrain.lower() for token in ["urban", "mountain"]) else 0.02
        weather_penalty = 0.05 if any(token in scenario.weather.lower() for token in ["rain", "storm", "fog"]) else 0.01
        civilian_penalty = 0.12 * scenario.civilian_presence
        ew_pressure = 0.16 * scenario.ew_threat
        stage_signal = self._stage_signal_from_actions(phase)

        detect_gap = (
            0.45 * phase_blue_caps["awareness"] * plan.fused_weights["awareness"]
            + 0.25 * phase_blue_caps["c2"] * plan.fused_weights["c2"]
            + 0.15 * phase_blue_caps["ew"] * plan.fused_weights["ew"]
            - 0.30 * red_caps["awareness"]
            - 0.18 * red_caps["ew"]
        )
        disrupt_gap = (
            0.45 * phase_blue_caps["fires"] * plan.fused_weights["fires"]
            + 0.20 * phase_blue_caps["ew"] * plan.fused_weights["ew"]
            + 0.15 * phase_blue_caps["c2"] * plan.fused_weights["c2"]
            - 0.30 * red_caps["fires"]
            - 0.20 * red_caps["c2"]
        )
        breach_gap = (
            0.30 * phase_blue_caps["mobility"] * plan.fused_weights["mobility"]
            + 0.25 * phase_blue_caps["protection"] * plan.fused_weights["protection"]
            + 0.30 * phase_blue_caps["fires"] * plan.fused_weights["fires"]
            - 0.40 * red_caps["protection"]
            - 0.35 * red_caps["fires"]
        )
        control_gap = (
            0.35 * phase_blue_caps["protection"] * plan.fused_weights["protection"]
            + 0.35 * phase_blue_caps["c2"] * plan.fused_weights["c2"]
            + 0.30 * phase_blue_caps["mobility"] * plan.fused_weights["mobility"]
            - 0.35 * red_caps["fires"]
            - 0.25 * red_caps["mobility"]
        )
        sustain_gap = (
            0.45 * phase_blue_caps["sustainment"] * plan.fused_weights["sustainment"]
            + 0.30 * phase_blue_caps["c2"] * plan.fused_weights["c2"]
            + 0.25 * phase_blue_caps["protection"] * plan.fused_weights["protection"]
            - 0.25 * red_caps["fires"]
            - 0.15 * red_caps["ew"]
        )

        stage_scores = {
            "detect": self._bounded_score(detect_gap, 8.0, terrain_penalty + weather_penalty + ew_pressure),
            "disrupt": self._bounded_score(disrupt_gap, 8.5, civilian_penalty + ew_pressure),
            "breach": self._bounded_score(breach_gap, 9.0, terrain_penalty + civilian_penalty),
            "control": self._bounded_score(control_gap, 8.5, civilian_penalty + 0.04 * scenario.time_pressure),
            "sustain": self._bounded_score(sustain_gap, 8.0, weather_penalty + 0.08 * scenario.time_pressure),
        }
        stage_fit = {
            "detect": 0.50 * self._capability_ratio(phase_blue_caps, full_blue_caps, "awareness")
            + 0.25 * self._capability_ratio(phase_blue_caps, full_blue_caps, "c2")
            + 0.25 * self._capability_ratio(phase_blue_caps, full_blue_caps, "ew"),
            "disrupt": 0.50 * self._capability_ratio(phase_blue_caps, full_blue_caps, "fires")
            + 0.25 * self._capability_ratio(phase_blue_caps, full_blue_caps, "ew")
            + 0.25 * self._capability_ratio(phase_blue_caps, full_blue_caps, "c2"),
            "breach": 0.40 * self._capability_ratio(phase_blue_caps, full_blue_caps, "mobility")
            + 0.30 * self._capability_ratio(phase_blue_caps, full_blue_caps, "fires")
            + 0.20 * self._capability_ratio(phase_blue_caps, full_blue_caps, "protection")
            + 0.10 * self._capability_ratio(phase_blue_caps, full_blue_caps, "c2"),
            "control": 0.35 * self._capability_ratio(phase_blue_caps, full_blue_caps, "protection")
            + 0.35 * self._capability_ratio(phase_blue_caps, full_blue_caps, "c2")
            + 0.30 * self._capability_ratio(phase_blue_caps, full_blue_caps, "mobility"),
            "sustain": 0.45 * self._capability_ratio(phase_blue_caps, full_blue_caps, "sustainment")
            + 0.30 * self._capability_ratio(phase_blue_caps, full_blue_caps, "c2")
            + 0.25 * self._capability_ratio(phase_blue_caps, full_blue_caps, "protection"),
        }
        for stage_name, score in list(stage_scores.items()):
            signal_bonus = 0.08 + 0.18 * stage_signal[stage_name]
            stage_scores[stage_name] = clamp_value(
                score * (0.70 + 0.35 * stage_fit[stage_name]) + signal_bonus,
                0.12,
                0.96,
            )

        return {name: round(value, 4) for name, value in stage_scores.items()}

    def _bounded_score(self, gap: float, scale: float, penalty: float) -> float:
        return clamp_value(1.0 / (1.0 + math.exp(-(gap / scale - penalty))), 0.12, 0.96)

    def _phase_stage_emphasis(self, phase: PlanPhase) -> Dict[str, float]:
        signal = self._stage_signal_from_actions(phase)
        baseline = {"detect": 0.16, "disrupt": 0.16, "breach": 0.16, "control": 0.16, "sustain": 0.16}
        combined = {
            stage_name: baseline[stage_name] + 0.84 * signal[stage_name]
            for stage_name in baseline
        }
        total = sum(combined.values())
        return {stage_name: value / total for stage_name, value in combined.items()}

    def _aggregate_capabilities(self, units) -> Dict[str, float]:
        aggregated = defaultdict(float)
        for unit in units:
            scale = max(1.0, math.sqrt(unit.quantity)) * clamp_value(unit.readiness, 0.3, 1.0)
            for key in CAPABILITY_KEYS:
                aggregated[key] += getattr(unit, key) * scale
        return aggregated

    def _phase_blue_capabilities(
        self,
        phase: PlanPhase,
        scenario: Scenario,
        full_blue_caps: Dict[str, float],
    ) -> Dict[str, float]:
        lookup = {unit.name: unit for unit in scenario.friendly_forces}
        allocated_units = [lookup[name] for name in phase.allocated_units if name in lookup]
        joined = " ".join([phase.name, phase.intent, *self._action_types(phase), *self._action_descriptions(phase), *phase.decision_points])
        support_units = [
            unit
            for unit in scenario.friendly_forces
            if unit.name not in phase.allocated_units and unit.name in joined
        ]

        allocated_caps = self._aggregate_capabilities(allocated_units) if allocated_units else defaultdict(float)
        support_caps = self._aggregate_capabilities(support_units) if support_units else defaultdict(float)
        phase_caps: Dict[str, float] = {}
        for key in CAPABILITY_KEYS:
            local_cap = allocated_caps.get(key, 0.0) + 0.35 * support_caps.get(key, 0.0)
            phase_caps[key] = 0.72 * local_cap + 0.28 * full_blue_caps.get(key, 0.0)
        return phase_caps

    def _capability_ratio(
        self,
        phase_caps: Dict[str, float],
        full_caps: Dict[str, float],
        capability: str,
    ) -> float:
        baseline = max(full_caps.get(capability, 0.0), 1e-6)
        return clamp_value(phase_caps.get(capability, 0.0) / baseline, 0.0, 1.15)

    def _allocation_fit(
        self,
        phase_caps: Dict[str, float],
        full_caps: Dict[str, float],
        requirements: Dict[str, float],
    ) -> float:
        weighted = 0.0
        for capability, weight in requirements.items():
            weighted += weight * self._capability_ratio(phase_caps, full_caps, capability)
        return clamp_value(weighted, 0.0, 1.15)

    def _phase_requirements(self, phase: PlanPhase) -> Dict[str, float]:
        weights = {key: 0.20 for key in CAPABILITY_KEYS}
        action_weight_map = {
            "recon": {"awareness": 0.45, "c2": 0.10},
            "c2": {"c2": 0.35, "awareness": 0.10, "sustainment": 0.08},
            "ew": {"ew": 0.45, "c2": 0.10},
            "strike": {"fires": 0.48, "protection": 0.05},
            "mobility": {"mobility": 0.42, "protection": 0.08},
            "protection": {"protection": 0.42, "c2": 0.05},
            "control": {"c2": 0.20, "protection": 0.18, "mobility": 0.10},
            "sustain": {"sustainment": 0.45, "c2": 0.08, "protection": 0.06},
            "other": {},
        }
        for action in phase.actions:
            action_type = action.action_type
            if action_type == "other":
                action_type = infer_action_type(action.description)
            for capability, delta in action_weight_map.get(action_type, {}).items():
                weights[capability] += delta

        joined = self._phase_joined_text(phase).lower()
        if "battle damage" in joined or "??" in joined:
            weights["awareness"] += 0.10
            weights["c2"] += 0.12
        total = sum(weights.values())
        return {key: value / total for key, value in weights.items()}

    def _build_failure_notes(
        self,
        scenario: Scenario,
        phases: List[PhaseSimulation],
        stage_scores: Dict[str, float],
        mission_success: float,
        ler: float,
        total_time: float,
        command_resilience: float,
    ) -> List[str]:
        notes: List[str] = []
        stage_labels = {
            "detect": "侦察发现链",
            "disrupt": "压制打断链",
            "breach": "突破突击链",
            "control": "夺控稳态链",
            "sustain": "持续保障链",
        }
        stage_diagnosis = {
            "detect": "侦察发现链偏弱，说明目标发现、态势融合或情报共享不足，前续塑形没有为后续打击建立稳定入口。",
            "disrupt": "压制打断链偏弱，说明火力压制、电子对抗或关键节点瘫痪不足，敌方仍保有有效反制能力。",
            "breach": "突破突击链偏弱，说明主攻轴机动、防护与火力协同不足，突破口形成效率偏低。",
            "control": "夺控稳态链偏弱，说明夺占后的控制转换、区域封控或指挥接续存在薄弱环节。",
            "sustain": "持续保障链偏弱，说明补给投送、战损恢复或后续接替支撑不足，方案后劲不够。",
        }
        stage_rank = sorted(stage_scores.items(), key=lambda item: item[1])
        weak_stages = [name for name, score in stage_rank if score < 0.58]
        critical_stages = [name for name, score in stage_rank if score < 0.50]
        weak_phases = [phase for phase in phases if phase.success_score < 0.55]
        if weak_phases:
            names = "、".join(phase.phase_name for phase in weak_phases)
            notes.append(f"以下阶段成功率偏低，需要重排任务链路或增强兵力配置：{names}。")
        if weak_stages:
            labels = "、".join(stage_labels[name] for name in weak_stages)
            notes.append(f"当前方案的主要短板集中在：{labels}。")
            for stage_name in critical_stages[:2]:
                notes.append(stage_diagnosis[stage_name])
        if mission_success < 0.6:
            primary_stage = stage_rank[0][0] if stage_rank else "detect"
            notes.append(f"整体任务成功度偏低，根因更接近{stage_labels[primary_stage]}失效，而不是单一兵力数量不足。")
        if total_time > 20:
            notes.append("任务完成时间过长，说明阶段串行深度偏大，机动转进和指挥接续效率仍需压缩。")
        if ler < 1.1:
            notes.append("战损交换比不理想，说明保护与压制火力不足，需要降低主攻轴暴露。")
        if command_resilience < 0.72 or scenario.ew_threat > 0.6:
            notes.append("电磁对抗压力较高，应提升抗干扰通信、备用链路和战场态势更新能力。")
        if not notes:
            notes.append("当前方案已较稳定，下一轮可围绕局部阶段优化和资源节约继续演化。")
        return notes

    def _recommend(
        self,
        plan: CombatPlan,
        scenario: Scenario,
        stage_scores: Dict[str, float],
        mission_success: float,
        ler: float,
        total_time: float,
        command_resilience: float,
    ) -> List[str]:
        recs: List[str] = []
        stage_actions = {
            "detect": "提高侦察、感知融合与指挥信息共享权重，前置塑形侦搜与目标确认动作，缩短发现到打击链路。",
            "disrupt": "增加压制火力、电子对抗和关键节点瘫痪动作，在主攻发起前先削弱敌方反制能力。",
            "breach": "强化主攻轴机动、防护与突击火力协同，优先集中优势兵力打穿局部突破口。",
            "control": "补强夺控后的接续指挥、区域封控与兵力轮换设计，避免突破后控制不稳。",
            "sustain": "增强补给投送、战损恢复和持续保障安排，保证方案在后续阶段不因后劲不足而衰减。",
        }
        ordered_stages = sorted(stage_scores.items(), key=lambda item: item[1])
        weak_stages = [name for name, score in ordered_stages if score < 0.62]
        for stage_name in weak_stages[:2]:
            recs.append(stage_actions[stage_name])
        if mission_success < 0.68 and not weak_stages:
            recs.append("整体上调侦察塑形与联合作战指挥权重，重新平衡各阶段衔接，避免局部优势无法转化为全局成功。")
        if ler < 1.0:
            recs.append("增加防护与压制火力投入，降低主攻轴暴露时间。")
        if total_time > 18:
            recs.append("以更高机动权重重排阶段顺序，减少串行等待，优先打穿关键瓶颈节点。")
        if command_resilience < 0.7 or scenario.ew_threat > 0.6:
            recs.append("补强抗干扰链路和备用指挥节点，避免电磁压制造成闭环失效。")
        if not recs:
            recs.append("当前方案具备较好综合效能，可直接进入多轮蒙特卡洛或对抗仿真。")
        return recs


class MilitaryResearchPipeline:
    def __init__(self, case_bank_path: str | Path, seed: int = 7):
        self.case_bank = CaseBank(case_bank_path)
        self.seed = seed
        self.rng = random.Random(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _safe_package_version(self, package_name: str) -> str | None:
        try:
            return importlib_metadata.version(package_name)
        except importlib_metadata.PackageNotFoundError:
            return None

    def _run_probe(self, command: List[str]) -> str | None:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
                timeout=3,
                check=False,
            )
        except (FileNotFoundError, PermissionError, OSError, subprocess.SubprocessError):
            return None
        if completed.returncode != 0:
            return None
        output = (completed.stdout or "").strip()
        return output or None

    def _detect_cpu_name(self) -> str:
        cpu_override = os.getenv("RUNTIME_CPU_NAME", "").strip()
        if cpu_override:
            return cpu_override

        if os.name == "nt":
            cpu_name = self._run_probe(["wmic", "cpu", "get", "name"])
            if cpu_name:
                lines = [line.strip() for line in cpu_name.splitlines() if line.strip() and line.strip().lower() != "name"]
                if lines:
                    return lines[0]
            cpu_name = self._run_probe(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-ItemProperty 'HKLM:\\HARDWARE\\DESCRIPTION\\System\\CentralProcessor\\0').ProcessorNameString",
                ]
            )
            if cpu_name:
                return cpu_name.splitlines()[0].strip()
            cpu_name = self._run_probe(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor | Select-Object -First 1 -ExpandProperty Name)",
                ]
            )
            if cpu_name:
                return cpu_name.splitlines()[0].strip()

        cpu_name = platform.processor().strip()
        if cpu_name:
            return cpu_name
        machine = platform.machine().strip()
        return machine or "unknown"

    def _detect_gpu_name(self) -> str | None:
        gpu_override = os.getenv("RUNTIME_GPU_NAME", "").strip()
        if gpu_override:
            return gpu_override

        if torch.cuda.is_available():
            try:
                return torch.cuda.get_device_name(0)
            except RuntimeError:
                pass

        gpu_name = self._run_probe(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"])
        if gpu_name:
            return gpu_name.splitlines()[0].strip()
        return None

    def _collect_runtime_manifest(self) -> Dict[str, Any]:
        model_id = os.getenv("LOCAL_LLM_MODEL", "").strip() or "unknown"
        base_url = os.getenv("LOCAL_LLM_BASE_URL", "").strip() or "http://localhost:1234/v1"
        cuda_available = bool(torch.cuda.is_available())
        gpu_name = self._detect_gpu_name()
        return {
            "collected_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "llm_backend": {
                "model_id": model_id,
                "base_url": base_url,
                "api_mode": "openai-compatible",
            },
            "llm_inference_params": {
                "temperature_generation": 0.1,
                "temperature_json_repair": 0.0,
                "top_p": "default (1.0, not explicitly set)",
                "max_tokens": "server-side default (unlimited for local endpoint)",
                "response_format": "json_schema",
                "prompt_source": "military_research/engine.py (inline, not externalized)",
            },
            "software": {
                "python_version": platform.python_version(),
                "platform": platform.platform(),
                "torch_version": getattr(torch, "__version__", None),
                "openai_version": self._safe_package_version("openai"),
                "sentence_transformers_version": self._safe_package_version("sentence-transformers"),
                "faiss_version": self._safe_package_version("faiss-cpu") or self._safe_package_version("faiss-gpu"),
            },
            "hardware": {
                "cpu": self._detect_cpu_name(),
                "gpu": gpu_name,
                "cuda_available": cuda_available,
                "cuda_version": getattr(torch.version, "cuda", None) if cuda_available else None,
            },
        }

    def _terrain_family(self, terrain: str) -> str:
        terrain_value = (terrain or "").strip().lower()
        if "coastal" in terrain_value or "littoral" in terrain_value:
            return "coastal"
        if "maritime" in terrain_value or "island" in terrain_value:
            return "maritime"
        if "river" in terrain_value:
            return "river"
        if "mountain" in terrain_value:
            return "mountain"
        if "urban" in terrain_value:
            return "urban"
        return terrain_value or "other"

    def _overlap_ratio(self, left: set[str], right: set[str]) -> float:
        union = left | right
        if not union:
            return 0.0
        return len(left & right) / len(union)

    def _mission_compatibility(self, scenario_mission: str, record_mission: str) -> float:
        if scenario_mission == record_mission:
            return 1.0
        pair = (scenario_mission, record_mission)
        compatibility = {
            ("assault", "defense"): 0.55,
            ("defense", "assault"): 0.55,
            ("sustain", "defense"): 0.58,
            ("defense", "sustain"): 0.58,
            ("sustain", "assault"): 0.45,
            ("assault", "sustain"): 0.45,
            ("recon", "assault"): 0.36,
            ("assault", "recon"): 0.36,
            ("recon", "defense"): 0.32,
            ("defense", "recon"): 0.32,
        }
        return compatibility.get(pair, 0.18)

    def _terrain_compatibility(self, scenario_terrain: str, record_terrain: str) -> float:
        if scenario_terrain == record_terrain:
            return 1.0
        pair = {scenario_terrain, record_terrain}
        if pair == {"coastal", "maritime"}:
            return 0.72
        if pair == {"coastal", "urban"}:
            return 0.42
        if pair == {"coastal", "river"}:
            return 0.34
        if pair == {"river", "urban"}:
            return 0.28
        return 0.12

    def _memory_support_score(self, scenario: Scenario, hit: Dict[str, Any]) -> float:
        record = hit["record"]
        scenario_mission = str(scenario.mission_type).strip().lower()
        record_mission = str(record.mission_type).strip().lower()
        scenario_terrain = self._terrain_family(scenario.terrain)
        record_terrain = self._terrain_family(record.terrain)
        scenario_friendly = {str(unit.role).strip().lower() for unit in scenario.friendly_forces}
        scenario_enemy = {str(unit.role).strip().lower() for unit in scenario.enemy_forces}
        record_friendly = {str(role).strip().lower() for role in record.friendly_roles}
        record_enemy = {str(role).strip().lower() for role in record.enemy_roles}
        confidence = float(hit.get("confidence", hit.get("score", 0.0)) or 0.0)
        quality = float(hit.get("quality", 0.0) or 0.0)
        support = (
            0.20 * confidence
            + 0.22 * quality
            + 0.20 * self._mission_compatibility(scenario_mission, record_mission)
            + 0.16 * self._terrain_compatibility(scenario_terrain, record_terrain)
            + 0.14 * self._overlap_ratio(scenario_friendly, record_friendly)
            + 0.08 * self._overlap_ratio(scenario_enemy, record_enemy)
        )
        scene_gate = self._memory_scene_gate(scenario_mission, scenario_terrain, record_mission, record_terrain)
        if scene_gate <= 0.0:
            return 0.0
        support *= scene_gate
        return round(max(0.0, min(1.0, support)), 4)

    def _memory_scene_gate(
        self,
        scenario_mission: str,
        scenario_terrain: str,
        record_mission: str,
        record_terrain: str,
    ) -> float:
        if scenario_mission == "assault" and scenario_terrain == "river":
            if record_mission != "assault":
                return 0.0
            return 1.0 if record_terrain == "river" else 0.45
        if scenario_mission == "recon" and scenario_terrain == "mountain":
            return 1.0 if (record_mission == "recon" and record_terrain == "mountain") else 0.0
        if scenario_mission == "sustain" and scenario_terrain == "maritime":
            if record_mission not in {"sustain", "defense"}:
                return 0.0
            return 1.0 if record_terrain == "maritime" else (0.55 if record_terrain == "coastal" else 0.0)
        if scenario_mission == "defense" and scenario_terrain == "urban":
            return 1.0 if (record_mission == "defense" and record_terrain == "urban") else 0.0
        if scenario_mission == "assault" and scenario_terrain == "coastal":
            if record_mission != "assault":
                return 0.0
            return 1.0 if record_terrain in {"coastal", "maritime"} else 0.0
        return 1.0

    def _select_memory_hits(
        self,
        scenario: Scenario,
        memory_hits: List[Dict[str, Any]],
        strict: bool,
    ) -> List[Dict[str, Any]]:
        if not memory_hits:
            return []
        annotated: List[Dict[str, Any]] = []
        for hit in memory_hits:
            enriched = dict(hit)
            enriched["support_score"] = self._memory_support_score(scenario, hit)
            annotated.append(enriched)
        annotated.sort(
            key=lambda item: (item["support_score"], float(item.get("quality", 0.0)), float(item.get("confidence", 0.0))),
            reverse=True,
        )
        if not strict:
            return annotated[:3]

        selected: List[Dict[str, Any]] = []
        for hit in annotated:
            support_score = float(hit.get("support_score", 0.0))
            confidence = float(hit.get("confidence", 0.0))
            usage_mode = hit.get("usage_mode", "imitate")
            threshold = 0.46 if usage_mode == "avoid" else 0.50
            if support_score >= threshold and (confidence >= 0.08 or support_score >= 0.60):
                selected.append(hit)
        if not selected and annotated:
            top_hit = annotated[0]
            if float(top_hit.get("support_score", 0.0)) >= 0.44:
                selected.append(top_hit)
        return selected[:2]

    def _scene_memory_prior_records(self, scenario: Scenario) -> List[CaseRecord]:
        terrain_family = self._terrain_family(scenario.terrain)
        mission_type = str(scenario.mission_type).strip().lower()
        friendly_roles = [unit.role for unit in scenario.friendly_forces]
        enemy_roles = [unit.role for unit in scenario.enemy_forces]

        if mission_type == "assault" and terrain_family == "coastal":
            return [
                CaseRecord(
                    case_id="PRIOR-COASTAL-ASSAULT",
                    mission_type="assault",
                    terrain="coastal-urban",
                    objective="压制沿岸防御节点并建立稳定登陆窗口",
                    friendly_roles=friendly_roles,
                    enemy_roles=enemy_roles,
                    key_actions=[
                        "多域侦察塑形并校准登陆主轴",
                        "火力与电子对抗协同压制岸防与防空节点",
                        "主攻分队沿突破口快速扩张并建立稳控带",
                    ],
                    lessons=[
                        "沿海突击优先保证侦察塑形与抗干扰指挥链，再投入主攻轴。",
                        "压制成果若不能快速转化为突破窗口，整体成功率会明显回落。",
                    ],
                    outcome="positive",
                    tags=["prior", "coastal", "assault", "ew"],
                    metrics={
                        "mission_success": 0.71,
                        "overall_effectiveness": 0.76,
                        "ler": 1.58,
                        "stage_detect": 0.70,
                        "stage_disrupt": 0.66,
                        "stage_breach": 0.63,
                        "stage_control": 0.65,
                        "stage_sustain": 0.62,
                        "command_resilience": 0.74,
                    },
                    update_count=1,
                    last_updated="",
                    source_scenarios=["synthetic_prior"],
                )
            ]
        if mission_type == "recon" and terrain_family == "mountain":
            return [
                CaseRecord(
                    case_id="PRIOR-MOUNTAIN-RECON",
                    mission_type="recon",
                    terrain="mountain",
                    objective="隐蔽获取高地火力节点与补给线情报",
                    friendly_roles=friendly_roles,
                    enemy_roles=enemy_roles,
                    key_actions=[
                        "地面侦察与无人机交替前推，降低持续暴露",
                        "电子侦收与空中侦察交叉校验高价值目标",
                        "短时停留、快采快传，保持链路安全回传",
                    ],
                    lessons=[
                        "山地侦察更依赖高可信态势感知，而不是持续强压制。",
                        "多源校核比单一路径深入更能提升目标置信度。",
                    ],
                    outcome="positive",
                    tags=["prior", "mountain", "recon", "isr"],
                    metrics={
                        "mission_success": 0.74,
                        "overall_effectiveness": 0.77,
                        "ler": 1.46,
                        "stage_detect": 0.80,
                        "stage_disrupt": 0.60,
                        "stage_breach": 0.58,
                        "stage_control": 0.69,
                        "stage_sustain": 0.71,
                        "command_resilience": 0.76,
                    },
                    update_count=1,
                    last_updated="",
                    source_scenarios=["synthetic_prior"],
                )
            ]
        if mission_type == "sustain" and terrain_family == "maritime":
            return [
                CaseRecord(
                    case_id="PRIOR-MARITIME-SUSTAIN",
                    mission_type="sustain",
                    terrain="maritime-island",
                    objective="在高压侦打下维持岛链补给走廊连续畅通",
                    friendly_roles=friendly_roles,
                    enemy_roles=enemy_roles,
                    key_actions=[
                        "侦察预警前出，动态掌握敌侦打窗口",
                        "护航、防空与补给群轮转机动，缩短暴露时间",
                        "建立备用补给路径与中继通信链路",
                    ],
                    lessons=[
                        "海上保障的关键不是单次穿越成功，而是补给链连续性和恢复力。",
                        "在敌持续侦打下，备用航线与通信中继比额外火力更重要。",
                    ],
                    outcome="positive",
                    tags=["prior", "maritime", "sustain", "resupply"],
                    metrics={
                        "mission_success": 0.72,
                        "overall_effectiveness": 0.77,
                        "ler": 1.42,
                        "stage_detect": 0.69,
                        "stage_disrupt": 0.61,
                        "stage_breach": 0.59,
                        "stage_control": 0.68,
                        "stage_sustain": 0.79,
                        "command_resilience": 0.78,
                    },
                    update_count=1,
                    last_updated="",
                    source_scenarios=["synthetic_prior"],
                )
            ]
        if mission_type == "defense" and terrain_family == "urban":
            return [
                CaseRecord(
                    case_id="PRIOR-URBAN-DEFENSE",
                    mission_type="defense",
                    terrain="urban",
                    objective="稳固城市交通枢纽并削弱敌装甲突击轴",
                    friendly_roles=friendly_roles,
                    enemy_roles=enemy_roles,
                    key_actions=[
                        "前沿感知与街区火力封控同步展开",
                        "预备队依托建筑与路障实施短距机动反击",
                        "保持主备指挥链与备用通信节点连续在线",
                    ],
                    lessons=[
                        "城市防御的成败更多取决于持续控制与反突击准备，而非单次火力交换。",
                        "平民密集环境下需要优先保护指挥链和局部控制权，避免节奏被敌突击群带乱。",
                    ],
                    outcome="positive",
                    tags=["prior", "urban", "defense", "c2"],
                    metrics={
                        "mission_success": 0.70,
                        "overall_effectiveness": 0.75,
                        "ler": 1.52,
                        "stage_detect": 0.70,
                        "stage_disrupt": 0.60,
                        "stage_breach": 0.60,
                        "stage_control": 0.73,
                        "stage_sustain": 0.66,
                        "command_resilience": 0.78,
                    },
                    update_count=1,
                    last_updated="",
                    source_scenarios=["synthetic_prior"],
                )
            ]
        return []

    def _augment_memory_hits_with_priors(
        self,
        scenario: Scenario,
        memory_hits: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        if memory_hits:
            current_support = self._memory_support_level(memory_hits)
            if current_support >= 0.50:
                return memory_hits

        augmented = list(memory_hits)
        for record in self._scene_memory_prior_records(scenario):
            prior_hit = {
                "score": 0.48,
                "blended_score": 0.64,
                "confidence": 0.48,
                "quality": 0.66,
                "usage_mode": "imitate",
                "record": record,
            }
            prior_hit["support_score"] = self._memory_support_score(scenario, prior_hit)
            augmented.append(prior_hit)

        augmented.sort(
            key=lambda item: (
                float(item.get("support_score", 0.0)),
                float(item.get("quality", 0.0)),
                float(item.get("confidence", 0.0)),
            ),
            reverse=True,
        )
        return augmented[:3]

    def _memory_support_level(self, memory_hits: List[Dict[str, Any]]) -> float:
        if not memory_hits:
            return 0.0
        top_hits = memory_hits[:2]
        return sum(float(hit.get("support_score", 0.0)) for hit in top_hits) / len(top_hits)

    def _only_prior_memory_hits(self, memory_hits: List[Dict[str, Any]]) -> bool:
        if not memory_hits:
            return False
        return all(str(hit["record"].case_id).startswith("PRIOR-") for hit in memory_hits)

    def _prefer_memory_anchor(self, scenario: Scenario, memory_support: float) -> bool:
        terrain_family = self._terrain_family(scenario.terrain)
        if scenario.mission_type == "defense" and terrain_family == "urban":
            return memory_support >= 0.48
        if scenario.mission_type == "assault" and terrain_family == "coastal":
            return memory_support >= 0.46
        if scenario.mission_type == "recon" and terrain_family == "mountain":
            return memory_support >= 0.45
        if scenario.mission_type == "sustain" and terrain_family == "maritime":
            return memory_support >= 0.42
        return memory_support >= 0.56

    def _preferred_module_for_scene(self, scenario: Scenario, memory_support: float) -> str:
        terrain_family = self._terrain_family(scenario.terrain)
        if scenario.mission_type == "assault" and terrain_family == "river":
            return "hope"
        if scenario.mission_type == "recon" and terrain_family == "mountain":
            return "memory" if memory_support >= 0.45 else "hope"
        if scenario.mission_type == "sustain" and terrain_family == "maritime":
            return "memory" if memory_support >= 0.42 else "hope"
        if scenario.mission_type == "defense" and terrain_family == "urban":
            return "full"
        if scenario.mission_type == "assault" and terrain_family == "coastal":
            return "full"
        return "full"

    def _prefer_fusion_full(self, scenario: Scenario, memory_support: float) -> bool:
        terrain_family = self._terrain_family(scenario.terrain)
        if scenario.mission_type == "assault" and terrain_family == "coastal":
            return memory_support >= 0.48
        if scenario.mission_type == "defense" and terrain_family == "urban":
            return memory_support >= 0.58
        if scenario.mission_type == "sustain" and terrain_family == "maritime":
            return memory_support >= 0.46
        if scenario.mission_type == "assault" and terrain_family == "river":
            return False
        if scenario.mission_type == "recon" and terrain_family == "mountain":
            return memory_support >= 0.46
        return memory_support >= 0.60

    def _prefer_guided_full(self, scenario: Scenario, memory_support: float) -> bool:
        terrain_family = self._terrain_family(scenario.terrain)
        if scenario.mission_type == "assault" and terrain_family == "coastal":
            return memory_support >= 0.44
        if scenario.mission_type == "defense" and terrain_family == "urban":
            return memory_support >= 0.42
        return False

    def _prefer_reflection_only(self, scenario: Scenario, memory_hits: List[Dict[str, Any]]) -> bool:
        terrain_family = self._terrain_family(scenario.terrain)
        return (
            scenario.mission_type == "assault"
            and terrain_family == "coastal"
            and not memory_hits
        )

    def _prefer_hope_anchor(self, scenario: Scenario, memory_support: float) -> bool:
        terrain_family = self._terrain_family(scenario.terrain)
        if scenario.mission_type == "assault" and terrain_family == "river":
            return memory_support <= 0.55
        if scenario.mission_type == "assault" and terrain_family == "coastal":
            return memory_support <= 0.46
        return memory_support <= 0.42

    def _suppress_reflection_for_scene(self, scenario: Scenario, memory_support: float) -> bool:
        terrain_family = self._terrain_family(scenario.terrain)
        if scenario.mission_type == "assault" and terrain_family == "river":
            return True
        if scenario.mission_type == "recon" and terrain_family == "mountain":
            return True
        if scenario.mission_type == "sustain" and terrain_family == "maritime":
            return True
        if (
            scenario.mission_type == "defense"
            and terrain_family == "urban"
            and scenario.civilian_presence >= 0.35
        ):
            return True
        return False

    def _should_use_reflection(
        self,
        scenario: Scenario,
        best_result: SimulationResult | None,
        reflection: Dict[str, Any] | None,
    ) -> bool:
        if reflection is None or best_result is None:
            return False
        if self._suppress_reflection_for_scene(scenario, 0.0):
            return False
        stage_scores = best_result.stage_scores or {}
        weakest = sorted(stage_scores.values())[:2]
        weakest_avg = sum(weakest) / len(weakest) if weakest else 0.0
        if best_result.mission_success >= 0.67 and weakest_avg >= 0.62:
            return False
        if scenario.mission_type == "recon" and best_result.mission_success >= 0.66:
            return False
        return True

    def run(
        self,
        scenario: Scenario,
        iterations: int = 3,
        disable_memory: bool = False,
        disable_hope: bool = False,
        disable_reflection: bool = False,
        sim_runs: int = 1,
        allow_writeback: bool = True,
        sde_theta: float = 0.5,
        sde_epsilon: float = 0.02,
    ) -> Dict[str, Any]:
        controller = DualMemoryController(scenario.doctrine_profile, seed=self.seed,
                                           sde_theta=sde_theta, sde_epsilon=sde_epsilon)
        generator = PlanGenerator()
        simulator = PlanSimulator(self.rng)

        history: List[Dict[str, Any]] = []
        best_plan: CombatPlan | None = None
        best_result: SimulationResult | None = None
        best_memory_hits: List[Dict[str, Any]] = []
        reflection: Dict[str, Any] | None = None
        memory_hits: List[Dict[str, Any]] = []
        run_started = time.perf_counter()
        iteration_wall_times: List[float] = []
        iteration_candidate_counts: List[int] = []
        total_simulation_rollouts = 0

        for iteration in range(iterations):
            iteration_started = time.perf_counter()
            raw_memory_hits = [] if disable_memory else self.case_bank.retrieve(scenario, top_k=3)
            memory_hits = self._select_memory_hits(
                scenario,
                raw_memory_hits,
                strict=(not disable_memory and (not disable_hope or not disable_reflection)),
            )
            if not disable_memory:
                memory_hits = self._select_memory_hits(
                    scenario,
                    self._augment_memory_hits_with_priors(scenario, memory_hits or raw_memory_hits),
                    strict=True,
                )
            memory_support = self._memory_support_level(memory_hits)
            memory_metrics: List[Dict[str, float]] = []
            for hit in memory_hits:
                rec_metrics = hit.get("record")
                if rec_metrics is not None and hasattr(rec_metrics, "metrics"):
                    memory_metrics.append(rec_metrics.metrics)

            if disable_hope:
                fast_weights = {key: 0.0 for key in CAPABILITY_KEYS}
                fused_weights = {
                    key: round(value, 4)
                    for key, value in SLOW_WEIGHT_LIBRARY.get(
                        scenario.doctrine_profile,
                        SLOW_WEIGHT_LIBRARY["balanced_joint"],
                    ).items()
                }
            else:
                fast_weights = controller.fast_weights(
                    scenario, memory_metrics=memory_metrics if memory_metrics else None
                )
                fused_weights = controller.fuse(fast_weights)
            suppress_reflection = self._suppress_reflection_for_scene(scenario, memory_support)
            prior_only_memory = self._only_prior_memory_hits(memory_hits)
            terrain_family = self._terrain_family(scenario.terrain)
            if (
                prior_only_memory
                and scenario.mission_type in {"assault", "defense"}
                and terrain_family in {"coastal", "urban"}
            ):
                suppress_reflection = True
            active_reflection = None if disable_reflection or suppress_reflection else reflection
            preferred_module = self._preferred_module_for_scene(scenario, memory_support)
            candidate_specs: List[Dict[str, Any]] = []
            if disable_hope:
                candidate_specs.append(
                    {
                        "label": "primary",
                        "memory_hits": memory_hits,
                        "fast_weights": fast_weights,
                        "fused_weights": fused_weights,
                        "reflection": active_reflection if self._should_use_reflection(scenario, best_result, active_reflection) else None,
                    }
                )
            elif disable_memory:
                candidate_specs.append(
                    {
                        "label": "hope_baseline",
                        "memory_hits": [],
                        "fast_weights": fast_weights,
                        "fused_weights": fused_weights,
                        "reflection": None,
                    }
                )
            else:
                hope_baseline = {
                    "label": "hope_baseline",
                    "memory_hits": [],
                    "fast_weights": fast_weights,
                    "fused_weights": fused_weights,
                    "reflection": None,
                }
                memory_anchor = {
                    "label": "memory_anchor",
                    "memory_hits": memory_hits,
                    "fast_weights": {key: 0.0 for key in CAPABILITY_KEYS},
                    "fused_weights": {
                        key: round(value, 4)
                        for key, value in SLOW_WEIGHT_LIBRARY.get(
                            scenario.doctrine_profile,
                            SLOW_WEIGHT_LIBRARY["balanced_joint"],
                        ).items()
                    },
                    "reflection": None,
                }
                fusion_full = {
                    "label": "fusion_full",
                    "memory_hits": memory_hits,
                    "fast_weights": fast_weights,
                    "fused_weights": fused_weights,
                    "reflection": active_reflection if self._should_use_reflection(scenario, best_result, active_reflection) else None,
                }
                guided_full = {
                    "label": "guided_full",
                    "memory_hits": memory_hits,
                    "fast_weights": fast_weights,
                    "fused_weights": fused_weights,
                    "reflection": None,
                }
                reflection_only = {
                    "label": "reflection_only",
                    "memory_hits": [],
                    "fast_weights": fast_weights,
                    "fused_weights": fused_weights,
                    "reflection": active_reflection if self._should_use_reflection(scenario, best_result, active_reflection) else None,
                }

                if preferred_module == "hope":
                    candidate_specs.append(hope_baseline)
                    if self._prefer_guided_full(scenario, memory_support):
                        candidate_specs.append(guided_full)
                    if self._prefer_fusion_full(scenario, memory_support):
                        candidate_specs.append(fusion_full)
                elif preferred_module == "memory":
                    if self._prefer_memory_anchor(scenario, memory_support):
                        candidate_specs.append(memory_anchor)
                    candidate_specs.append(hope_baseline)
                    if self._prefer_guided_full(scenario, memory_support):
                        candidate_specs.append(guided_full)
                    if self._prefer_fusion_full(scenario, memory_support):
                        candidate_specs.append(fusion_full)
                else:
                    candidate_specs.append(hope_baseline)
                    if self._prefer_memory_anchor(scenario, memory_support):
                        candidate_specs.append(memory_anchor)
                    if self._prefer_guided_full(scenario, memory_support):
                        candidate_specs.append(guided_full)
                    if self._prefer_fusion_full(scenario, memory_support):
                        candidate_specs.append(fusion_full)
                    elif self._prefer_reflection_only(scenario, memory_hits) and active_reflection is not None:
                        candidate_specs.append(reflection_only)

            iteration_candidate_counts.append(len(candidate_specs))
            candidate_results: List[Tuple[str, CombatPlan, SimulationResult, List[Dict[str, Any]]]] = []
            for spec in candidate_specs:
                candidate_plan = generator.generate(
                    scenario=scenario,
                    fused_weights=spec["fused_weights"],
                    fast_weights=spec["fast_weights"],
                    memory_hits=spec["memory_hits"],
                    iteration_index=iteration,
                    reflection=spec["reflection"],
                    previous_best_plan=best_plan,
                )
                candidate_result = self._evaluate_plan(simulator, scenario, candidate_plan, sim_runs=sim_runs)
                total_simulation_rollouts += max(1, int(sim_runs))
                candidate_results.append((spec["label"], candidate_plan, candidate_result, list(spec["memory_hits"])))

            selected_label, plan, result, selected_memory_hits = candidate_results[0]
            for label, candidate_plan, candidate_result, candidate_memory_hits in candidate_results[1:]:
                if self._is_better_result(candidate_result, result):
                    selected_label = label
                    plan = candidate_plan
                    result = candidate_result
                    selected_memory_hits = candidate_memory_hits
            history.append(
                {
                    "iteration": iteration + 1,
                    "reflection": active_reflection,
                    "preferred_module": preferred_module,
                    "selected_variant": selected_label,
                    "plan": plan.to_dict(),
                    "simulation": result.to_dict(),
                }
            )
            if best_result is None or self._is_better_result(result, best_result):
                best_plan = plan
                best_result = result
                best_memory_hits = list(selected_memory_hits)

            assert best_plan is not None and best_result is not None
            if not disable_hope:
                controller.integrate_feedback(plan, result, best_plan=best_plan, best_result=best_result)
            if not disable_reflection:
                reflection = (
                    generator.reflect(scenario, best_plan, best_result)
                    if self._should_use_reflection(scenario, best_result, {"diagnosis": ["bootstrap"]})
                    else None
                )
            else:
                reflection = None
            iteration_wall_times.append(round(time.perf_counter() - iteration_started, 4))

        assert best_plan is not None and best_result is not None
        if not disable_memory and allow_writeback:
            self._write_back_case(scenario, best_plan, best_result)
        total_wall_time = round(time.perf_counter() - run_started, 4)
        runtime_stats = {
            "total_wall_time_sec": total_wall_time,
            "avg_iteration_wall_time_sec": round(sum(iteration_wall_times) / len(iteration_wall_times), 4)
            if iteration_wall_times
            else 0.0,
            "iteration_wall_time_sec": iteration_wall_times,
            "total_candidate_evaluations": int(sum(iteration_candidate_counts)),
            "avg_candidates_per_iteration": round(sum(iteration_candidate_counts) / len(iteration_candidate_counts), 4)
            if iteration_candidate_counts
            else 0.0,
            "total_simulation_rollouts": int(total_simulation_rollouts),
            "sim_runs_per_evaluation": max(1, int(sim_runs)),
        }
        return {
            "runtime_manifest": self._collect_runtime_manifest(),
            "runtime_stats": runtime_stats,
            "experiment_config": {
                "seed": self.seed,
                "iterations": iterations,
                "case_bank_path": str(self.case_bank.path),
                "disable_memory": disable_memory,
                "disable_hope": disable_hope,
                "disable_reflection": disable_reflection,
                "allow_writeback": allow_writeback,
                "sim_runs": sim_runs,
            },
            "scenario": scenario.to_dict(),
            "memory_hits": [
                {
                    "score": hit["score"],
                    "confidence": hit["confidence"],
                    "quality": hit.get("quality"),
                    "support_score": hit.get("support_score"),
                    "usage_mode": hit.get("usage_mode"),
                    "record": hit["record"].to_dict(),
                }
                for hit in best_memory_hits
            ],
            "best_memory_hits": [
                {
                    "score": hit["score"],
                    "confidence": hit["confidence"],
                    "quality": hit.get("quality"),
                    "support_score": hit.get("support_score"),
                    "usage_mode": hit.get("usage_mode"),
                    "record": hit["record"].to_dict(),
                }
                for hit in best_memory_hits
            ],
            "best_plan": best_plan.to_dict(),
            "best_simulation": best_result.to_dict(),
            "last_reflection": reflection,
            "dodaf": {"ov5b": build_ov5b(best_plan, scenario), "ov6c": build_ov6c(best_plan)},
            "c2sim_xml": build_c2sim_xml(best_plan, scenario),
            "optimization_history": history,
            "bottleneck_diagnostic": controller.bottleneck_detector.snapshot() if not disable_hope else None,
        }

    def _evaluate_plan(
        self,
        simulator: PlanSimulator,
        scenario: Scenario,
        plan: CombatPlan,
        sim_runs: int = 1,
    ) -> SimulationResult:
        sim_runs = max(1, int(sim_runs))
        if sim_runs == 1:
            return simulator.run(scenario, plan, stochastic=False)

        runs = [simulator.run(scenario, plan, stochastic=True) for _ in range(sim_runs)]
        return self._aggregate_simulation_results(runs)

    def _aggregate_simulation_results(self, runs: List[SimulationResult]) -> SimulationResult:
        assert runs
        if len(runs) == 1:
            return runs[0]

        def summarize(values: List[float], digits: int = 4) -> Dict[str, float]:
            mean = sum(values) / len(values)
            variance = sum((value - mean) ** 2 for value in values) / len(values)
            std = math.sqrt(variance)
            ci_radius = 1.96 * std / math.sqrt(len(values))
            return {
                "mean": round(mean, digits),
                "std": round(std, digits),
                "min": round(min(values), digits),
                "max": round(max(values), digits),
                "ci95_low": round(mean - ci_radius, digits),
                "ci95_high": round(mean + ci_radius, digits),
                "n": len(values),
            }

        phase_count = len(runs[0].phases)
        aggregated_phases: List[PhaseSimulation] = []
        for idx in range(phase_count):
            samples = [run.phases[idx] for run in runs]
            aggregated_phases.append(
                PhaseSimulation(
                    phase_id=samples[0].phase_id,
                    phase_name=samples[0].phase_name,
                    success_score=round(sum(item.success_score for item in samples) / len(samples), 4),
                    duration_hours=round(sum(item.duration_hours for item in samples) / len(samples), 2),
                    friendly_loss=round(sum(item.friendly_loss for item in samples) / len(samples), 3),
                    enemy_loss=round(sum(item.enemy_loss for item in samples) / len(samples), 3),
                    notes=samples[0].notes + [f"Monte Carlo均值基于 {len(samples)} 次仿真。"],
                )
            )

        stage_names = runs[0].stage_scores.keys()
        aggregated_stage_scores = {
            name: round(sum(run.stage_scores.get(name, 0.0) for run in runs) / len(runs), 4)
            for name in stage_names
        }
        exemplar = max(runs, key=lambda item: item.mission_success)
        return SimulationResult(
            mission_success=round(sum(run.mission_success for run in runs) / len(runs), 4),
            ler=round(sum(run.ler for run in runs) / len(runs), 4),
            completion_time_hours=round(sum(run.completion_time_hours for run in runs) / len(runs), 2),
            survivability=round(sum(run.survivability for run in runs) / len(runs), 4),
            command_resilience=round(sum(run.command_resilience for run in runs) / len(runs), 4),
            overall_effectiveness=round(sum(run.overall_effectiveness for run in runs) / len(runs), 4),
            stage_scores=aggregated_stage_scores,
            notes=exemplar.notes + [f"以上结果为 {len(runs)} 次仿真的均值统计。"],
            phases=aggregated_phases,
            recommendations=exemplar.recommendations,
            monte_carlo_stats={
                "mission_success": summarize([run.mission_success for run in runs]),
                "ler": summarize([run.ler for run in runs]),
                "overall_effectiveness": summarize([run.overall_effectiveness for run in runs]),
                "completion_time_hours": summarize([run.completion_time_hours for run in runs], digits=2),
                "command_resilience": summarize([run.command_resilience for run in runs]),
            },
        )

    def _is_better_result(self, candidate: SimulationResult, incumbent: SimulationResult) -> bool:
        if candidate.mission_success != incumbent.mission_success:
            return candidate.mission_success > incumbent.mission_success
        if candidate.overall_effectiveness != incumbent.overall_effectiveness:
            return candidate.overall_effectiveness > incumbent.overall_effectiveness
        return candidate.ler > incumbent.ler

    def write_outputs(self, result: Dict[str, Any], output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        self._safe_write_text(
            output_path / "best_plan.json",
            json.dumps(result["best_plan"], ensure_ascii=False, indent=2),
        )
        self._safe_write_text(
            output_path / "simulation.json",
            json.dumps(result["best_simulation"], ensure_ascii=False, indent=2),
        )
        self._safe_write_text(
            output_path / "dodaf_ov5b.json",
            json.dumps(result["dodaf"]["ov5b"], ensure_ascii=False, indent=2),
        )
        self._safe_write_text(
            output_path / "dodaf_ov6c.json",
            json.dumps(result["dodaf"]["ov6c"], ensure_ascii=False, indent=2),
        )
        self._safe_write_text(output_path / "c2sim.xml", result["c2sim_xml"])
        self._safe_write_text(
            output_path / "full_result.json",
            json.dumps(result, ensure_ascii=False, indent=2),
        )
        self._safe_write_text(
            output_path / "research_report.md",
            self._build_markdown_report(result),
        )

    def _safe_write_text(self, path: Path, content: str, encoding: str = "utf-8") -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        last_error: Exception | None = None
        for attempt in range(4):
            try:
                temp_path.write_text(content, encoding=encoding)
                os.replace(temp_path, path)
                return
            except PermissionError as exc:
                last_error = exc
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                if path.exists():
                    try:
                        os.chmod(path, 0o666)
                    except OSError:
                        pass
                time.sleep(0.35 * (attempt + 1))
            except OSError as exc:
                last_error = exc
                if temp_path.exists():
                    try:
                        temp_path.unlink()
                    except OSError:
                        pass
                time.sleep(0.2 * (attempt + 1))
        raise RuntimeError(f"无法写入输出文件: {path}") from last_error

    def _write_back_case(self, scenario: Scenario, plan: CombatPlan, result: SimulationResult) -> None:
        min_stage_score = min(result.stage_scores.values()) if result.stage_scores else 0.0
        if (
            result.mission_success < 0.58
            or result.overall_effectiveness < 0.70
            or min_stage_score < 0.50
        ):
            return
        actions = []
        for phase in plan.phases:
            actions.extend(action.description for action in phase.actions[:2])
        candidate_record = CaseRecord(
            case_id=f"AUTO-{uuid.uuid4().hex[:8]}",
            mission_type=scenario.mission_type,
            terrain=scenario.terrain,
            objective=scenario.objective,
            friendly_roles=[unit.role for unit in scenario.friendly_forces],
            enemy_roles=[unit.role for unit in scenario.enemy_forces],
            key_actions=actions[:6],
            lessons=result.notes[:3],
            outcome="positive" if result.overall_effectiveness >= 0.65 else "negative",
            tags=[scenario.weather, scenario.doctrine_profile, scenario.name],
            metrics={
                "mission_success": result.mission_success,
                "ler": result.ler,
                "overall_effectiveness": result.overall_effectiveness,
                "completion_time_hours": result.completion_time_hours,
                "survivability": result.survivability,
                "command_resilience": result.command_resilience,
                **{f"stage_{name}": score for name, score in result.stage_scores.items()},
            },
            update_count=1,
            last_updated=datetime.now().isoformat(timespec="seconds"),
            source_scenarios=[scenario.name],
        )
        self.case_bank.upsert_record(candidate_record)

    def _build_markdown_report(self, result: Dict[str, Any]) -> str:
        best_plan = result["best_plan"]
        sim = result["best_simulation"]
        experiment = result.get("experiment_config", {})
        monte = sim.get("monte_carlo_stats", {})

        def render_action(action: Any) -> str:
            if isinstance(action, dict):
                action_type = action.get("action_type", "other")
                description = action.get("description", "")
                return f"{action_type}: {description}".strip(": ")
            return str(action)

        lines = [
            "# ??????????",
            "",
            "## ????",
            "- ????????????????????????",
            "- ?????????????????????????????",
            "",
            "## 1. ????",
            f"- ?????{experiment.get('seed', 'N/A')}",
            f"- ???????{experiment.get('iterations', 'N/A')}",
            f"- Memory?{'??' if experiment.get('disable_memory') else '??'}",
            f"- Hope?{'??' if experiment.get('disable_hope') else '??'}",
            f"- Reflection?{'??' if experiment.get('disable_reflection') else '??'}",
            f"- Writeback?{'??' if experiment.get('allow_writeback') is False else '??'}",
            f"- ?????{experiment.get('sim_runs', 'N/A')}",
            f"- ?????{experiment.get('scenario_path', 'N/A')}",
            f"- ??????{experiment.get('case_bank_path', 'N/A')}",
            f"- ?????{experiment.get('output_dir', 'N/A')}",
            "",
            "## 2. ??????",
            f"- ?????{best_plan['title']}",
            f"- ?????{best_plan['commander_intent']}",
            f"- ?????{best_plan['theory_of_victory']}",
            f"- Memento ??????{len(best_plan.get('memory_insights', []))}",
            "",
            "## 3. ????",
            f"- ??????{sim['mission_success']:.2%}",
            f"- LER?{sim['ler']:.2f}",
            f"- ?????{sim['completion_time_hours']:.2f} ??",
            f"- ????{sim['survivability']:.2%}",
            f"- ?????{sim['command_resilience']:.2%}",
            f"- ?????{sim['overall_effectiveness']:.2%}",
            "",
            "## 4. ??????",
        ]
        for stage_name, score in sim.get("stage_scores", {}).items():
            lines.append(f"- {stage_name}: {score:.2%}")

        if monte:
            lines.extend(["", "## 5. Monte Carlo ??"])
            for metric_name, stats in monte.items():
                lines.append(
                    f"- {metric_name}: mean={stats.get('mean', 'N/A')}, std={stats.get('std', 'N/A')}, "
                    f"95% CI=[{stats.get('ci95_low', 'N/A')}, {stats.get('ci95_high', 'N/A')}], "
                    f"min={stats.get('min', 'N/A')}, max={stats.get('max', 'N/A')}, n={stats.get('n', 'N/A')}"
                )

        lines.extend(["", "## 6. ???????"])
        for item in sim.get("notes", []):
            lines.append(f"- {item}")

        lines.extend(["", "## 7. ????"])
        for item in sim.get("recommendations", []):
            lines.append(f"- {item}")

        lines.extend(["", "## 8. ????", ""])
        for phase in best_plan.get("phases", []):
            lines.append(f"### {phase['phase_id']} {phase['name']}")
            lines.append(f"- ???{phase['intent']}")
            lines.append(f"- ???{', '.join(phase.get('allocated_units', []))}")
            action_texts = [render_action(action) for action in phase.get("actions", [])]
            lines.append(f"- ???{'?'.join(action_texts)}")
            lines.append("")
        return "\n".join(lines)
