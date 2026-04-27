from __future__ import annotations

import json
import math
import os
import random
import re
import uuid
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .case_memory import CaseBank, CaseRecord
from .domain import CAPABILITY_KEYS, CombatPlan, PhaseSimulation, PlanPhase, Scenario, SimulationResult, clamp_value
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


class HOPEAdapter(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        
        # 输入到隐藏层
        self.i2h = nn.Linear(input_dim, hidden_dim)
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
        
        # 隐藏状态
        self.hidden = nn.Parameter(torch.zeros(hidden_dim))
        
        # 时间追踪器
        self.register_buffer('t_current', torch.tensor(1.0))
        self.register_buffer('t_create', torch.ones(output_dim))
        
        # 隐藏到输出
        self.h2o = nn.Linear(hidden_dim, output_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [batch, input_dim]
        h = self.hidden.unsqueeze(0).expand(x.size(0), -1)  # [batch, hidden_dim]
        
        # 推进物理时间
        if self.training:
            self.t_current += 1.0
        
        # 1. 记忆价值估计 V(c_t) = I / H 的神经网络逼近
        val_input = torch.cat([x, h], dim=1)  # [batch, input_dim + hidden_dim]
        v = self.value_estimator(val_input)   # [batch, output_dim]
        
        # 2. 年龄计算 log(t_current / t_create)
        age = torch.clamp(self.t_current - self.t_create, min=1.0) # [output_dim]
        log_age = torch.log(age).unsqueeze(0).expand(x.size(0), -1) # [batch, output_dim]
        
        # 3. 双维度遗忘门控
        # 保证 delta 始终为负，以确保随时间衰减的物理意义
        delta_neg = -F.relu(-self.forget_delta) 
        forget = torch.sigmoid(self.forget_gamma * v + delta_neg * log_age + self.forget_beta)  # [batch, output_dim]
        
        # 输入门与候选状态
        input_gate = torch.sigmoid(self.i2h(x))  # [batch, hidden_dim]
        candidate = torch.tanh(self.i2h(x))      # [batch, hidden_dim]
        
        # 更新隐藏状态 (batch-wise, then average)
        new_h = (1 - input_gate) * h + input_gate * candidate  # [batch, hidden_dim]
        self.hidden.data = new_h.mean(dim=0).detach()  # 更新参数为平均值
        
        # 更新 t_create: 若信息有大量翻新 (这里用 input_gate 的平均强度代理)，则创造时间前移
        if self.training:
            refresh_rate = input_gate.mean().item()
            self.t_create = (1 - refresh_rate) * self.t_create + refresh_rate * self.t_current
        
        # 输出
        output = self.i2o(x) + self.h2o(h)  # [batch, output_dim]
        
        # 应用基于价值和时间的遗忘
        output = forget * output
        
        return output


class DualMemoryController(nn.Module):
    def __init__(self, doctrine_profile: str):
        super().__init__()
        self.doctrine_profile = doctrine_profile
        
        # 初始化slow_weights为可学习参数
        initial_slow = SLOW_WEIGHT_LIBRARY.get(doctrine_profile, SLOW_WEIGHT_LIBRARY["balanced_joint"])
        self.slow_weights = nn.Parameter(torch.tensor(list(initial_slow.values()), dtype=torch.float32))
        
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
        self.sde_theta = 0.5      # 回复系数 (遗忘率)
        self.sde_epsilon = 0.02   # 噪声强度
        self.sde_dt = 1.0         # 离散化时间步长
        self.register_buffer('current_fast', torch.zeros(output_dim))
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
    
    def fast_weights(self, scenario: Scenario) -> Dict[str, float]:
        self.current_features = self._encode_scenario(scenario)
        target_fast = self.adapter(self.current_features).squeeze(0)  # [output_dim]
        
        # SDE Euler-Maruyama 离散化步进: 
        # dW_fast = theta * (F(W_slow, Phi) - W_fast) * dt + epsilon * sqrt(dt) * noise
        noise = torch.randn_like(self.current_fast)
        with torch.no_grad():
            dW = self.sde_theta * (target_fast - self.current_fast) * self.sde_dt + \
                 self.sde_epsilon * math.sqrt(self.sde_dt) * noise
            self.current_fast += dW
            
        weights = {key: float(adj) for key, adj in zip(CAPABILITY_KEYS, self.current_fast)}
        return {key: round(value, 4) for key, value in weights.items()}
    
    def fuse(self, fast_weights: Dict[str, float]) -> Dict[str, float]:
        # 快慢权重耦合，输出时保护性限制
        fast_tensor = torch.tensor(list(fast_weights.values()), dtype=torch.float32)
        fused_tensor = torch.clamp(self.slow_weights + fast_tensor, 0.2, 1.0)
        fused = {key: round(float(val), 4) for key, val in zip(CAPABILITY_KEYS, fused_tensor)}
        return fused
    
    def integrate_feedback(self, best_plan: CombatPlan, result: SimulationResult) -> None:
        target_weights = torch.tensor(list(best_plan.fused_weights.values()), dtype=torch.float32)
        
        # 1. 恢复计算图：通过 current_features 得到最新的 F(W_slow, Phi)
        if self.current_features is not None:
            target_fast = self.adapter(self.current_features).squeeze(0)
        else:
            target_fast = torch.zeros_like(self.slow_weights)
            
        # 2. Adapter 损失 (仅反向传播更新 adapter)
        expected_fast = target_weights - self.slow_weights.detach()
        loss_adapter = F.mse_loss(target_fast, expected_fast)
        
        self.adapter_optimizer.zero_grad()
        loss_adapter.backward()
        self.adapter_optimizer.step()
        
        # 3. 慢权重的多目标损失构建 (代理损失函数)
        # 任务总基准：贴近最佳方案
        L_task = F.mse_loss(self.slow_weights, target_weights)
        
        # 时间维度：若时间长，需强化 mobility (1) 和 fires (0)
        target_time = target_weights.clone().detach()
        target_time[1] = 1.0; target_time[0] = 1.0
        L_time = F.mse_loss(self.slow_weights, target_time) * (result.completion_time_hours / 24.0)
        
        # 指挥韧性维度：若低，强化 c2 (6), ew (4), awareness (3)
        target_res = target_weights.clone().detach()
        target_res[6] = 1.0; target_res[4] = 1.0; target_res[3] = 1.0
        L_res = F.mse_loss(self.slow_weights, target_res) * (1.0 - result.command_resilience)
        
        # 生存与交换比维度：若低，强化 protection (2), sustainment (5)
        target_surv = target_weights.clone().detach()
        target_surv[2] = 1.0; target_surv[5] = 1.0
        ler_penalty = 1.0 / max(result.ler, 0.1)
        L_surv = F.mse_loss(self.slow_weights, target_surv) * ler_penalty
        
        # 4. 计算各个独立梯度
        grads = []
        for L in [L_task, L_time, L_res, L_surv]:
            if self.slow_weights.grad is not None:
                self.slow_weights.grad.zero_()
            L.backward(retain_graph=True)
            if self.slow_weights.grad is not None:
                grads.append(self.slow_weights.grad.clone())
        
        # 5. PCGrad (冲突梯度投影) 寻找帕累托最优方向
        if len(grads) > 0:
            import random
            random.shuffle(grads)  # 消除顺序偏差
            pc_grads = [grads[0].clone()]
            for i in range(1, len(grads)):
                g_i = grads[i].clone()
                for g_j in pc_grads:
                    dot_product = torch.dot(g_i, g_j)
                    if dot_product < 0:
                        # 冲突投影
                        g_i = g_i - (dot_product / (torch.norm(g_j)**2 + 1e-8)) * g_j
                pc_grads.append(g_i)
            
            # 综合帕累托梯度
            g_pareto = sum(pc_grads)
            
            # 6. 时滞补偿 (Time-Delay Correction)
            W_t = self.slow_weights.data.clone()
            delay_diff = W_t - self.past_slow_weights
            lambda_delay = 0.1
            g_final = g_pareto + lambda_delay * delay_diff
            
            # 更新历史状态
            self.past_slow_weights.data.copy_(W_t)
            
            # 7. 黎曼流形对数障碍收回 (Riemannian Retraction)
            lr = 0.05
            a, b = 0.2, 1.0
            with torch.no_grad():
                W_safe = torch.clamp(self.slow_weights.data, a + 1e-5, b - 1e-5)
                ratio = (b - W_safe) / (W_safe - a)
                W_new = a + (b - a) / (1.0 + ratio * torch.exp(-lr * g_final))
                self.slow_weights.data = W_new
                
        if self.slow_weights.grad is not None:
            self.slow_weights.grad.zero_()


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

    def _extract_json_block(self, text: str) -> str:
        stripped = text.strip()
        if stripped.startswith("```"):
            stripped = re.sub(r"^```[^\n]*\n", "", stripped)
            stripped = re.sub(r"\n```$", "", stripped).strip()
        match = re.search(r"\{[\s\S]*\}", stripped)
        return match.group(0) if match else stripped

    def _format_case_brief(self, hit: Dict[str, Any]) -> Dict[str, Any]:
        record: CaseRecord = hit["record"]
        return {
            "case_id": record.case_id,
            "similarity": hit["score"],
            "confidence": hit["confidence"],
            "mission_type": record.mission_type,
            "terrain": record.terrain,
            "objective": record.objective,
            "friendly_roles": record.friendly_roles,
            "enemy_roles": record.enemy_roles,
            "key_actions": record.key_actions,
            "lessons": record.lessons,
            "outcome": record.outcome,
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
            "phase_actions": {phase.phase_id: phase.actions for phase in plan.phases},
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
            f"- 约束：{'；'.join(scenario['constraints'])}",
            "",
            "友军单元名称：",
            f"- {'；'.join(unit['name'] for unit in scenario['friendly_forces'])}",
            "",
            "敌军单元名称：",
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
                f"- {case['case_id']} | 相似度 {case['similarity']:.2f} | 关键动作：{'；'.join(case['key_actions'][:4])} | 教训：{'；'.join(case['lessons'][:2])}"
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
                "- 只生成 4 个阶段。",
                "- phase_id 必须是 P1 到 P4。",
                "- 每个阶段名称必须是正常中文短语。",
                "- allocated_units 只能从给定友军单元名称中选择。",
                "- 每个阶段至少 2 个 allocated_units，且全方案必须覆盖全部友军单元。",
                "- 每个阶段至少 2 条 actions、2 条 decision_points、2 条 expected_effects。",
                "- actions、decision_points、expected_effects 都必须是简洁中文。",
                "- commander_intent、theory_of_victory、risk_controls 必须是专业作战语言。",
            ]
        )
        return "\n".join(lines)

    def _build_reflection_user_prompt(self, payload: Dict[str, Any]) -> str:
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
            phase.actions = self._ensure_minimum_items(
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
            target_stage = self._preferred_stage_for_unit(unit_name, scenario)
            phases[target_stage].allocated_units.append(unit_name)

        for idx, phase in enumerate(phases):
            stage_name = self._infer_stage_name(idx)
            phase.allocated_units = self._normalize_units(
                phase.allocated_units,
                friendly_units,
                fallback_units[stage_name],
            )

        return phases

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

    def _ensure_minimum_items(self, values: List[str], fallback_values: List[str], minimum: int) -> List[str]:
        cleaned = [item.strip() for item in values if item and item.strip()]
        for candidate in fallback_values:
            candidate = candidate.strip()
            if candidate and candidate not in cleaned:
                cleaned.append(candidate)
            if len(cleaned) >= minimum:
                break
        return cleaned[: max(minimum, len(cleaned))]

    def _infer_stage_name(self, index: int) -> str:
        return ["detect", "disrupt", "breach", "control"][min(index, 3)] if index < 4 else "sustain"

    def _default_actions(self, stage_name: str, phase: PlanPhase, scenario: Scenario) -> List[str]:
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
            insights.append(f"案例{record.case_id}（相似度 {hit['score']:.2f}）提示：{lessons}")
        return insights

    def _build_phase(self, payload: Dict[str, Any], default_index: int) -> PlanPhase:
        phase_id = str(payload.get("phase_id") or f"P{default_index}")
        return PlanPhase(
            phase_id=phase_id,
            name=str(payload.get("name") or f"阶段{default_index}"),
            intent=str(payload.get("intent") or ""),
            actions=self._ensure_list(payload.get("actions"), fallback=["待补充动作"]),
            allocated_units=self._ensure_list(payload.get("allocated_units"), fallback=["待分配兵力"]),
            decision_points=self._ensure_list(payload.get("decision_points"), fallback=["待补充决策点"]),
            expected_effects=self._ensure_list(payload.get("expected_effects"), fallback=["待补充预期效果"]),
        )

    def _ensure_list(self, value: Any, fallback: List[str]) -> List[str]:
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


class PlanSimulator:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def run(self, scenario: Scenario, plan: CombatPlan) -> SimulationResult:
        blue_caps = self._aggregate_capabilities(scenario.friendly_forces)
        red_caps = self._aggregate_capabilities(scenario.enemy_forces)
        phases: List[PhaseSimulation] = []
        total_friendly_loss = 0.0
        total_enemy_loss = 0.0
        total_time = 0.0
        stage_accumulator: Dict[str, List[float]] = defaultdict(list)

        for phase in plan.phases:
            requirements = self._phase_requirements(phase)
            stage_scores = self._evaluate_kill_chain(phase, scenario, plan, blue_caps, red_caps, requirements)
            emphasis = self._phase_stage_emphasis(phase)
            success = sum(stage_scores[name] * weight for name, weight in emphasis.items())
            blue_power = self._blue_effectiveness(blue_caps, plan.fused_weights, requirements)
            red_power = self._red_effectiveness(red_caps, scenario, requirements)

            duration = max(
                1.0,
                len(phase.actions) * 1.05 + 2.8 * scenario.time_pressure - 1.8 * plan.fused_weights["mobility"],
            )
            friendly_loss = max(0.005, red_power * (1 - success) * 0.006)
            enemy_loss = max(0.008, blue_power * success * 0.008)
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

    def _evaluate_kill_chain(
        self,
        phase: PlanPhase,
        scenario: Scenario,
        plan: CombatPlan,
        blue_caps: Dict[str, float],
        red_caps: Dict[str, float],
        requirements: Dict[str, float],
    ) -> Dict[str, float]:
        terrain_penalty = 0.06 if any(token in scenario.terrain.lower() for token in ["urban", "mountain"]) else 0.02
        weather_penalty = 0.05 if any(token in scenario.weather.lower() for token in ["rain", "storm", "fog"]) else 0.01
        civilian_penalty = 0.12 * scenario.civilian_presence
        ew_pressure = 0.16 * scenario.ew_threat

        detect_gap = (
            0.45 * blue_caps["awareness"] * plan.fused_weights["awareness"]
            + 0.25 * blue_caps["c2"] * plan.fused_weights["c2"]
            + 0.15 * blue_caps["ew"] * plan.fused_weights["ew"]
            - 0.30 * red_caps["awareness"]
            - 0.18 * red_caps["ew"]
        )
        disrupt_gap = (
            0.45 * blue_caps["fires"] * plan.fused_weights["fires"]
            + 0.20 * blue_caps["ew"] * plan.fused_weights["ew"]
            + 0.15 * blue_caps["c2"] * plan.fused_weights["c2"]
            - 0.30 * red_caps["fires"]
            - 0.20 * red_caps["c2"]
        )
        breach_gap = (
            0.30 * blue_caps["mobility"] * plan.fused_weights["mobility"]
            + 0.25 * blue_caps["protection"] * plan.fused_weights["protection"]
            + 0.30 * blue_caps["fires"] * plan.fused_weights["fires"]
            - 0.40 * red_caps["protection"]
            - 0.35 * red_caps["fires"]
        )
        control_gap = (
            0.35 * blue_caps["protection"] * plan.fused_weights["protection"]
            + 0.35 * blue_caps["c2"] * plan.fused_weights["c2"]
            + 0.30 * blue_caps["mobility"] * plan.fused_weights["mobility"]
            - 0.35 * red_caps["fires"]
            - 0.25 * red_caps["mobility"]
        )
        sustain_gap = (
            0.45 * blue_caps["sustainment"] * plan.fused_weights["sustainment"]
            + 0.30 * blue_caps["c2"] * plan.fused_weights["c2"]
            + 0.25 * blue_caps["protection"] * plan.fused_weights["protection"]
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

        joined = " ".join(phase.actions + [phase.name, phase.intent])
        if "侦察" in joined or "塑形" in joined:
            stage_scores["detect"] = clamp_value(stage_scores["detect"] + 0.05)
        if "火力" in joined or "压制" in joined:
            stage_scores["disrupt"] = clamp_value(stage_scores["disrupt"] + 0.05)
        if "突击" in joined or "夺控" in joined or "突破" in joined:
            stage_scores["breach"] = clamp_value(stage_scores["breach"] + 0.05)
            stage_scores["control"] = clamp_value(stage_scores["control"] + 0.03)
        if "稳控" in joined or "投送" in joined or "补给" in joined:
            stage_scores["sustain"] = clamp_value(stage_scores["sustain"] + 0.05)

        return {name: round(value, 4) for name, value in stage_scores.items()}

    def _bounded_score(self, gap: float, scale: float, penalty: float) -> float:
        return clamp_value(1.0 / (1.0 + math.exp(-(gap / scale - penalty))), 0.12, 0.96)

    def _phase_stage_emphasis(self, phase: PlanPhase) -> Dict[str, float]:
        emphasis = {"detect": 0.20, "disrupt": 0.20, "breach": 0.20, "control": 0.20, "sustain": 0.20}
        joined = " ".join(phase.actions + [phase.name, phase.intent])
        if "侦察" in joined or "塑形" in joined:
            emphasis = {"detect": 0.40, "disrupt": 0.20, "breach": 0.15, "control": 0.10, "sustain": 0.15}
        elif "火力" in joined or "压制" in joined:
            emphasis = {"detect": 0.15, "disrupt": 0.40, "breach": 0.20, "control": 0.10, "sustain": 0.15}
        elif "突击" in joined or "夺控" in joined or "突破" in joined:
            emphasis = {"detect": 0.10, "disrupt": 0.20, "breach": 0.40, "control": 0.20, "sustain": 0.10}
        elif "稳控" in joined or "补给" in joined or "投送" in joined:
            emphasis = {"detect": 0.10, "disrupt": 0.10, "breach": 0.15, "control": 0.25, "sustain": 0.40}
        return emphasis

    def _aggregate_capabilities(self, units) -> Dict[str, float]:
        aggregated = defaultdict(float)
        for unit in units:
            scale = max(1.0, math.sqrt(unit.quantity)) * clamp_value(unit.readiness, 0.3, 1.0)
            for key in CAPABILITY_KEYS:
                aggregated[key] += getattr(unit, key) * scale
        return aggregated

    def _phase_requirements(self, phase: PlanPhase) -> Dict[str, float]:
        weights = {key: 0.35 for key in CAPABILITY_KEYS}
        joined = " ".join(phase.actions + [phase.name, phase.intent])
        lower = joined.lower()
        if "侦察" in joined or "态势" in joined:
            weights["awareness"] += 0.35
            weights["c2"] += 0.08
        if "电子" in joined or "电磁" in joined:
            weights["ew"] += 0.35
            weights["c2"] += 0.06
        if "火力" in joined or "打击" in joined or "导弹" in joined:
            weights["fires"] += 0.45
        if "突击" in joined or "机动" in joined or "突破" in joined:
            weights["mobility"] += 0.35
            weights["protection"] += 0.12
        if "补给" in joined or "稳控" in joined or "恢复" in joined:
            weights["sustainment"] += 0.35
        if "防护" in joined or "防御" in joined:
            weights["protection"] += 0.35
        if "评估" in joined or "battle damage" in lower:
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
        self.rng = random.Random(seed)

    def run(self, scenario: Scenario, iterations: int = 3) -> Dict[str, Any]:
        controller = DualMemoryController(scenario.doctrine_profile)
        generator = PlanGenerator()
        simulator = PlanSimulator(self.rng)

        history: List[Dict[str, Any]] = []
        best_plan: CombatPlan | None = None
        best_result: SimulationResult | None = None
        reflection: Dict[str, Any] | None = None
        memory_hits: List[Dict[str, Any]] = []

        for iteration in range(iterations):
            memory_hits = self.case_bank.retrieve(scenario, top_k=3)
            fast_weights = controller.fast_weights(scenario)
            fused_weights = controller.fuse(fast_weights)
            plan = generator.generate(
                scenario=scenario,
                fused_weights=fused_weights,
                fast_weights=fast_weights,
                memory_hits=memory_hits,
                iteration_index=iteration,
                reflection=reflection,
                previous_best_plan=best_plan,
            )
            result = simulator.run(scenario, plan)
            history.append(
                {
                    "iteration": iteration + 1,
                    "reflection": reflection,
                    "plan": plan.to_dict(),
                    "simulation": result.to_dict(),
                }
            )
            if best_result is None or self._is_better_result(result, best_result):
                best_plan = plan
                best_result = result

            assert best_plan is not None and best_result is not None
            controller.integrate_feedback(best_plan, best_result)
            reflection = generator.reflect(scenario, best_plan, best_result)

        assert best_plan is not None and best_result is not None
        self._write_back_case(scenario, best_plan, best_result)
        return {
            "scenario": scenario.to_dict(),
            "memory_hits": [
                {
                    "score": hit["score"],
                    "confidence": hit["confidence"],
                    "record": hit["record"].to_dict(),
                }
                for hit in memory_hits
            ],
            "best_plan": best_plan.to_dict(),
            "best_simulation": best_result.to_dict(),
            "last_reflection": reflection,
            "dodaf": {"ov5b": build_ov5b(best_plan, scenario), "ov6c": build_ov6c(best_plan)},
            "c2sim_xml": build_c2sim_xml(best_plan, scenario),
            "optimization_history": history,
        }

    def _is_better_result(self, candidate: SimulationResult, incumbent: SimulationResult) -> bool:
        if candidate.mission_success != incumbent.mission_success:
            return candidate.mission_success > incumbent.mission_success
        if candidate.overall_effectiveness != incumbent.overall_effectiveness:
            return candidate.overall_effectiveness > incumbent.overall_effectiveness
        return candidate.ler > incumbent.ler

    def write_outputs(self, result: Dict[str, Any], output_dir: str | Path) -> None:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        (output_path / "best_plan.json").write_text(
            json.dumps(result["best_plan"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "simulation.json").write_text(
            json.dumps(result["best_simulation"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "dodaf_ov5b.json").write_text(
            json.dumps(result["dodaf"]["ov5b"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "dodaf_ov6c.json").write_text(
            json.dumps(result["dodaf"]["ov6c"], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "c2sim.xml").write_text(result["c2sim_xml"], encoding="utf-8")
        (output_path / "full_result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        (output_path / "research_report.md").write_text(
            self._build_markdown_report(result),
            encoding="utf-8",
        )

    def _write_back_case(self, scenario: Scenario, plan: CombatPlan, result: SimulationResult) -> None:
        actions = []
        for phase in plan.phases:
            actions.extend(phase.actions[:2])
        self.case_bank.add_record(
            CaseRecord(
                case_id=f"AUTO-{uuid.uuid4().hex[:8]}",
                mission_type=scenario.mission_type,
                terrain=scenario.terrain,
                objective=scenario.objective,
                friendly_roles=[unit.role for unit in scenario.friendly_forces],
                enemy_roles=[unit.role for unit in scenario.enemy_forces],
                key_actions=actions[:6],
                lessons=result.notes[:3],
                outcome="positive" if result.overall_effectiveness >= 0.65 else "negative",
                tags=[scenario.weather, scenario.doctrine_profile],
                metrics={
                    "mission_success": result.mission_success,
                    "ler": result.ler,
                    "overall_effectiveness": result.overall_effectiveness,
                },
            )
        )

    def _build_markdown_report(self, result: Dict[str, Any]) -> str:
        best_plan = result["best_plan"]
        sim = result["best_simulation"]
        lines = [
            "# 毕设研究原型运行报告",
            "",
            "## 1. 双重记忆认知智能体",
            f"- 方案名称：{best_plan['title']}",
            f"- 慢权重（Hope 长时策略）主导能力：{', '.join(sorted(best_plan['doctrine_weights'], key=best_plan['doctrine_weights'].get, reverse=True)[:3])}",
            f"- 快权重（Hope 快速适应）主导能力：{', '.join(sorted(best_plan['fast_weights'], key=best_plan['fast_weights'].get, reverse=True)[:3])}",
            f"- Memento 记忆启发数：{len(best_plan['memory_insights'])}",
            "",
            "## 2. 标准化方案输出",
            f"- DoDAF OV-5b 活动数：{len(result['dodaf']['ov5b']['activities'])}",
            f"- DoDAF OV-6c 事件数：{len(result['dodaf']['ov6c']['event_traces'])}",
            f"- C2SIM XML 已生成：是",
            "",
            "## 3. 仿真评估结果",
            f"- 任务成功度：{sim['mission_success']:.2%}",
            f"- 战损交换比 LER：{sim['ler']:.2f}",
            f"- 完成时间：{sim['completion_time_hours']:.2f} 小时",
            f"- 生存性：{sim['survivability']:.2%}",
            f"- 指挥韧性：{sim['command_resilience']:.2%}",
            f"- 综合效能：{sim['overall_effectiveness']:.2%}",
            "",
            "## 4. 杀伤链分阶段得分",
        ]
        for name, score in sim.get("stage_scores", {}).items():
            lines.append(f"- {name}: {score:.2%}")
        lines.extend(["", "## 5. 失败教训与反思"])
        for item in sim["notes"]:
            lines.append(f"- {item}")
        lines.extend(["", "## 6. 优化建议"])
        for item in sim["recommendations"]:
            lines.append(f"- {item}")
        lines.extend(["", "## 7. 典型阶段", ""])
        for phase in best_plan["phases"]:
            lines.append(f"### {phase['phase_id']} {phase['name']}")
            lines.append(f"- 意图：{phase['intent']}")
            lines.append(f"- 兵力：{', '.join(phase['allocated_units'])}")
            lines.append(f"- 动作：{'；'.join(phase['actions'])}")
            lines.append("")
        return "\n".join(lines)
