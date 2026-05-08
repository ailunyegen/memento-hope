from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List


OUTPUT_PATH = Path("data/military_case_bank_generalization_v2.jsonl")


@dataclass(frozen=True)
class TaskGroup:
    code: str
    mission_type: str
    terrain: str
    objective: str
    friendly_roles: List[str]
    enemy_roles: List[str]
    positive_actions: List[str]
    negative_actions: List[str]
    positive_lessons: List[str]
    negative_lessons: List[str]
    tags: List[str]
    stage_focus: Dict[str, float]


TASK_GROUPS: List[TaskGroup] = [
    TaskGroup(
        code="coastal_joint_assault",
        mission_type="assault",
        terrain="coastal",
        objective="在虚拟沿海场景中夺控登陆窗口并压制岸防感知链路",
        friendly_roles=["两栖突击分队", "远程火力群", "无人侦察分队", "电子对抗分队", "联合指挥通信单元"],
        enemy_roles=["岸防火力节点", "防空分队", "机步预备队", "电子干扰节点"],
        positive_actions=[
            "以无人侦察和多源情报校核建立登陆窗口态势图",
            "同步压制岸防火力与防空感知节点",
            "以电子对抗遮断敌方目标指示和协同通信",
            "主攻轴在压制窗口内完成快速机动与夺控",
            "预置补给与通信中继保证后续波次持续展开",
        ],
        negative_actions=[
            "未完成岸防节点确认即发起主攻",
            "火力压制与电子对抗窗口不同步",
            "登陆后缺少稳定通信中继和补给接续",
        ],
        positive_lessons=[
            "沿海突击需要先固化侦察-压制-突破的时间窗口，再投入主攻轴。",
            "电子对抗与远程火力同步时，breach 与 control 阶段更稳定。",
            "持续补给和指挥通信会决定压制成果能否转化为控制效果。",
        ],
        negative_lessons=[
            "压制链路与机动链路脱节会导致 breach 阶段收益回落。",
            "忽视通信中继会使 sustain 阶段成为后续短板。",
        ],
        tags=["joint", "coastal", "assault", "ew", "breach"],
        stage_focus={"detect": 0.72, "disrupt": 0.75, "breach": 0.76, "control": 0.70, "sustain": 0.66},
    ),
    TaskGroup(
        code="river_crossing_breakthrough",
        mission_type="assault",
        terrain="river",
        objective="在虚拟江河强渡场景中建立渡河通路并扩大突破纵深",
        friendly_roles=["工程保障分队", "机动突击分队", "远程火力群", "防空掩护分队", "通信保障分队"],
        enemy_roles=["河岸防御分队", "反机动火力节点", "无人侦察节点", "预备反击分队"],
        positive_actions=[
            "先行侦察河岸火力、通行条件和敌反机动节点",
            "压制河岸观察点与反机动火力，建立短时强渡窗口",
            "工程保障分队构设虚拟通路并同步组织机动掩护",
            "防空与电子对抗压缩敌无人侦察回传链路",
            "突破后快速展开补给、通信和纵深控制节点",
        ],
        negative_actions=[
            "未压制河岸观察点就暴露渡河组织",
            "工程通路与机动突击不同步",
            "突破后未建立纵深补给和通信保障",
        ],
        positive_lessons=[
            "江河突破的关键是把 disrupt 成果转化为工程通路和机动节奏。",
            "breach 阶段需要机动、火力、防护和保障同时达标。",
            "强渡后 sustain 不稳会快速削弱前期突破收益。",
        ],
        negative_lessons=[
            "只强化火力而忽略工程保障，会使 breach 阶段停滞。",
            "缺少纵深控制节点会使突破成果难以保持。",
        ],
        tags=["river", "assault", "mobility", "engineering", "sustain"],
        stage_focus={"detect": 0.70, "disrupt": 0.72, "breach": 0.78, "control": 0.68, "sustain": 0.72},
    ),
    TaskGroup(
        code="urban_hub_defense",
        mission_type="defense",
        terrain="urban",
        objective="在虚拟城市枢纽场景中迟滞突入并保持关键节点控制",
        friendly_roles=["城市防御分队", "反装甲分队", "无人侦察分队", "预备机动分队", "通信保障分队"],
        enemy_roles=["装甲突击分队", "无人侦察群", "火力支援节点", "渗透分队"],
        positive_actions=[
            "以分布式侦察持续识别突入方向和渗透通道",
            "构建分层火力与障碍体系迟滞敌机动节奏",
            "预备队按触发条件实施局部封堵和反冲击",
            "保持主备通信链路与分散指挥节点",
            "在关键枢纽周边维持控制、救援和补给通道",
        ],
        negative_actions=[
            "把所有力量集中在单一街区造成侧翼空隙",
            "未准备备用通信链路导致局部指挥失联",
            "反冲击时机过早造成预备队消耗",
        ],
        positive_lessons=[
            "城市防御应优先保证 detect 与 control 的连续性。",
            "反装甲、障碍、通信和预备队协同能稳定 disrupt 阶段。",
            "分布式指挥能降低局部失联对整体防御的影响。",
        ],
        negative_lessons=[
            "单点堆叠容易被绕行，control 阶段会被持续侵蚀。",
            "缺少备用通信链会放大电子压制后的指挥脆弱性。",
        ],
        tags=["urban", "defense", "c2", "control", "resilience"],
        stage_focus={"detect": 0.74, "disrupt": 0.73, "breach": 0.64, "control": 0.78, "sustain": 0.70},
    ),
    TaskGroup(
        code="mountain_corridor_recon",
        mission_type="recon",
        terrain="mountain",
        objective="在虚拟山地走廊中隐蔽获取节点态势并保持侦察链路",
        friendly_roles=["侦察分队", "无人机群", "电子侦收分队", "山地保障分队", "通信中继分队"],
        enemy_roles=["山地警戒分队", "机动补给分队", "火力观察节点", "电子侦察节点"],
        positive_actions=[
            "多路隐蔽接近并建立分层观察点",
            "无人机与电子侦收交叉验证高价值节点",
            "以低暴露通信方式回传目标可信度",
            "预设撤收路线和补给点维持侦察持续性",
            "发现敌补给与火力节点后形成可共享态势图",
        ],
        negative_actions=[
            "侦察路线单一导致暴露风险集中",
            "只依赖单一无人平台确认目标",
            "缺少撤收与补给预案造成侦察链中断",
        ],
        positive_lessons=[
            "山地侦察的收益来自多源校核和持续回传，而非单次发现。",
            "通信中继与补给节点会决定 sustain 阶段稳定性。",
            "低暴露路线规划能提高 detect 阶段有效性。",
        ],
        negative_lessons=[
            "单源情报容易造成误判，后续 disrupt 和 control 会受影响。",
            "忽视撤收与补给会使侦察优势不可持续。",
        ],
        tags=["mountain", "recon", "isr", "c2", "sustain"],
        stage_focus={"detect": 0.82, "disrupt": 0.65, "breach": 0.62, "control": 0.70, "sustain": 0.74},
    ),
    TaskGroup(
        code="island_resupply_corridor",
        mission_type="sustain",
        terrain="maritime",
        objective="在虚拟岛链补给场景中维持走廊通畅和节点保障",
        friendly_roles=["海上补给分队", "防空掩护分队", "无人侦察分队", "通信保障分队", "抢修保障分队"],
        enemy_roles=["远程火力节点", "海空侦察节点", "电子干扰节点", "小规模袭扰分队"],
        positive_actions=[
            "持续侦察补给航路、气象窗口和敌远程火力威胁",
            "以分段护航和防空掩护维持补给走廊韧性",
            "电子对抗压缩敌侦察回传和目标指示链路",
            "设置备用航线、临时补给点和抢修节点",
            "通过主备通信链同步补给节奏与防护状态",
        ],
        negative_actions=[
            "只规划单一补给航线",
            "忽视敌侦察链导致补给节点持续暴露",
            "没有抢修与备用通信方案",
        ],
        positive_lessons=[
            "岛链补给的核心是 sustain，但 detect 和 c2 决定保障韧性。",
            "备用航线与临时节点能降低单点中断风险。",
            "防护、通信和抢修同步时 overall_effectiveness 更稳定。",
        ],
        negative_lessons=[
            "单航线依赖会使敌远程火力威胁被放大。",
            "忽略电子干扰会让补给节奏失去同步。",
        ],
        tags=["maritime", "sustain", "resupply", "c2", "protection"],
        stage_focus={"detect": 0.72, "disrupt": 0.66, "breach": 0.64, "control": 0.70, "sustain": 0.82},
    ),
]


def _stage_metrics(group: TaskGroup, index: int, positive: bool) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    wave = ((index - 1) % 8) * 0.006
    if positive:
        for stage, base in group.stage_focus.items():
            metrics[f"stage_{stage}"] = round(min(0.90, base + wave), 4)
        mission_success = 0.67 + ((index - 1) % 10) * 0.012
        overall = 0.72 + ((index - 1) % 8) * 0.011
        ler = 1.45 + ((index - 1) % 9) * 0.11
        command = 0.68 + ((index - 1) % 7) * 0.018
    else:
        for stage, base in group.stage_focus.items():
            penalty = 0.14 + ((index - 1) % 5) * 0.018
            metrics[f"stage_{stage}"] = round(max(0.38, base - penalty), 4)
        mission_success = 0.43 + ((index - 1) % 6) * 0.018
        overall = 0.50 + ((index - 1) % 5) * 0.022
        ler = 0.88 + ((index - 1) % 5) * 0.09
        command = 0.46 + ((index - 1) % 5) * 0.025
    metrics.update(
        {
            "mission_success": round(mission_success, 4),
            "ler": round(ler, 4),
            "overall_effectiveness": round(overall, 4),
            "command_resilience": round(command, 4),
        }
    )
    return metrics


def _select_window(items: List[str], index: int, width: int) -> List[str]:
    return [items[(index + offset) % len(items)] for offset in range(width)]


def build_record(group: TaskGroup, index: int, positive: bool) -> Dict[str, object]:
    case_kind = "P" if positive else "N"
    actions = group.positive_actions if positive else group.negative_actions
    lessons = group.positive_lessons if positive else group.negative_lessons
    action_width = 4 if positive else 3
    lesson_width = 3 if positive else 2
    key_actions = _select_window(actions, index - 1, action_width)
    selected_lessons = _select_window(lessons, index - 1, lesson_width)
    if positive:
        selected_lessons.append("本案例仅用于虚拟场景下的方案生成与仿真评估研究。")
    else:
        selected_lessons.append("负样本用于规避失败链路，不作为可直接模仿方案。")

    return {
        "case_id": f"GEN2-{group.code.upper().replace('-', '_')}-{case_kind}{index:03d}",
        "mission_type": group.mission_type,
        "terrain": group.terrain,
        "objective": f"{group.objective}（变体{index:02d}）",
        "friendly_roles": group.friendly_roles,
        "enemy_roles": group.enemy_roles,
        "key_actions": key_actions,
        "lessons": selected_lessons,
        "outcome": "positive" if positive else "negative",
        "tags": group.tags + [f"group:{group.code}", "generalization-v2", "virtual-research-only"],
        "metrics": _stage_metrics(group, index, positive),
        "update_count": 1,
        "last_updated": "2026-05-08T00:00:00",
        "source_scenarios": [group.code],
    }


def build_records() -> Iterable[Dict[str, object]]:
    for group in TASK_GROUPS:
        for index in range(1, 41):
            yield build_record(group, index, positive=True)
        for index in range(1, 11):
            yield build_record(group, index, positive=False)


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    records = list(build_records())
    with OUTPUT_PATH.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(OUTPUT_PATH), "record_count": len(records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
