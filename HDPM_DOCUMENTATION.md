````markdown
# HDPM: Hierarchical Dual-Pathway Memory for ProtAgent

## Table of Contents
1.  [Introduction (English)](#introduction-english)
2.  [Core Concepts (English)](#core-concepts-english)
3.  [How to Use (English)](#how-to-use-english)
    *   [Initialization](#initialization-english)
    *   [Post-Task: Updating Memory](#post-task-updating-memory-english)
    *   [Pre-Task: Generating Prompts](#pre-task-generating-prompts-english)
4.  [API Reference (English)](#api-reference-english)
    *   [Main Classes](#main-classes-english)
    *   [Key Methods](#key-methods-english)
5.  [Example Code (English)](#example-code-english)
6.  [Performance Evaluation Suggestions (English)](#performance-evaluation-suggestions-english)
7.  [--- 中文文档 ---](#---中文文档---)
8.  [简介 (中文)](#简介-中文)
9.  [核心概念 (中文)](#核心概念-中文)
10. [如何使用 (中文)](#如何使用-中文)
    *   [初始化 (中文)](#初始化-中文)
    *   [任务后：更新记忆 (中文)](#任务后更新记忆-中文)
    *   [任务前：生成提示 (中文)](#任务前生成提示-中文)
11. [API 参考 (中文)](#api-参考-中文)
    *   [主要类 (中文)](#主要类-中文)
    *   [关键方法 (中文)](#关键方法-中文)
12. [代码示例 (中文)](#代码示例-中文)
13. [性能评估建议 (中文)](#性能评估建议-中文)

---

## Introduction (English)

The Hierarchical Dual-Pathway Memory (HDPM) framework is designed to enhance multi-agent systems like ProtAgent by enabling them to learn from past experiences. It achieves this by maintaining two distinct, three-layered memory stores: one for successful tasks (Positive Memory) and one for failed tasks (Negative Memory). After each task, a `ReflectAgent` processes the task trajectory, extracts knowledge, and stores it hierarchically. When a new task arrives, HDPM retrieves relevant positive and negative experiences to construct role-specific prompts (or "scaffolds") for each agent in the ProtAgent team (Planner, Assistant, Critic), facilitating contextual learning and improved decision-making.

**Key Features:**
*   **Dual Pathways:** Separates successful and failed experiences for targeted learning.
*   **Hierarchical Structure:** Organizes knowledge into Atomic Evidence Cards (details), Policy Pathways (workflows), and Insights (high-level principles/traps).
*   **Reflection Agent:** Automates post-task knowledge extraction and storage.
*   **Role-Specific Context:** Provides tailored guidance to Planner, Assistant, and Critic agents based on retrieved memories.

## Core Concepts (English)

*   **`AtomicEvidenceCard`**: The most granular piece of information, representing a single action, observation, or feedback from an agent within a task. Contains content, the type of agent that generated it, and a timestamp.
*   **`PolicyPathway`**: An ordered sequence of `AtomicEvidenceCard` IDs that represents the complete workflow or strategy followed during a specific task. It's linked to a high-level `Insight`.
*   **`Insight`**: High-level, distilled knowledge derived from a `PolicyPathway`. For positive experiences, it's a "success principle"; for negative ones, a "failure trap."
*   **`MemoryStore`**: A container for one type of experience (either positive or negative). It holds dictionaries of `Insight`s, `PolicyPathway`s, and `AtomicEvidenceCard`s.
*   **`HDPM`**: The main class, holding one `MemoryStore` for positive experiences and another for negative experiences.
*   **`ReflectAgent`**: A component responsible for processing a completed task's `trajectory` and its `result` (success/failure) to update the `HDPM` instance.
*   **Trajectory (`ProtAgentTrajectory`)**: A list of dictionaries representing the sequence of steps/events that occurred during a task. Each event typically includes `agent_type`, `action_content`, and `timestamp`.

## How to Use (English)

### Initialization (English)
First, create an instance of the HDPM system and the ReflectAgent.

```python
from hdpm import HDPM, ReflectAgent, ProtAgentTrajectory

# Initialize the main HDPM system
hdpm_system = HDPM()

# Initialize the ReflectAgent (used for post-task processing)
# "simple_keywords" is a placeholder for LLM distillation logic
reflect_agent = ReflectAgent(llm_simulation_mode="simple_keywords")
```

### Post-Task: Updating Memory (English)
After a ProtAgent task is completed, use the `ReflectAgent` to process its trajectory and outcome, then update the HDPM.

```python
# Assume 'task_trajectory' is a list of dicts from ProtAgent
# e.g., task_trajectory: ProtAgentTrajectory = [
#    {'agent_type': 'Planner', 'action_content': 'Initial plan...', 'timestamp': '...'},
#    {'agent_type': 'Assistant', 'action_content': 'Executed tool X...', 'timestamp': '...'},
#    {'agent_type': 'Critic', 'action_content': 'Evaluation result...', 'timestamp': '...'}
# ]
# Assume 'task_result' is 1 for success, -1 for failure
# task_result = 1 # or -1

# Example successful trajectory
sample_successful_trajectory: ProtAgentTrajectory = [
    {'agent_type': 'Planner', 'action_content': 'Decided to use ProteinFoldTool.', 'timestamp': '2023-01-15T10:00:00Z'},
    {'agent_type': 'Assistant', 'action_content': 'Used ProteinFoldTool successfully.', 'timestamp': '2023-01-15T10:05:00Z'},
    {'agent_type': 'Critic', 'action_content': 'Fold quality is excellent. Success.', 'timestamp': '2023-01-15T10:10:00Z'},
]
task_result = 1 # Success

# Update HDPM with this experience
insight, pathway, cards = reflect_agent.update_memory(
    trajectory=sample_successful_trajectory,
    result=task_result,
    hdpm_instance=hdpm_system
)

if insight:
    print(f"Stored new insight: {insight.content}")
```

### Pre-Task: Generating Prompts (English)
When a new task (query) is received, use the `HDPM` instance to generate role-specific prompts.

```python
new_query = "Design a stable variant of protein AlphaFold"

# Generate prompts for Planner, Assistant, and Critic
# top_k_retrieval specifies how many relevant items to fetch from each memory store
role_prompts = hdpm_system.generate_role_specific_prompts(
    query=new_query,
    top_k_retrieval=1 # Retrieve the most relevant positive and negative experience
)

planner_prompt = role_prompts["Planner"]
assistant_prompt = role_prompts["Assistant"]
critic_prompt = role_prompts["Critic"]

print("\n--- Planner Prompt ---")
print(planner_prompt)
# This prompt can now be fed to the Planner agent.
# Similarly for Assistant and Critic.
```

## API Reference (English)

### Main Classes (English)
*   **`hdpm.HDPM()`**:
    *   Manages `positive_memory: MemoryStore` and `negative_memory: MemoryStore`.
    *   Provides methods for retrieval and prompt generation.
*   **`hdpm.ReflectAgent(llm_simulation_mode: str = "simple_keywords")`**:
    *   Processes task outcomes to update an `HDPM` instance.
*   **`hdpm.MemoryStore()`**:
    *   Stores `insights`, `pathways`, and `evidence` (AtomicEvidenceCards).
*   **`hdpm.Insight(content: str, source_pathway_id: str, insight_id: Optional[str] = None)`**:
    *   Represents a distilled piece of knowledge.
*   **`hdpm.PolicyPathway(evidence_card_ids: List[str], pathway_id: Optional[str] = None, linked_insight_id: Optional[str] = None)`**:
    *   Represents an ordered workflow via IDs of evidence cards.
*   **`hdpm.AtomicEvidenceCard(content: str, agent_type: str, timestamp: Optional[datetime.datetime] = None, id: Optional[str] = None)`**:
    *   Represents a single event/action in a trajectory.

### Key Methods (English)
*   **`ReflectAgent.update_memory(trajectory: ProtAgentTrajectory, result: int, hdpm_instance: HDPM) -> Tuple[Optional[Insight], Optional[PolicyPathway], List[AtomicEvidenceCard]]`**:
    *   The primary method for storing new experiences. It atomizes the trajectory, serializes it into a pathway, distills an insight, and stores all components in the appropriate (positive/negative) memory store of the `hdpm_instance`.
*   **`HDPM.generate_role_specific_prompts(query: str, top_k_retrieval: int = 1) -> Dict[str, str]`**:
    *   Retrieves relevant past experiences (both positive and negative) based on the `query` and generates tailored textual prompts for the "Planner", "Assistant", and "Critic" agents.
*   **`HDPM.global_co_retrieval(query: str, top_k: int = 1) -> Tuple[List[Insight], List[PolicyPathway], List[Insight], List[PolicyPathway]]`**:
    *   Performs retrieval from both positive and negative stores. (Primarily for internal use by `generate_role_specific_prompts` but can be used directly).
*   Serialization/Deserialization:
    *   `HDPM.to_dict() -> Dict` and `HDPM.from_dict(data: Dict) -> HDPM`
    *   `MemoryStore.to_dict() -> Dict` and `MemoryStore.from_dict(data: Dict) -> MemoryStore`
    *   (And similar methods for `Insight`, `PolicyPathway`, `AtomicEvidenceCard`) for saving and loading the memory state.

## Example Code (English)

This comprehensive example demonstrates initialization, storing both a positive and a negative experience, and then retrieving prompts for a new query.

```python
import datetime
from hdpm import HDPM, ReflectAgent, AtomicEvidenceCard, PolicyPathway, Insight, ProtAgentTrajectory

# 1. Initialize HDPM and ReflectAgent
hdpm_system = HDPM()
reflect_agent = ReflectAgent(llm_simulation_mode="simple_keywords") # Uses placeholder LLM

# 2. Simulate and store a successful experience
positive_trajectory: ProtAgentTrajectory = [
    {'agent_type': 'Planner', 'action_content': 'Plan: Use SuperFold for protein Alpha.', 'timestamp': '2023-01-01T10:00:00Z'},
    {'agent_type': 'Assistant', 'action_content': 'Execute: SuperFold params {a=1, b=2} -> good_fold.pdb', 'timestamp': '2023-01-01T10:05:00Z'},
    {'agent_type': 'Critic', 'action_content': 'Critique: good_fold.pdb is high quality. Success.', 'timestamp': '2023-01-01T10:10:00Z'}
]
reflect_agent.update_memory(positive_trajectory, 1, hdpm_system)

# 3. Simulate and store a failed experience
negative_trajectory: ProtAgentTrajectory = [
    {'agent_type': 'Planner', 'action_content': 'Plan: Use QuickDock for ligand Beta.', 'timestamp': '2023-01-02T11:00:00Z'},
    {'agent_type': 'Assistant', 'action_content': 'Execute: QuickDock default params -> steric_clash_error', 'timestamp': '2023-01-02T11:05:00Z'},
    {'agent_type': 'Critic', 'action_content': 'Critique: Steric clashes indicate failure for this ligand type.', 'timestamp': '2023-01-02T11:10:00Z'}
]
reflect_agent.update_memory(negative_trajectory, -1, hdpm_system)

print("--- HDPM State after storing experiences ---")
print(hdpm_system)
# print(f"Positive Insights: {list(hdpm_system.positive_memory.insights.values())}")
# print(f"Negative Insights: {list(hdpm_system.negative_memory.insights.values())}")

# 4. New query: Generate role-specific prompts
new_query = "Design a high-quality fold for protein Alpha variant"
role_prompts = hdpm_system.generate_role_specific_prompts(query=new_query, top_k_retrieval=1)

print(f"\n--- Prompts for Query: '{new_query}' ---")
print("\n[Planner Prompt]")
print(role_prompts["Planner"])

print("\n[Assistant Prompt]")
print(role_prompts["Assistant"])

print("\n[Critic Prompt]")
print(role_prompts["Critic"])

# 5. (Optional) Save and Load HDPM state
# hdpm_dict = hdpm_system.to_dict()
# # import json; with open("hdpm_state.json", "w") as f: json.dump(hdpm_dict, f, indent=2)
# # loaded_hdpm = HDPM.from_dict(hdpm_dict) # Or load from file
# # print("\n--- Reloaded HDPM State ---")
# # print(loaded_hdpm)
```

## Performance Evaluation Suggestions (English)

To demonstrate the performance improvement HDPM brings to ProtAgent, consider the following:

1.  **Baseline Comparison:**
    *   **ProtAgent-Stateless:** The original ProtAgent without any memory.
    *   **ProtAgent-HDPM:** ProtAgent integrated with the HDPM framework.
    *   **(Optional) ProtAgent-RAG (Global):** ProtAgent with a simpler memory system where all experiences (positive/negative) are stored in a single undifferentiated vector store, and generic context is retrieved.

2.  **Task Benchmark:**
    *   Design a suite of protein design tasks. Include:
        *   Tasks where solutions are similar to previously successful stored experiences.
        *   Tasks that contain "traps" or failure modes similar to previously failed stored experiences.
        *   Novel tasks to assess generalization.

3.  **Metrics:**
    *   **First-Attempt Success Rate:** Percentage of tasks solved correctly on the first try without needing iterative refinement by the user or system. This is key for HDPM's goal of learning from past mistakes/successes.
    *   **Task Completion Time/Efficiency:**
        *   Total interaction rounds or steps taken by the agent team.
        *   Number of planning retries or significant plan revisions.
        *   Computational resources consumed (if measurable).
    *   **Quality of Solution:** (Domain-specific) e.g., stability scores, binding affinity, structural correctness of designed proteins.
    *   **Role-Specific Improvements (Ablation or Targeted Analysis):**
        *   *Assistant's Tool Accuracy:* Reduction in erroneous tool calls or parameter settings if specific negative examples about tool misuse are in memory.
        *   *Critic's Defect Detection Rate:* Improved ability to flag known issues if similar defect patterns are in negative memory.
        *   *Planner's Strategy Quality:* Qualitative assessment or reduction in steps if planner gets good strategic hints.

4.  **Ablation Studies (as proposed in your document):**
    *   **HDPM w/o Hierarchy:** Flatten memory to only atomic evidence to test the value of pathways and insights.
    *   **HDPM w/o Negative Pathway:** Only use the positive memory store to quantify the contribution of learning from failures.
    *   **HDPM w/o Role-specific Prompts:** Provide all agents with the same generic context (e.g., only what the Planner gets) to test the value of differentiated scaffolding.

5.  **Data Collection:**
    *   Log all agent interactions, retrieved memories, and generated prompts for qualitative analysis.
    *   Track task outcomes and the metrics above systematically.

By comparing these metrics across different configurations, you can quantify the benefits of the full HDPM framework and its individual components.

---
## --- 中文文档 ---
---

## 简介 (中文)

分层式双轨记忆（Hierarchical Dual-Pathway Memory, HDPM）框架旨在通过使 ProtAgent 等多智能体系统能够从过去的经验中学习来增强其能力。它通过维护两个独立的三层记忆库来实现这一目标：一个用于成功任务（正向记忆库），一个用于失败任务（负向记忆库）。每个任务完成后，一个 `ReflectAgent`（反思智能体）会处理任务轨迹，提取知识，并将其分层存储。当新任务到达时，HDPM 会检索相关的正反经验，为 ProtAgent团队中的每个智能体（Planner规划者, Assistant助手, Critic评论家）构建角色定制化的提示（或称“脚手架”），从而促进上下文学习并改进决策。

**主要特性:**
*   **双轨记忆:** 区分成功和失败经验，进行针对性学习。
*   **分层结构:** 将知识组织成原子证据卡片（细节）、策略路径（工作流）和洞见（高阶原则/陷阱）。
*   **反思智能体:** 自动化任务后知识提取和存储。
*   **角色化上下文:** 基于检索到的记忆，为规划者、助手和评论家智能体提供量身定制的指导。

## 核心概念 (中文)

*   **`AtomicEvidenceCard` (原子证据卡片)**: 最细粒度的信息单元，代表任务中单个智能体的行动、观察或反馈。包含内容、生成者智能体类型和时间戳。
*   **`PolicyPathway` (策略路径)**: `AtomicEvidenceCard` ID 的有序序列，代表在特定任务期间遵循的完整工作流或策略。它链接到一个高阶的 `Insight`。
*   **`Insight` (洞见)**: 从 `PolicyPathway` 中提炼出来的高阶、抽象知识。对于成功经验，它是“成功原则”；对于失败经验，则是“失败陷阱”。
*   **`MemoryStore` (记忆库)**: 一种经验类型（正向或负向）的容器。它包含 `Insight`、`PolicyPathway` 和 `AtomicEvidenceCard` 的字典。
*   **`HDPM` (分层式双轨记忆系统)**: 主类，包含一个用于正向经验的 `MemoryStore` 和一个用于负向经验的 `MemoryStore`。
*   **`ReflectAgent` (反思智能体)**: 负责处理已完成任务的 `trajectory`（轨迹）及其 `result`（结果，成功/失败），以更新 `HDPM` 实例。
*   **Trajectory (`ProtAgentTrajectory`, 任务轨迹)**: 一个字典列表，代表任务期间发生的步骤/事件序列。每个事件通常包括 `agent_type`（智能体类型）、`action_content`（行动内容）和 `timestamp`（时间戳）。

## 如何使用 (中文)

### 初始化 (中文)
首先，创建 HDPM 系统和 ReflectAgent 的实例。

```python
from hdpm import HDPM, ReflectAgent, ProtAgentTrajectory

# 初始化 HDPM 主系统
hdpm_system = HDPM()

# 初始化 ReflectAgent (用于任务后处理)
# "simple_keywords" 是 LLM 蒸馏逻辑的占位符
reflect_agent = ReflectAgent(llm_simulation_mode="simple_keywords")
```

### 任务后：更新记忆 (中文)
ProtAgent 任务完成后，使用 `ReflectAgent` 处理其轨迹和结果，然后更新 HDPM。

```python
# 假设 'task_trajectory' 是来自 ProtAgent 的字典列表
# 例如: task_trajectory: ProtAgentTrajectory = [
#    {'agent_type': 'Planner', 'action_content': '初始计划...', 'timestamp': '...'},
#    {'agent_type': 'Assistant', 'action_content': '执行工具 X...', 'timestamp': '...'},
#    {'agent_type': 'Critic', 'action_content': '评估结果...', 'timestamp': '...'}
# ]
# 假设 'task_result' 成功为 1, 失败为 -1
# task_result = 1 # 或 -1

# 示例成功轨迹
sample_successful_trajectory: ProtAgentTrajectory = [
    {'agent_type': 'Planner', 'action_content': '决定使用 ProteinFoldTool。', 'timestamp': '2023-01-15T10:00:00Z'},
    {'agent_type': 'Assistant', 'action_content': '成功使用 ProteinFoldTool。', 'timestamp': '2023-01-15T10:05:00Z'},
    {'agent_type': 'Critic', 'action_content': '折叠质量极好。成功。', 'timestamp': '2023-01-15T10:10:00Z'},
]
task_result = 1 # 成功

# 用此经验更新 HDPM
insight, pathway, cards = reflect_agent.update_memory(
    trajectory=sample_successful_trajectory,
    result=task_result,
    hdpm_instance=hdpm_system
)

if insight:
    print(f"已存储新洞见: {insight.content}")
```

### 任务前：生成提示 (中文)
当收到新任务（查询）时，使用 `HDPM` 实例生成角色定制化的提示。

```python
new_query = "设计蛋白质 AlphaFold 的稳定变体"

# 为 Planner, Assistant, Critic 生成提示
# top_k_retrieval 指定从每个记忆库中获取多少相关项
role_prompts = hdpm_system.generate_role_specific_prompts(
    query=new_query,
    top_k_retrieval=1 # 检索最相关的一个正向和一个负向经验
)

planner_prompt = role_prompts["Planner"]
assistant_prompt = role_prompts["Assistant"]
critic_prompt = role_prompts["Critic"]

print("\n--- 规划器提示 ---")
print(planner_prompt)
# 此提示现在可以提供给规划器智能体。
# 助手和评论家也类似。
```

## API 参考 (中文)

### 主要类 (中文)
*   **`hdpm.HDPM()`**:
    *   管理 `positive_memory: MemoryStore` (正向记忆库) 和 `negative_memory: MemoryStore` (负向记忆库)。
    *   提供检索和提示生成的方法。
*   **`hdpm.ReflectAgent(llm_simulation_mode: str = "simple_keywords")`**:
    *   处理任务结果以更新 `HDPM` 实例。
*   **`hdpm.MemoryStore()`**:
    *   存储 `insights` (洞见), `pathways` (路径), 和 `evidence` (原子证据卡片)。
*   **`hdpm.Insight(content: str, source_pathway_id: str, insight_id: Optional[str] = None)`**:
    *   代表一条提炼出的知识。
*   **`hdpm.PolicyPathway(evidence_card_ids: List[str], pathway_id: Optional[str] = None, linked_insight_id: Optional[str] = None)`**:
    *   通过证据卡片的 ID 列表代表一个有序的工作流。
*   **`hdpm.AtomicEvidenceCard(content: str, agent_type: str, timestamp: Optional[datetime.datetime] = None, id: Optional[str] = None)`**:
    *   代表轨迹中的单个事件/行动。

### 关键方法 (中文)
*   **`ReflectAgent.update_memory(trajectory: ProtAgentTrajectory, result: int, hdpm_instance: HDPM) -> Tuple[Optional[Insight], Optional[PolicyPathway], List[AtomicEvidenceCard]]`**:
    *   存储新经验的主要方法。它将轨迹原子化，序列化为路径，提炼出洞见，并将所有组件存储到 `hdpm_instance` 的相应（正向/负向）记忆库中。
*   **`HDPM.generate_role_specific_prompts(query: str, top_k_retrieval: int = 1) -> Dict[str, str]`**:
    *   根据 `query` 检索相关的过去经验（包括正向和负向），并为 "Planner"、"Assistant" 和 "Critic" 智能体生成定制化的文本提示。
*   **`HDPM.global_co_retrieval(query: str, top_k: int = 1) -> Tuple[List[Insight], List[PolicyPathway], List[Insight], List[PolicyPathway]]`**:
    *   从正向和负向记忆库执行检索。（主要供 `generate_role_specific_prompts` 内部使用，但也可直接使用）。
*   序列化/反序列化:
    *   `HDPM.to_dict() -> Dict` 和 `HDPM.from_dict(data: Dict) -> HDPM`
    *   `MemoryStore.to_dict() -> Dict` 和 `MemoryStore.from_dict(data: Dict) -> MemoryStore`
    *   （以及 `Insight`, `PolicyPathway`, `AtomicEvidenceCard` 的类似方法）用于保存和加载记忆状态。

## 代码示例 (中文)

这个综合示例演示了初始化、存储一个正向和一个负向经验，然后为一个新查询检索提示。

```python
import datetime
from hdpm import HDPM, ReflectAgent, AtomicEvidenceCard, PolicyPathway, Insight, ProtAgentTrajectory

# 1. 初始化 HDPM 和 ReflectAgent
hdpm_system = HDPM()
reflect_agent = ReflectAgent(llm_simulation_mode="simple_keywords") # 使用占位符 LLM

# 2. 模拟并存储一次成功经验
positive_trajectory: ProtAgentTrajectory = [
    {'agent_type': 'Planner', 'action_content': '计划: 使用 SuperFold 处理蛋白质 Alpha。', 'timestamp': '2023-01-01T10:00:00Z'},
    {'agent_type': 'Assistant', 'action_content': '执行: SuperFold 参数 {a=1, b=2} -> good_fold.pdb', 'timestamp': '2023-01-01T10:05:00Z'},
    {'agent_type': 'Critic', 'action_content': '评审: good_fold.pdb 质量很高。成功。', 'timestamp': '2023-01-01T10:10:00Z'}
]
reflect_agent.update_memory(positive_trajectory, 1, hdpm_system)

# 3. 模拟并存储一次失败经验
negative_trajectory: ProtAgentTrajectory = [
    {'agent_type': 'Planner', 'action_content': '计划: 使用 QuickDock 处理配体 Beta。', 'timestamp': '2023-01-02T11:00:00Z'},
    {'agent_type': 'Assistant', 'action_content': '执行: QuickDock 默认参数 -> steric_clash_error (空间位阻冲突错误)', 'timestamp': '2023-01-02T11:05:00Z'},
    {'agent_type': 'Critic', 'action_content': '评审: 空间位阻冲突表明此类配体处理失败。', 'timestamp': '2023-01-02T11:10:00Z'}
]
reflect_agent.update_memory(negative_trajectory, -1, hdpm_system)

print("--- 存储经验后的 HDPM 状态 ---")
print(hdpm_system)
# print(f"正向洞见: {list(hdpm_system.positive_memory.insights.values())}")
# print(f"负向洞见: {list(hdpm_system.negative_memory.insights.values())}")

# 4. 新查询: 生成角色定制化提示
new_query = "为蛋白质 Alpha 变体设计高质量的折叠结构"
role_prompts = hdpm_system.generate_role_specific_prompts(query=new_query, top_k_retrieval=1)

print(f"\n--- 查询 '{new_query}' 的提示 ---")
print("\n[规划器提示]")
print(role_prompts["Planner"])

print("\n[助手提示]")
print(role_prompts["Assistant"])

print("\n[评论家提示]")
print(role_prompts["Critic"])

# 5. (可选) 保存和加载 HDPM 状态
# hdpm_dict = hdpm_system.to_dict()
# # import json; with open("hdpm_state.json", "w", encoding="utf-8") as f: json.dump(hdpm_dict, f, indent=2, ensure_ascii=False)
# # loaded_hdpm = HDPM.from_dict(hdpm_dict) # 或从文件加载
# # print("\n--- 重新加载的 HDPM 状态 ---")
# # print(loaded_hdpm)
```

## 性能评估建议 (中文)

为了证明 HDPM 给 ProtAgent 带来的性能提升，可以考虑以下方面：

1.  **基线比较:**
    *   **ProtAgent-Stateless:** 原始的、无任何记忆能力的 ProtAgent。
    *   **ProtAgent-HDPM:** 集成了 HDPM 框架的 ProtAgent。
    *   **(可选) ProtAgent-RAG (Global):** ProtAgent 使用一个更简单的记忆系统，其中所有经验（无论成败）都存储在单个未区分的向量库中，并检索通用上下文。

2.  **任务基准:**
    *   设计一套蛋白质设计任务。包括：
        *   解决方案与先前存储的成功经验相似的任务。
        *   包含与先前存储的失败经验相似的“陷阱”或失败模式的任务。
        *   全新的任务以评估泛化能力。

3.  **评估指标:**
    *   **首次尝试成功率:** 无需用户或系统迭代调整，一次性正确解决任务的百分比。这是 HDPM 从过去的错误/成功中学习目标的关键。
    *   **任务完成时间/效率:**
        *   智能体团队采取的总交互轮次或步骤数。
        *   规划重试或重大计划修订的次数。
        *   消耗的计算资源（如果可测量）。
    *   **解决方案质量:** (领域特定) 例如，蛋白质设计的稳定性得分、结合亲和力、结构正确性。
    *   **角色特定改进 (消融研究或针对性分析):**
        *   *助手工具准确性:* 如果记忆中有关于工具误用的特定负面案例，则错误工具调用或参数设置的减少情况。
        *   *评论家缺陷检测率:* 如果负面记忆中有类似的缺陷模式，则改进识别已知问题的能力。
        *   *规划器策略质量:* 如果规划器获得良好的战略提示，则进行定性评估或减少步骤。

4.  **消融研究 (如您的提案中所述):**
    *   **HDPM w/o Hierarchy (无分层结构):** 将记忆扁平化，仅保留原子证据，以测试路径和洞见的价值。
    *   **HDPM w/o Negative Pathway (无负向路径):** 仅使用正向记忆库，以量化从失败中学习部分的贡献。
    *   **HDPM w/o Role-specific Prompts (无角色化提示):** 为所有智能体提供相同的通用上下文（例如，仅规划器获得的内容），以测试差异化脚手架的价值。

5.  **数据收集:**
    *   记录所有智能体交互、检索到的记忆和生成的提示，以进行定性分析。
    *   系统地跟踪任务结果和上述指标。

通过比较这些指标在不同配置下的表现，您可以量化完整 HDPM 框架及其各个组件所带来的益处。
````
