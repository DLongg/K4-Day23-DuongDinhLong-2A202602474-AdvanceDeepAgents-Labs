# LLM Agents and Tool Use: A Comprehensive Technical Survey

## TL;DR
- **Tool Integration Paradigms:** Tool integration in LLMs has evolved from prompt-based interleaved reasoning loops like ReAct [1] and self-supervised API loss filtering in Toolformer [2] to neural API retrieval architectures over 16,000+ RESTful tools in ToolLLM [3], retrieval-aware fine-tuned API generators in Gorilla [4], and frozen-LLM token embedding extensions in ToolkenGPT [5].
- **Planning and Multi-Step Execution:** Agents resolve complex long-horizon tasks using tree-search planners like Tree of Thoughts [6] and Language Agent Tree Search [7], parallel dependency-graph compilers achieving 3.7x latency speedups [8], operating-system-inspired context paging in MemGPT [9], and episodic self-reflection in Reflexion [10] which achieves 91.0% pass@1 on HumanEval.
- **Multi-Agent Collaboration Frameworks:** Frameworks coordinate multi-agent tool execution via Standard Operating Procedures in MetaGPT [11], actor-based conversational loops in AutoGen [12], dual-role chat chains in ChatDev [13], and dynamic multi-stage recruitment in AgentVerse [14], governed by emerging standards such as the Model Context Protocol (MCP) and Agent-to-Agent (A2A) protocols [15][16].
- **Real-World Execution Capabilities:** Evaluation benchmarks reveal a stark performance drop when transitioning from static QA to interactive environments—agents achieve only 1.74%–1.96% success on SWE-bench [17], 14.41% on WebArena [18], and 15.0% on GAIA [19], limited by trajectory drift, invalid tool syntax, and execution error recovery failures [20][21].
- **Security & Authorization Guardrails:** Indirect prompt injection (IPI) poses severe risks to tool-integrated agents, achieving up to 45% attack success rates on InjecAgent [22] and 84.3% under mixed strategies on Agent Security Bench [23]. Formally enforcing dynamic session authorization and intent binding via Agentic Principal Chains (APC) eliminates data exfiltration on benchmark attacks [24].

## Background
Large Language Models (LLMs) have evolved from static next-token prediction engines into interactive autonomous agents capable of perceiving environment states, formulating multi-step plans, invoking external tools, and evaluating execution outcomes [1][15]. Early tool-augmented language models demonstrated that augmenting parametric memory with symbolic tools—such as web search engines, calculators, and database APIs—prevents factual hallucinations, extends mathematical precision, and connects models to real-time information systems [2][20].

The formal paradigm for agentic tool use relies on closed-loop feedback: the LLM generates an explicit tool call (including tool name and JSON-formatted parameters), an external execution runtime runs the tool, and the environment returns execution results (stdout, API JSON responses, or status errors) back into the prompt context window [1][4]. As tool integration expanded from single-function invocations to extensive API catalogs containing thousands of endpoints [3], research shifted toward architectural optimizations, deliberate planning algorithms [6][7], multi-agent role specialization [11][12], execution benchmarking [17][18], and robust authorization security [22][24].

## Architectures and Function-Calling Mechanisms
The interface between language models and external APIs requires effective representations for tool discovery, parameter generation, and call execution. Tool-calling architectures are broadly categorized into prompt-based interleaved loops, self-supervised API prediction, neural tool retrieval over large catalogs, fine-tuned function generators, and token embedding extensions.

```
+-----------------------------------------------------------------------------------+
|                           LLM Tool Invocation Paradigms                           |
+------------------------------------+----------------------------------------------+
| Paradigm                           | Key Architectural Mechanism                  |
+------------------------------------+----------------------------------------------+
| Interleaved Loop (ReAct [1])       | Prompt-based Thought -> Action -> Observation|
| Self-Supervised (Toolformer [2])   | Perplexity loss filtering on candidate APIs  |
| Neural Retrieval (ToolLLM [5])     | Dense retriever over 16,000+ RapidAPI tools  |
| Fine-Tuned (Gorilla [3])           | Retrieval-aware tuning on APIBench catalog   |
| Token Embedding (ToolkenGPT [4])   | Dedicated vocabulary 'toolkens' for APIs     |
+------------------------------------+----------------------------------------------+
```

### Interleaved Reasoning and Self-Supervised Tool Discovery
ReAct [1] established the foundational paradigm of interleaving reasoning traces ("Thought") and environment actions ("Action") in a continuous execution loop. By prompting the model to explicitly state its rationale prior to outputting a tool command, ReAct enables models to track progress, adapt to unexpected observation outputs, and update execution sub-goals. On the ALFWorld interactive decision-making benchmark, ReAct achieved a 71% success rate, outperforming imitation learning baselines (37%) by 34% [1]. On WebShop, ReAct achieved a 40% success rate with 1-2 in-context demonstrations [1].

Toolformer [2] introduced self-supervised tool learning, demonstrating that a 6.7B parameter GPT-J model can self-annotate API calls across unlabeled text corpora (CCNet). Toolformer samples candidate API calls (Calculator, Q&A, Wikipedia search, Calendar) and retains only those calls that demonstrably reduce cross-entropy loss on subsequent tokens. In zero-shot evaluations across LAMA and MathQA, Toolformer performed competitively with a 175B parameter GPT-3 baseline without requiring manual annotation [2].

### Scaled API Retrieval and Parameter Generation
Scaling tool use to real-world environments requires navigating thousands of candidate APIs. Gorilla [4] addressed API call hallucination by fine-tuning LLaMA-7B on APIBench—a dataset of 1,645 API calls across HuggingFace (928), TorchHub (94), and TensorHub (623). Gorilla incorporates a document retriever into its fine-tuning and evaluation loops, allowing the model to generate correct API calls even as API documentation changes dynamically [4].

ToolLLM [3] scaled tool integration to 16,464 RESTful APIs from RapidAPI across 49 domains. ToolLLM pairs a dense neural API retriever with ToolLLaMA (a fine-tuned LLaMA-7B). To handle complex multi-step execution paths, ToolLLM introduces Depth-First Search Decision Trees (DFSDT), allowing agents to evaluate multiple API trajectories and recover from invalid calls [3].

To decouple tool expansion from context length limits, ToolkenGPT [5] represents external APIs as specialized token embeddings ("toolkens") appended to the LLM's vocabulary. Keeping the backbone LLM frozen, ToolkenGPT only trains the toolken embeddings. When a toolken is predicted, execution transfers to an argument generation mode, enabling plug-and-play support for over 200+ tools while avoiding full parameter fine-tuning [5].

API-Bank [53 system tools, 264 domain APIs] established a three-level evaluation framework covering single-turn calling (Level 1), joint retrieval and call generation (Level 2), and multi-step planning (Level 3), demonstrating that while models achieve high precision on explicit calls, joint retrieval and multi-step execution remain primary bottlenecks [20].

## Planning, Reasoning, and Multi-Step Execution
Complex agent tasks require multi-step planning, state tracking, self-reflection, and efficient memory management to avoid trajectory collapse over long horizons.

### Search Trees and Parallel Execution Compilers
While sequential CoT and ReAct execution paths can get stuck in dead ends, Tree of Thoughts (ToT) [6] frames problem-solving as tree search. ToT decomposes tasks into intermediate semantic thoughts, evaluates state heuristics ("sure/maybe/impossible"), and applies Breadth-First Search (BFS) or Depth-First Search (DFS) with backtracking. On the Game of 24 benchmark, standard CoT achieved a 4.0% success rate, whereas ToT (BFS, $b=5$) reached 74.0% [6].

Language Agent Tree Search (LATS) [7] unifies reasoning, acting, and planning by combining Monte Carlo Tree Search (MCTS) with ReAct execution steps, LLM value functions, and verbal self-reflections. During MCTS leaf node evaluation, LATS generates verbal diagnostics on failure, saving reflections to episodic memory. LATS achieved 92.7% pass@1 on HumanEval with GPT-4, doubled ReAct accuracy on HotPotQA, and reached 75.9 on WebShop [7].

To address latency constraints in sequential function calling, LLM Compiler [8] introduces a parallel execution architecture. It consists of a Function Calling Planner (generating a task Directed Acyclic Graph [DAG] with variable placeholders like `$1`, `$2`), a Task Fetching Unit, and a Parallel Executor. LLM Compiler achieved up to 3.7x latency speedups, 6.7x cost savings, and ~9% higher accuracy over sequential ReAct on parallel function calling workloads [8].

```
+-----------------------------------------------------------------------------------+
|                     Multi-Step Planning & Reflection Frameworks                   |
+-------------------+---------------------------------------+-----------------------+
| Method            | Primary Control Mechanism             | Key Benchmark Impact  |
+-------------------+---------------------------------------+-----------------------+
| Reflexion [7]     | Verbal feedback in episodic memory    | 91.0% HumanEval pass@1|
| Tree of Thoughts[8| BFS/DFS tree search with state scoring| 74.0% Game of 24      |
| LATS [9]          | MCTS + ReAct + Verbal self-reflection | 92.7% HumanEval pass@1|
| LLM Compiler [11] | Task DAG parallel execution engine    | 3.7x latency speedup  |
| MemGPT [10]       | Hierarchical main/external context RAM| Long-context stability|
+-------------------+---------------------------------------+-----------------------+
```

### Self-Reflection and Hierarchical Memory Management
Reflexion [10] replaces gradient updates with verbal reinforcement learning. An Actor generates action trajectories, an Evaluator scores outputs, and a Self-Reflection model generates diagnostic feedback stored in a bounded context buffer (1–3 past reflections). This verbal memory provides a "semantic gradient" for iterative policy improvement, pushing HumanEval pass@1 from 80% (GPT-4 baseline) to 91.0%, and improving HotPotQA accuracy by 20% [10]. Analysis of search and reflection mechanisms shows that while verbal self-reflection excels when explicit scalar or binary execution feedback is present, high-cardinality state spaces require explicit structural tree search (MCTS/ToT) to prevent local minima traps [25].

MemGPT [9] addresses fixed context window constraints by adapting operating system virtual memory management. MemGPT divides memory into Main Context (in-prompt tokens representing physical RAM, containing core memory and message queues) and External Context (archival storage and historical recall logs). Agents invoke explicit memory editing functions (`core_memory_append`, `archival_memory_search`) and use a `request_heartbeat=true` flag to chain function executions without intermediate user intervention, maintaining consistency across extended interactions [9].

## Multi-Agent Frameworks and Tool Collaboration
When individual agents encounter complex tasks, multi-agent frameworks divide labor among specialized roles, orchestrate communication topologies, and share tool resources.

```
+-----------------------------------------------------------------------------------+
|                            Multi-Agent Framework Topologies                       |
+-------------------+-----------------------------------+---------------------------+
| Framework         | Orchestration Topology            | Key Mechanism             |
+-------------------+-----------------------------------+---------------------------+
| MetaGPT [13]      | Sequential SOP Pipeline           | Async Pub/Sub message pool|
| AutoGen [14]      | Actor-based Conversational Graphs | Dynamic speaker selection |
| ChatDev [15]      | Sequential Chat-Chain (4 phases)  | Bilateral paired sub-chats|
| AgentVerse [16]   | 4-Stage Iterative Loop            | Dynamic role recruitment  |
+-------------------+-----------------------------------+---------------------------+
```

### Organizational Workflows and Conversational Topologies
MetaGPT [11] incorporates Standard Operating Procedures (SOPs) from software engineering into multi-agent workflows. Specialized agents (Product Manager, Architect, Engineer, QA) communicate via an asynchronous publish-subscribe message pool. Agents observe role-relevant artifacts (Product Requirement Documents, API specifications) rather than unstructured chat logs, ensuring formal output schema compliance and reducing token redundancy [11].

AutoGen [12] models multi-agent execution as customizable actor-based conversational graphs. It features `AssistantAgent` (reasoning), `UserProxyAgent` (code/tool execution), and `GroupChatManager` (dynamic conversation routing). AutoGen supports diverse topologies—from static sequential chains to nested dynamic graphs—enabling flexible human-in-the-loop oversight and automated error recovery [12].

ChatDev [13] structures virtual software development through a sequential Chat-Chain covering Designing, Coding, Testing, and Documenting. Within each phase, pairs of agents conduct focused bilateral sub-chats (e.g., Code Reviewer and Programmer), utilizing shared project file repositories and self-reflection loops to correct code defects [13].

AgentVerse [14] introduces dynamic group recruitment across four stages: Recruitment, Decision-Making, Execution, and Evaluation. Instead of static teams, AgentVerse dynamically composes specialized agent configurations based on task demands, utilizing collaborative debate and evaluation feedback loops to outperform single-agent baselines on complex reasoning tasks [14].

### Multi-Agent Interoperability Protocols
Emerging surveys highlight a structural shift toward standardized multi-agent protocols [15][16]. Multi-agent systems utilize communication protocols such as the Contract Net Protocol (CNP) for task allocation/bidding, Agent-to-Agent (A2A), and Agent Network Protocol (ANP) [15]. Furthermore, industry adoption of the Model Context Protocol (MCP) provides a unified abstraction layer connecting heterogeneous agent runtimes with external tool registries, databases, and sandboxed execution environments [16].

## Benchmarks, Evaluation Frameworks, and Security
Evaluating and securing tool-using LLM agents requires interactive benchmark environments and rigorous authorization guardrails.

### Benchmark Capabilities and Environment Performance Gaps
Static LLM benchmarks fail to measure dynamic environment execution, parameter validation, and multi-turn error recovery. Recent benchmarks establish realistic interactive evaluation environments:

```
+-----------------------------------------------------------------------------------+
|                        Interactive Agent Evaluation Benchmarks                    |
+------------------+---------------------------------------+------------------------+
| Benchmark        | Environment Domain & Scope            | SOTA Agent Performance |
+------------------+---------------------------------------+------------------------+
| SWE-bench [19]   | 2,942 real GitHub software issues     | 1.74% (GPT-4) [19]     |
| WebArena [20]    | 812 web tasks (e-commerce, GitLab)    | 14.41% (GPT-4 ReAct)   |
| GAIA [21]        | 466 multimodal general assistant tasks| 15.0% (GPT-4 + tools)  |
| AgentBench [22]  | 8 domains (OS, SQL, Web, Games, KG)   | 4.10 / 5 (GPT-4 API)   |
+------------------+---------------------------------------+------------------------+
```

- **SWE-bench [17]:** Evaluates agents on 2,942 real-world software engineering issues from GitHub Python repositories. Agents must navigate codebases, edit multiple files, and pass unit tests. Initial evaluations revealed severe performance limitations, with Claude 2 solving 1.96% and GPT-4 solving 1.74% of issues [17].
- **WebArena [18]:** Evaluates end-to-end task execution across e-commerce, social forum, GitLab, and CMS web applications. The best GPT-4 ReAct agent achieved a success rate of 14.41%, compared to human performance of 78.24%, suffering from web action execution errors and state recovery failures [18].
- **GAIA [19]:** Evaluates multimodal AI assistants on 466 verifiable real-world questions requiring web browsing, code execution, and document parsing. GPT-4 equipped with tools achieved 15.0% overall accuracy (Level 1: 30%, Level 2: 15%, Level 3: 0%), compared to human accuracy of 92.0% [19].
- **AgentBench [21]:** Assesses 27 LLMs across 8 interactive domains (Operating Systems, SQL databases, Knowledge Graphs, WebShop, ALFWorld). Results show a wide capability gap between proprietary API models (GPT-4 scoring 4.10/5) and open-source models (LLaMA-2-70B scoring 0.88/5), identifying invalid tool call syntax and instruction drift as primary failure modes [21].

### Security Vulnerabilities and Authorization Guardrails
Integrating LLMs with external tools introduces attack vectors, particularly indirect prompt injection (IPI) through untrusted web content or file inputs [22][23].

```
+-----------------------------------------------------------------------------------+
|                        Agent Security Threats & Guardrails                        |
+-------------------+------------------------------------+--------------------------+
| Threat / Framework| Mechanism / Objective              | Attack Rate / Defense    |
+-------------------+------------------------------------+--------------------------+
| InjecAgent [23]   | IPI via untrusted tool outputs     | 24% - 45% ASR (GPT-4)    |
| ASB [24]          | Mixed attacks (IPI, memory poison) | Up to 84.30% ASR         |
| APC Guardrail [25]| Session auth & intent binding      | Exfiltration dropped to 0%|
+-------------------+------------------------------------+--------------------------+
```

InjecAgent [22] benchmarked indirect prompt injection across 1,054 test cases covering 17 user tools and 62 attacker tools. Attacks targeted User Harm (unauthorized transactions) and Data Exfiltration (stealing private files/emails). ReAct-prompted GPT-4 suffered a 24.0% attack success rate (ASR), rising above 45% when attackers applied prompt reinforcement techniques [22].

Agent Security Bench (ASB) [23] formalized attacks across 10 domains, 400+ tools, and 13 LLM backbones. Models reached average attack success rates up to 84.30% under mixed attack strategies (IPI, memory poisoning, privilege escalation). ASB demonstrated that standard prompt guards fail to prevent unauthorized tool execution, requiring system-level authorization runtime checks [23].

Bounded Agents [24] introduced the Agentic Principal Chain (APC), a delegation authorization architecture enforcing dynamic session budget tracking, intent binding, and composition closure. Evaluated against 3,154 attack cases across InjecAgent, AgentDojo, and ASB, APC reduced data exfiltration from 75–100% down to 0% and lowered unauthorized state manipulation from 90.5% to 12.1% with only 0.24 ms authorization overhead at the 99th percentile [24].

## Trends and Open Problems
1. **Protocol Standardization:** The agent ecosystem is standardizing interface protocols, with Model Context Protocol (MCP) becoming the standard for tool interfaces and Agent-to-Agent (A2A) protocols enabling cross-framework multi-agent interoperability [15][16].
2. **Mitigating Trajectory Drift:** Agent execution on complex benchmarks (SWE-bench, WebArena) remains constrained by compounding trajectory errors. Developing hybrid planning architectures that combine MCTS tree search [7], parallel task graph compilation [8], and verified tool execution feedback [10] is critical to bridging the gap to human performance.
3. **Formal Security Enforcement:** Standard prompt safety filters fail against indirect prompt injection in tool contexts. Future agent deployment mandates formal authorization layers—such as Agentic Principal Chains [24] and sandboxed execution environments—that enforce privilege constraints regardless of model context state.
4. **Hierarchical Memory and Context Paging:** As agents engage in multi-turn interactive tasks, combining OS-inspired memory paging (MemGPT [9]) with fine-tuned token embeddings (ToolkenGPT [5]) provides a scalable alternative to expanding physical context window limits.

## References
[1] ReAct: Synergizing Reasoning and Acting in Language Models. arxiv. https://arxiv.org/abs/2210.03629 (2022-10-06)
[2] Toolformer: Language Models Can Teach Themselves to Use Tools. arxiv. https://arxiv.org/abs/2302.04761 (2023-02-09)
[3] ToolLLM: Facilitating Large Language Models to Master 16000+ Real-world APIs. arxiv. https://arxiv.org/abs/2307.16789 (2023-07-31)
[4] Gorilla: Large Language Model Connected with Massive APIs. hf-daily. https://huggingface.co/papers/2305.15334 (2023-05-24)
[5] ToolkenGPT: Augmenting Frozen Language Models with Massive Tools via Tool Embeddings. hf-daily. https://huggingface.co/papers/2305.11554 (2023-05-19)
[6] Tree of Thoughts: Deliberate Problem Solving with Large Language Models. arxiv. https://arxiv.org/abs/2305.10601 (2023-05-17)
[7] Language Agent Tree Search Unifies Reasoning, Acting, and Planning in Language Models. arxiv. https://arxiv.org/abs/2310.04406 (2023-10-06)
[8] An LLM Compiler for Parallel Function Calling. arxiv. https://arxiv.org/abs/2312.04511 (2023-12-07)
[9] MemGPT: Towards LLMs as Operating Systems. arxiv. https://arxiv.org/abs/2310.08560 (2023-10-12)
[10] Reflexion: Language Agents with Verbal Reinforcement Learning. arxiv. https://arxiv.org/abs/2303.11366 (2023-03-20)
[11] MetaGPT: Meta Programming for Multi-Agent Collaborative Framework. hf-search. https://huggingface.co/papers/2308.00352 (2023-08-01)
[12] AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation Framework. hf-search. https://huggingface.co/papers/2308.08155 (2023-08-16)
[13] Communicative Agents for Software Development. hf-search. https://huggingface.co/papers/2307.07924 (2023-07-16)
[14] AgentVerse: Facilitating Multi-Agent Collaboration and Exploring Emergent Behaviors. hf-search. https://huggingface.co/papers/2308.10848 (2023-08-21)
[15] Agentic AI Frameworks: Architectures, Protocols, and Design Challenges. web. https://arxiv.org/abs/2508.10146 (2025-08-14)
[16] LLM-Based Multi-Agent Orchestration: A Survey of Frameworks, Communication Protocols, and Emerging Patterns. web. https://ideas.repec.org/a/gam/jftint/v18y2026i6p326-d1967830.html (2026-06-01)
[17] SWE-bench: Can Language Models Resolve Real-World GitHub Issues?. arxiv. https://arxiv.org/abs/2310.06770 (2023-10-10)
[18] WebArena: A Realistic Web Environment for Building Autonomous Agents. arxiv. https://arxiv.org/abs/2307.13854 (2023-07-25)
[19] GAIA: A Benchmark for General AI Assistants. arxiv. https://arxiv.org/abs/2311.12983 (2023-11-21)
[20] API-Bank: A Comprehensive Benchmark for Tool-Augmented LLMs. hf-daily. https://huggingface.co/papers/2304.08244 (2023-04-14)
[21] AgentBench: Evaluating LLMs as Agents. hf-daily. https://huggingface.co/papers/2308.03688 (2023-08-07)
[22] InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. hf-daily. https://huggingface.co/papers/2403.02691 (2024-03-05)
[23] Agent Security Bench (ASB): Formalizing and Benchmarking Attacks and Defenses in LLM-based Agents. hf-daily. https://huggingface.co/papers/2410.02644 (2024-10-03)
[24] Bounded Agents: Delegation Security for Multi-Agent AI Systems. arxiv. https://arxiv.org/abs/2608.15888 (2026-08-16)
[25] Analysis of Reflection & Search Mechanisms in Agent Execution. web. https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf (2023-12-15)
