# Survey on Efficient Inference and Small Language Models

## TL;DR
- **Ultra-Low Bit Quantization & Custom KV Caches:** Post-training quantization (PTQ) methods like AWQ [1], GPTQ [2], SmoothQuant [3], and LieQ [4] compress model weights down to 2–4 bits, while ternary architectures like BitNet b1.58 [5] eliminate floating-point matrix multiplications entirely. Asymmetric KV cache quantization methods such as KIVI [6] and KVarN [7] reduce decoding memory footprints by up to 2.6×–4× without accuracy collapse.
- **Architectural Refinements & Compact SLMs:** Modern sub-10B small language models (SLMs) achieve capabilities once exclusive to 70B+ models through structural innovations including Grouped-Query Attention (GQA) [8], Multi-Head Latent Attention (MLA) [9], interleaved local-global attention in Gemma 2 [10], deep-and-thin network configurations in MobileLLM [11], and hybrid non-Transformer state space architectures like Mamba-2 [12].
- **System-Level Acceleration & Speculative Decoding:** Systems software innovations—including PagedAttention in vLLM [13], RadixAttention in SGLang [14], asynchronous hardware utilization in FlashAttention-3 [15], and streaming attention sinks in StreamingLLM [16]—boost batch serving throughput by 2×–6.4×. Speculative decoding frameworks like Medusa [17] eliminate multi-model coordination to speed up generation by 1.9×–2.3×.
- **Teacher-to-Student Distillation & PEFT Serving:** Synthetic dataset distillation from frontier teachers (e.g., DeepSeek-R1 [18] and MiniCPM [19]) enables 1.5B–32B SLMs to achieve high reasoning performance. Parameter-efficient fine-tuning via QLoRA [20] and multi-adapter execution systems like S-LoRA [21] allow hosting thousands of task-specific adapters with minimal memory overhead.

## Background
The rapid progression of large language models (LLMs) has demonstrated exceptional zero-shot reasoning, mathematical computation, and task instruction adherence. However, deploying multi-billion parameter autoregressive models presents severe computational challenges: standard 16-bit floating-point (FP16/BF16) model execution demands high GPU VRAM capacity, while the autoregressive generation loop is inherently memory-bandwidth bound. During generation, every generated token requires loading gigabytes of model weights from High Bandwidth Memory (HBM) into fast SRAM cache, creating significant latency bottlenecks during single-sequence decoding and severe memory exhaustion during multi-tenant batch serving.

To overcome these constraints, the artificial intelligence community has established two primary research directions: **Small Language Models (SLMs)** (typically ranging from sub-1B to 14B parameters) and **Efficient Inference Techniques**. Small language models aim to maximize token efficiency, data quality, and architectural density, delivering competitive benchmark accuracy under strict compute budgets [10][22][11][19]. Efficient inference encompasses a multi-layered stack of optimizations:
1. *Quantization*: Compressing weight parameters, activation tensors, and Key-Value (KV) caches to low-bit integer formats [5][4][1][3][2][6][7].
2. *Architectural Innovation*: Structuring Transformer attention mechanisms and State Space Models (SSMs) to minimize memory footprints and scale linearly with sequence length [9][10][12][8][11].
3. *System Serving Frameworks & Acceleration Algorithms*: Eliminating memory fragmentation, parallelizing decoding passes, and exploiting GPU hardware primitives [13][14][15][17][16][23][24].
4. *Distillation, Pruning, and Adapter Execution*: Transferring capabilities from frontier models, stripping redundant network layers, and serving thousands of specialized adapters concurrently [18][25][26][27][20][21].

This literature survey provides a comprehensive technical taxonomy and synthesis of recent advances across these four domains.

## Model Compression and Quantization Paradigms

Quantization reduces the bit-precision of model weights, activation states, and intermediate tensors, directly lowering VRAM requirements and boosting hardware compute density.

### Post-Training Quantization (PTQ) for Weights and Activations
Post-Training Quantization operates directly on pre-trained FP16/BF16 checkpoints without requiring full model retraining:
- **Weight-Only PTQ:** GPTQ [2] formulates layer-wise weight quantization as an inverse Hessian-based squared error minimization problem $\arg\min_{\hat{W}} \| W X - \hat{W} X \|_2^2$. By updating unquantized weights along columns using approximate second-order information, GPTQ compresses 175B models to 3-bit or 4-bit precision in 4 GPU hours, delivering 3.25×–4.5× end-to-end inference speedups. AWQ [1] observes that protecting the top 1% of salient weight channels drastically suppresses quantization error. Instead of mixed-precision kernels, AWQ derives per-channel scaling factors $S_x = s_X^\alpha$ offline from activation magnitudes to execute uniform INT3/INT4 matrix multiplications, producing over 3× latency speedups on edge GPUs like Jetson Orin.
- **Weight-Activation PTQ:** Quantizing activations to 8-bit or 4-bit integer precision (W8A8/W4A4) is complicated by systemic activation outliers concentrated in specific channels. SmoothQuant [3] resolves this by applying an offline per-channel mathematically equivalent transformation $Y = (X \cdot \text{diag}(s)^{-1}) \cdot (\text{diag}(s) \cdot W)$, migrating quantization difficulty from activations to weights. SmoothQuant enables standard INT8 GEMM execution across OPT, LLaMA, and Mistral models, delivering 1.56× speedups and 2× VRAM reduction with zero accuracy degradation.
- **SLM-Specific PTQ:** Sub-7B SLMs are particularly sensitive to uniform low-bit quantization. LieQ [4] addresses performance collapse in sub-7B models by evaluating three layer-wise sensitivity metrics: Perplexity Drop, Representational Compactness, and Top-$k$ Energy Gain. On Qwen3-4B at 2.05-bit precision, LieQ retains 95.9% of FP16 accuracy, outperforming uniform GPTQ by 19.7% and AWQ by 18.1%.

### Quantization-Aware Training (QAT) and 1-Bit Architectures
While PTQ excels at 4-bit and 8-bit precision, reducing weights to 1–2 bits often demands Quantization-Aware Training (QAT). The BitNet b1.58 [5] framework introduces a ternary weight LLM architecture where every parameter is constrained to $\{-1, 0, 1\}$, represented in $\log_2(3) \approx 1.58$ bits. Linear layer floating-point matrix multiplications are completely replaced by integer additions. BitNet b1.58 matches full-precision FP16 Transformer perplexity starting at 3B parameters while achieving a 2.4×–4.1× memory reduction, 2.7× speedup, and up to 7.16× lower energy consumption.

### Low-Bit Key-Value (KV) Cache Quantization
During long-context decoding, the memory consumption of stored Key and Value tensors often exceeds weight memory:
- **KIVI [6]:** Implements a tuning-free 2-bit asymmetric KV cache quantization scheme. Observing distinct structural distributions in Key vs. Value matrices, KIVI quantizes Key caches per-channel and Value caches per-token. This reduces overall peak GPU VRAM by up to 2.6× and enables up to a 4× increase in batch sizes with 2.35×–3.47× throughput gains.
- **KVarN [7]:** Targets error accumulation in long-horizon reasoning tasks (e.g., AIME and MATH-500). KVarN uses randomized Hadamard rotations to smooth out channel outliers alongside dual-axis variance normalization, preventing accuracy collapse at 2-bit KV precision without requiring calibration datasets.

## Modern SLM Architectural Innovations and State-Space Designs

Beyond quantization, fundamental structural alterations to model backbones significantly enhance inference efficiency and context scalability.

### Attention Optimizations: GQA, MLA, and Interleaved Windows
Standard Multi-Head Attention (MHA) allocates independent Key and Value heads for every Query head, causing linear VRAM growth with context length and batch size:
- **Grouped-Query Attention (GQA) [8]:** Groups $N$ query heads into $G$ subgroups, sharing a single Key-Value pair per subgroup. Uptraining MHA models into GQA using mean-pooled projections requires only 5% of original pre-training compute, achieving decoding speeds comparable to Multi-Query Attention (MQA) while retaining full MHA reasoning performance.
- **Multi-Head Latent Attention (MLA) [9]:** Introduced in DeepSeek-V2, MLA projects attention KV states into a low-rank compressed latent vector $c_{t}^{KV}$, paired with decoupled Rotary Position Embedding ($k_{t,i}^R$) key branches. MLA compresses the KV cache footprint by 93.3% compared to standard MHA, allowing DeepSeek-V2 to scale to 128K context windows while boosting generation decoding throughput by 5.76×.
- **Interleaved Local-Global Attention [10]:** Google's Gemma 2 alternates between local sliding-window attention (4096 tokens) and global self-attention (8192 tokens) every layer. Combined with logit soft-capping, Gemma 2 (2.6B to 27.2B) reduces KV cache growth during sequence generation while matching models 2×–3× larger.
- **Context Scaling via LongRoPE [22]:** Microsoft's Phi-3-mini (3.8B parameters) employs LongRoPE context scaling to extend native 4K context to 128K tokens without fine-tuning performance drop on short contexts. Quantized to 4-bit AWQ/GGUF format, Phi-3-mini fits within <2.4 GB VRAM and executes locally on mobile device chips at >12 tokens/sec.

### State Space Models: Mamba-2 and Hybrid Sequence Decoders
To bypass the quadratic compute and $O(N)$ memory complexity of Transformer self-attention, State Space Models (SSMs) offer constant-state $O(1)$ memory during decoding. Mamba-2 [12] introduces Structured State Space Duality (SSD), establishing a theoretical bridge between selective SSMs and causal linear attention via semiseparable matrix factorizations. By reformulating recurrences into dense matrix-multiplication (GEMM) blocks, Mamba-2 yields 2×–8× faster state computation throughput than Mamba-1 on GPUs and TPUs, scaling linearly with sequence length while preserving generation quality.

### On-Device Sub-B Parameter Topology
MobileLLM [11] investigates optimal design principles for sub-1B parameter models (125M and 350M parameters). Empirical evaluations demonstrate that "deep and thin" architectures (e.g., 30 layers for 125M, 42 layers for 350M) significantly outperform conventional shallow and wide configurations. Incorporating input/output embedding parameter sharing and immediate block-wise weight sharing allows MobileLLM models to operate within <1 GB RAM under 4-bit quantization on mobile platforms.

## Inference Acceleration, KV Cache Management, and Systems Infrastructure

System software and kernel execution engines translate algorithmic compressions into physical hardware throughput gains.

### Dynamic KV Cache Memory Management: PagedAttention and RadixAttention
Traditional LLM serving frameworks pre-allocate contiguous memory blocks for maximum context lengths, resulting in 60%–80% VRAM waste due to internal and external fragmentation:
- **PagedAttention & vLLM [13]:** Inspired by operating system virtual memory paging, PagedAttention partitions KV caches into fixed-size physical memory blocks. It enables copy-on-write KV sharing across beam-search candidates and shared prompt prefixes, reducing VRAM waste to under 4% and increasing serving throughput by 2×–4× over FasterTransformer.
- **RadixAttention & SGLang [14]:** SGLang organizes prompt prefixes, agent trajectory histories, and dynamic generation trees into a radix tree structure. RadixAttention automatically manages KV cache reuse across multiple generation calls via LRU eviction and cache-aware request scheduling, achieving up to 6.4× higher throughput and 3.7× lower first-token latency than vLLM.
- **StreamingLLM [16]:** Addresses the "attention sink" phenomenon, where softmax normalization forces disproportionate attention weight onto initial tokens regardless of semantic content. StreamingLLM retains 4–8 initial sink tokens alongside a sliding window of recent tokens, enabling models trained on 2K–4K context to execute infinite stream generation (>4 million tokens) with up to 22.2× speedups over re-computation.

### Low-Precision Async Kernels & GPU Hardware Exploitation
- **FlashAttention-3 [15]:** Redesigns fused attention for NVIDIA Hopper GPUs (H100). FlashAttention-3 introduces producer-consumer warp specialization to hide memory latency, overlaps FP32 softmax exponential operations with asynchronous WGMMA matrix multiplications, and incorporates FP8 block quantization with Hadamard transformations. FlashAttention-3 reaches 840 TFLOPs/s (85% utilization) in FP16 and up to 1.3 PFLOPs/s in FP8.
- **llama.cpp CUDA Graph Support [24]:** Integrating CUDA Graph execution into GGML's backend evaluation pipeline captures GPU kernel launch sequences into single computational graphs, eliminating CPU scheduling overhead during token decoding. Paired with GGUF integer quantization (2–8 bit k-quants), CUDA Graphs deliver up to 1.2× wall-clock speedups for single-batch Llama 7B inference.

### Speculative Decoding & Edge Runtime Engines
- **Medusa [17]:** Bypasses the coordination overhead of separate draft models by attaching lightweight single-layer heads ("Medusa heads") directly onto the target model's final hidden state. Medusa heads generate candidate tokens in parallel, which are verified in a single forward pass using custom tree-attention masks. Medusa achieves 1.9×–2.3× wall-clock speedups without output quality loss.
- **ExecuTorch [23]:** Meta's PyTorch-native edge deployment framework exports models via `torch.export` into a restricted Edge Dialect IR covering <300 Core ATen operators. ExecuTorch enables static memory planning, operator fusion, and TorchAO quantization, delegating subgraphs to specialized hardware microprocessors (Qualcomm Hexagon DSP, Apple Neural Engine, Vulkan GPU) to execute C++ binary packages without Python runtime dependencies.

## Knowledge Distillation, Pruning, and Parameter-Efficient Adaptation

Compressing model weights during pre-training or fine-tuning allows developers to build specialized SLMs tailored for specific deployment constraints.

### Distillation from Frontier Models
Synthetic data distillation has emerged as a dominant methodology for training high-performing SLMs:
- **DeepSeek-R1 Distillation [18]:** DeepSeek-R1 distills 800,000 synthetic reasoning trajectories generated by a 671B MoE teacher model into dense student SLMs (1.5B to 70B parameters) via direct SFT. The resulting DeepSeek-R1-Distill-Qwen-32B achieves 72.6% pass@1 on AIME 2024 and 94.3% on MATH-500, outperforming OpenAI o1-mini and proving that teacher distillation is vastly more sample- and compute-efficient than training SLMs from scratch via RL.
- **MiniCPM Scalable Training [19]:** Utilizes a Warmup-Stable-Decay (WSD) learning rate scheduler and model wind-tunnel experiments. The 2.4B MiniCPM base model performs on par with 7B–13B baselines (Llama2-7B, Mistral-7B) across general benchmarks, establishing data-to-parameter scaling ratios beyond standard Chinchilla optimality.

### One-Shot and Structured Network Pruning
Model pruning removes redundant weight parameters to accelerate execution:
- **Unstructured Pruning via SparseGPT [25]:** Solves layer-wise pruning as a one-shot optimization problem without retraining. SparseGPT prunes 50%–60% of parameters from 175B models in under 4.5 GPU hours on a single A100 GPU with minimal perplexity degradation.
- **Structured Pruning via Sheared LLaMA & LLM-Pruner [26][27]:** Sheared LLaMA [26] uses targeted structured pruning to remove entire layers, attention heads, and intermediate hidden dimensions, compacting LLaMA2-7B into Sheared-LLaMA-1.3B (81.4% parameter reduction) and 2.7B models using only 3% of original pre-training compute. LLM-Pruner [27] uses first-order gradients and approximate Hessian information to prune coupled structure components, achieving 94.97% accuracy recovery after 3 hours of LoRA fine-tuning on 50k samples.

### Parameter-Efficient Task Adaptation (PEFT) and Multi-Adapter Serving
- **QLoRA [20]:** Backpropagates gradients through a frozen 4-bit NormalFloat (NF4) base model into 16-bit Low-Rank Adapters (LoRA), incorporating Double Quantization and Paged Optimizers. QLoRA enables fine-tuning a 65B parameter model on a single 48GB GPU, reducing 7B VRAM footprint down to 5 GB (compared to 26 GB for FP16 baselines).
- **S-LoRA Concurrent Adapter Serving [21]:** Hosts thousands of task-specific LoRA adapters on a single base model using Unified Paging (managing dynamic adapter weights and KV cache tensors in a shared memory pool). S-LoRA delivers up to 4× higher throughput than vLLM with naive LoRA support and up to 30× throughput improvement over HuggingFace PEFT, serving over 2,000 adapters concurrently per GPU.

## Trends and Open Problems

Despite significant progress in efficient inference and small language models, several key challenges and research vectors remain:

1. **Ultra-Low Bit Accuracy Floor (1–2 Bit Regime):** While 4-bit PTQ (AWQ [1], GPTQ [2]) is widely deployed in production, moving to 2-bit or ternary precision (BitNet [5], LieQ [4]) introduces accuracy degradation on complex mathematical and symbolic reasoning tasks. Developing hardware-native 2-bit vector execution units and error-tolerant QAT training recipes represents a primary open frontier.
2. **Context Window vs. KV Cache Bounds:** Long-context handling (>100K tokens) remains heavily memory-bound. While MLA [9], GQA [8], KIVI [6], and KVarN [7] substantially compress KV cache sizes, streaming sliding-window approaches like StreamingLLM [16] drop middle context tokens. Reconciling infinite context retention with sub-linear memory growth remains an active area of research.
3. **Hardware-Aware SSM and Transformer Hybrids:** While Mamba-2 [12] provides $O(1)$ decoding memory and sub-quadratic sequence scaling, pure SSMs sometimes lag Transformers on complex multi-hop retrieval and in-context learning tasks. Hybrid architectures combining GQA Transformer layers with Mamba-2 SSM layers are being explored to achieve high task accuracy alongside linear generation throughput.
4. **On-Device Mobile Deployment Standards:** Runtime frameworks such as ExecuTorch [23] and llama.cpp [24] are bridging the gap between desktop GPUs and mobile edge processors. Standardizing low-precision INT4/INT2 operator kernels across heterogeneous NPU/DSP hardware remains essential for seamless edge AI adoption.

## References
[1] AWQ: Activation-aware Weight Quantization for LLM Compression and Acceleration. arxiv. https://arxiv.org/abs/2306.00978 (2023-06-01)
[2] GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers. arxiv. https://arxiv.org/abs/2210.17323 (2022-10-31)
[3] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models. arxiv. https://arxiv.org/abs/2211.10438 (2022-11-18)
[4] LieQ: Exploring Layer-wise Information Effectiveness for Post-Training Quantization in Small Language Models. hf-daily. https://huggingface.co/papers/2508.03332 (2025-08-05)
[5] The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits. hf-daily. https://huggingface.co/papers/2402.17764 (2024-02-27)
[6] KIVI: A Tuning-Free Asymmetric 2bit Quantization for KV Cache. arxiv. https://arxiv.org/abs/2402.02750 (2024-02-04)
[7] KVarN: Variance-Normalized KV-Cache Quantization Mitigates Error Accumulation in Reasoning Tasks. hf-daily. https://huggingface.co/papers/2606.03458 (2026-06-02)
[8] GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints. hf-search. https://huggingface.co/papers/2305.13245 (2023-05-22)
[9] DeepSeek-V2: A Strong, Economical, and Efficient Mixture-of-Experts Language Model. arxiv. https://arxiv.org/abs/2405.04434 (2024-05-07)
[10] Gemma 2: Improving Open Language Models at a Practical Size. hf-search. https://huggingface.co/papers/2408.00118 (2024-07-31)
[11] MobileLLM: Optimizing Sub-billion Parameter Language Models for On-Device Use Cases. arxiv. https://arxiv.org/abs/2402.14905 (2024-02-22)
[12] Transformers are SSMs: Generalized Models and Efficient Algorithms Through Structured State Space Duality. hf-search. https://huggingface.co/papers/2405.21060 (2024-05-31)
[13] Efficient Memory Management for Large Language Model Serving with PagedAttention. arxiv. https://arxiv.org/abs/2309.06180 (2023-09-12)
[14] SGLang: Efficient Execution of Structured Language Model Programs. arxiv. https://arxiv.org/abs/2312.07104 (2023-12-11)
[15] FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision. arxiv. https://arxiv.org/abs/2407.08608 (2024-07-12)
[16] Efficient Streaming Language Models with Attention Sinks. arxiv. https://arxiv.org/abs/2309.17453 (2023-09-29)
[17] Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads. arxiv. https://arxiv.org/abs/2401.10774 (2024-01-19)
[18] DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning. web. https://arxiv.org/abs/2501.12948 (2025-01-22)
[19] MiniCPM: Unveiling the Potential of Small Language Models with Scalable Training Strategies. hf-daily. https://huggingface.co/papers/2404.06395 (2024-04-09)
[20] QLoRA: Efficient Finetuning of Quantized LLMs. web. https://arxiv.org/abs/2305.14314 (2023-05-23)
[21] S-LoRA: Serving Thousands of Concurrent LoRA Adapters. hf-daily. https://huggingface.co/papers/2311.03285 (2023-11-06)
[22] Phi-3 Technical Report: A Highly Capable Language Model Locally on Your Phone. arxiv. https://arxiv.org/abs/2404.14219 (2024-04-22)
[23] ExecuTorch: A Unified PyTorch Solution to Run AI Models On-Device. arxiv. https://arxiv.org/abs/2605.08195 (2026-05-11)
[24] Optimizing llama.cpp AI Inference with CUDA Graphs. web. https://developer.nvidia.com/blog/optimizing-llama-cpp-ai-inference-with-cuda-graphs/ (2024-08-07)
[25] SparseGPT: Massive Language Models Can Be Accurately Pruned in One-Shot. web. https://arxiv.org/abs/2301.00774 (2023-01-02)
[26] Sheared LLaMA: Accelerating Language Model Pre-training via Structured Pruning. hf-daily. https://huggingface.co/papers/2310.06694 (2023-10-10)
[27] LLM-Pruner: On the Structural Pruning of Large Language Models. hf-daily. https://huggingface.co/papers/2305.11627 (2023-05-19)
