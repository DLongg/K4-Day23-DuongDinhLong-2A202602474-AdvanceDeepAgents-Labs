# Comprehensive Survey on Video and Multimodal Generation: Architectures, Control, Benchmarks, and Open Challenges

## TL;DR
- Modern video and multimodal generative modeling has transitioned from 2D U-Nets augmented with temporal convolutions to 3D Spatial-Temporal Diffusion Transformers (STDiTs) paired with high-compression Causal 3D VAEs [1][2][3][4].
- Cross-modal conditioning mechanisms span early-fusion discrete tokenization, LLM signal token routing, and twin Diffusion Transformers with blockwise bidirectional cross-attention, enabling high-definition Text-to-Video (T2V), Image-to-Video (I2V), Audio-to-Video (A2V), and Any-to-Any generation [5][6][7][8][9].
- Precise motion control and temporal coherence are realized through decoupled camera extrinsics and object trajectory modules, entity-level latent drag representations, optical-flow feature consistency constraints, and sliding temporal windowing [10][11][12][13][14][15][16].
- Quantitative evaluation has evolved from unidimensional scalar metrics (FVD, IS, CLIP-score) toward human-aligned multi-dimensional benchmark suites (VBench, VBench++, EvalCrafter) and million-scale prompt galleries (VidProM) [17][18][19][20].
- Key open challenges center on physical commonsense modeling (VideoPhy), long-context temporal drift, inference acceleration via post-training quantization and step distillation (ViDiT-Q, TurboDiffusion), and robust in-generation digital watermarking (VideoShield, SynthID) [21][22][23][24][25].

## Background
Generative video and multimodal synthesis aims to generate temporally coherent, visually crisp, and semantically aligned video streams conditioned on diverse modalities, including text prompts, static images, dynamic audio, and trajectory paths. Early approaches adapted 2D Latent Diffusion Models (LDMs) by grafting 1D temporal convolution and temporal attention layers onto pre-trained image UNets, such as Stable Video Diffusion (SVD) [4]. While effective for short clips, UNet-based architectures faced scalability limits in resolution and temporal modeling.

The field has undergone a major paradigm shift toward Diffusion Transformers (DiTs), Flow Matching frameworks, and Causal 3D Variational Autoencoders (3D VAEs) [1][2][26]. By factorizing or jointly attending across spatial and temporal dimensions, modern backbones scale parameter counts past 10 billion while generating high-definition (768p--1080p) videos extending to 10--16 seconds [2][5]. Simultaneously, the scope of multimodal generation has expanded from isolated Text-to-Video (T2V) to unified Any-to-Any cross-modal models capable of processing and generating interleaved text, vision, and audio streams [5][6][7].

## Neural Architectures for Generative Video

### Spatial-Temporal Diffusion Transformers (DiT & STDiT)
Modern generative video backbones predominantly leverage Spatial-Temporal Diffusion Transformers (STDiTs) to process compressed video latents. CogVideoX introduces an Expert Transformer featuring Expert Adaptive LayerNorm (Expert Adaptive LN) to decouple text-visual cross-attention from temporal dynamics, generating 10-second videos at 768x1360 resolution [1]. HunyuanVideo scales open video generation to over 13 billion parameters using a dual-stream to single-stream DiT architecture, achieving state-of-the-art text-video alignment through joint text guidance from LLaMA3-8B and CLIP encoders [2]. Open-Sora 2.0 demonstrates that commercial-grade STDiT models can be trained within a $200k compute budget [27].

### Latent Spatiotemporal Compression: 3D VAEs
To reduce the computational burden of quadratic attention over thousands of video tokens, high-efficiency 3D VAEs compress spatial and temporal dimensions concurrently:
- **HunyuanVideo Causal 3D VAE**: Employs a $4 \times 8 \times 8$ spatiotemporal downsampling factor to preserve fine motion details and temporal causality [2].
- **LTX-Video 3D VAE**: Pushes spatial-temporal video compression to an extreme $1:192$ compression ratio ($32 \times 32 \times 8$ factor) by moving the patchification step into the Video-VAE input stage, enabling faster-than-real-time generation (5 seconds of 24 fps video in 2 seconds on an H100 GPU) [3].
- **KVAE Tokenizer Family**: Analyzes tradeoffs in $4 \times 16 \times 16$ and $4 \times 8 \times 8$ video VAE tokenizers, balancing continuous latent channel dimensions against downstream DiT convergence rates [28].

### Flow Matching and Resolution-Pyramid Trajectories
Flow matching and rectified flow trajectories have largely supplanted standard DDPM sampling schedules due to straighter sampling paths and faster inference convergence. Pyramidal Flow Matching structures the denoising trajectory across resolution pyramid stages, executing full-resolution temporal attention only at the final stage to produce 768p 24 FPS videos in 20.7k A100 GPU hours [26]. Movie Gen leverages a 30-billion parameter Transformer-based Flow Matching model to generate 1080p videos with synchronized high-fidelity audio [5].

### Autoregressive Video Models & In-Flight KV Caching
Beyond iterative diffusion models, autoregressive video modeling continues to advance. Cosmos World Foundation Models combine Diffusion World Models with discrete Autoregressive World Models trained on FSQ/VQ-VAE video tokenizers for physical AI simulations [29]. To address the latency of long-context autoregressive video generation, FlashForward introduces an in-flight KV cache with clean anchors, directly reusing attention Key-Value states calculated during denoising forwards to eliminate dedicated KV reconstruction overhead [30].

## Multimodal and Cross-Modal Generation Paradigms

### Text-to-Video (T2V) and Image-to-Video (I2V)
Conditioning video generation on text prompts and static images requires effective cross-modal feature injection:
- **Explicit Image Factorization**: Emu Video factorizes text-to-video synthesis into a two-stage process: generating a high-resolution image conditioned on text (T2I), followed by video frame generation conditioned on both the text prompt and the generated static image (I2V), achieving 81%--96% human evaluation win rates over single-stage baselines [31].
- **U-ViT Latent Diffusion**: Vidu utilizes a U-ViT (U-Net Vision Transformer) backbone to generate 1080p clips up to 16 seconds in a single unified diffusion pass [32].
- **Modular Animation**: AnimateDiff inserts a plug-and-play temporal motion module into frozen text-to-image diffusion models, animating personalized image checkpoints without retraining spatial parameters [10].

### Audio-to-Video (A2V) and Video-to-Audio (V2A)
Integrating audio dynamics into visual generation is essential for realistic talking-head synthesis and soundtrack synchronization:
- **EMO (Emote Portrait Alive)**: Generates expressive portrait videos from a single reference image and audio stream by injecting wav2vec 2.0 audio features into UNet blocks via cross-attention, bypassing intermediate 3D structural representations or landmark constraints [8].
- **Movie Gen Audio**: Synchronizes high-fidelity audio generation (V2A) with generated video streams via joint spatial-temporal attention [5].
- **Ovi Twin-DiT Architecture**: Ovi establishes joint audio-video generation using twin Latent Diffusion Transformers (twin-DiTs) linked via blockwise bidirectional cross-modal attention, utilizing shared T5 text embeddings and aligned RoPE position scaling to produce 1080p videos with lip-synced speech and sound effects [9].

### Any-to-Any Multimodal Architecture Comparison
Modern cross-modal architectures fall into three primary integration paradigms:

| Model | Architecture Paradigm | Primary Modalities Supported | Key Conditioning Mechanism |
| :--- | :--- | :--- | :--- |
| **Chameleon** [6] | Early-Fusion Autoregressive LLM | Interleaved Text & Image | Discrete VQ-VAE visual tokenization into a unified vocabulary space |
| **NExT-GPT** [7] | Parameter-Efficient Signal Routing | Text, Image, Video, Audio | Vicuna LLM core outputs latent signal tokens (`[Vid]`) to diffusion decoders |
| **Movie Gen** [5] | Flow Matching Transformer | Text, Video, Audio, Image | Multi-stage text cross-attention and visual reference encoders |
| **Ovi** [9] | Twin-DiT Cross-Modal Fusion | Text, Video, Audio | Blockwise bidirectional cross-attention with temporal RoPE scaling |

## Fine-Grained Motion Control and Temporal Consistency

### Controllable Video Generation
Controlling dynamic object movements and camera trajectories is critical for cinematic video creation:
- **MotionCtrl**: Decouples motion control into a Camera Motion Control Module (CMCM), which integrates camera extrinsics sequences into temporal transformers, and an Object Motion Control Module (OMCM), which injects object movement trajectories into UNet layers [11].
- **Trajectory Attention**: Models trajectory attention as an auxiliary branch alongside standard temporal attention, explicitly performing attention along pixel trajectories across frames for fine-grained camera and object motion control [12].
- **DragAnything**: Utilizes entity representation extracted from diffusion latent features rather than single pixel drag points, preventing background distortion and outperforming point-based drag methods like DragNUWA by 26% in user preference studies [13].

### Zero-Shot and Instruction-Based Video Editing
- **TokenFlow**: Enforces temporal consistency in text-driven video editing by enforcing linear feature consistency across diffusion internal feature maps guided by inter-frame optical flow trajectories [14].
- **InstructVideo**: Recasts reward fine-tuning for video diffusion models as an instruction-driven editing process using diffusion corruption to align text-to-video synthesis with human feedback [33].
- **Tune-A-Video**: Introduces One-Shot Video Tuning, inflating 2D image diffusion models into spatio-temporal UNets using sparse spatio-temporal attention and DDIM inversion for structural guidance [34].

### Long-Video Extrapolation and Temporal Coherence
Generating extended video sequences (hundreds to thousands of frames) without visual degradation or scene cuts requires dedicated temporal management:
- **Gen-L-Video**: Employs temporal co-denoising to extend short video diffusion models by treating long videos as overlapping temporal windows denoised in parallel and smoothly merged [35].
- **FreeNoise**: Discovers the critical role of initial noise correlation across time, introducing noise rescheduling and temporal attention fusion to generate consistent 64+ frame videos with only ~17% inference latency overhead compared to ~255% for window-averaging methods [15].
- **StreamingT2V**: Combines a Conditional Attention Module (CAM) for short-term memory across adjacent chunks with an Appearance Preservation Module (APM) for long-term anchor memory, enabling consistent long video generation past 1200 frames [16].

## Benchmarks, Evaluation Metrics, and Inference Acceleration

### Comprehensive Benchmark Suites
Evaluating video generation quality requires multi-dimensional diagnostics:
- **VBench & VBench++**: Dissects video quality into 16 disentangled dimensions across **Video Quality** (motion smoothness, temporal flickering, subject consistency, dynamic degree) and **Video-Condition Consistency** (text-video alignment, attribute binding) [17]. VBench++ expands evaluation to I2V tasks, trustworthiness, and VBench-Long [18].
- **EvalCrafter**: Evaluates models across 700 diverse prompts and 17 objective metrics, using linear regression coefficient optimization against human ratings to achieve superior Spearman ($\rho_s$) rank correlation [19].
- **VidProM Dataset**: Curates 1.67 million real-user T2V prompts and 6.69 million generated videos across 4 SOTA diffusion backbones, featuring 3072-dimensional text embeddings and NSFW safety scores [20].

### Mathematical Formulations of Key Metrics

1. **Fréchet Video Distance (FVD)**: Measures distribution divergence between generated video feature representations and ground truth video features extracted from an Inflated 3D ConvNet (I3D) [17][19]:
$$\text{FVD} = \|\mu_r - \mu_g\|^2 + \text{Tr}(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2})$$

2. **CLIP-Score (Text-Video Alignment)**: Computes the average cosine similarity between frame visual embeddings $e_{v,t}$ and prompt text embedding $e_{\text{text}}$ across $T$ frames [17]:
$$\text{CLIP-Score} = \frac{1}{T} \sum_{t=1}^{T} \frac{e_{v,t} \cdot e_{\text{text}}}{\|e_{v,t}\| \|e_{\text{text}}\|}$$

3. **Motion Smoothness**: Evaluates optical flow vector variance across consecutive frame pairs [17]:
$$\text{Smoothness} = 1 - \frac{1}{T-1} \sum_{t=1}^{T-1} \|\mathbf{W}_{t \to t+1} - \mathbf{W}_{t-1 \to t}\|$$

### Inference Speedup: Quantization & Distillation
- **ViDiT-Q**: Identifies time-varying activation channel imbalances in Video DiTs and introduces static-dynamic channel balancing and dynamic online scaling, achieving W8A8 and W4A8 post-training quantization with $2.0\text{--}2.5\times$ memory savings and $1.4\text{--}1.7\times$ speedup [21].
- **QVD (Quantized Video Diffusion)**: Employs High Temporal Discriminability Quantization (HTDQ) with logarithmic temporal quantization and Scattered Channel Range Integration (SCRI) to prevent FVD collapse under low-bit quantization [36].
- **TurboDiffusion**: Combines step distillation (reducing denoising steps to 1--4 steps), 3D bi-directional attention acceleration, and FP8/INT8 KV-cache quantization to achieve a $100\text{--}200\times$ speedup over standard iterative diffusion models [22].

## Trends and Open Problems

### Physical Commonsense and World Simulation
Despite significant scaling, current video models struggle with intuitive physics:
- **VideoPhy Benchmark**: Evaluates model performance on real-world physical dynamics (gravity, fluid dynamics, rigid body collisions, momentum conservation) [23]. Results reveal that even scaled DiT architectures suffer from severe physical hallucinations (e.g., floating objects, unnatural liquid flow, interpenetrating solid objects) that cannot be resolved purely through prompt engineering or conventional data scaling [23].
- **Physical AI World Models**: Systems like NVIDIA Cosmos build foundation models explicitly conditioned on physical actions, camera trajectories, and robotic control inputs to bridge the gap between visual synthesis and embodied physical simulation [29].

### Spatial-Temporal Hallucinations & Long-Context Drift
Autoregressive and sliding-window long-video generators remain prone to cumulative drift:
- **Spatial-Temporal Hallucination**: Spontaneous object disappearance/reappearance, character identity distortion, and temporal flickering persist over long generation horizons [16][18].
- **Context Window Bottlenecks**: Maintaining long-range semantic anchors over thousands of frames requires memory architectures like StreamingT2V's APM [16] or KV-caching like FlashForward [30] to prevent scene degradation.

### Watermarking, Safety, and Provenance Verification
As video generation reaches photorealistic quality, robust provenance tracking and watermarking are mandatory:
- **VideoShield**: Introduces a training-free in-generation watermarking scheme that maps binary payloads to initial Gaussian noise latents prior to iterative denoising [24]. Utilizing DDIM inversion for blind payload extraction, VideoShield enables both spatial (frame-level) and temporal (frame order / insertion) tamper localization [24].
- **SynthID for Video**: Google DeepMind's SynthID embeds an imperceptible digital watermark directly into frame pixel latents during generation, maintaining resilience against cropping, compression, and frame-rate conversions without degrading video fidelity [25].

```
+-----------------------------------------------------------------------------------+
|                        MODERN VIDEO GENERATION TAXONOMY                           |
+-----------------------------------------------------------------------------------+
|  Neural Backbones     | 3D DiT / STDiT (CogVideoX, HunyuanVideo, Open-Sora 2.0)   |
|                       | Flow Matching (Movie Gen, Pyramidal Flow Matching)        |
|                       | Autoregressive / World Models (Cosmos, FlashForward)      |
+-----------------------+-----------------------------------------------------------+
|  Compression Layer    | Causal 3D VAEs (HunyuanVideo 4x8x8, LTX-Video 1:192)      |
+-----------------------+-----------------------------------------------------------+
|  Multimodal Fusion    | Early Fusion (Chameleon), Signal Tokens (NExT-GPT)        |
|                       | Twin-DiT Cross-Attention (Ovi), Factorized I2V (Emu Video)|
+-----------------------+-----------------------------------------------------------+
|  Motion & Control     | Trajectory Attention, MotionCtrl (CMCM/OMCM), DragAnything|
|                       | Editing (TokenFlow, InstructVideo, Tune-A-Video)          |
|                       | Extrapolation (FreeNoise, StreamingT2V, Gen-L-Video)      |
+-----------------------+-----------------------------------------------------------+
|  Benchmarking         | VBench / VBench++, EvalCrafter, VidProM Prompt Gallery    |
+-----------------------+-----------------------------------------------------------+
|  Speedup & Safety     | TurboDiffusion, ViDiT-Q, QVD, VideoShield, SynthID        |
+-----------------------------------------------------------------------------------+
```

## References
[1] CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer. arxiv. https://arxiv.org/abs/2408.06072 (2024-08-12)
[2] HunyuanVideo: A Systematic Framework For Large Video Generative Models. hf-daily. https://huggingface.co/papers/2412.03603 (2024-12-03)
[3] LTX-Video: Realtime Video Latent Diffusion. hf-daily. https://huggingface.co/papers/2501.00103 (2024-12-30)
[4] Stable Video Diffusion: Scaling Latent Video Diffusion Models to Large Datasets. arxiv. https://arxiv.org/abs/2311.15127 (2023-11-25)
[5] Movie Gen: A Cast of Media Foundation Models. arxiv. https://arxiv.org/abs/2410.13720 (2024-10-04)
[6] Chameleon: Mixed-Modal Early-Fusion Foundation Models. hf-search. https://huggingface.co/papers/2405.09818 (2024-05-16)
[7] NExT-GPT: Any-to-Any Multimodal LLM. arxiv. https://arxiv.org/abs/2309.05519 (2023-09-11)
[8] EMO: Emote Portrait Alive - Generating Expressive Portrait Videos with Audio2Video Diffusion Model under Weak Conditions. arxiv. https://arxiv.org/abs/2402.17485 (2024-02-27)
[9] Ovi: Twin Backbone Cross-Modal Fusion for Audio-Video Generation. hf-search. https://huggingface.co/papers/2510.01284 (2025-09-30)
[10] AnimateDiff: Animate Your Personalized Text-to-Image Diffusion Models without Specific Tuning. arxiv. https://arxiv.org/abs/2307.04725 (2023-07-10)
[11] MotionCtrl: A Unified and Flexible Motion Controller for Video Generation. arxiv. https://arxiv.org/abs/2312.03641 (2023-12-06)
[12] Trajectory Attention for Fine-grained Video Motion Control. arxiv. https://arxiv.org/abs/2411.19324 (2024-11-28)
[13] DragAnything: Motion Control for Anything using Entity Representation. arxiv. https://arxiv.org/abs/2403.07420 (2024-03-15)
[14] TokenFlow: Consistent Diffusion Features for Consistent Video Editing. arxiv. https://arxiv.org/abs/2307.10373 (2023-07-19)
[15] FreeNoise: Tuning-Free Longer Video Diffusion via Noise Rescheduling. web. https://proceedings.iclr.cc/paper_files/paper/2024/file/15ce8e7afe5ee95bad56e3b9be28d3d1-Paper-Conference.pdf (2024-05-07)
[16] StreamingT2V: Consistent, Dynamic, and Extendable Long Video Generation from Text. arxiv. https://arxiv.org/abs/2403.14773 (2024-03-27)
[17] VBench: Comprehensive Benchmark Suite for Video Generative Models. web. https://arxiv.org/abs/2311.17982 (2023-11-29)
[18] VBench++: Comprehensive and Versatile Benchmark Suite for Video Generative Models. web. https://arxiv.org/abs/2411.13503 (2024-11-20)
[19] EvalCrafter: Benchmarking and Evaluating Large Video Generation Models. hf-daily. https://huggingface.co/papers/2310.11440 (2023-10-17)
[20] VidProM: A Million-scale Real Prompt-Gallery Dataset for Text-to-Video Diffusion Models. hf-daily. https://huggingface.co/papers/2403.06098 (2024-03-10)
[21] ViDiT-Q: Efficient and Accurate Quantization of Diffusion Transformers for Image and Video Generation. hf-daily. https://huggingface.co/papers/2406.02540 (2024-06-04)
[22] TurboDiffusion: Accelerating Video Diffusion Models by 100-200 Times. hf-daily. https://huggingface.co/papers/2512.16093 (2025-12-18)
[23] VideoPhy: Evaluating Physical Commonsense for Video Generation. hf-daily. https://huggingface.co/papers/2406.03520 (2024-06-05)
[24] VideoShield: Regulating Diffusion-based Video Generation Models via Watermarking. web. https://arxiv.org/abs/2501.14195 (2025-01-23)
[25] Watermarking AI-generated text and video with SynthID. web. https://deepmind.google/blog/watermarking-ai-generated-text-and-video-with-synthid/ (2024-05-14)
[26] Pyramidal Flow Matching for Efficient Video Generative Modeling. hf-daily. https://huggingface.co/papers/2410.05954 (2024-10-08)
[27] Open-Sora 2.0: Training a Commercial-Level Video Generation Model in $200k. hf-daily. https://huggingface.co/papers/2503.09642 (2025-03-12)
[28] KVAE: Family of Tokenizers for Multimodal Generative Models. arxiv. https://arxiv.org/abs/2608.05798 (2026-08-06)
[29] Cosmos World Foundation Model Platform for Physical AI. hf-daily. https://huggingface.co/papers/2501.03575 (2025-01-07)
[30] In-Flight KV Cache with Clean Anchors for Faster Autoregressive Video Diffusion. arxiv. https://arxiv.org/abs/2609.32540 (2026-09-26)
[31] Emu Video: Factorizing Text-to-Video Generation by Explicit Image Conditioning. hf-search. https://huggingface.co/papers/2311.10709 (2023-11-17)
[32] Vidu: a Highly Consistent, Dynamic and Skilled Text-to-Video Generator with Diffusion Models. hf-search. https://huggingface.co/papers/2405.04233 (2024-05-07)
[33] InstructVideo: Instructing Video Diffusion Models with Human Feedback. arxiv. https://arxiv.org/abs/2312.12490 (2023-12-19)
[34] Tune-A-Video: One-Shot Tuning of Image Diffusion Models for Text-to-Video Generation. web. https://tuneavideo.github.io/ (2023-03-17)
[35] Gen-L-Video: Multi-Text to Long Video Generation via Temporal Co-Denoising. arxiv. https://arxiv.org/abs/2305.18264 (2023-05-30)
[36] QVD: Post-training Quantization for Video Diffusion Models. web. https://arxiv.org/pdf/2407.11585 (2024-07-16)
