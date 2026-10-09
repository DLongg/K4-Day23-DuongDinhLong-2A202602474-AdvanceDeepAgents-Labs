# World Models in Artificial Intelligence: Foundations, Architectures, Applications, and Open Challenges

## TL;DR
- **Paradigm Shift in State Dynamics**: World models have transitioned from classical VAE and Recurrent State Space Models (RSSM) to non-generative representation models like Joint Embedding Predictive Architectures (JEPA) [1][2][3][4][5][6], reducing computational overhead by 2.5x–10x while eliminating pixel reconstruction noise [5][6].
- **Sample-Efficient Model-Based RL & Robotics**: Latent imagination enables reinforcement learning agents (DreamerV1–V3, DayDreamer) to learn behaviors entirely within hallucinated latent state spaces, achieving sample-efficient zero-shot transfer and physical robot manipulation/locomotion in 1–2 hours of online interaction [1][2][3][7][8].
- **Neural Simulators & Video/Driving Models**: Foundation world models (Genie, DreamDojo, GAIA-1/2, Cosmos, Drive-WM) utilize autoregressive transformers, latent diffusion, and 3D/4D occupancy grids to generate interactive, action-controllable neural environments for embodied AI and autonomous vehicles [9][10][11][12][13][14][15][16][17].
- **Diagnostics & Physical Realism Gap**: Diagnostic benchmarks (WorldScore, WBench, PhysBench, EgoTaskQA, MMWorld) reveal that while generative video models achieve high visual fidelity, long-horizon imagined rollouts often capture kinematic trajectories without capturing underlying physical dynamics, leading to severe multi-turn compounding errors [18][19][20][21][22][23][24].

## Background
A world model is an internal representation of environment dynamics that enables an artificial intelligence agent to predict future world states, simulate the consequences of candidate actions, and plan or learn policies within an internal mental workspace [1][25]. Seminal formulations in cognitive science and reinforcement learning established that intelligent agents construct compact latent representations of high-dimensional sensorimotor inputs to internalize environmental physics and causal mechanisms [1][4].

Formally, a canonical world model decomposes the interactive decision-making loop into four primary components [1][25]:
1. **Observation Encoder ($E_\phi$)**: Maps high-dimensional visual or sensory observations $x_t \in \mathbb{R}^{H \times W \times C}$ into a lower-dimensional latent state representation $z_t \in \mathcal{Z}$.
2. **Latent Dynamics Predictor ($T_\theta$)**: Models the state transition distribution $\mathcal{P}(z_{t+1} \mid z_t, a_t, h_t)$ conditioned on prior latent state $z_t$, hidden memory $h_t$, and executed action $a_t$.
3. **Reward/State Predictor ($R_\psi$)**: Predicts scalar environment rewards $\hat{r}_t = R_\psi(z_t)$ or future sensory observations $\hat{x}_t = D(z_t)$.
4. **Policy or Planner ($\pi_\omega$)**: Selects optimal actions $a_t \sim \pi_\omega(z_t, h_t)$ by maximizing predicted cumulative reward across imagined latent trajectories.

Early world models relied on modular architectures combining Variational Autoencoders (VAEs) for visual spatial compression with Mixture Density Recurrent Neural Networks (MDN-RNNs) to capture temporal transitions [1]. Recent advances have expanded this definition across reinforcement learning, foundation video generation, and autonomous driving, sparking a fundamental debate between pixel-reconstructive generative models and energy-based representation models [4][5][6][25][17].

## Taxonomy and Latent Dynamics Architectures: Generative vs Non-Generative Paradigms
The literature categorizes world model architectures along two fundamental paradigms: **Generative World Models** and **Non-Generative Predictive Representation Models** [4][5][25].

```
+-----------------------------------------------------------------------------------+
|                                WORLD MODEL TAXONOMY                               |
+-----------------------------------------------------------------------------------+
                                         |
     +-----------------------------------+-----------------------------------+
     |                                                                       |
     v                                                                       v
+------------------------------------+                     +------------------------------------+
|     Generative World Models        |                     | Non-Generative Representation      |
|  (Pixel/Token Reconstruction)      |                     |            (JEPA / RSSM)           |
+------------------------------------+                     +------------------------------------+
| * VAE + MDN-RNN (Ha & Schmidhuber) |                     | * I-JEPA (Image Latent Prediction) |
| * RSSM (Dreamer V1-V3 continuous)  |                     | * V-JEPA (Video Latent Prediction) |
| * Video Diffusion & Transformers   |                     | * Latent Energy-Based Models (EBM) |
|   (Sora, GAIA-1, Genie, Cosmos)    |                     | * Discrete Categorical RSSM        |
+------------------------------------+                     +------------------------------------+
```

### 1. Generative Visual World Models
Generative world models model the full conditional distribution $P(x_{t+1} \mid x_t, a_t)$ in pixel space or tokenized latent space [1][25][9][17]. Ha and Schmidhuber's foundational architecture [1] utilized a VAE to map $64 \times 64 \times 3$ RGB frames into a Gaussian latent vector $z_t \in \mathbb{R}^{32}$, while an MDN-RNN with 256 hidden units predicted future latent Gaussian mixtures. Modern generative video world models, such as OpenAI Sora, Google Genie, and NVIDIA Cosmos, scale this concept using Spatiotemporal VQ-VAEs or Video Latent Diffusion Transformers (DiT), operating on millions of video frame tokens [9][13][17]. However, pixel reconstruction forces the encoder and dynamics network to spend capacity modeling unpredictable high-frequency background noise (e.g., rustling tree leaves, water ripples), which leads to compounding visual artifacts during multi-step rollouts [4][5][21].

### 2. Recurrent State Space Models (RSSM)
To stabilize latent dynamics, the Recurrent State Space Model (RSSM) introduced in Dreamer [2] decouples state representations into a deterministic recurrent component $h_t = f(h_{t-1}, s_{t-1}, a_{t-1})$ (using a 200-unit GRU) and a stochastic latent component $s_t$ [2][3]. DreamerV1 employed continuous Gaussian latents [2], whereas DreamerV2 and DreamerV3 transitioned to categorical discrete representations (32 categorical variables with 32 classes each, amounting to 1024 discrete dimensions) optimized via Straight-Through Gumbel-Softmax gradient estimation [3][7]. Discrete latents prevent representation collapse, bound prediction variance, and handle multimodal transition dynamics significantly better than continuous Gaussians [3][7]. Furthermore, DreamerV2 introduced KL Balancing (weighting prior KL loss updates at 0.8 and posterior updates at 0.2) to eliminate underfitting [3].

### 3. Non-Generative Joint Embedding Predictive Architectures (JEPA)
Formulated by Yann LeCun [4], Joint Embedding Predictive Architectures (JEPA) reject pixel reconstruction altogether. Instead of decoding raw frames, JEPA maps context $x$ and target $y$ into abstract representation spaces $s_x = E_x(x)$ and $s_y = E_y(y)$ using Vision Transformers (ViT), predicting target embeddings directly via a predictor network $P(s_x, z, a)$ [4][5][6]. 
- **I-JEPA** [5] demonstrates image-based representation prediction: a Context Encoder processes an unmasked 55% image patch, while a Target Encoder (updated via Exponential Moving Average with $\tau=0.996 \to 1.0$) processes 15–20% target patches. I-JEPA achieves 81.0% top-1 accuracy on ImageNet-1K with ViT-H/14 while requiring 2.5x to 10x fewer training FLOPs than pixel-reconstructive Masked Autoencoders (MAE) [5].
- **V-JEPA** [6] extends this paradigm to video dynamics by masking up to 90% of spatio-temporal video blocks. By focusing capacity exclusively on physical motion and temporal transformation vectors, V-JEPA achieves 1.5x to 6x higher sample efficiency on fine-grained action classification tasks (Kinetics-400, Something-Something V2) compared to generative video baselines [6].

| Architecture | Representation Space | Prediction Level | Reconstruction Loss | Key Strengths | Key Weaknesses |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **VAE + MDN-RNN** [1] | Continuous Latent ($z \in \mathbb{R}^{32}$) | Latent Gaussian Mixture | Pixel MSE / VAE KL | Simple, modular | Low resolution, severe visual blur |
| **Dreamer RSSM (V2/V3)** [3][7] | Discrete Categorical (1024-d) | Discrete Latent State | ST-Gumbel KL / Symlog | Handles multimodality, stable RL | Limited pixel rollout quality |
| **I-JEPA / V-JEPA** [5][6] | ViT Latent Feature Space | Abstract Embedding | L2 Latent Distance | 2.5x–10x FLOP efficiency, no pixel noise | Cannot render direct visual RGB frames |
| **Generative Video DiT** [9][13][17] | Tokenized Visual Latent | Spatiotemporal Tokens | Diffusion Score / Cross-Entropy | Photorealistic visual simulation | High compute cost, compounding error |

## Model-Based Reinforcement Learning and Embodied Robotics
World models serve as neural simulators inside which reinforcement learning agents acquire complex behaviors without requiring exhaustive physical interactions [1][2][3][7][8][9][10].

### Latent Imagination and Policy Optimization
In model-based reinforcement learning (MBRL), algorithms such as DreamerV1–V3 optimize actor-critic neural networks entirely inside imagined trajectories generated by the world model [2][3][7]. Over an imagined horizon $H=15$, the agent projects future latent states $\hat{s}_{t+1:t+H}$ using the RSSM prior $p(s_\tau \mid h_\tau)$ and evaluates analytic value gradients $\nabla_\omega \sum_{\tau=t}^{t+H} \gamma^{\tau-t} v_\psi(s_\tau)$ backpropagated through time [2][3].
- **DreamerV2** [3] became the first world-model agent to achieve human-level performance across all 55 Atari 2600 games, matching model-free SOTA (Rainbow, IQN) with significantly higher sample efficiency.
- **DreamerV3** [7] introduced symlog scaling for rewards and target normalization, enabling a single fixed hyperparameter configuration to collect diamonds in Minecraft from scratch without human demonstrations or domain-specific reward shaping [7].

### Real-World Robot Hardware: DayDreamer
While MBRL is traditionally validated in simulation, **DayDreamer** [8] demonstrated that latent world models can learn control policies directly on physical robot hardware. Operating without pre-collected datasets, offline pre-training, or physics simulators, DayDreamer trained an Unitree A1 quadruped robot to walk from scratch in under 1 hour, and a UR5 robotic arm to perform visual pick-and-place manipulation in 1 to 2 hours [8]. Online continuous policy updates within the RSSM latent workspace prevented mechanical wear and accelerated physical learning [8].

### Generalist Foundation Simulators for Robotics
Scaling world models using massive web-scale datasets has unlocked generalist interactive simulators:
- **Genie** [9]: An 11B parameter generative world model trained on 200,000 hours of unlabelled internet video game footage. Genie combines a Spatiotemporal VQ-VAE tokenizer, a Latent Action Model (LAM) that infers unobserved discrete actions frame-to-frame, and an autoregressive MaskGIT dynamics backbone. Users can control generated environments interactively from a single image prompt [9].
- **DreamDojo** [10]: Pre-trained on 44,000 hours of egocentric human manipulation videos, DreamDojo utilizes continuous latent action conditioning and real-time model distillation to serve as a generalist neural simulator for robotic manipulation, enabling zero-shot policy adaptation and physical transfer [10].

## Generative Video, Multi-View Surround, and Occupancy Driving World Models
In autonomous driving and spatial AI, generative world models function as high-fidelity closed-loop simulators for testing corner cases and evaluating autonomous vehicle planning [11][12][13][14][15][16].

### Autoregressive Video & Diffusion World Models
Autonomous driving world models synthesize temporal camera streams conditioned on vehicle steering angles, acceleration, HD maps, and natural language commands [11][12][13][14].
- **GAIA-1** [11]: A 9.6B parameter model combining autoregressive transformers with video decoders to generate multi-second driving scenarios conditioned on text, action tokens, and historical video frames. GAIA-1 demonstrates emergent understanding of vehicle interactions and 3D geometry [11].
- **GAIA-2** [12]: Extends GAIA-1 to surround-view multi-camera generation using latent video diffusion architectures conditioned on 3D bounding boxes and HD map layouts, enabling fine-grained user control over traffic participants and weather conditions [12].
- **NVIDIA Cosmos Platform** [13]: A scalable suite of World Foundation Models utilizing Diffusion and Mixture-of-Transformers (MoT) architectures. Cosmos processes multi-sensory physical observations (video streams, depth, LiDAR, ego-actions) to forecast visual states, providing closed-loop evaluation for physical AI [13].
- **Drive-WM** [14]: Introduces multi-view spatial-temporal cross-attention to maintain geometric consistency across surround cameras. Using a tree-based trajectory sampler, Drive-WM predicts multi-future visual outcomes for candidate control trajectories, allowing planners to evaluate collision risk visually before executing maneuvers [14].

```
+------------------------------------------------------------------------------------+
|               GENERIC AUTONOMOUS DRIVING WORLD MODEL PIPELINE                      |
+------------------------------------------------------------------------------------+
  [Ego-Vehicle Trajectory / Actions (a_t)]  +  [3D Map / Surround Videos (x_t)]
                                    |
                                    v
+------------------------------------------------------------------------------------+
|                         Spatial-Temporal Cross-Attention                           |
|                      & 4D Volumetric Occupancy Dynamics                            |
|                       (GenieDrive [16] / DOME [17])                                |
+------------------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------------------+
|             Multi-View Latent Diffusion / Transformer Predictor                  |
|                      (GAIA-2 [13] / Drive-WM [15])                                 |
+------------------------------------------------------------------------------------+
                                    |
                                    v
+------------------------------------------------------------------------------------+
|     Synthesized Multi-Camera Video Rollouts & Collision Risk Evaluation              |
+------------------------------------------------------------------------------------+
```

### 3D/4D Occupancy World Models
Pure 2D video generation frequently suffers from perspective distortion and depth inconsistency [15][16][17]. To overcome this, recent driving world models integrate explicit 3D and 4D volumetric occupancy grids [15][16]:
- **DOME** [16]: Reformulates 3D occupancy prediction into a temporal generative forecasting problem using Spatial-Temporal Diffusion Transformers (DiT), operating directly over 3D voxel lattices to ensure multi-step geometric stability [16].
- **GenieDrive** [15]: Couples a 4D volumetric occupancy VAE with a video diffusion backbone using Mutual Control Attention. Conditioning video synthesis on explicit 4D occupancy dynamics significantly improves physical plausibility, depth alignment, and temporal consistency [15].

## Benchmarks, Diagnostic Metrics, and Long-Horizon Error Dynamics
Evaluating world models requires probing not only perceptual visual quality, but also action controllability, temporal stability, and adherence to physical laws [18][19][20][21][22][23][24].

### Diagnostic Evaluation Benchmarks
Standard video metrics like Fréchet Video Distance (FVD) or Structural Similarity (SSIM) fail to capture action alignment or causal consistency [17][19]. Modern evaluation suites target specific functional dimensions [18][19][20][22][24]:
- **WorldScore** [19]: Evaluates world generation along three decoupled axes: action controllability, visual quality, and dynamic consistency across next-scene transitions [19].
- **WBench** [22]: A multi-turn interactive benchmark containing 289 test cases and 1,058 interaction turns. WBench measures action compliance, dynamic continuity, and spatial memory over long-horizon interactive turns [22].
- **PhysBench** [18]: Evaluates world model compliance with fundamental physical laws (rigid body dynamics, fluid dynamics, gravity, friction, and collisions) using structured diagnostic visual probes [18].
- **EgoTaskQA** [20]: Tests temporal reasoning in egocentric task videos across descriptive, predictive, explanatory, and counterfactual ("what-if") queries [20].
- **MMWorld** [24]: Multi-discipline benchmark covering physics, mechanics, optics, chemistry, and biology across multi-faceted causal reasoning tasks [24].

### Compounding Errors and the Kinematic-vs-Dynamic Gap
A central challenge in world model research is the severe degradation of imagined rollouts over extended time horizons [21][22][23].

```
Compounding Error over Horizon T:
  E(T) = sum_{t=1}^T || z_t - hat{z}_t || <= C * (1 + epsilon)^T
```

As horizon $T$ increases, small prediction inaccuracies $\epsilon$ accumulate exponentially, causing hallucinated visual frames to drift from valid environment states [21][22].

Recent analytical work by Imagined Rollouts Diagnosis [21] uncovered a fundamental cause for this failure: **Imagined Rollouts are Kinematic, Not Dynamic**. By introducing the Imagined Kinematic-Consistency Error (iKCE) metric, researchers revealed that latent dynamics models learn smooth visual and kinematic motion trajectories while completely failing to model underlying physical forces (e.g., surface friction, mass inertia, momentum transfer) [21]. Over multi-step rollouts, kinematic consistency errors remain artificially low, masking the fact that downstream policy performance completely collapses when encountering dynamic regime shifts or physical contact boundaries [21].

Furthermore, in multi-agent environments, **ME-World** [23] demonstrated that independent egocentric video streams rapidly diverge without explicit cross-agent view synchronization mechanisms, leading to severe spatial-temporal inconsistencies across agent viewpoints [23].

## Trends and Open Problems
The field of world models is rapidly advancing toward physical AI and foundational neural simulation, yet key open challenges persist across theoretical and practical dimensions [4][25][13][17][18][21]:

1. **Bridging the Kinematic-to-Dynamic Gap**: Existing video world models prioritize photorealism over Newtonian mechanics [18][21]. Incorporating differentiable physics priors, contact-aware energy losses, or neural-symbolic constraints is critical to ensure physical fidelity in imagined trajectories [4][18][21].
2. **Persistent Latent Memory for Multi-Turn Consistency**: Current interactive video models suffer severe scene drift after 5+ interaction turns [22]. Architectures must incorporate long-term spatial keyframe memory, persistent volumetric maps, or retrieval-augmented dynamics to maintain consistency across extended horizons [16][22].
3. **Generative Synthesis vs Non-Generative Representation**: While generative models produce intuitive visual rollouts, non-generative representation models (JEPA) achieve superior FLOP efficiency, lower planning latency, and immunity to pixel noise [4][5][6][25]. Merging JEPA representation spaces with lightweight on-demand decoders remains an active research direction [4][6][25].
4. **Multi-Agent & Cross-View Grounding**: Transitioning from single-agent egocentric prediction to multi-agent collaborative world models (e.g., ME-World) requires fine-grained cross-view state synchronization and multi-perspective action conditioning [23].
5. **Unified Sim-to-Real Benchmarking**: Closed-loop evaluation of foundation world models on real robot hardware (building on DayDreamer and DreamDojo) requires standardized benchmarks that measure zero-shot policy transfer success rather than static video metrics [8][10][19].

## References
[1] World Models. arxiv. https://arxiv.org/abs/1803.10122 (2018-03-27)
[2] Dream to Control: Learning Behaviors by Latent Imagination. arxiv. https://arxiv.org/abs/1912.01603 (2019-12-03)
[3] Mastering Atari with Discrete World Models. arxiv. https://arxiv.org/abs/2010.02193 (2020-10-05)
[4] Introduction to Latent Variable Energy-Based Models: A Path Towards Autonomous Machine Intelligence. arxiv. https://arxiv.org/abs/2306.02572 (2023-06-05)
[5] Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture. arxiv. https://arxiv.org/abs/2301.08243 (2023-01-19)
[6] V-JEPA: The next step toward Yann LeCun's vision of AGI. web. https://ai.meta.com/blog/v-jepa-yann-lecun-ai-model-video/ (2024-02-15)
[7] Mastering Diverse Domains through World Models. hf-daily. https://huggingface.co/papers/2301.04104 (2023-01-10)
[8] DayDreamer: World Models for Physical Robot Learning. arxiv. https://arxiv.org/abs/2206.14176 (2022-06-28)
[9] Genie: Generative Interactive Environments. hf-search. https://huggingface.co/papers/2402.15391 (2024-02-23)
[10] DreamDojo: A Generalist Robot World Model from Large-Scale Human Videos. hf-daily. https://huggingface.co/papers/2602.06949 (2026-02-06)
[11] GAIA-1: A Generative World Model for Autonomous Driving. hf-search. https://huggingface.co/papers/2309.17080 (2023-09-29)
[12] GAIA-2: A Controllable Multi-View Generative World Model for Autonomous Driving. hf-search. https://huggingface.co/papers/2503.20523 (2025-03-26)
[13] Cosmos World Foundation Model Platform for Physical AI. hf-search. https://huggingface.co/papers/2501.03575 (2025-01-07)
[14] Driving into the Future: Multiview Visual Forecasting and Planning with World Model for Autonomous Driving. hf-search. https://huggingface.co/papers/2311.17918 (2023-11-29)
[15] GenieDrive: Towards Physics-Aware Driving World Model with 4D Occupancy Guided Video Generation. hf-search. https://huggingface.co/papers/2512.12751 (2025-12-14)
[16] DOME: Taming Diffusion Model into High-Fidelity Controllable Occupancy World Model. hf-search. https://huggingface.co/papers/2410.10429 (2024-10-14)
[17] Is Sora a World Simulator? A Comprehensive Survey on General World Models and Beyond. hf-search. https://huggingface.co/papers/2405.03520 (2024-05-06)
[18] PhysBench: Benchmarking and Enhancing Vision-Language Models for Physical World Understanding. hf-search. https://huggingface.co/papers/2501.16411 (2025-01-27)
[19] WorldScore: A Unified Evaluation Benchmark for World Generation. hf-search. https://huggingface.co/papers/2504.00983 (2025-04-01)
[20] EgoTaskQA: Understanding Human Tasks in Egocentric Videos. hf-search. https://huggingface.co/papers/2210.03929 (2022-10-08)
[21] Imagined Rollouts are Kinematic, Not Dynamic: A Diagnosis of Long-Horizon World-Model Failure. hf-search. https://huggingface.co/papers/2607.05966 (2026-07-07)
[22] WBench: A Comprehensive Multi-turn Benchmark for Interactive Video World Model Evaluation. hf-search. https://huggingface.co/papers/2605.25874 (2026-05-25)
[23] Multi-Agent Egocentric World Model with Fine-Grained Embodied Interaction (ME-World). hf-daily. https://huggingface.co/papers/2610.12299 (2026-10-08)
[24] MMWorld: Towards Multi-discipline Multi-faceted World Model Evaluation in Videos. hf-search. https://huggingface.co/papers/2406.08407 (2024-06-12)
[25] Understanding World or Predicting Future? A Comprehensive Survey of World Models. arxiv. https://arxiv.org/abs/2411.14499 (2024-11-21)
