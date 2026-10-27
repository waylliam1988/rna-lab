# config.py
# -*- coding: utf-8 -*-
import os
import sys
import math
import torch


class Config:
    VERSION = "Nezha-Bio-Min"

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(SCRIPT_DIR, "data")
    EVOLUTION_DIR = os.path.join(SCRIPT_DIR, "evolution_data")
    CHECKPOINT_DIR = os.path.join(EVOLUTION_DIR, "checkpoints")

    # Protocol dimensions
    PROTO_DIM = 256
    GHOST_CORTEX_DIM = 512
    GHOST_CORTEX_TIME_STEPS = 8

    # Device
    GHOST_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # DDP
    DDP_ENABLE = False
    DDP_BACKEND = "gloo" if os.name == 'nt' else "nccl"
    DDP_FIND_UNUSED = True
    DDP_BROADCAST_BUFFERS = False
    DDP_TIMEOUT_MIN = 30

    # -----------------------------
    # Numeric helpers
    # -----------------------------
    @staticmethod
    def _finfo(dtype: torch.dtype) -> torch.finfo:
        if dtype is None:
            dtype = torch.float32
        if not dtype.is_floating_point:
            dtype = torch.float32
        return torch.finfo(dtype)

    @staticmethod
    def py_eps() -> float:
        return float(getattr(sys, "float_info").epsilon)

    @staticmethod
    def eps(dtype: torch.dtype = None) -> float:
        return float(Config._finfo(dtype).eps)

    @staticmethod
    def tiny(dtype: torch.dtype = None) -> float:
        return float(Config._finfo(dtype).tiny)

    @staticmethod
    def max_exp_arg(dtype: torch.dtype = None) -> float:
        fin = Config._finfo(dtype)
        return float(math.log(float(fin.max)))

    @staticmethod
    def eps_like(x: torch.Tensor) -> torch.Tensor:
        if not isinstance(x, torch.Tensor):
            return torch.tensor(Config.py_eps(), dtype=torch.float32)
        return x.new_tensor(Config.eps(x.dtype))

    # -----------------------------
    # Core learning (Ghost)
    # -----------------------------
    CORE_LR = 1e-5
    CORE_CLIP_NORM = 1.0
    CORE_TRAIN_CONTROL_BUS = True
    CORE_TRAIN_INTENT_NORM = True
    CORE_TRAIN_CORTEX = False

    # -----------------------------
    # Ghost control activity settings
    # -----------------------------
    GHOST_ACTIVITY_TARGET_PER_WINDOW = 0.0
    GHOST_ACTIVITY_HOMEOSTASIS_TAU_STEPS = 256.0
    GHOST_ACTIVITY_HETEROGENEITY_CV = 0.2
    GHOST_ACTIVITY_TRACE_TAU_STEPS = 8.0
    GHOST_ACTIVITY_TRACE_TAIL_PROB = 0.0
    GHOST_ACTIVITY_THRESH_TAIL_PROB = 0.0
    GHOST_ACTIVITY_THRESH_MIN = 0.0
    GHOST_ACTIVITY_THRESH_MAX = 0.0
    GHOST_ACTIVITY_TAU_MIN_STEPS = 1.0
    GHOST_ACTIVITY_VRMS_TAU = 64.0
    GHOST_ACTIVITY_TRACE_RMS_TAU = 64.0
    GHOST_ACTIVITY_MOD_RMS_TAU = 64.0
    GHOST_ACTIVITY_OMEGA_MAX = 0.0
    GHOST_ACTIVITY_STDP = None
    GHOST_ACTIVITY_FORCE_SCALE_TAU_TOKENS = 64.0

    GHOST_ACTIVITY_ENCODER_MODE_INFER = "rate"
    GHOST_ACTIVITY_ENCODER_MODE_TRAIN = "rate"
    GHOST_ACTIVITY_PRETRACE_MODE_WHEN_RATE = "pdm"
    GHOST_ACTIVITY_PRETRACE_ENABLED_WHEN_RATE = False
    GHOST_ACTIVITY_PDM_ROUNDING = True
    GHOST_ACTIVITY_DETERMINISTIC_THRESHOLD = 0.5
    GHOST_CORTEX_USE_GRAD = True
    GHOST_CONTROL_ACTIVITY_BOUND = 8.0
    GHOST_GOAL_INPUT_SCALE = 0.35
    GHOST_RWKV_KEY_MODE = "softplus_zero"
    GHOST_RWKV_PAIR_STATE_CAP = 1.0
    GHOST_RWKV_PAIR_STATE_CAP_MODE = "soft"

    # -----------------------------
    # Ghost Cortex Auto-Unfreeze
    # -----------------------------
    # 说明：不再手动切 CORE_TRAIN_CORTEX，而是按训练步数自动放开
    # head/RWKV -> stage2 -> stage1，逐层恢复学习能力
    RNA_CORTEX_AUTO_UNFREEZE_ENABLE = True
    RNA_CORTEX_HEAD_START_STEP = 1000
    RNA_CORTEX_HEAD_WARMUP_STEPS = 1500
    RNA_CORTEX_HEAD_MAX_SCALE = 1.0
    RNA_CORTEX_HEAD_LR_SCALE = 1.0
    GHOST_CORTEX_STAGE2_START_STEP = 3000
    GHOST_CORTEX_STAGE2_WARMUP_STEPS = 2000
    GHOST_CORTEX_STAGE2_MAX_SCALE = 0.75
    GHOST_CORTEX_STAGE2_LR_SCALE = 0.5
    GHOST_CORTEX_STAGE1_START_STEP = 6000
    GHOST_CORTEX_STAGE1_WARMUP_STEPS = 3000
    GHOST_CORTEX_STAGE1_MAX_SCALE = 0.5
    GHOST_CORTEX_STAGE1_LR_SCALE = 0.25

    # -----------------------------
    # Kernel defaults (kept for kernels.py)
    # -----------------------------
    KTO_BETA_GAIN = 1.0
    KTO_KL_TAIL_PROB = 0.0
    PHYS_CURVATURE_MASS_GAIN = 2.0
    PHYS_FORCE_TAIL_PROB = 0.0
    PHYS_MAX_EXP_ARG = 0.0
    RWKV_DEN_EPS = 1e-6
    RWKV_MAX_EXP_ARG = 50.0
    SAMPLING_TEMP_BASE = 0.0
    SPINE_GHOST_RMS_TAU = 64.0
    VAE_LOGVAR_MAX_EXP_ARG = 20.0


    # -----------------------------
    # Bio++: Adaptive physics control (ported from kernels1/ghost1 ideas)
    # -----------------------------
    PHYS_ADAPTIVE_ENABLE = True

    # Riemannian / preconditioner (diagonal metric)
    PHYS_METRIC_BETA = 0.99
    PHYS_GINV_CLIP_MIN = 0.05
    PHYS_GINV_CLIP_MAX = 20.0
    PHYS_FORCE_TAIL_PROB = 0.0          # 0 => use tail_prob_default(dim)

    # Dynamic force clip (scalar clip on max|F|)
    PHYS_FORCE_CLIP_FLOOR = 0.25
    PHYS_FORCE_CLIP_CAP = 5.0

    # Modulator (learned scheduler)
    BIO_MOD_ENABLE = True
    BIO_MOD_HIDDEN = 128
    BIO_DT_SCALE_MIN = 0.90
    BIO_DT_SCALE_MAX = 1.10
    BIO_DT_SCALE_SMOOTH = 0.2  # 0=no smoothing, 1=always previous
    BIO_GAMMA_SCALE_MIN = 0.5
    BIO_GAMMA_SCALE_MAX = 2.0
    BIO_TEMP_SCALE_MIN = 0.5
    BIO_TEMP_SCALE_MAX = 2.0
    BIO_PORT_SCALE_MIN = 0.5
    BIO_PORT_SCALE_MAX = 2.0
    BIO_ALIGN_SCALE_MIN = 0.5
    BIO_ALIGN_SCALE_MAX = 1.5
    BIO_JREG_SCALE_MIN = 0.2
    BIO_JREG_SCALE_MAX = 1.0

    # Global auto loss reweighting (all auxiliary losses)
    RNA_LOSS_AUTO_ENABLE = False#True  # 自动权重
    RNA_LOSS_AUTO_TARGET_FRAC = 0.05
    RNA_LOSS_AUTO_EMA_TAU = 256.0
    RNA_LOSS_AUTO_MIN = 0.0
    RNA_LOSS_AUTO_MAX = 1.0

    # Safety
    RNA_XI_DETACH = True

    # -----------------------------
    # RNA / Bio System
    # -----------------------------
    # train_sequences.csv 包含序列信息
    RNA_DATA_PATH = os.path.join(DATA_DIR, "train_sequences.csv")
    # train_labels.csv 包含 3D 坐标信息
    RNA_COORDS_PATH = os.path.join(DATA_DIR, "train_labels.csv")
    # validation_sequences / validation_labels: 仅用于训练时额外监控，不参与 best-model 决策
    RNA_VAL_ENABLE = True
    RNA_VAL_DATA_PATH = os.path.join(DATA_DIR, "validation_sequences.csv")
    RNA_VAL_COORDS_PATH = os.path.join(DATA_DIR, "validation_labels.csv")
    RNA_VAL_SEED = 42
    RNA_VAL_MAX_SAMPLES = None
    # mmCIF 二级结构先验（从 data/PDB_RNA/*.cif 解析）
    RNA_USE_CIF_PAIR_MASK = True
    RNA_CIF_PAIR_DIR = os.path.join(DATA_DIR, "PDB_RNA")
    # mmCIF pair prior 构建的安全上限（避免超长序列申请 LxL 巨矩阵导致 OOM）
    # 例如 125580 长度会产生约 58.7 GiB 的 float32 矩阵，直接爆内存。
    RNA_CIF_PAIR_MAX_LEN = 8192
    # ---------------------------------------------------------------
    # MSA 预处理特征 (Bias Map + Guided Edges)
    # ---------------------------------------------------------------
    # Bias Map: [L, L] 共变异软偏置 (不改标签，只做 soft prior)
    RNA_MSA_BIAS_ENABLE = True
    RNA_MSA_BIAS_PATH = r"E:\Nezha\Nezha-bio\data\msa_bias"
    RNA_MSA_BIAS_SCALE = 1.0
    RNA_MSA_BIAS_DROP_P = 0.20#0.5   # 训练时 MSA Dropout: 0.5 => 一半样本不使用 MSA
    # MSA Bias -> J (默认关闭：避免模型“查表”)
    RNA_MSA_BIAS_TO_J_ENABLE = False
    RNA_MSA_BIAS_TO_J_SCALE = 0.1
    # MSA Bias -> NFF k (物理参数调制)
    RNA_MSA_BIAS_TO_NFF_ENABLE = True
    RNA_MSA_BIAS_TO_NFF_SCALE = 1.0
    RNA_MSA_BIAS_TO_NFF_MODE = "mul"  # "mul" or "add"

    # Guided Long-Range Loss (采样引导)
    RNA_MSA_GUIDED_LONG_LOSS = True
    RNA_MSA_EDGES_PATH = r"E:\Nezha\Nezha-bio\data\msa_edges"
    RNA_MSA_GUIDED_FRAC = 0.7   # long-loss 中 guided 采样比例
    RNA_MSA_GUIDED_MAX = 4096   # 每样本最多 guided pair 数 (0=全部)
    RNA_MSA_GUIDED_MIN_SCORE = 0.0
    RNA_MSA_GUIDED_DROP_P = 0.20#0.5 # 训练时 guided 也做 dropout

    # MSA Entropy (1D) & Dynamic Gating
    RNA_MSA_ENTROPY_ENABLE = True
    RNA_MSA_ENTROPY_PATH = r"E:\Nezha\Nezha-bio\data\msa_entropy"
    RNA_MSA_ENTROPY_DROP_P = 0.6
    RNA_MSA_ENTROPY_LOSS_LAMBDA = 0.0#0.05
    RNA_MSA_ENTROPY_TEMP_ALPHA = 0.5

    RNA_MSA_GATE_ENABLE = True
    RNA_MSA_GATE_USE_ENERGY = True
    RNA_MSA_GATE_MIN = 0.2
    RNA_MSA_GATE_ALPHA = 0.5
    RNA_MSA_GATE_BASE = 5.0
    # ---------------------------------------------------------------
    # PSSM (来自 MSA 的 1D 进化特征，L x 4)
    # ---------------------------------------------------------------
    RNA_PSSM_ENABLE = True
    RNA_PSSM_PATH = r"E:\Nezha\Nezha-bio\data\msa_pssm"
    RNA_PSSM_DROP_P = 0.4
    RNA_PSSM_SCALE = 1.0
    RNA_2D_BPSEQ_DIR = r"E:\Nezha\Nezha-bio\data\bpseqFiles"
    RNA_BPSEQ_CACHE_DIR = r"E:\Nezha\Nezha-bio\data\bpseq_cache"

    # ---------------------------------------------------------------
    # BPSEQ simplemix auxiliary route
    # current best strict mainline simulation:
    # 3D, BPSEQ-only, react, BPSEQ-only
    # training-time only; inference does not depend on 2D data
    # ---------------------------------------------------------------
    RNA_BPSEQ_SIMPLEMIX_ENABLE = True
    RNA_BPSEQ_SIMPLEMIX_MIN_LEN = 160
    RNA_BPSEQ_SIMPLEMIX_MAX_LEN = 512
    RNA_BPSEQ_SIMPLEMIX_PAIRCTX_WEIGHT = 0.06
    RNA_BPSEQ_SIMPLEMIX_PAIRCTX_GATE_MODE = "graph_precision"
    RNA_BPSEQ_SIMPLEMIX_TEACHER_ALPHA = 0.55
    # 独立 2D 比例：x 表示每取 1 个 3D batch，对应混入 x 个 BPSEQ-only batch
    # 当前默认赢家组合是 2D=2.0, 1D=1.0，对应 3D -> BPSEQ -> React -> BPSEQ
    RNA_BPSEQ_RATIO = 0.1#0.25#2.0
    # ---------------------------------------------------------------
    # Sparse Distillation (仅在 MSA 被 Drop 时学习 Top-K 边)
    # ---------------------------------------------------------------
    RNA_MSA_DISTILL_ENABLE = True
    RNA_MSA_DISTILL_LAMBDA = 0.0#0.1
    # ---------------------------------------------------------------
    # Reactivity 训练数据配置
    # ---------------------------------------------------------------
    RNA_REACT_DATA_PATH = r"E:\Nezha\Nezha-bio\data\train_data_QUICK_START.csv"
    # 独立 1D 比例：x 表示每取 1 个 3D batch，对应混入 x 个 React batch
    RNA_REACT_RATIO = 0.1#0.25#1.0#0.5
    
    # Reactivity Loss 的权重 (Masked MSE)
    # 当前主线不依赖 raw react regression 头带来收益，因此保持关闭。
    RNA_REACT_LAMBDA = 0.0#0.15
    RNA_REACT_USE_GEOM = False
    # Reactivity SNR weighting (基于 reactivity_error_*)
    RNA_REACT_SNR_ENABLE = True
    RNA_REACT_SNR_EPS = 1e-4
    RNA_REACT_SNR_MAX = 10.0
    RNA_REACT_SNR_MIN = 0.0

    RNA_BATCH_SIZE = 8#6#2
    RNA_ACCUM_STEPS = 2#3#16
    RNA_PHYS_DIM = 6

    # Encoder depth:
    # - 20: 扩深容量
    # - 旧 16 层 checkpoint 加载时，可配合下方 EXPAND_COPY_* 自动复制后 4 层权重做热启动
    RNA_ENCODER_LAYERS = 20
    RNA_ENCODER_EXPAND_COPY_ENABLE = True
    RNA_ENCODER_EXPAND_COPY_MODE = "tail"   # "tail" | "cycle" | "last1"
    # Data basis (see 测试脚本/analyze_config_data_basis.py):
    # validation/test 长度分布里，L<=512 覆盖率约 92.86%，在显存与覆盖率之间性价比较好。
    RNA_MAX_LEN = 512

    # -----------------------------
    # [Triangle Update] 轻量级三角更新 (Lite Evoformer)
    # -----------------------------
    RNA_TRI_ENABLE = True           # 总开关：建议开启，显存占用很低
    # 384 对 validation/test 的三角分支覆盖率约 89.29%，同时显存风险明显低于 512。
    RNA_TRI_MAX_LEN = 384           # 长度熔断：超过此长度自动跳过三角更新，防止显存爆炸
    RNA_TRI_PAIR_DIM = 32#16           # Pair通道数：16足够捕捉几何一致性
    RNA_TRI_MID_DIM = 8             # 中间层维度
    RNA_TRI_BLOCKS = 4              # 堆叠层数：2层性价比最高
    RNA_TRI_CHUNK = 48              # 分块计算大小：用于削减显存峰值
    RNA_TRI_USE_CHECKPOINT = True   # 训练时开启，用计算换显存
    RNA_TRI_RES_SCALE = 1.0         # 残差修正的缩放系数
    RNA_TRI_LOGIT_GAIN_LEARNABLE = True
    RNA_TRI_LOGIT_GAIN_INIT = 2.0
    RNA_TRI_LOGIT_GAIN_MIN = 1.0
    RNA_TRI_LOGIT_GAIN_MAX = 10.0

    RNA_RECYCLE_STEPS = 1#3 # 循环精炼次
    RNA_NFF_INIT_K = 0.05  # 神经力场初始劲度系数 (0.01=Soft, 1.0=Hard)

    # -----------------------------
    # [Root Fix] NFF d0 range (unit: nm)
    # -----------------------------
    # 标签坐标在 rna_system.py 里会做 *0.1（Å -> nm），并且 export_pdb 会把 nm *10 输出为 Å。
    # 所以内部长度单位是 nm。
    RNA_NFF_D0_MIN = 0.30     # 0.30 nm = 3 Å
    RNA_NFF_D0_MAX = 2.50     # 2.50 nm = 25 Å
    RNA_NFF_INIT_D0 = 1.00    # 初始 d0（基于统计报告）

    # NFF 稀疏化：阶段性关闭 J 门控，让力场先学习长程拓扑
    RNA_NFF_USE_J_WEIGHT = True
    RNA_NFF_J_POWER = 1.5
    RNA_NFF_J_FLOOR = 0.0
    RNA_NFF_J_SCHEDULE_ENABLE = True
    RNA_NFF_J_SCHEDULE_STEPS = 900
    RNA_NFF_J_SCHEDULE_START = 0.1
    RNA_NFF_J_SCHEDULE_END = 1.0
    RNA_NFF_K_MAX = 2.0

    # -----------------------------
    # [Root Fix] Initialization
    # -----------------------------
    # q0 初态：straight / coil / helix
    RNA_INIT_MODE = "straight"
    RNA_INIT_SPACING = 0.55        # nm，用于 straight
    # straight 模式下给出极小、确定性的 3D 厚度，避免严格 1D 退化
    RNA_INIT_STRAIGHT_THICKNESS = 0.02
    RNA_INIT_STRAIGHT_THICKNESS_FREQ = 0.1
    RNA_INIT_COIL_STEP = 0.55      # nm，用于 random-walk coil
    RNA_INIT_COIL_BIAS_Z = 0.05
    RNA_INIT_HELIX_RADIUS = 0.82   # nm
    RNA_INIT_HELIX_PITCH = 0.28    # nm per residue
    RNA_J_GLOBAL_BIAS_INIT = -1.0

    # -----------------------------
    # [Double-Sided Langevin]
    # -----------------------------
    # 说明：用“物理一致”的 Langevin 扰动替代盲目高斯噪声。
    # - Input Side：发车前松弛，主要解开原子重叠（用 NFF 能量面）
    # - Target Side：训练时扰动 GT 目标，抑制死记硬背（纯物理，不走 NFF）

    # 1) Input Side (训练用，推理 warmup 也复用)
    RNA_INPUT_LANGEVIN_ENABLE = False#True
    RNA_INPUT_LANGEVIN_T = 0.05       # 温度（越大扰动越强）
    RNA_INPUT_LANGEVIN_DT = 0.02      # 步长
    RNA_INPUT_LANGEVIN_STEPS = 2      # 每次 forward 的 warmup 步数（建议 1~4）

    # 2) Target Side (训练用：在 loss 内扰动 targets)
    RNA_TARGET_LANGEVIN_ENABLE = False#True
    RNA_TARGET_LANGEVIN_T = 0.05#0.01    # 0.05极低温度微扰，对应约 0.3~0.4 埃的真实热涨
    RNA_TARGET_LANGEVIN_DT = 0.01        # 显式设定 DT，配合 T=0.05 达到完美的步长比例
    RNA_TARGET_LANGEVIN_STEPS = 1        # 整型！1步足以产生微小抖动和局部弛豫
    RNA_TARGET_LANGEVIN_PROB = 0.5       # 50%的概率触发，保留一半原始硬标签

    # 3) 数值安全阀
    RNA_LANGEVIN_FORCE_CLIP = 5.0     # 对每个 residue 的 6D 力做 clip（0=不 clip）
    RNA_LANGEVIN_MAX_DRIFT = 0.20     # 单步 position 最大位移（nm），防止 overlap 时飞掉（0=不限制）
    RNA_LANGEVIN_POS_ONLY_NOISE = True  # True=只给位置加噪声，法线只做 drift+投影
    RNA_LANGEVIN_NOISE_CLIP_SIGMA = 3.0 # 噪声分量 clip 到 ±(sigma_clip * sigma)，0=不 clip

    # 4) REMD 推理 warmup（替代 pos_noise_std 高斯噪声）
    RNA_REMD_LANGEVIN_WARMUP_ENABLE = True
    RNA_REMD_LANGEVIN_WARMUP_STEPS = 100

    # -----------------------------
    # Advanced Physics & Loss Controls
    # -----------------------------
    # Target Langevin: 在对目标进行物理扰动时，是否切断配对吸引力 (推荐 True)
    # 这能防止模型死记硬背错误的拓扑，强迫它学习局部几何
    RNA_TARGET_LANGEVIN_J_ZERO = True

    # Atomic Clash Loss (几何排斥惩罚)
    # 即使没有 3D 标签，也要惩罚原子重叠 (Geometry Prior)
    RNA_CLASH_LAMBDA = 0.1#0.5        # 建议权重 0.1 ~ 1.0
    RNA_CLASH_DIST_NM = 0.35      # 3.5 Å 以下产生强烈惩罚

    # -----------------------------
    # Q0 Refine Head & Aux Losses
    # -----------------------------
    # 1. Refine Head 开关与参数
    RNA_Q0_REFINE_ENABLE = True    # 是否启用局部精修头
    RNA_Q0_REFINE_HIDDEN = 128     # 中间层维度
    RNA_Q0_REFINE_KERNEL = 7       # 局部感受野大小 (奇数)
    RNA_Q0_REFINE_DROPOUT = 0.1    # 防止过拟合

    # -----------------------------
    # q0 Refinement (pre-simulation small correction)
    # -----------------------------
    RNA_Q0_REFINE_MAX_STEP_INFER = 0.3   # nm，推理默认 0.3（≈3Å）
    RNA_Q0_REFINE_MAX_STEP_TRAIN = 1.0#2.0   # nm，训练可先放宽（可选）
    RNA_Q0_REFINE_WARMUP_STEPS = 0#500       # 0 表示不做 warmup 调度；需要的话填 500/1000

    # -----------------------------
    # Coarse Structure / Pair Decoupling
    # -----------------------------
    RNA_PAIR_LATENT_DIM = 32
    RNA_COARSE_HEAD_ENABLE = True
    RNA_COARSE_HEAD_HIDDEN = 256
    # pair mainline routing:
    # - coarse 路增强 pair_ctx，让 pair_latent 更容易进入初始折叠主路
    # - refine 路只注入轻量无参残差，避免把局部修正变成 pair-only 分支
    RNA_PAIR_CTX_SOURCE_SCALE = 1.0
    RNA_PAIR_CTX_COARSE_SCALE = 1.5
    RNA_PAIR_CTX_REFINE_FEAT_SCALE = 0.30
    
    # 最佳短适应语义：learnable pair->single bridge + post-moment match，仅在 refine 前消费。
    RNA_PAIR_CTX_REFINE_FEAT_RECIPE = "postmom"   # "raw" | "stdmatch" | "postmom"
    RNA_PAIR_CTX_REFINE_PROJECTOR_ENABLE = True
    RNA_PAIR_CTX_REFINE_PROJECTOR_INIT = "scale_0.35"  # "identity" | "half" | "quarter" | "zero" | "scale_xx"
    RNA_PAIR_CTX_REFINE_PROJECTOR_INIT_SCALE = 0.35
    
    RNA_HYPOTHESIS_COUNT = 5
    RNA_HYPOTHESIS_DIM = 16
    RNA_HYPOTHESIS_TRAIN_RANDOM = True
    
    RNA_MULTI_CAND_TRAIN_ENABLE =  False#True
    RNA_MULTI_CAND_TRAIN_COUNT = 5
    RNA_MULTI_CAND_TRAIN_REFINE = True
    RNA_MULTI_CAND_TRAIN_PHYSICS_ENABLE = True
    RNA_MULTI_CAND_TRAIN_SIM_STEPS = 6
    RNA_MULTI_CAND_TRAIN_TEMP = 0.05
    RNA_MULTI_CAND_TRAIN_POS_NOISE_STD = 0.015
    RNA_MULTI_CAND_DIVERSITY_ENABLE = True
    RNA_MULTI_CAND_DIVERSITY_GROWTH = 0.35
    RNA_MULTI_CAND_DIVERSITY_KEEP_MAIN = True
    RNA_MULTI_CAND_DIVERSITY_RECOMPUTE_NORMALS = True

    # -----------------------------
    # Native multibasin candidate path
    # -----------------------------
    # 原生多构象分支槽位。
    # 开启后，多候选路径不再依赖 hypothesis slots，而是由共享 head 直接生成 basin residual / prior / expand。
    RNA_NATIVE_MULTIBASIN_ENABLE = True
    RNA_NATIVE_MULTIBASIN_BRANCH_COUNT = 5
    RNA_NATIVE_MULTIBASIN_TRAIN_ENABLE = True
    RNA_NATIVE_MULTIBASIN_TRAIN_COUNT = 5
    RNA_NATIVE_MULTIBASIN_LATENT = 96
    RNA_NATIVE_MULTIBASIN_HIDDEN = 128
    RNA_NATIVE_MULTIBASIN_POS_SCALE = 0.12
    RNA_NATIVE_MULTIBASIN_AUX_DIM = 11
    RNA_NATIVE_MULTIBASIN_LEARN_TRUST = False
    RNA_NATIVE_MULTIBASIN_EXPAND_BIAS_INIT = -2.0
    RNA_NATIVE_MULTIBASIN_DELTA_REF = 0.035
    RNA_NATIVE_MULTIBASIN_DELTA_BLEND = 0.75
    RNA_NATIVE_MULTIBASIN_POLICY_BASE_BUDGET = 1
    RNA_NATIVE_MULTIBASIN_POLICY_MAX_BUDGET = 2
    RNA_NATIVE_MULTIBASIN_POLICY_TOP2_RATIO_FLOOR = 0.95
    RNA_NATIVE_MULTIBASIN_POLICY_TOP2_GAP_CEIL = 0.0075
    RNA_NATIVE_MULTIBASIN_SINGLE_ENABLE = True
    RNA_NATIVE_MULTIBASIN_SINGLE_MAX_BUDGET = 1
    RNA_NATIVE_MULTIBASIN_SORT_MODE = "quality_energy_hybrid"
    RNA_NATIVE_MULTIBASIN_SORT_ENERGY_WEIGHT = 0.02
    RNA_NATIVE_MULTIBASIN_INCUMBENT_QUALITY_MARGIN = 0.003
    RNA_NATIVE_MULTIBASIN_INCUMBENT_SWITCH_MARGIN = 0.010
    RNA_NATIVE_MULTIBASIN_INCUMBENT_WEAK_PENALTY = 0.020

    # Progressive x native multibasin unification.
    # Inference uses progressive as the incumbent path, then late-forks only the
    # final new segment through native multibasin branches.
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_ENABLE = True
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_MULTI_ENABLE = True
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_SINGLE_ENABLE = False
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_TRAIN_ENABLE = True
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_SIM_STEPS = 6
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_POS_NOISE_STD = 0.0
    RNA_PROGRESSIVE_NATIVE_LATE_FORK_REFINE = True
    
    RNA_QUALITY_HEAD_ENABLE = True
    RNA_QUALITY_HEAD_HIDDEN = 128
    RNA_QUALITY_LOSS_LAMBDA = 0.02
    RNA_QUALITY_WARMUP_STEPS = 0
    RNA_QUALITY_RANK_LOSS_LAMBDA = 0.02
    RNA_QUALITY_RANK_TARGET_MARGIN = 0.02
    RNA_QUALITY_RANK_PRED_MARGIN = 0.02
    RNA_QUALITY_RANK_WARMUP_STEPS = 0
    
    RNA_CANDIDATE_SORT_MODE = "quality_energy_hybrid"  # quality | energy | quality_energy_hybrid
    RNA_CANDIDATE_HYBRID_ENERGY_WEIGHT = 0.02
    RNA_CANDIDATE_INCUMBENT_QUALITY_MARGIN = 0.003
    RNA_CANDIDATE_INCUMBENT_SWITCH_MARGIN = 0.010
    RNA_CANDIDATE_INCUMBENT_WEAK_PENALTY = 0.020

    # 2. Q0 几何正则化 (无监督，适用于所有数据)
    # 推荐开启，特别是当你混合了 reactivity 数据时，防止无监督数据结构崩塌
    RNA_Q0_GEOM_LAMBDA = 0.01#0.1       # 总权重
    RNA_Q0_GEOM_BOND_R0 = 0.56#0.60     # 训练集相邻残基中位距离约 0.565 nm
    RNA_Q0_GEOM_BOND_BETA = 0.1    # SmoothL1 的 beta
    RNA_Q0_GEOM_CURV_LAMBDA = 0.5  # 曲率平滑权重的相对比例

    # -----------------------------
    # [Q0 Direct Supervision]
    # -----------------------------
    # 给 encoder 产出的 q0 直接加结构监督（辅助 Loss）
    # 建议权重 0.03，既能引导初始结构，又不会抢占物理引擎主导权
    RNA_Q0_STRUCT_LAMBDA = 0.05
    # Warmup 步数：前 100 步权重从 0 线性增加到目标值
    RNA_Q0_STRUCT_WARMUP_STEPS = 0#100
    # 局部监督范围：只监督 |i-j| <= 64 的距离，节省显存并专注于局部几何
    # 设为 0 表示监督全量距离矩阵（显存消耗大）
    RNA_Q0_STRUCT_LOCAL_K = 64

    # -----------------------------
    # [Debug] Gradient / safety toggles
    # -----------------------------
    RNA_ORI_NORM_NO_GRAD = False   # True 则 orientation normalize 不反传；排障期建议 False
    RNA_DEBUG_GRAD_CHECK = True    # 训练时强制检查 q.requires_grad

    # -----------------------------
    # [Annealing] (optional but recommended)
    # -----------------------------
    RNA_ANNEAL_ENABLE = True
    RNA_ANNEAL_TEMP_START = 1.5
    RNA_ANNEAL_TEMP_END = 0.05
    RNA_ANNEAL_BP_GAIN_START = 0.3
    RNA_ANNEAL_BP_GAIN_END = 1.0
    RNA_ANNEAL_STACK_K_START = 0.2
    RNA_ANNEAL_STACK_K_END = 1.0
    RNA_NFF_K_SCALE_START = 0.3
    RNA_NFF_K_SCALE_END = 1.0

    # Simulation
    RNA_SIM_STEPS = 8#64
    RNA_DT = 0.05
    RNA_GAMMA_DYN = 0.35
    RNA_PORT_GAIN = 0.05#0.02
    RNA_TARGET_TEMP = 0.0
    RNA_PHYSICS_DETERMINISTIC_TRAIN = True

    # -----------------------------
    # Progressive Mainline (default)
    # Preset below matches the current tuned progressive folding route:
    # `transition_aux_weight=0.0010, segment_kernel=13, max_segments=1`.
    # -----------------------------
    RNA_PROGRESSIVE_ENABLE = True
    RNA_PROGRESSIVE_PREFIX_START = 96
    RNA_PROGRESSIVE_PREFIX_STEP = 96
    RNA_PROGRESSIVE_MAX_STAGES = 2
    RNA_PROGRESSIVE_EARLY_STAGE_STEPS = 1
    RNA_PROGRESSIVE_CARRY_ALPHA = 0.08
    RNA_PROGRESSIVE_STAGE_POLICY = "spread"
    RNA_PROGRESSIVE_DETACH_BETWEEN_STAGES = True
    RNA_PROGRESSIVE_CARRY_MODE = "fixed"
    RNA_PROGRESSIVE_CARRY_RATIO_START = 0.30
    RNA_PROGRESSIVE_CARRY_RATIO_END = 0.85
    RNA_PROGRESSIVE_CARRY_RATIO_POWER = 1.0
    RNA_PROGRESSIVE_CARRY_DENSITY_LOW = 4.5
    RNA_PROGRESSIVE_CARRY_DENSITY_HIGH = 6.0
    RNA_PROGRESSIVE_CARRY_DENSITY_POWER = 1.0
    RNA_PROGRESSIVE_CARRY_COMBINE_MODE = "mul"
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_MOVE_LOW = 0.40
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_MOVE_HIGH = 1.20
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_FLOOR_MULT = 0.35
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_CEIL_MULT = 1.90
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_ALPHA_CAP = 0.16
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_RATIO_WEIGHT = 0.35
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_DENSITY_WEIGHT = 0.35
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_STABILITY_WEIGHT = 0.30
    RNA_PROGRESSIVE_CARRY_QUALITY_LOW = 0.32
    RNA_PROGRESSIVE_CARRY_QUALITY_HIGH = 0.38
    RNA_PROGRESSIVE_CARRY_QUALITY_POWER = 1.0
    RNA_PROGRESSIVE_CARRY_ADAPTIVE_QUALITY_WEIGHT = 0.55
    RNA_PROGRESSIVE_FRAGMENT_KERNEL = 5
    RNA_PROGRESSIVE_FRAGMENT_SHAPE = "legacy"
    RNA_PROGRESSIVE_FRAGMENT_TEMP = 0.75
    RNA_PROGRESSIVE_LOWFREQ_KERNEL = 9
    RNA_PROGRESSIVE_KEEP_FRAC = 0.25
    RNA_PROGRESSIVE_SEGMENT_KERNEL = 13
    RNA_PROGRESSIVE_MAX_SEGMENTS = 1
    RNA_PROGRESSIVE_MIN_SEGMENT_LEN = 8
    RNA_PROGRESSIVE_GAP_BRIDGE = 2
    RNA_PROGRESSIVE_SOFT_TRAIN = True
    RNA_PROGRESSIVE_SOFT_TEMP = 0.08
    RNA_PROGRESSIVE_HARD_EVAL = True
    RNA_PROGRESSIVE_CARRY_REPR = "anchor"
    RNA_PROGRESSIVE_ANCHOR_KERNEL = 33
    RNA_PROGRESSIVE_TRANSITION_SCORE_STABILITY_WEIGHT = 0.35
    RNA_PROGRESSIVE_TRANSITION_SCORE_WEIGHT = 0.35
    RNA_PROGRESSIVE_CURRENT_SUPPORT_WEIGHT = 0.15
    RNA_PROGRESSIVE_PERSIST_WEIGHT = 0.15
    RNA_PROGRESSIVE_TRANSITION_AUX_WEIGHT = 0.0010
    RNA_PROGRESSIVE_TRANSITION_AUX_KERNEL = 9
    RNA_PROGRESSIVE_TRANSITION_AUX_INIT_WEIGHT = 0.35
    RNA_PROGRESSIVE_TRANSITION_AUX_ROLL_WEIGHT = 0.65
    # paironly_m0 mainline merge:
    # stage_ctx symlog aux + full-stage hybrid pairwise rerank
    RNA_PROGRESSIVE_STAGE_CTX_AUX_ENABLE = True
    RNA_PROGRESSIVE_STAGE_CTX_AUX_WEIGHT = 0.010
    RNA_PROGRESSIVE_STAGE_CTX_AUX_LR = 5e-4
    RNA_PROGRESSIVE_STAGE_CTX_AUX_HIDDEN = 256
    RNA_PROGRESSIVE_STAGE_CTX_AUX_PHYS_WEIGHT = 1.0
    RNA_PROGRESSIVE_STAGE_CTX_AUX_PRESSURE_WEIGHT = 0.0
    RNA_PROGRESSIVE_STAGE_CTX_AUX_PHYS_FLOOR = 1e-2
    RNA_PROGRESSIVE_STAGE_CTX_AUX_PRESSURE_FLOOR = 1e-2
    RNA_PROGRESSIVE_STAGE_CTX_AUX_PHYS_CLIP = 0.0
    RNA_PROGRESSIVE_STAGE_CTX_AUX_PRESSURE_CLIP = 0.0
    RNA_PROGRESSIVE_STAGE_CTX_AUX_LOSS_SPACE = "symlog"
    RNA_PROGRESSIVE_STAGE_CTX_AUX_NORM_MODE = "ratio_add"
    RNA_PROGRESSIVE_CANDIDATE_RERANK_ENABLE = True
    RNA_PROGRESSIVE_CANDIDATE_RERANK_WEIGHT = 0.010
    RNA_PROGRESSIVE_CANDIDATE_RERANK_LR = 1e-3
    RNA_PROGRESSIVE_CANDIDATE_RERANK_HIDDEN = 128
    RNA_PROGRESSIVE_CANDIDATE_RERANK_PAIR_SCALE = 384.0
    RNA_PROGRESSIVE_CANDIDATE_RERANK_PAIR_WEIGHT = 1.0
    RNA_PROGRESSIVE_CANDIDATE_RERANK_CE_WEIGHT = 0.0
    RNA_PROGRESSIVE_CANDIDATE_RERANK_TEACHER_MARGIN = 0.0
    RNA_PROGRESSIVE_CANDIDATE_RERANK_TEACHER_MODE = "full_stage"
    RNA_PROGRESSIVE_CANDIDATE_RERANK_PREVIEW_STEPS = 1
    RNA_PROGRESSIVE_CANDIDATE_RERANK_PREVIEW_CAP_TO_STAGE = True
    RNA_PROGRESSIVE_CANDIDATE_RERANK_CANDIDATE_MODE = "global_segment_mix"
    RNA_PROGRESSIVE_CANDIDATE_RERANK_GLOBAL_SCALES = (0.0, 0.5, 1.0)
    RNA_PROGRESSIVE_CANDIDATE_RERANK_SEGMENT_TOPKS = (1, 2, 3)
    RNA_PROGRESSIVE_CANDIDATE_RERANK_SEGMENT_SCALES = (1.0,)
    RNA_PROGRESSIVE_CANDIDATE_RERANK_SEGMENT_MIN_LEN = 8
    RNA_PROGRESSIVE_CANDIDATE_RERANK_SEGMENT_REL_THRESHOLD = 0.35
    RNA_PROGRESSIVE_CANDIDATE_RERANK_EVAL_MARGIN = 0.0

    # Learning
    RNA_LR = 5e-6#2e-5
    RNA_ENCODER_Q0_PROJ_LR_SCALE = 1.0
    RNA_ENCODER_Q0_PROJ_START_STEP = 0
    RNA_ENCODER_Q0_PROJ_WARMUP_STEPS = 0

    # -----------------------------
    # LR Scheduler (Cosine Annealing)
    # -----------------------------
    # 是否启用主训练的 LR 调度器
    RNA_LR_SCHED_ENABLE = False#True
    # 调度器类型："cosine_wr" (突破平台期推荐) 或 "cosine"
    RNA_LR_SCHED_KIND = "cosine_wr"
    # cosine_wr 参数：每 T0 个 Epoch 重启一次，之后周期乘以 TMULT
    RNA_LR_WR_T0 = 10
    RNA_LR_WR_TMULT = 2
    RNA_LR_TMAX = 50#100 # cosine (非重启) 模式的总周期数
    RNA_LR_MIN = 1e-6
    # 调度步长单位："update" (每步更新) 或 "epoch" (每轮更新)
    RNA_LR_SCHED_STEP_UNIT = "update"
    # 打印频率
    RNA_LR_LOG_EVERY_UPDATES = 50

    # Cosine模式下，若为True且未恢复状态，自动推断T_max，防止学习率跳变
    RNA_LR_TMAX_AUTO = True
    # Cosine模式下，限制步数不超过T_max，防止学习率在训练后期回升
    RNA_LR_COSINE_CLAMP = True

    RNA_SAVE_EVERY_STEPS = 500 # 自动保存检查点的步数间隔 (默认为 500)
    RNA_LOSS_MIX_ENABLE = False # 关闭 SmoothL1 -> MSE 的切换，全程用 SmoothL1 (更稳健)
    RNA_LOSS_MIX_TO_MSE_STEPS = 900
    RNA_LOSS_MIX_START = 0.0
    RNA_LOSS_MIX_END = 1.0

    # Force field
    RNA_MIN_SEQ_SEP = 4  # report
    RNA_PAIR_CONTACT_THRESHOLD_NM = 1.00#0.75  # 10A gives a learnable positive density on the training set
    RNA_PAIR_CONTACT_SOFT_WIDTH_NM = 0.10  # soft transition around threshold for smoother BCE targets
    RNA_BP_R0 = 1.05     # report
    RNA_BP_SIGMA = 0.25  # report
    RNA_BP_GAIN = 1.00
    RNA_BP_TAIL_ENABLE = False
    RNA_BP_TAIL_SIGMA = 1.20
    RNA_BP_TAIL_GAIN = 0.20
    RNA_REP_R = 0.35
    RNA_REP_SOFT = 0.08
    RNA_REP_GAIN = 0.35
    RNA_REP_POWER = 2.0
    RNA_BOND_R0 = 0.56#0.58
    RNA_BOND_K = 2.00
    RNA_STACK_K = 0.5 
    RNA_BEND_K = 0.25
    RNA_CONTACT_RC = 1.0#0.9

    # Ports
    RNA_PORT_K = 8
    RNA_PORT_GAIN_TRAIN_SCALE = 1.0
    RNA_PORT_VEC_RES_ENABLE = True
    RNA_PORT_VEC_RES_SCALE = 0.05
    RNA_PORT_LOCAL_ENABLE = True
    RNA_PORT_LOCAL_HIDDEN = 64
    RNA_PORT_LOCAL_CAP = 0.18
    RNA_PORT_LOCAL_FIELD_BLEND = 0.75
    # Learnable intervention policy drives the recovery controller path.
    RNA_PORT_POLICY_ENABLE = True
    RNA_PORT_POLICY_HIDDEN = 128
    RNA_PORT_POLICY_INIT_PROB = 0.35
    RNA_PORT_POLICY_TOPK = 0
    RNA_PORT_POLICY_NEIGHBOR = 1
    RNA_PORT_POLICY_PROTECT_LEAK = 0.65
    RNA_PORT_POLICY_ABSTAIN_WEIGHT = 0.20
    RNA_PORT_LOCAL_SUPPORT_INIT_PROB = 0.15
    RNA_PORT_LOCAL_PROTECT_INIT_PROB = 0.10
    RNA_PORT_LOCAL_REPAIR_CAP = 0.18
    RNA_PORT_LOCAL_REPAIR_GAIN_SCALE = 2.00
    RNA_PORT_LOCAL_PRIOR_GAIN_SCALE = 1.00
    RNA_PORT_DEFAULT_GOAL_SUPPORT_ALPHA = 2.50
    RNA_PORT_DEFAULT_GOAL_SUPPORT_RMS_TAU = 0.020
    RNA_PORT_DEFAULT_GOAL_SUPPORT_RMS_BETA = 80.0
    RNA_PORT_DEFAULT_GOAL_SUPPORT_LEN_POWER = 3.0
    RNA_PORT_DEFAULT_GOAL_SUPPORT_LEN_REF = 16.0
    RNA_PORT_TEMP_COOL_SCALE = 0.35
    RNA_PORT_TEMP_COOL_FLOOR = 0.55
    RNA_PORT_TEMP_COOL_IMPLICIT_SCALE = 0.65
    RNA_PORT_TEMP_COOL_IMPLICIT_FLOOR = 0.50
    RNA_PORT_TEMP_COOL_IMPLICIT_PORT_GAIN_MAX = 0.50
    RNA_EXPLICIT_GOAL_CODE_CONSISTENT_SCALE = 1.00
    RNA_EXPLICIT_GOAL_CODE_INCONSISTENT_SCALE = 1.00
    RNA_EXPLICIT_GOAL_CODE_ONLY_SCALE = 1.00
    RNA_EXPLICIT_GOAL_CODE_TOKEN_BACKFILL = 0.00
    RNA_EXPLICIT_GOAL_CODE_CONSISTENCY_TAU = 0.85
    RNA_EXPLICIT_GOAL_CODE_TOKEN_ROUTE_BLEND = 0.25
    RNA_EXPLICIT_GOAL_CODE_ONLY_ROUTE_BLEND = 0.80
    RNA_EXPLICIT_GOAL_CODE_ROUTE_FOCUS_FLOOR = 0.25
    RNA_EXPLICIT_GOAL_CODE_ROUTE_FOCUS_CAP = 2.00
    RNA_EXPLICIT_GOAL_CODE_CORTEX_BLEND = 0.25
    RNA_PORT_EXPLICIT_GOAL_SUPPORT_ALPHA = 0.90
    RNA_PORT_EXPLICIT_GOAL_SUPPORT_CAP = 0.95
    RNA_PORT_EXPLICIT_GOAL_SUPPORT_TEMP = 0.08
    RNA_PORT_EXPLICIT_GOAL_LOCAL_FORCE_GAIN = 0.25
    RNA_PORT_EXEC_DETACH_MAIN_TRAIN = True
    RNA_PORT_POLICY_INTERVENE_WEIGHT = 0.25
    RNA_PORT_POLICY_SUPPORT_WEIGHT = 0.40
    RNA_PORT_POLICY_PROTECT_WEIGHT = 0.12
    RNA_RECOVERY_ROUTE_GAIN = 0.75
    RNA_RECOVERY_GATE_FLOOR = 0.08
    RNA_RECOVERY_LOCAL_GATE_FLOOR = 0.06
    RNA_RECOVERY_GATE_TEMP = 0.06
    RNA_RECOVERY_GATE_SMOOTH = 0.65
    RNA_PORT_SAFETY_ENABLE = False
    RNA_PORT_SAFETY_TRAIN_ENABLE = False
    RNA_PORT_SAFETY_SCORE_THRESH = 1.50
    RNA_PORT_SAFETY_SCORE_TEMP = 0.10
    RNA_PORT_SAFETY_GATE_FLOOR = 0.30

    # Conditioning
    RNA_COND_ENABLE = True
    RNA_COND_DIM = 32
    RNA_COND_UNKNOWN_TOKEN = "__UNK__"
    RNA_COND_USE_EXPERIMENT_TYPE = True
    RNA_COND_USE_DATASET_NAME = True

    # Target normalization
    # RNA_TARGET_NORM = "global" (用于需要预测 reactivity 的比赛)
    RNA_TARGET_NORM = "none" #"global"
    RNA_TARGET_NORM_EPS = 1e-6

    # PDB export (virtual atom for base orientation)
    # 建议使用标准元素名以提高解析器兼容性（如 "O" 或 "C"）
    RNA_PDB_VIRTUAL_ATOM = "O"
    RNA_PDB_VIRTUAL_ELEMENT = "O"

    # Diagnostics
    RNA_DIAG_GRAD_EVERY = 100 # 0为关闭, 100,表示每 100 步打印一次

    # Geometry
    RNA_GEOM_USE_INVARIANTS = True
    RNA_INV_DIM = 13
    RNA_GEOM_DIM = 4
    RNA_SUMMARY_DIM = 5

    # [GVP] Geometry-aware perception (Route A)
    RNA_GVP_PERCEPTION_ENABLE = True # 是否启用几何感知器 (GVP)，让 Ghost 能看到切向和法向

    # [Route A] Ghost 感知链路的半开梯度门
    # 说明：obs 主链和 Ghost 感知前各有一层梯度阀门，默认都走“限流回传”
    RNA_GVP_PERCEPTION_OBS_GRAD_MAX_SCALE = 0.15
    RNA_GVP_PERCEPTION_OBS_GRAD_WARMUP_STEPS = 1500
    RNA_GVP_PERCEPTION_GEOM_GRAD_MAX_SCALE = 0.08
    RNA_GVP_PERCEPTION_GEOM_GRAD_WARMUP_STEPS = 2500
    RNA_GVP_PERCEPTION_GRAD_INFER_FULL_SCALE = False

    # MLM (Masked Language Modeling) 设置
    RNA_MLM_ENABLE = True           # 是否开启 MLM 训练
    RNA_MLM_PROB = 0.15             # 遮盖概率 (15%)
    RNA_MLM_WEIGHT = 0.005#0.01#0.03            # MLM Loss 在总 Loss 中的权重

    # Cycle Consistency (Inverse Folding)
    RNA_CYCLE_ENABLE = True
    RNA_CYCLE_WEIGHT = 0.005#0.01#0.03
    RNA_CYCLE_WARMUP_STEPS = 400

    # 前期只训练 cycle 头部（inverse_head + cycle_gvp_aux），避免早期强梯度扰动 encoder/q0
    RNA_CYCLE_PRETRAIN_STEPS = 200

    # 梯度门控：grad_scale 从 0->1 的爬升步数
    RNA_CYCLE_GRAD_RAMP_STEPS = 400

    # 降低过拟合与数据先验偏置
    RNA_CYCLE_LABEL_SMOOTH = 0.05
    RNA_CYCLE_GVP_USE_FEAT = True
    RNA_CYCLE_GVP_FEAT_PROJ_DIM = 32
    RNA_CYCLE_GVP_RESIDUAL_SCALE = 0.10#0.15

    # 全局梯度裁剪（对 encoder + core_params）
    RNA_GRAD_CLIP_NORM = 5.0 #1.0

    # Ghost activity homeostasis gain (threshold update)
    GHOST_ACTIVITY_HOMEOSTASIS_GAIN = 0.5

    # REMD
    RNA_REMD_SWAP_INTERVAL = 16
    RNA_REMD_TMAX_RATIO = 4.0


    # J regularization
    RNA_J_REG_ENABLE = True
    RNA_J_REG_STRENGTH = 0.0005
    RNA_J_REG_WARMUP_STEPS = 200
    RNA_J_ROW_SUM_CAP = 4.0
    RNA_J_ENTROPY_REG = 0.01

    # Energy loss
    RNA_LAMBDA_EMA_TAU = 256.0
    RNA_ENERGY_LAMBDA = 0.5
    RNA_ENERGY_AUTO_SCALE = False
    RNA_ENERGY_TARGET_FRAC = 0.20
    RNA_ENERGY_LAM_MIN = 0.01
    RNA_ENERGY_LAM_MAX = 1.0  # 更稳健的上限，避免能量权重过大导致训练抖动
    RNA_ENERGY_WARMUP_STEPS = 0#200
    RNA_ENERGY_NORM = True
    RNA_ENERGY_NORM_MODE = "residue"   # pair/sqrt_pair/residue/none
    RNA_ENERGY_LOG_SCALE = 10.0
    RNA_ENERGY_GRAD_REDUCTION = 0.2   # 高能区线性惩罚的梯度缩放（0~1）
    RNA_FORCE_SAFE_THRESH = 2500.0
    RNA_STABILITY_LAMBDA = 0.0#0.01
    # Observation energy feature for modulator (length-robust)
    RNA_OBS_ENERGY_NORM_MODE = "residue"   # "" follows RNA_ENERGY_NORM_MODE
    RNA_OBS_ENERGY_ASINH_SCALE = 10.0

    # Ghost stabilizer
    RNA_GHOST_ALIGN_GATE = True
    RNA_GHOST_ALIGN_TAU = 64.0
    RNA_GHOST_ALIGN_FLOOR = 0.10
    RNA_GHOST_FORCE_CLIP = 2.0
    RNA_GHOST_WARMUP_STEPS = 500
    RNA_GHOST_KINETIC_SUPPRESS_THRESH = 1.0
    RNA_GHOST_MOBILITY_LOG_BOUND = 0.50
    RNA_GHOST_ORI_LOG_BOUND = 0.25
    RNA_GHOST_DAMP_MAX = 0.25
    RNA_GHOST_ORI_DAMP_MAX = 0.15
    RNA_GHOST_CONTROL_ENERGY_BUDGET = 0.20
    RNA_GHOST_CONTROLLER_LAMBDA = 0.0#0.10
    RNA_GHOST_CONTROLLER_ROLLOUT_LOSS_ENABLE = True
    RNA_GHOST_CONTROLLER_TEACHER_ONLY_START_STEP = 20000
    RNA_GHOST_EXPLICIT_CONTRACT_AUX_WEIGHT = 0.00
    RNA_GHOST_EXPLICIT_CONTRACT_ZERO_RATIO = 0.990
    RNA_GHOST_EXPLICIT_CONTRACT_WRONG_RATIO = 0.988
    RNA_GHOST_EXPLICIT_CONTRACT_TOKEN_RATIO = 0.998
    RNA_GHOST_EXPLICIT_CONTRACT_DETERMINISTIC = True
    RNA_GHOST_INTENT_NORM_FREEZE_STEP = 20000

    # Alignment loss
    RNA_ALIGN_LAMBDA = 0.0#0.05
    RNA_ALIGN_WARMUP_STEPS = 1000
    RNA_ALIGN_DETACH_FIELD = True

    # 3D Contact Map Loss (J Matrix)
    RNA_CONTACT_MAP_LAMBDA = 0.0#0.15#0.10
    RNA_CONTACT_MAP_WARMUP_STEPS = 2000

    # Inference / Validation Sampling
    RNA_EVAL_SAMPLES = 5           # 验证时的采样次数（与 submission 的 best-of-5 对齐）
    RNA_EVAL_TEMP = 1.0            # 验证时的采样温度
    RNA_EVAL_GAMMA = 0.5           # 验证时的阻尼
    RNA_INFER_BOLTZMANN_TEMP = 5.0 # 推理时能量加权的温度系数

    # 长序列 Tiling 分块推理配置
    RNA_TILE_ENABLE = True
    
    # 分块窗口大小 (必须 <= RNA_MAX_LEN，等于RNA_TRI_MAX_LEN)
    RNA_TILE_WINDOW = 384
    # 重叠区域大小 (建议 64~96，保证对齐稳定性)
    RNA_TILE_OVERLAP = 96
    
    # 针对极长序列的性能优化：根据长度自动降低模拟步数
    RNA_TILE_SIM_STEPS_SHORT = 64      # 长度 <= 2k
    RNA_TILE_SIM_STEPS_MED = 32        # 2k < 长度 <= 20k
    RNA_TILE_SIM_STEPS_LONG = 16       # 长度 > 20k

    # -----------------------------
    # Tiling + Local Refinement (Boundary polishing)
    # -----------------------------
    RNA_TILE_REFINE_ENABLE = True

    # 边界窗口半径 K：实际精修窗口大小为 2*K (例如 64*2=128)
    RNA_TILE_REFINE_K = 64

    # 边界精修的模拟步数 (建议 15-25，不宜过长以免破坏整体结构)
    RNA_TILE_REFINE_STEPS = 15

    # 写回时的安全边距 (Margin)：窗口两端各留多少残基不写回
    # 保护已对齐好的区域不被精修破坏
    RNA_TILE_REFINE_WRITE_MARGIN = 16

    # 精修温度：低温松弛 (0.05 接近绝对零度但保持微弱活性)
    RNA_TILE_REFINE_TEMP = 0.05


    # Nussinov 算法参数（当前主训练/验证路径未调用 nussinov_pairs；保留为备用参数）
    RNA_NUSSINOV_WINDOW = 300
    RNA_NUSSINOV_STRIDE = 150

    # Kabsch Loss 权重
    # 当前 resume 高步数 checkpoint 上，5.0 会让对齐项长期压过主任务；
    # 先用 1.5 保留全局拓扑约束，同时避免 late-stage 训练被坐标项主导。
    RNA_KABSCH_LAMBDA = 1.0
    RNA_KABSCH_WARMUP_STEPS = 300

    # ---- Sentinel monitor (TM-Score monitoring) ----
    # 以 train_sequences.csv 当前规模(5716)为基准，约采样 64 条哨兵：
    # 64 / 5716 ~= 0.0112
    RNA_MONITOR_RATIO = 0.0112
    RNA_MONITOR_SEED = 42
    # 固定上限，控制每轮监控开销和时间抖动
    RNA_MONITOR_MAX_SAMPLES = 64

    # ---- Train holdout (target_id-level generalization monitor) ----
    # 仅用于额外泛化监控，不参与 best-model 选择
    RNA_HOLDOUT_ENABLE = True
    RNA_HOLDOUT_RATIO = 0.05
    RNA_HOLDOUT_SEED = 42
    RNA_HOLDOUT_MAX_SAMPLES = 64

    # ---- Train window mode ----
    # 第一阶段默认走 fixed_window：长序列为每个 target 固定一个可复现窗口反复学习
    # False 时回到 coverage-like 训练：窗口随 epoch 轮换覆盖
    RNA_TRAIN_FIXED_WINDOW_ENABLE = True
    RNA_TRAIN_WINDOW_BASE_SEED = 42

    # ---- Long-target coverage crop (CPU-side only; does not increase VRAM) ----
    # 目标：长序列不要每个 epoch 只随机撞到一个窗口，而是按 stride 轮转覆盖。
    RNA_COVERAGE_CROP_ENABLE = True
    RNA_COVERAGE_CROP_STRIDE = 256
    RNA_COVERAGE_CROP_JITTER = 0

    # ---- Difficulty-aware 3D sampling (CPU-side only; does not increase VRAM) ----
    # 使用有效长度 + 接触密度 + 长程接触比例，给更难的样本更高采样概率。
    RNA_DIFF_SAMPLING_ENABLE = True
    RNA_DIFF_SAMPLING_LEN_W = 0.15
    RNA_DIFF_SAMPLING_CONTACT_W = 0.45
    RNA_DIFF_SAMPLING_LONG_W = 0.40
    RNA_DIFF_SAMPLING_REPLACEMENT = True
    RNA_DIFF_SAMPLING_MIN_W = 0.50
    RNA_DIFF_SAMPLING_MAX_W = 2.00
    RNA_DIFF_SAMPLING_Q_LOW = 0.10
    RNA_DIFF_SAMPLING_Q_HIGH = 0.90
    
    # 训练范围控制：
    # - all: 保持原主线全量训练
    # - pairctx_proj_q0_nff_phys: 最佳 tri 白嫖方案，只训 projector + q0_refine + NFF/physics
    # - pairctx_proj_q0_nff_phys_encoder_q0proj: 在上面的基础上额外放开 encoder.q0_proj
    # - pairctx_proj_q0_nff_phys_encoder: 在上面的基础上放开整个 encoder（包含 q0_proj）
    RNA_TRAIN_SCOPE = "pairctx_proj_q0_nff_phys_encoder"

    # -----------------------------
    # Flow Matching (SE(3) equivariant)
    # -----------------------------
    RNA_FLOW_ENABLE = False              # 总开关：是否启用 Flow Head（默认关闭，节省显存）
    RNA_FLOW_TRAIN_ENABLE = False        # 训练时是否计算 Flow Loss（默认关闭）
    RNA_FLOW_LAMBDA = 0.40#0.08#0.05#0.02  # keep Flow useful, but stop it from dominating the main geometry loss

    # FlowRobust: 允许 Flow Head 获得梯度（禁用硬跳过），仅做软压缩
    RNA_FLOW_LOSS_CAP = 80.0        # 超过此值进行软压缩 (保留梯度方向)
    RNA_FLOW_LOSS_SKIP_TH = 0.0     # 0=禁用硬跳过，避免 Flow Head 长期空转

    # Flow Loss Warmup：像其它模块一样，训练前期从 0 逐步打开，避免早期梯度不稳定
    # - 设为 0 表示不做 warmup（立即生效）
    # - kind: "linear" 或 "cosine"
    RNA_FLOW_WARMUP_STEPS = 2000
    RNA_FLOW_WARMUP_KIND = "cosine"
    RNA_FLOW_LOSS_FAILSAFE = True              # True=Flow loss 出错/NaN 时自动跳过并报警
    RNA_FLOW_WARMUP_DELAY_STEPS = 300
    
    # Flow 能量一致性正则（轻量物理约束，默认关闭）
    RNA_FLOW_ENERGY_REG_ENABLE = True  # False 暂时关闭物理引擎的梯度拉扯，看纯 MSE 极限
    RNA_FLOW_ENERGY_REG_LAMBDA = 0.001#0.05#0.01    # 正则强度（建议 0.005~0.05）
    RNA_FLOW_ENERGY_REG_PROB = 0.25      # 触发概率（降低计算开销）
    RNA_FLOW_ENERGY_REG_MAX_LEN = 512    # 超长序列可跳过（0=不限制）


    # J 矩阵的 Dropout 概率 (建议 0.10 ~ 0.20)
    # 迫使 Flow Head 在没有接触图信息时也能工作，增强鲁棒性
    RNA_FLOW_J_DROPOUT_PROB = 0.10      # 0.0 暂时给模型全天候的 2D 视力，消除盲猜方差
    RNA_FLOW_HIDDEN = 256               # EGNN 内部隐藏层维度
    RNA_FLOW_LAYERS = 4                 # 消息传递层数
    RNA_FLOW_USE_J = True               # 是否利用 Encoder 的 J 矩阵作为边特征
    RNA_FLOW_J_POWER = 1.0              # J 的指数调节

    # ================================
    # Flow on R^3 x S^2 (Manifold Flow)
    # ================================
    RNA_FLOW_S2_ENABLE = False         # S2 法线预测开关
    RNA_FLOW_S2_LOSS_LAMBDA = 1.0      # 法线 Loss 权重 0.05 → 0.1 → 0.25 → 1.0
    RNA_FLOW_S2_FD_EPS = 1e-3          # 训练差分 epsilon
    RNA_FLOW_PREDICT_NORMAL = False    # 显式预测 Normal开关
    RNA_FLOW_SLERP_EPS = 1e-6
    RNA_FLOW_NORMAL_LOSS_WEIGHT = 0.25 # 法线损失权重
    RNA_FLOW_S2_STITCH_MISSING_NORMAL = True # 缺失坐标区域强制 n1=n0，避免伪旋转目标

    # =========================
    # Flow 注入 (inject) 行为开关
    # =========================
    RNA_FLOW_POS_UPDATE_MODE = "residual"   # "residual" | "overwrite"
    RNA_FLOW_POS_UPDATE_SCALE = 0.05        # residual 模式下吸收比例（建议 0.05~0.20）
    # Dynamic per-residue confidence gate for Flow injection:
    # q_init = q0 + gate * (q_flow - q0), where gate is predicted per residue.
    # Gate is initialized to recover static behavior (around RNA_FLOW_POS_UPDATE_SCALE).
    RNA_FLOW_CONF_GATE_ENABLE = True
    RNA_FLOW_CONF_GATE_HIDDEN = 128
    RNA_FLOW_CONF_GATE_DETACH_FEAT = True
    # gate = base_scale * (1 + gain * (2*raw-1)); raw in [0,1]
    RNA_FLOW_CONF_GATE_GAIN = 1.0
    RNA_FLOW_CONF_GATE_MIN = 0.0
    RNA_FLOW_CONF_GATE_MAX = 1.0
    # Auxiliary supervision to train gate confidence during flow matching.
    RNA_FLOW_CONF_GATE_AUX_ENABLE = True
    RNA_FLOW_CONF_GATE_AUX_LAMBDA = 0.02
    RNA_FLOW_CONF_GATE_AUX_TAU = 0.25
    RNA_FLOW_INJECT_DETACH_QFLOW = True     # 注入阶段是否切断 q_flow 反传
    RNA_FLOW_INJECT_CLONE_CURRENT_Q = True  # 注入时是否 clone current_q（避免 in-place 污染）
    RNA_FLOW_INJECT_FORCE_DERIVE_NORMALS = True  # 强制由坐标重建法线（忽略 Flow 法线）
    RNA_FLOW_INJECT_RECENTER = True         # 注入后做 masked recenter，抑制整体漂移

    # ================================
    # Memory Optimization (Sparse Loss)
    # ================================
    RNA_STRUCT_LOSS_MODE = "knn"       # 结构损失模式: "knn" (省显存) 或 "dense"
    RNA_STRUCT_LOCAL_K = 64#32            # 局部窗口 K (只监督前后 K 个邻居)
    RNA_STRUCT_BETA = 5.0              # SmoothL1 Beta
    RNA_STRUCT_LONG_MIN_SEP_EXTRA = 4  # 长程采样最小距离 = local_k + extra
    RNA_STRUCT_LONG_DEDUP = True       # 长程采样去重（降低噪声）
    RNA_CLASH_CHUNK = 128              # 排斥力计算分块大小

    # best-of-5 / TM-like 辅助（默认开启，低权重）
    RNA_BEST_OF_AUX_ENABLE = True
    RNA_BEST_OF_AUX_WEIGHT = 0.1
    RNA_BEST_OF_AUX_WARMUP_STEPS = 0
    RNA_BEST_OF_TAU = 0.5
    RNA_BEST_OF_BETA = 10.0

    RNA_TM_AUX_ENABLE = True
    RNA_TM_AUX_WEIGHT = 0.10
    RNA_TM_AUX_WARMUP_STEPS = 0
    RNA_TM_AUX_TAU = 0.5
    
    # ================================
    # Path Loss (Physics Supervision)
    # ================================
    RNA_PATH_LOSS_ENABLE = False#True
    RNA_PATH_LOSS_WEIGHT = 0.02
    RNA_PATH_LOSS_WARMUP_STEPS = 0#900
    RNA_PATH_LOSS_V_COEF = 1.0         # 惩罚势能积分 (寻找低能路径)
    RNA_PATH_LOSS_H_COEF = 0.0         # 惩罚总能量积分 (Hamiltonian),最小作用量原理设为0
    RNA_PATH_LOSS_W_COEF = 0.1         # 惩罚 Ghost 做功 (最小功原理)

    # 做功惩罚模式 (W_Mode): 决定 Ghost 如何看待“控制代价”
    # "abs"      : (默认) 绝对值惩罚。无论推还是拉，只要出力就产生 Loss。
    #              -> 效果：最严格的“最小干预”，迫使 Ghost 寻找最顺滑的自然路径。
    # "relu"     : 仅惩罚正功 (消耗能量)。如果顺应势能滑行 (做负功/被拖拽)，不惩罚。
    #              -> 效果：允许 Ghost "借力打力"，利用物理场加速，只在逆势时受罚。
    # "softplus" : Relu 的平滑版 (Log-Sum-Exp)。全程可导，对负功有微弱惩罚。
    #              -> 效果：比 Relu 训练更稳定，梯度更平滑，推荐用于精细调优。
    RNA_PATH_W_MODE = "relu"
    RNA_PATH_LOG_SCALE = 10.0

    # -----------------------------
    # GNO-style Kernel Force Head (Neural Operator-ish)
    # -----------------------------
    RNA_GNO_ENABLE = True            # 总开关 (设为 True 以启用)
    RNA_GNO_FORCE_SCALE = 0.05#0.01        # ΔF 的总增益 (建议 0.05~0.20)
    RNA_GNO_WARMUP_STEPS = 0#900        # 约 10 epoch 完成 warmup
    RNA_GNO_TOPK = 32                 # 稀疏交互：每个残基只看 J 最大的 K 个邻居
    RNA_GNO_HIDDEN = 128              # MLP 隐藏层维度
    RNA_GNO_LAYERS = 2                # MLP 层数
    RNA_GNO_J_POWER = 1.0             # J 权重的指数 (例如 2.0 会让强配对更重要)

    # Graph Construction (Sparse)
    RNA_FLOW_GRAPH_MODE = "knn"         # 图构建模式: "knn" 或 "radius" (推荐 knn)
    RNA_FLOW_KNN_K = 24                 # KNN 的 K 值 (每节点邻居数)
    RNA_FLOW_KNN_J_TOPK = 12            # 混合图：每个目标节点额外注入的 Top-J 长程边数（0=关闭）
    RNA_FLOW_KNN_J_MIN = 0.05           # 混合图：注入边的最小 J 阈值（过滤弱边）
    RNA_PAIR_INJECT_ENABLE = True       # 将外部 pair_mask 先验注入图构建权重
    RNA_PAIR_INJECT_MODE = "max"        # "max" | "add"
    RNA_PAIR_INJECT_WEIGHT = 0.35       # 注入强度（建议 0.2~0.6）
    RNA_PAIR_INJECT_MIN_SEP = 4         # 仅保留 |i-j|>=min_sep 的配对先验
    RNA_PAIR_INJECT_DROPOUT_P = 0.10    # 训练时先验边随机丢弃比例（验证/推理自动关闭）
    RNA_PAIR_INJECT_SAMPLE_DROP_P = 0.10  # 训练时整张先验图随机关闭的样本比例
    # 先验可信度估计 + 自适应注入强度（降低错误先验污染）
    RNA_PAIR_INJECT_ADAPTIVE_ENABLE = True
    RNA_PAIR_INJECT_ADAPTIVE_MIN_SCALE = 0.05   # conf=0 时的强度比例（相对 RNA_PAIR_INJECT_WEIGHT）
    RNA_PAIR_INJECT_ADAPTIVE_MAX_SCALE = 1.00   # conf=1 时的强度比例
    RNA_PAIR_INJECT_DENSITY_TARGET = 0.08       # 经验先验密度目标（长程区域平均值）
    RNA_PAIR_INJECT_DENSITY_TOL = 0.10          # 偏离容忍度（越小越严格）
    RNA_PAIR_INJECT_AGREE_WEIGHT = 0.35         # 与模型 J 一致性的权重上限
    RNA_PAIR_INJECT_AGREE_WARMUP_STEPS = 4000   # 一致性权重 warmup（早期减少自举偏差）
    RNA_PAIR_TARGET_SOFT_ENABLE = True          # pair_target 使用多构象软聚合
    RNA_PAIR_TARGET_MAX_CONFS = 2               # 聚合的最大构象数（CPU 侧，尽量不增显存）
    RNA_PAIR_TARGET_MIN_SEP = 4                 # pair_target 监督仅保留 |i-j|>=min_sep
    RNA_PAIR_TARGET_POS_WEIGHT = 10.0           # pair_target BCE 的正样本权重
    RNA_FLOW_DIST_CHUNK = 256           # KNN 计算时的分块大小 (防显存溢出)
    RNA_FLOW_RADIUS = 2.5               # 半径图阈值 (nm)
    RNA_FLOW_MAX_EDGES_PER_NODE = 64    # 半径图最大边数限制

    # Graph Enhancements
    RNA_FLOW_GRAPH_BIDIR = True              # KNN/Radius 图是否双向化（推荐 True）
    RNA_FLOW_GRAPH_CACHE_ENABLE = True       # 是否在 ODE 采样中缓存邻接（速度优化，推荐 True）
    RNA_FLOW_GRAPH_REBUILD_EVERY = 3         # 每 N 次 forward 重建一次邻接（Cache 模式下）
    
    # Adaptive & Scaling (Flow + Physics)
    RNA_FLOW_DIST_CHUNK_ADAPT = False#True         # 根据序列长度自适应 Chunk 大小
    RNA_FLOW_X1_SCALE = 1.0                  # 坐标缩放因子（默认为 1.0）
    RNA_FLOW_INIT_FOR_PHYSICS = False#True    # 推理时：先用 Flow 生成粗结构，再交给物理引擎
    RNA_FLOW_PHYSICS_REFINE_STEPS = 20       # Flow 生成后，物理引擎精修的步数
    RNA_FLOW_PHYS_SOFT_LANDING_ENABLE = False # Flow->Physics 过渡缓冲总开关
    RNA_FLOW_PHYS_SOFT_LANDING_STEPS = 4      # 软着陆生效的物理步数
    RNA_FLOW_PHYS_SOFT_LANDING_DT_SCALE = 0.25 # 软着陆时 dt 乘子 (0,1]
    RNA_FLOW_PHYS_SOFT_LANDING_GAMMA_SCALE = 4.0 # 软着陆时 gamma 乘子 (>=1)

    # Inference Strategy
    RNA_FLOW_INFER_STEPS = 50           # 推理时的 ODE 积分步数
    RNA_FLOW_INFER_USE_HEUN = True      # 使用 Heun 方法 (二阶) 还是 Euler (一阶)
    RNA_FLOW_INIT_STD = 1.0             # 初始高斯噪声的标准差
    
    # =========================================================
    # Bridge Flow: use encoder q0 as flow start (no extra VRAM)
    # =========================================================
    # True: x0/x_init uses encoder q0 (detached, no extra training graph/VRAM)
    # False: fallback to original Gaussian x0
    RNA_FLOW_BRIDGE_ENABLE = True    # 必须开启！让 4 层网络只需做“塑形”而不是“从零创造”

    # Bridge 起点破对称噪声（单位 nm）
    RNA_FLOW_BRIDGE_NOISE_STD = 0.01#0.15

    # If q0 has normal (q[...,3:6]), whether to use q0 normal as n0 in inference
    RNA_FLOW_BRIDGE_USE_Q0_NORMAL = True

    # steps<=0: truly disable flow integration and return q0 directly
    RNA_FLOW_INFER_STEPS_ALLOW_ZERO = True

    # Optional: disable J for early inference time to avoid early mismatch
    # 0.0 disables warmup; 0.2 means J=None when t<0.2
    RNA_FLOW_J_WARMUP_T = 0.2

    # 观测向量是否切断物理梯度（稳定/省显存）
    RNA_OBS_DETACH = False
    RNA_OBS_GRAD_MAX_SCALE = 0.15
    RNA_OBS_GRAD_WARMUP_STEPS = 1500
    RNA_OBS_MICRO_GRAD_MAX_SCALE = 0.10
    RNA_OBS_MICRO_GRAD_WARMUP_STEPS = 2500
    RNA_OBS_ENERGY_GRAD_MAX_SCALE = 0.05
    RNA_OBS_ENERGY_GRAD_WARMUP_STEPS = 3000
    RNA_OBS_ENERGY_GRAD_INFER_FULL_SCALE = False
    RNA_OBS_GRAD_INFER_FULL_SCALE = False
    RNA_FLOW_CENTER_EACH_STEP = True    # 每步去质心 (消除平移漂移)

    # -----------------------------
    # Checkpoint Loading Strategy (存档加载策略)
    # -----------------------------
    
    # === 场景说明 (请根据当前情况修改下面的 True/False) ===
    
    # 【场景 1: 正常断点续训】 (最常用)
    #   情况：训练跑了一半断电了/显存爆了/被杀后台了，想接着刚才的进度丝滑继续。
    #   设置：全部 True
    #
    # 【场景 2: 架构升级 / 微调】 
    #   情况：代码改了(加了GNO/Flow)，或者参数变了，想复用旧权重，但优化器需要重置以适应新参数。
    #   设置：Step=True(为了看日志连续), 其它=False(重置优化器状态)
    #
    # 【场景 3: 彻底重头再来】
    #   情况：只想借用旧权重的“神韵”做初始化，但步数归零，当作一个全新的训练任务。
    #   设置：全部 False
    
    # === 当前配置区 ===
    
    # 1. 恢复步数: True = 日志从断点处继续(如 Step 50001); False = 日志从 Step 0 开始
    #    (建议: 只要不是换数据集，通常设为 True 方便看总进度)
    RNA_LOAD_GLOBAL_STEP = True
    
    # 2. 恢复优化器(AdamW状态):
    #    当前主线已支持按参数组/形状做部分恢复；断点续训默认开启。
    #    如果你是刻意换训练任务或想重置动量，再手动设为 False。
    RNA_LOAD_OPTIM = True
    
    # 3. 恢复调度器(LR Scheduler): 
    #    (建议: 配合优化器，如果重置了优化器，最好也重置调度器)
    RNA_LOAD_SCHEDULER = True
    
    # 4. 恢复混合精度(Scaler): 
    #    (建议: 只要重置了优化器，这个也可以重置，影响不大)
    RNA_LOAD_SCALER = True

    # 5. 统计量加载开关
    # True = 断点续训 (使用 ckpt 中的统计量，保持分布一致)
    # False = 微调新数据 (使用当前数据的统计量，适应新分布)
    RNA_LOAD_NORM_STATS = True
