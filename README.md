# Causal HIL-RL: 基于因果模型可控奖励引导的在线强化学习

本仓库（`causal_code`）是面向真实机器人操作任务的 **Human-in-the-Loop Reinforcement Learning (HIL-RL)** 完整可部署代码，基于纯 **JAX / Flax + Agentlace 异步 Actor-Learner** 架构，集成了 **Distilled-Object CID Causal Reward（离线蒸馏目标定位 + 纯低维 13D 5-MLP Ensemble + 同方差基底/认知方差修正 CID 因果奖励）**。

---

## 核心特性

- **离线蒸馏“小眼睛” + “小脑”，在线零外部 Tracker**：
  - 仅在训练前对 demo 离线执行一次目标物体提取（支持 5 折交叉验证与 `npy_coords` 原始帧对齐），利用冻结 ResNet-10 + SpatialSoftmax 保留空间位置信息，蒸馏出纯 JAX 轻量视觉定位头 $g_\phi$ 与 5-MLP Ensemble 动力学模型；
  - 动作保持原始有界尺度 `[-1, 1]`（不做 demo `std` 放大），状态标准化设置下限并 `clip(-5, 5)` 提升在线鲁棒性；
  - 在线 RL 阶段 **0 外部 Tracker、0 PyTorch 依赖**，Actor 10Hz 实机控制完全零改动。
- **确定性固定动作集 & 零 JIT 重编译**：
  - 使用固定 $K=128$ 反事实动作库（Common Random Numbers），相同状态输出 100% 确定、低方差的 $C(s)$；
  - 在线推理统一 pad 到固定 `jit_batch_size`（默认 32），XLA 全程仅编译 1 次，彻底消除动态 batch 反复重编译卡顿。
- **势函数塑形（PBRS）+ 因果优先经验回放（CID-PER），双开关独立可控**：
  - `TrainerServer` 后台线程仅写原始数据（0 GPU 竞争）；Learner 主线程仅对新增未打分样本成对计算 $(C(s_t), C(s_{t+1}))$ 并缓存在 Buffer 的 `cid_scores` / `next_cid_scores` 中；
  - 默认同时开启 **CID 奖励塑形**（`use_cid_reward=True`，Ng et al. 1999 PBRS `F(s, s') = gamma * mask * Phi(s') - Phi(s)`）与 **CID 优先经验回放**（`use_cid_replay=True`，基于排名的 $p_i \propto \operatorname{rank}(i)^{-\alpha}$ + 均匀混合防饥饿，默认作用于 online replay buffer）；
  - 两个开关完全解耦，可通过配置、环境变量或 CLI 参数自由组合消融：
    - 两者全开（默认）：`USE_CID_REWARD=1 USE_CID_REPLAY=1`
    - 仅因果奖励：`USE_CID_REWARD=1 USE_CID_REPLAY=0`（或 `--use_cid_reward=true --use_cid_replay=false`）
    - 仅因果经验回放：`USE_CID_REWARD=0 USE_CID_REPLAY=1`（或 `--use_cid_reward=false --use_cid_replay=true`）
    - 纯 RLPD 基线：`USE_CID_REWARD=0 USE_CID_REPLAY=0`（或 `--use_cid_reward=false --use_cid_replay=false`）

---

## 目录结构

```text
causal_code/
├── serl_launcher/
│   └── serl_launcher/
│       ├── agents/continuous/      # 纯净 SAC / Hybrid SAC+Grasp-DQN 策略
│       ├── data/                   # 支持 cid_scores 线程安全增量懒打分的 ReplayBuffer & DataStore
│       ├── vision/                 # 冻结 ImageNet-1K ResNet-10 视觉编码器
│       └── utils/cid_reward_utils.py  # 纯 JAX CID 因果模型离线训练 & 在线打分核心库
├── serl_robot_infra/               # UR / Franka 实机驱动、RealSense 相机、SpaceMouse 干预封装
├── examples/
│   ├── prepare_causal_model.py     # 一键离线清洗 Demo、训练 cid_bundle.pkl 与留出验证可视化
│   ├── train_rlpd.py               # 异步 Actor-Learner RLPD + CID 因果奖励在线训练主脚本
│   ├── record_demos.py             # SpaceMouse 演示录制脚本
│   └── experiments/pick_place_banana/
│       ├── config.py               # 任务与 CID 奖励超参配置
│       ├── run_prepare_causal.sh   # Step 1: 离线准备因果模型
│       ├── run_tmux.sh             # Step 2: 启动 Learner + Actor 分屏训练
│       ├── run_learner.sh          # 单独启动 Learner
│       ├── run_actor.sh            # 单独启动 Actor
│       └── run_eval.sh             # 实机评估脚本
└── tests/
    └── test_cid_reward_pipeline.py # 自动化单元测试（含无 JAX 与完整 JAX 两层测试）
```

---

## 快速上手（Linux GPU 真机部署）

### 1. 安装环境

```bash
conda create -n causal_hil python=3.10 -y
conda activate causal_hil

# 安装 JAX (CUDA 12 示例)
pip install --upgrade "jax[cuda12]==0.6.2"

# 安装本地包
cd serl_launcher && pip install -e . && pip install -r requirements.txt && cd ..
cd serl_robot_infra && pip install -e . && cd ..
```

*(或使用仓库自带的 `uv sync` / `env_create.sh` 复现环境。)*

### 2. 运行离线准备脚本（从 Demo 生成 `cid_bundle.pkl` 并检查验证图）

```bash
cd examples/experiments/pick_place_banana

export DEMO_PATH=/abs/path/to/pick_place_banana_10_demos.pkl
export OBJ_EXTRACTOR=hsv_banana          # 可选: hsv_banana | hsv_red_ring | hsv_eggplant | npy_coords
export OUTPUT_DIR=./cid_causal_ckpt/pick_place_banana

bash run_prepare_causal.sh
```

运行完成后将输出：
- `./cid_causal_ckpt/pick_place_banana/cid_bundle.pkl`（冻结因果模型包）
- `./cid_causal_ckpt/pick_place_banana/cid_metrics.json`（留出集定位 $R^2$、动力学 $R^2$、AUC、信噪比）
- `./cid_causal_ckpt/pick_place_banana/pick_place_banana_cid_validation.png`（关键帧定位与 `r_bonus` 可视化诊断图）

> 若不提前运行此脚本，也可以直接启动 `run_tmux.sh`：Learner 在未检测到 `CID_MODEL_PATH` 时会自动使用 `--demo_path` 在 `<checkpoint_path>/cid_causal_model/cid_bundle.pkl` 完成一次性预训练。

### 3. 启动在线 HIL-RL 训练（Learner + Actor TMUX 分屏）

```bash
cd examples/experiments/pick_place_banana

export NEW_RUN=1
export RUN_ID=banana_cid_$(date +%Y%m%d_%H%M%S)
export DEMO_PATH=/abs/path/to/pick_place_banana_10_demos.pkl
export CID_MODEL_PATH=./cid_causal_ckpt/pick_place_banana/cid_bundle.pkl

export WANDB_API_KEY=your_wandb_api_key
export UR_ROBOT_IP=192.168.56.101
export HAND_SERIAL=218622271809
export EXTERNAL_SERIAL=254622076156

./run_tmux.sh
```

### 4. 运行自动化单元测试

```bash
pytest tests/test_cid_reward_pipeline.py
```
