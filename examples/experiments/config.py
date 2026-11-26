from abc import abstractmethod
from typing import List

class DefaultTrainingConfig:
    """Default training configuration. """

    agent: str = "drq"
    max_traj_length: int = 100
    batch_size: int = 128
    cta_ratio: int = 2
    discount: float = 0.97

    max_steps: int = 1000000
    replay_buffer_capacity: int = 200000

    random_steps: int = 0
    training_starts: int = 100
    steps_per_update: int = 50

    log_period: int = 10
    eval_period: int = 2000

    # "resnet" for ResNet10 from scratch and "resnet-pretrained" for frozen ResNet10 with pretrained weights
    encoder_type: str = "resnet-pretrained"
    demo_path: str = None
    checkpoint_period: int = 0
    buffer_period: int = 0

    eval_checkpoint_step: int = 0
    eval_n_trajs: int = 5

    image_keys: List[str] = None
    classifier_keys: List[str] = None
    proprio_keys: List[str] = None
    
    # "single-arm-learned-gripper", "dual-arm-learned-gripper" for with learned gripper, 
    # "single-arm-fixed-gripper", "dual-arm-fixed-gripper" for without learned gripper (i.e. pregrasped)
    setup_mode: str = "single-arm-fixed-gripper"

    # Distilled-Object CID Causal Reward & Experience Replay configuration.
    # Independent toggles (both default True) for ablation:
    #   - use_cid_reward: Potential-Based CID reward shaping
    #   - use_cid_replay: Causal Influence-based Prioritized Experience Replay (CID-PER)
    use_cid_reward: bool = True
    use_cid_replay: bool = True
    cid_replay_mode: str = "ranking"  # "ranking" | "softmax" | "proportional"
    cid_replay_alpha: float = 0.6     # priority exponent / temperature (0.0 -> uniform)
    cid_replay_eps: float = 1e-3      # floor epsilon for proportional mode
    cid_replay_uniform_mix: float = 0.2  # uniform safety mixture to avoid starvation
    cid_replay_on_demo: bool = False  # apply prioritization to online buffer by default
    cid_model_path: str = ""
    cid_obj_extractor: str = "hsv_banana"
    cid_external_coords_path: str = ""
    cid_external_key: str = "external"
    cid_wrist_key: str = "wrist"
    cid_pca_dim: int = 32
    cid_ensemble_size: int = 5
    cid_obj_steps: int = 800
    cid_dyn_steps: int = 1200
    cid_action_samples: int = 128
    cid_action_sampling_mode: str = "uniform"  # "uniform" | "support"
    cid_jit_batch_size: int = 32
    cid_state_min_std: float = 0.05
    cid_state_clip: float = 5.0
    cid_use_spatial_features: bool = True
    cid_cv_folds: int = 5
    cid_sigma0_sq: float = 0.25
    cid_ood_beta: float = 2.0
    cid_bonus_weight: float = 0.05
    cid_bonus_min_weight: float = 0.0
    cid_bonus_max: float = 0.5
    cid_bonus_decay_steps: int = 15000
    cid_reward_mode: str = "potential"  # "potential" (PBRS) | "additive_decay" | "step_penalty_capped"
    cid_step_penalty: float = 0.02

    @abstractmethod
    def get_environment(self, fake_env=False, save_video=False, classifier=False):
        raise NotImplementedError
    
    @abstractmethod
    def process_demos(self, demo):
        raise NotImplementedError
    
