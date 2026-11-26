"""Pure-JAX Distilled-Object CID Causal Reward module for real-world HIL-RL.

Key engineering features:
- Demo cleaning with preserved start indices (keeps external `npy_coords` aligned).
- Tight HSV object masks with largest connected-component filtering.
- Frozen ImageNet ResNet-10 with SpatialSoftmax + global pooling (preserves 2D position info).
- No demo-std amplification on actions (actions stay in bounded [-1, 1] scale).
- State standardization with `state_min_std` floor and [-5, 5] clipping for online OOD stability.
- Fixed deterministic action bank (Common Random Numbers, K=128 default) -> repeatable C(s).
- Fixed static batch padding in OnlineCIDScorer -> compiles JAX XLA graph exactly ONCE (0 recompiles).
- Potential-Based Reward Shaping (PBRS): F(s, s') = gamma * mask * Phi(s') - Phi(s) (default),
  plus optional `additive_decay` and `step_penalty_capped` modes.
- 5-fold trajectory cross-validation evaluated against independent GT object motion ||Delta s_o||.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
import pickle as pkl
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover - optional for headless tests without cv2
    cv2 = None


# =============================================================================
# 1. Pure-NumPy Demo Cleaning, Object Extractors, PCA, and Reward Shaping
# =============================================================================


def clean_demo_trajectories(
    trajectories: Sequence[List[Dict[str, Any]]],
    action_eps: float = 0.02,
    grip_eps: float = 0.5,
    return_start_indices: bool = False,
):
    """Trim leading stationary frames (e.g., human hand placing objects at t=0..15).

    If `return_start_indices=True`, returns `(cleaned_trajs, kept_raw_indices, start_indices)`
    so external coordinate arrays indexed by raw demo frames stay 100% aligned.
    """
    cleaned: List[List[Dict[str, Any]]] = []
    kept_raw_indices: List[int] = []
    start_indices: List[int] = []
    for raw_idx, traj in enumerate(trajectories):
        if not traj:
            continue
        actions = np.stack(
            [np.asarray(t["actions"], dtype=np.float32).reshape(-1) for t in traj],
            axis=0,
        )
        xyz_dim = min(3, actions.shape[1])
        xyz_moving = np.linalg.norm(actions[:, :xyz_dim], axis=1) > float(action_eps)
        if actions.shape[1] > 3:
            grip_moving = np.abs(actions[:, -1]) > float(grip_eps)
            moving = np.where(xyz_moving | grip_moving)[0]
        else:
            moving = np.where(xyz_moving)[0]
        start = max(0, int(moving[0]) - 1) if len(moving) > 0 else 0
        sub = traj[start:]
        if len(sub) >= 2:
            cleaned.append(sub)
            kept_raw_indices.append(raw_idx)
            start_indices.append(start)
    if return_start_indices:
        return cleaned, kept_raw_indices, start_indices
    return cleaned


def split_flat_transitions(transitions: Sequence[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    """Split a flat transition list into episode trajectories on `dones`."""
    trajs: List[List[Dict[str, Any]]] = []
    current: List[Dict[str, Any]] = []
    for transition in transitions:
        current.append(transition)
        if bool(transition.get("dones", False)):
            trajs.append(current)
            current = []
    if current:
        trajs.append(current)
    return trajs


def _require_cv2(name: str):
    if cv2 is None:
        raise ImportError(
            f"OpenCV (cv2) is required for HSV object extractor '{name}'. "
            "Install opencv-python or pass external .npy object coordinates."
        )


def _keep_largest_component(mask_u8: np.ndarray, min_area: int = 6) -> np.ndarray:
    """Keep only the largest connected component to reject stray rim/background noise."""
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
    if num_labels <= 1:
        return mask_u8
    areas = stats[1:, cv2.CC_STAT_AREA]
    best_label = int(np.argmax(areas)) + 1
    if int(areas[best_label - 1]) < min_area:
        return np.zeros_like(mask_u8)
    return (labels == best_label).astype(np.uint8)


def _hsv_banana_mask(img_rgb: np.ndarray) -> np.ndarray:
    """Yellow banana mask (H in [20, 38] excludes the orange pan rim at H ~ 12..18)."""
    _require_cv2("hsv_banana")
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = (
        (hsv[:, :, 0] >= 20)
        & (hsv[:, :, 0] <= 38)
        & (hsv[:, :, 1] > 95)
        & (hsv[:, :, 2] > 80)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _keep_largest_component(mask, min_area=10)


def _hsv_red_ring_mask(img_rgb: np.ndarray) -> np.ndarray:
    _require_cv2("hsv_red_ring")
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = (
        ((hsv[:, :, 0] <= 10) | (hsv[:, :, 0] >= 165))
        & (hsv[:, :, 1] > 85)
        & (hsv[:, :, 2] > 55)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return _keep_largest_component(mask, min_area=6)


def _hsv_eggplant_mask(img_rgb: np.ndarray) -> np.ndarray:
    _require_cv2("hsv_eggplant")
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = (
        (hsv[:, :, 0] >= 120)
        & (hsv[:, :, 0] <= 165)
        & (hsv[:, :, 1] > 55)
        & (hsv[:, :, 2] > 35)
    ).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    return _keep_largest_component(mask, min_area=10)


MASK_EXTRACTORS: Dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "hsv_banana": _hsv_banana_mask,
    "hsv_red_ring": _hsv_red_ring_mask,
    "hsv_eggplant": _hsv_eggplant_mask,
}


def extract_object_centroids_from_mask_fn(
    images_nhwc: np.ndarray,
    mask_fn: Callable[[np.ndarray], np.ndarray],
    min_area: int = 6,
    smooth: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract normalized 2D object centroids in [-1, 1]^2 with occlusion interpolation."""
    imgs = np.asarray(images_nhwc)
    if imgs.ndim == 5:
        imgs = imgs[:, -1]
    T, H, W, _ = imgs.shape
    coords = np.full((T, 2), np.nan, dtype=np.float32)
    areas = np.zeros((T,), dtype=np.float32)
    for t in range(T):
        mask = mask_fn(imgs[t])
        area = int(np.sum(mask > 0))
        areas[t] = float(area)
        if area >= min_area:
            ys, xs = np.where(mask > 0)
            cx = (float(xs.mean()) / max(1, W - 1)) * 2.0 - 1.0
            cy = (float(ys.mean()) / max(1, H - 1)) * 2.0 - 1.0
            coords[t, 0] = cx
            coords[t, 1] = cy

    idx = np.arange(T)
    for dim in range(2):
        valid = ~np.isnan(coords[:, dim])
        if np.any(valid) and not np.all(valid):
            coords[:, dim] = np.interp(idx, idx[valid], coords[valid, dim])
        elif not np.any(valid):
            coords[:, dim] = 0.0

    if smooth and T >= 3:
        smoothed = coords.copy()
        smoothed[1:-1] = 0.25 * coords[:-2] + 0.5 * coords[1:-1] + 0.25 * coords[2:]
        coords = smoothed
    return coords, areas


def extract_object_trajectory(
    images_nhwc: np.ndarray,
    extractor_name: str = "hsv_banana",
    external_coords: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Dispatch object trajectory extraction by preset name or external coordinates."""
    if extractor_name == "npy_coords":
        if external_coords is None:
            raise ValueError("extractor_name='npy_coords' requires external_coords.")
        coords = np.asarray(external_coords, dtype=np.float32)
        if coords.shape[0] != images_nhwc.shape[0]:
            raise ValueError(
                f"external_coords length {coords.shape[0]} != frames length {images_nhwc.shape[0]}"
            )
        return coords, np.ones((coords.shape[0],), dtype=np.float32)

    if extractor_name in MASK_EXTRACTORS:
        return extract_object_centroids_from_mask_fn(
            images_nhwc, MASK_EXTRACTORS[extractor_name]
        )

    raise ValueError(
        f"Unsupported obj_extractor '{extractor_name}'. "
        f"Available presets: {sorted(MASK_EXTRACTORS.keys()) + ['npy_coords']}."
    )


def slice_aligned_external_coords(
    external_coords_all: Any,
    raw_trajs: Sequence[List[Dict[str, Any]]],
    raw_idx: int,
    start_idx: int,
    seq_len: int,
) -> np.ndarray:
    """Slice external coordinates indexed in RAW demo coordinates so trimming never misaligns.

    Supports:
    - Object array / list of per-trajectory arrays of length `len(raw_traj)` or `len(raw_traj) + 1`
    - Flat 2D array aligned to all raw transitions (`sum(len(t) for t in raw_trajs)`)
      or all raw frames (`sum(len(t) + 1 for t in raw_trajs)`).
    """
    if isinstance(external_coords_all, list) or (
        isinstance(external_coords_all, np.ndarray) and external_coords_all.dtype == object
    ):
        traj_coords = np.asarray(external_coords_all[raw_idx], dtype=np.float32)
        raw_len = len(raw_trajs[raw_idx])
        if len(traj_coords) == raw_len:
            traj_coords = np.concatenate([traj_coords, traj_coords[-1:]], axis=0)
        return traj_coords[start_idx : start_idx + seq_len]

    arr = np.asarray(external_coords_all, dtype=np.float32)
    total_raw_transitions = sum(len(t) for t in raw_trajs)
    total_raw_frames = sum(len(t) + 1 for t in raw_trajs)
    if arr.shape[0] == total_raw_frames:
        offset = sum(len(raw_trajs[i]) + 1 for i in range(raw_idx)) + start_idx
        return arr[offset : offset + seq_len]
    if arr.shape[0] == total_raw_transitions:
        offset = sum(len(raw_trajs[i]) for i in range(raw_idx))
        traj_coords = arr[offset : offset + len(raw_trajs[raw_idx])]
        traj_coords = np.concatenate([traj_coords, traj_coords[-1:]], axis=0)
        return traj_coords[start_idx : start_idx + seq_len]
    raise ValueError(
        f"Cannot align external_coords of length {arr.shape[0]} with raw demos "
        f"(total transitions={total_raw_transitions}, total frames={total_raw_frames})."
    )


def fit_pca_params(X: np.ndarray, n_components: int = 32) -> Dict[str, np.ndarray]:
    """Fit standardized PCA projection on training features."""
    X = np.asarray(X, dtype=np.float32)
    mu = X.mean(axis=0, keepdims=True)
    Xc = X - mu
    _, s_vals, vt = np.linalg.svd(Xc, full_matrices=False)
    k = min(int(n_components), vt.shape[0])
    W = vt[:k].T.astype(np.float32)
    projected = Xc @ W
    std = np.maximum(projected.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    total_var = float(np.sum(s_vals ** 2)) + 1e-12
    explained = float(np.sum(s_vals[:k] ** 2) / total_var)
    return {
        "mean": mu.astype(np.float32),
        "components": W,
        "std": std,
        "explained_variance_ratio": np.asarray(explained, dtype=np.float32),
    }


def apply_pca_np(X: np.ndarray, pca_params: Mapping[str, np.ndarray]) -> np.ndarray:
    return (
        (np.asarray(X, dtype=np.float32) - pca_params["mean"])
        @ pca_params["components"]
    ) / pca_params["std"]


def compute_roc_auc_np(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Pure NumPy ROC-AUC helper for offline validation diagnostics."""
    y = np.asarray(y_true, dtype=np.int32).reshape(-1)
    s = np.asarray(scores, dtype=np.float64).reshape(-1)
    pos = int(np.sum(y == 1))
    neg = int(len(y) - pos)
    if pos == 0 or neg == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    tpr = np.concatenate([[0.0], tps / float(pos)])
    fpr = np.concatenate([[0.0], fps / float(neg)])
    return float(np.trapz(tpr, fpr))


def make_fixed_action_bank(
    n_samples: int,
    action_dim: int,
    action_low: Optional[np.ndarray] = None,
    action_high: Optional[np.ndarray] = None,
    seed: int = 12345,
) -> np.ndarray:
    """Generate a fixed, deterministic action bank (Common Random Numbers) in [low, high]."""
    rng = np.random.RandomState(int(seed))
    low = (
        np.full((action_dim,), -1.0, dtype=np.float32)
        if action_low is None
        else np.asarray(action_low, dtype=np.float32).reshape(action_dim)
    )
    high = (
        np.full((action_dim,), 1.0, dtype=np.float32)
        if action_high is None
        else np.asarray(action_high, dtype=np.float32).reshape(action_dim)
    )
    u = rng.uniform(0.0, 1.0, size=(int(n_samples), int(action_dim))).astype(np.float32)
    return low[None, :] + u * (high[None, :] - low[None, :])


def compute_cid_lambda(
    step: int,
    weight: float,
    min_weight: float = 0.0,
    decay_steps: int = 15000,
) -> float:
    """Linear decay schedule from `weight` down to `min_weight` over `decay_steps`."""
    if decay_steps <= 0:
        return float(weight)
    progress = min(1.0, max(0.0, float(step) / float(decay_steps)))
    return float(weight) * (1.0 - progress) + float(min_weight) * progress


def compute_shaped_cid_rewards(
    rewards: Any,
    cid_scores: Any,
    dones: Any,
    *,
    step: int,
    baseline_score: float,
    next_cid_scores: Optional[Any] = None,
    masks: Optional[Any] = None,
    discount: float = 0.97,
    weight: float = 0.05,
    min_weight: float = 0.0,
    max_score: float = 0.5,
    decay_steps: int = 15000,
    mode: str = "potential",
    step_penalty: float = 0.02,
    xp: Any = np,
) -> Tuple[Any, Any, float]:
    """Combine extrinsic rewards with CID causal shaping.

    Modes:
    - 'potential' (default, Ng et al. 1999 PBRS):
        Phi(s) = lambda_t * clip(max(0, C(s) - c_base), 0, c_max)
        F(s, s') = discount * mask * Phi(s') - Phi(s)
        Strictly policy-invariant: rewards *entering* high-influence states,
        never rewards *lingering* in place to farm bonus.
    - 'additive_decay':
        r_total = rewards + Phi(s)
    - 'step_penalty_capped':
        non-terminal: min(rewards - step_penalty + Phi(s), 0.0); terminal: rewards.

    Returns `(shaped_rewards, shaping_term, lambda_t)`.
    """
    lambda_t = compute_cid_lambda(
        step=step,
        weight=weight,
        min_weight=min_weight,
        decay_steps=decay_steps,
    )
    safe_curr = xp.where(xp.isfinite(cid_scores), cid_scores, float(baseline_score))
    phi_curr = float(lambda_t) * xp.clip(
        xp.maximum(0.0, safe_curr - float(baseline_score)), 0.0, float(max_score)
    )

    if mode == "potential":
        if next_cid_scores is None:
            raise ValueError("cid_reward_mode='potential' requires `next_cid_scores`.")
        safe_next = xp.where(
            xp.isfinite(next_cid_scores), next_cid_scores, float(baseline_score)
        )
        phi_next = float(lambda_t) * xp.clip(
            xp.maximum(0.0, safe_next - float(baseline_score)), 0.0, float(max_score)
        )
        if masks is not None:
            mask_val = masks.astype(phi_next.dtype)
        else:
            mask_val = (1.0 - dones.astype(phi_next.dtype))
        shaping = float(discount) * mask_val * phi_next - phi_curr
        shaped = rewards + shaping
    elif mode == "additive_decay":
        shaping = phi_curr
        shaped = rewards + shaping
    elif mode == "step_penalty_capped":
        is_done = dones.astype(bool)
        non_terminal = xp.minimum(rewards - float(step_penalty) + phi_curr, 0.0)
        shaped = xp.where(is_done, rewards, non_terminal)
        shaping = shaped - rewards
    else:
        raise ValueError(
            f"Unsupported cid_reward_mode '{mode}'. "
            "Use 'potential', 'additive_decay', or 'step_penalty_capped'."
        )
    return shaped, shaping, lambda_t


def _strip_obs_stack_image(img: np.ndarray) -> np.ndarray:
    """Convert (B, T, H, W, C) or (T, H, W, C) or (H, W, C) to (B, H, W, C)."""
    arr = np.asarray(img)
    if arr.ndim == 5:
        return arr[:, -1]
    if arr.ndim == 4:
        return arr
    if arr.ndim == 3:
        return arr[None, ...]
    raise ValueError(f"Unexpected image shape: {arr.shape}")


def _flatten_obs_state(state: np.ndarray) -> np.ndarray:
    """Convert (B, T, D) or (B, D) or (D,) to (B, D_flat)."""
    arr = np.asarray(state, dtype=np.float32)
    if arr.ndim == 1:
        return arr[None, :]
    if arr.ndim >= 2:
        return arr.reshape(arr.shape[0], -1)
    raise ValueError(f"Unexpected state shape: {arr.shape}")


# =============================================================================
# 2. Bundle Dataclass & Pure-Pickle Persistence
# =============================================================================


@dataclass
class CIDCausalModelBundle:
    """Self-contained frozen weights & normalization stats for online CID reward."""

    image_keys: Tuple[str, str]  # (external_key, wrist_key)
    state_dim: int
    action_dim: int
    obj_dim: int
    ensemble_size: int
    obj_hidden_dims: Tuple[int, ...]
    dyn_hidden_dims: Tuple[int, ...]
    resnet_params: Any
    pca_ext: Dict[str, np.ndarray]
    pca_wrist: Dict[str, np.ndarray]
    obj_reg_params: Any
    dyn_ensemble_params: Any
    obj_mean: np.ndarray
    obj_std: np.ndarray
    state_mean: np.ndarray
    state_std: np.ndarray
    dobj_std: np.ndarray
    baseline_score: float
    action_low: np.ndarray
    action_high: np.ndarray
    fixed_action_bank: np.ndarray
    use_spatial_features: bool = True
    state_clip: float = 5.0
    sigma0_sq: float = 0.25
    ood_beta: float = 2.0
    n_action_samples: int = 128
    metrics: Optional[Dict[str, float]] = None


def _to_numpy_tree(tree: Any) -> Any:
    if isinstance(tree, dict):
        return {k: _to_numpy_tree(v) for k, v in tree.items()}
    if isinstance(tree, (list, tuple)):
        converted = [_to_numpy_tree(v) for v in tree]
        return tuple(converted) if isinstance(tree, tuple) else converted
    if hasattr(tree, "__array__"):
        return np.asarray(tree)
    return tree


def save_cid_bundle(path: str, bundle: CIDCausalModelBundle) -> str:
    """Serialize a CIDCausalModelBundle as pure NumPy/pickle for easy portability."""
    abs_path = os.path.abspath(path)
    os.makedirs(os.path.dirname(abs_path), exist_ok=True)
    payload = {
        "format_version": 2,
        "image_keys": tuple(bundle.image_keys),
        "state_dim": int(bundle.state_dim),
        "action_dim": int(bundle.action_dim),
        "obj_dim": int(bundle.obj_dim),
        "ensemble_size": int(bundle.ensemble_size),
        "obj_hidden_dims": tuple(bundle.obj_hidden_dims),
        "dyn_hidden_dims": tuple(bundle.dyn_hidden_dims),
        "resnet_params": _to_numpy_tree(bundle.resnet_params),
        "pca_ext": _to_numpy_tree(bundle.pca_ext),
        "pca_wrist": _to_numpy_tree(bundle.pca_wrist),
        "obj_reg_params": _to_numpy_tree(bundle.obj_reg_params),
        "dyn_ensemble_params": _to_numpy_tree(bundle.dyn_ensemble_params),
        "obj_mean": np.asarray(bundle.obj_mean, dtype=np.float32),
        "obj_std": np.asarray(bundle.obj_std, dtype=np.float32),
        "state_mean": np.asarray(bundle.state_mean, dtype=np.float32),
        "state_std": np.asarray(bundle.state_std, dtype=np.float32),
        "dobj_std": np.asarray(bundle.dobj_std, dtype=np.float32),
        "baseline_score": float(bundle.baseline_score),
        "action_low": np.asarray(bundle.action_low, dtype=np.float32),
        "action_high": np.asarray(bundle.action_high, dtype=np.float32),
        "fixed_action_bank": np.asarray(bundle.fixed_action_bank, dtype=np.float32),
        "use_spatial_features": bool(bundle.use_spatial_features),
        "state_clip": float(bundle.state_clip),
        "sigma0_sq": float(bundle.sigma0_sq),
        "ood_beta": float(bundle.ood_beta),
        "n_action_samples": int(bundle.n_action_samples),
        "metrics": dict(bundle.metrics or {}),
    }
    tmp_path = f"{abs_path}.tmp"
    with open(tmp_path, "wb") as f:
        pkl.dump(payload, f)
    os.replace(tmp_path, abs_path)
    return abs_path


def load_cid_bundle(path: str) -> CIDCausalModelBundle:
    """Load a serialized CIDCausalModelBundle."""
    with open(path, "rb") as f:
        payload = pkl.load(f)
    ad = int(payload["action_dim"])
    n_samples = int(payload.get("n_action_samples", 128))
    action_low = np.asarray(
        payload.get("action_low", np.full((ad,), -1.0, dtype=np.float32)), dtype=np.float32
    )
    action_high = np.asarray(
        payload.get("action_high", np.full((ad,), 1.0, dtype=np.float32)), dtype=np.float32
    )
    fixed_bank = payload.get("fixed_action_bank", None)
    if fixed_bank is None:
        fixed_bank = make_fixed_action_bank(n_samples, ad, action_low, action_high)
    return CIDCausalModelBundle(
        image_keys=tuple(payload["image_keys"]),
        state_dim=int(payload["state_dim"]),
        action_dim=ad,
        obj_dim=int(payload["obj_dim"]),
        ensemble_size=int(payload["ensemble_size"]),
        obj_hidden_dims=tuple(payload["obj_hidden_dims"]),
        dyn_hidden_dims=tuple(payload["dyn_hidden_dims"]),
        resnet_params=payload["resnet_params"],
        pca_ext=payload["pca_ext"],
        pca_wrist=payload["pca_wrist"],
        obj_reg_params=payload["obj_reg_params"],
        dyn_ensemble_params=payload["dyn_ensemble_params"],
        obj_mean=np.asarray(payload["obj_mean"], dtype=np.float32),
        obj_std=np.asarray(payload["obj_std"], dtype=np.float32),
        state_mean=np.asarray(payload["state_mean"], dtype=np.float32),
        state_std=np.asarray(payload["state_std"], dtype=np.float32),
        dobj_std=np.asarray(payload["dobj_std"], dtype=np.float32),
        baseline_score=float(payload["baseline_score"]),
        action_low=action_low,
        action_high=action_high,
        fixed_action_bank=np.asarray(fixed_bank, dtype=np.float32),
        use_spatial_features=bool(payload.get("use_spatial_features", True)),
        state_clip=float(payload.get("state_clip", 5.0)),
        sigma0_sq=float(payload.get("sigma0_sq", 0.25)),
        ood_beta=float(payload.get("ood_beta", 2.0)),
        n_action_samples=n_samples,
        metrics=dict(payload.get("metrics", {})),
    )


# =============================================================================
# 3. Pure-JAX / Flax Models, Durrieu var_prod_approx GMM-KL, Training & Scoring
# =============================================================================

try:
    import flax.linen as nn
    import jax
    import jax.numpy as jnp
    import optax

    _JAX_AVAILABLE = True
except ImportError:  # pragma: no cover - allows non-JAX unit tests on CPU-only machines
    nn = None
    jax = None
    jnp = None
    optax = None
    _JAX_AVAILABLE = False


def _require_jax():
    if not _JAX_AVAILABLE:
        raise ImportError(
            "JAX, Flax, and Optax are required for CID neural model training/inference."
        )


def pool_spatial_features_jax(fmap, use_spatial: bool = True):
    """Extract spatial-preserving frozen features from ResNet-10 (B, H, W, C) fmap.

    Concatenates global average pool (C,) with parameter-free SpatialSoftmax
    expected coordinates (E[x], E[y]) per channel (2*C,) -> (B, 3*C).
    """
    avg = jnp.mean(fmap, axis=(-3, -2))  # (B, C)
    if not use_spatial:
        return avg
    B, H, W, C = fmap.shape
    flat = fmap.reshape(B, H * W, C)
    weights = jax.nn.softmax(flat, axis=1)  # (B, H*W, C)
    ys = jnp.linspace(-1.0, 1.0, H)
    xs = jnp.linspace(-1.0, 1.0, W)
    grid_y, grid_x = jnp.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(1, H * W, 1)
    grid_y = grid_y.reshape(1, H * W, 1)
    exp_x = jnp.sum(weights * grid_x, axis=1)  # (B, C)
    exp_y = jnp.sum(weights * grid_y, axis=1)  # (B, C)
    return jnp.concatenate([avg, exp_x, exp_y], axis=-1)


def gaussian_entropy_jax(v):
    d = v.shape[-1]
    return 0.5 * (d * (1.0 + math.log(2.0 * math.pi)) + jnp.sum(jnp.log(v), axis=-1))


def gaussian_product_log_norm_jax(m1, v1, m2, v2):
    d = v1.shape[-1]
    var_sum = v1 + v2
    return -0.5 * (
        d * math.log(2.0 * math.pi)
        + jnp.sum(jnp.log(var_sum), axis=-1)
        + jnp.sum((m1 - m2) ** 2 / var_sum, axis=-1)
    )


def diag_gaussian_kl_jax(m1, v1, m2, v2):
    d = v1.shape[-1]
    return 0.5 * (
        -d
        + jnp.sum(
            (v1 + (m2 - m1) ** 2) / v2 + jnp.log(v2) - jnp.log(v1),
            axis=-1,
        )
    )


def kl_var_prod_approx_jax(m1, v1, m2, v2):
    """Exact Durrieu et al. (2012) var_prod_approx KL(N(m1, v1) || GMM(m2, v2)).

    Args:
        m1, v1: (B, K, D) component means and diagonal variances.
        m2, v2: (B, 1, M, D) mixture means and diagonal variances.
    Returns:
        (B, K) estimated KL divergences.
    """
    _require_jax()
    M = m2.shape[2]
    m1_e = m1[:, :, None, :]
    v1_e = v1[:, :, None, :]

    kls = diag_gaussian_kl_jax(m1_e, v1_e, m2, v2)
    kl_var = jnp.log(float(M)) - jax.scipy.special.logsumexp(-kls, axis=-1)

    log_z = gaussian_product_log_norm_jax(m1_e, v1_e, m2, v2)
    ent = gaussian_entropy_jax(v1)
    kl_prod = jnp.maximum(
        0.0,
        jnp.log(float(M)) - ent - jax.scipy.special.logsumexp(log_z, axis=-1),
    )
    return 0.5 * (kl_var + kl_prod)


if _JAX_AVAILABLE:

    class VisualObjRegressor(nn.Module):
        """Lightweight distilled visual object locator g_phi([z_ext_pca, z_wri_pca]) -> hat_s_o."""

        hidden_dims: Tuple[int, ...] = (128, 64)
        out_dim: int = 2
        dropout_rate: float = 0.1

        @nn.compact
        def __call__(self, x, train: bool = False):
            for dim in self.hidden_dims:
                x = nn.Dense(dim)(x)
                x = nn.silu(x)
                x = nn.Dropout(rate=self.dropout_rate, deterministic=not train)(x)
            return nn.Dense(self.out_dim)(x)

    class SingleDynamicsMLP(nn.Module):
        """Single ensemble member predicting Delta s_o from [hat_s_o, state, raw_action]."""

        hidden_dims: Tuple[int, ...] = (256, 256)
        out_dim: int = 2

        @nn.compact
        def __call__(self, x):
            for dim in self.hidden_dims:
                x = nn.Dense(
                    dim,
                    kernel_init=nn.initializers.orthogonal(scale=math.sqrt(2.0)),
                )(x)
                x = nn.tanh(x)
            return nn.Dense(self.out_dim)(x)


def load_pretrained_resnet10_weights(
    public: bool = True, weights_path: Optional[str] = None
) -> Dict[str, Any]:
    """Load ImageNet-1K pretrained ResNet-10 params (reusing ~/.serl/resnet10_params.pkl)."""
    if weights_path is not None and os.path.exists(weights_path):
        with open(weights_path, "rb") as f:
            return pkl.load(f)

    file_name = "resnet10_params.pkl"
    if not public:
        with open(file_name, "rb") as f:
            return pkl.load(f)

    cache_dir = os.path.expanduser("~/.serl")
    os.makedirs(cache_dir, exist_ok=True)
    file_path = os.path.join(cache_dir, file_name)
    if not os.path.exists(file_path):
        import requests
        from tqdm import tqdm

        url = f"https://github.com/rail-berkeley/serl/releases/download/resnet10/{file_name}"
        print(f"[cid_reward] Downloading ResNet-10 weights from {url}")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total_size = int(response.headers.get("content-length", 0))
        t = tqdm(total=total_size, unit="iB", unit_scale=True)
        with open(file_path, "wb") as f:
            for chunk in response.iter_content(1024):
                t.update(len(chunk))
                f.write(chunk)
        t.close()
    with open(file_path, "rb") as f:
        return pkl.load(f)


def init_frozen_resnet10(
    image_size: Tuple[int, int] = (128, 128),
    public: bool = True,
    weights_path: Optional[str] = None,
):
    """Initialize frozen resnetv1-10-frozen encoder with ImageNet-1K weights."""
    _require_jax()
    from serl_launcher.vision.resnet_v1 import resnetv1_configs

    resnet_def = resnetv1_configs["resnetv1-10-frozen"](image_size=image_size)
    dummy = jnp.zeros((1, image_size[0], image_size[1], 3), dtype=jnp.uint8)
    variables = resnet_def.init(jax.random.PRNGKey(0), dummy, train=False)
    params = dict(variables["params"])
    pretrained = load_pretrained_resnet10_weights(public=public, weights_path=weights_path)
    matched = 0
    for k in list(params.keys()):
        if k in pretrained:
            params[k] = pretrained[k]
            matched += 1
    if matched == 0:
        raise RuntimeError("No matching keys found when loading ResNet-10 pretrained weights.")
    return resnet_def, params


def extract_frozen_resnet10_features_batched(
    resnet_def,
    resnet_params,
    images_nhwc: np.ndarray,
    batch_size: int = 64,
    use_spatial: bool = True,
) -> np.ndarray:
    """Extract frozen ResNet-10 spatial+avg features with fixed-size padded JIT batches."""
    _require_jax()
    imgs = _strip_obs_stack_image(images_nhwc)

    @jax.jit
    def _encode_fixed(batch_imgs):
        fmap = resnet_def.apply({"params": resnet_params}, batch_imgs, train=False)
        return pool_spatial_features_jax(fmap, use_spatial=use_spatial)

    chunks = []
    N = len(imgs)
    for start in range(0, N, batch_size):
        chunk = imgs[start : min(N, start + batch_size)]
        valid = len(chunk)
        if valid < batch_size:
            pad = np.repeat(chunk[-1:], batch_size - valid, axis=0)
            chunk = np.concatenate([chunk, pad], axis=0)
        feat = np.asarray(_encode_fixed(jnp.asarray(chunk)), dtype=np.float32)[:valid]
        chunks.append(feat)
    return np.concatenate(chunks, axis=0)


class OnlineCIDScorer:
    """Deterministic, fixed-batch-padded pure-JAX online CID scorer (compiles XLA once)."""

    def __init__(
        self,
        bundle: CIDCausalModelBundle,
        jit_batch_size: int = 32,
        sigma0_sq: Optional[float] = None,
        ood_beta: Optional[float] = None,
    ):
        _require_jax()
        from serl_launcher.vision.resnet_v1 import resnetv1_configs

        self.bundle = bundle
        self.ext_key, self.wrist_key = bundle.image_keys
        self.action_dim = int(bundle.action_dim)
        self.jit_batch_size = max(1, int(jit_batch_size))
        self.sigma0_sq = float(bundle.sigma0_sq if sigma0_sq is None else sigma0_sq)
        self.ood_beta = float(bundle.ood_beta if ood_beta is None else ood_beta)
        self.use_spatial = bool(bundle.use_spatial_features)
        self.state_clip = float(bundle.state_clip)

        self.resnet_def = resnetv1_configs["resnetv1-10-frozen"](image_size=(128, 128))
        self.obj_reg_def = VisualObjRegressor(
            hidden_dims=bundle.obj_hidden_dims,
            out_dim=bundle.obj_dim,
        )
        self.dyn_def = SingleDynamicsMLP(
            hidden_dims=bundle.dyn_hidden_dims,
            out_dim=bundle.obj_dim,
        )

        self._resnet_params = jax.tree.map(jnp.asarray, bundle.resnet_params)
        self._pca_ext_mean = jnp.asarray(bundle.pca_ext["mean"], dtype=jnp.float32)
        self._pca_ext_comp = jnp.asarray(bundle.pca_ext["components"], dtype=jnp.float32)
        self._pca_ext_std = jnp.asarray(bundle.pca_ext["std"], dtype=jnp.float32)
        self._pca_wri_mean = jnp.asarray(bundle.pca_wrist["mean"], dtype=jnp.float32)
        self._pca_wri_comp = jnp.asarray(bundle.pca_wrist["components"], dtype=jnp.float32)
        self._pca_wri_std = jnp.asarray(bundle.pca_wrist["std"], dtype=jnp.float32)
        self._obj_reg_params = jax.tree.map(jnp.asarray, bundle.obj_reg_params)
        self._dyn_ens_params = jax.tree.map(jnp.asarray, bundle.dyn_ensemble_params)
        self._obj_mean = jnp.asarray(bundle.obj_mean, dtype=jnp.float32)
        self._obj_std = jnp.asarray(bundle.obj_std, dtype=jnp.float32)
        self._state_mean = jnp.asarray(bundle.state_mean, dtype=jnp.float32)
        self._state_std = jnp.asarray(bundle.state_std, dtype=jnp.float32)
        # Fixed action bank (Common Random Numbers): shape (K, action_dim)
        self._fixed_actions = jnp.asarray(bundle.fixed_action_bank, dtype=jnp.float32)

        self._score_fixed_fn = self._build_jit_scorer()

    def _build_jit_scorer(self):
        resnet_def = self.resnet_def
        obj_reg_def = self.obj_reg_def
        dyn_def = self.dyn_def
        sigma0_sq = self.sigma0_sq
        ood_beta = self.ood_beta
        use_spatial = self.use_spatial
        state_clip = self.state_clip

        resnet_params = self._resnet_params
        pca_ext_mean, pca_ext_comp, pca_ext_std = (
            self._pca_ext_mean,
            self._pca_ext_comp,
            self._pca_ext_std,
        )
        pca_wri_mean, pca_wri_comp, pca_wri_std = (
            self._pca_wri_mean,
            self._pca_wri_comp,
            self._pca_wri_std,
        )
        obj_reg_params = self._obj_reg_params
        dyn_ens_params = self._dyn_ens_params
        obj_mean, obj_std = self._obj_mean, self._obj_std
        state_mean, state_std = self._state_mean, self._state_std
        fixed_actions = self._fixed_actions  # (K, action_dim)
        n_samples = fixed_actions.shape[0]

        @jax.jit
        def _score_fixed_batch(ext_imgs, wri_imgs, states):
            B = states.shape[0]
            f_ext = pool_spatial_features_jax(
                resnet_def.apply({"params": resnet_params}, ext_imgs, train=False),
                use_spatial=use_spatial,
            )
            f_wri = pool_spatial_features_jax(
                resnet_def.apply({"params": resnet_params}, wri_imgs, train=False),
                use_spatial=use_spatial,
            )
            z_ext = ((f_ext - pca_ext_mean) @ pca_ext_comp) / pca_ext_std
            z_wri = ((f_wri - pca_wri_mean) @ pca_wri_comp) / pca_wri_std
            hat_obj = obj_reg_def.apply(
                {"params": obj_reg_params},
                jnp.concatenate([z_ext, z_wri], axis=-1),
                train=False,
            )
            obj_n = jnp.clip((hat_obj - obj_mean) / obj_std, -state_clip, state_clip)
            st_n = jnp.clip((states - state_mean) / state_std, -state_clip, state_clip)
            compact_s = jnp.concatenate([obj_n, st_n], axis=-1)  # (B, D_compact)

            # Fixed deterministic action bank broadcast to (B, K, action_dim) — NO std blowup!
            actions = jnp.broadcast_to(fixed_actions[None, :, :], (B, n_samples, fixed_actions.shape[-1]))
            s_rep = jnp.broadcast_to(compact_s[:, None, :], (B, n_samples, compact_s.shape[-1]))
            sa = jnp.concatenate([s_rep, actions], axis=-1).reshape(B * n_samples, -1)

            preds = jax.vmap(lambda p: dyn_def.apply({"params": p}, sa))(dyn_ens_params)
            preds = preds.reshape(preds.shape[0], B, n_samples, -1)  # (E, B, K, obj_dim)
            means = jnp.mean(preds, axis=0)  # (B, K, obj_dim)
            epistemic_var = jnp.var(preds, axis=0)  # (B, K, obj_dim)
            v_eff = sigma0_sq + ood_beta * epistemic_var

            kls = kl_var_prod_approx_jax(means, v_eff, means[:, None, :, :], v_eff[:, None, :, :])
            cid_scores = jnp.mean(kls, axis=1)  # (B,)
            epist_mean = jnp.mean(epistemic_var, axis=(1, 2))  # (B,)
            return cid_scores, epist_mean, hat_obj

        return _score_fixed_batch

    def score_observations(
        self,
        observations: Mapping[str, Any],
        rng_key: Any = None,  # kept for backwards API compatibility; scoring is deterministic
        return_details: bool = False,
    ):
        """Score observations in fixed padded chunks of `self.jit_batch_size` (0 JIT recompiles)."""
        ext_imgs = _strip_obs_stack_image(observations[self.ext_key])
        wri_imgs = _strip_obs_stack_image(observations[self.wrist_key])
        states = _flatten_obs_state(observations["state"])

        N = states.shape[0]
        bs = self.jit_batch_size
        all_scores, all_epist, all_hat = [], [], []
        for start in range(0, N, bs):
            end = min(N, start + bs)
            valid = end - start
            ext_c = ext_imgs[start:end]
            wri_c = wri_imgs[start:end]
            st_c = states[start:end]
            if valid < bs:
                pad_n = bs - valid
                ext_c = np.concatenate([ext_c, np.repeat(ext_c[-1:], pad_n, axis=0)], axis=0)
                wri_c = np.concatenate([wri_c, np.repeat(wri_c[-1:], pad_n, axis=0)], axis=0)
                st_c = np.concatenate([st_c, np.repeat(st_c[-1:], pad_n, axis=0)], axis=0)
            sc, ep, ho = self._score_fixed_fn(
                jnp.asarray(ext_c),
                jnp.asarray(wri_c),
                jnp.asarray(st_c),
            )
            all_scores.append(np.asarray(sc[:valid], dtype=np.float32))
            all_epist.append(np.asarray(ep[:valid], dtype=np.float32))
            all_hat.append(np.asarray(ho[:valid], dtype=np.float32))

        scores_np = np.concatenate(all_scores, axis=0)
        if not return_details:
            return scores_np
        return (
            scores_np,
            np.concatenate(all_epist, axis=0),
            np.concatenate(all_hat, axis=0),
        )


def _fit_one_split(
    train_trajs: Sequence[Dict[str, np.ndarray]],
    val_trajs: Sequence[Dict[str, np.ndarray]],
    *,
    resnet_params: Any,
    image_keys: Tuple[str, str],
    pca_dim: int,
    obj_hidden_dims: Tuple[int, ...],
    dyn_hidden_dims: Tuple[int, ...],
    ensemble_size: int,
    obj_steps: int,
    dyn_steps: int,
    batch_size: int,
    n_action_samples: int,
    action_sampling_mode: str,
    state_min_std: float,
    state_clip: float,
    sigma0_sq: float,
    ood_beta: float,
    use_spatial_features: bool,
    seed: int,
):
    """Train g_phi + 5-MLP ensemble on one train/val split and evaluate against GT ||Delta s_o||."""
    ext_key, wrist_key = image_keys

    def _cat(flist, k):
        return np.concatenate([x[k] for x in flist], axis=0)

    tr_ze, va_ze = _cat(train_trajs, "z_ext"), _cat(val_trajs, "z_ext")
    tr_zw, va_zw = _cat(train_trajs, "z_wri"), _cat(val_trajs, "z_wri")
    tr_st, va_st = _cat(train_trajs, "state"), _cat(val_trajs, "state")
    tr_act, va_act = _cat(train_trajs, "action"), _cat(val_trajs, "action")
    tr_obj, tr_obj_n = _cat(train_trajs, "obj"), _cat(train_trajs, "obj_next")
    va_obj, va_obj_n = _cat(val_trajs, "obj"), _cat(val_trajs, "obj_next")

    pca_ext = fit_pca_params(tr_ze, n_components=pca_dim)
    pca_wri = fit_pca_params(tr_zw, n_components=pca_dim)
    tr_pe, va_pe = apply_pca_np(tr_ze, pca_ext), apply_pca_np(va_ze, pca_ext)
    tr_pw, va_pw = apply_pca_np(tr_zw, pca_wri), apply_pca_np(va_zw, pca_wri)

    # ---- Stage 1: Distill visual object locator g_phi ----
    reg_in_tr = jnp.asarray(np.concatenate([tr_pe, tr_pw], axis=1), dtype=jnp.float32)
    reg_in_va = jnp.asarray(np.concatenate([va_pe, va_pw], axis=1), dtype=jnp.float32)
    reg_y_tr = jnp.asarray(tr_obj, dtype=jnp.float32)
    reg_y_va = jnp.asarray(va_obj, dtype=jnp.float32)

    obj_dim = tr_obj.shape[1]
    obj_reg_def = VisualObjRegressor(hidden_dims=obj_hidden_dims, out_dim=obj_dim)
    rng = jax.random.PRNGKey(seed)
    rng, init_key = jax.random.split(rng)
    reg_params = obj_reg_def.init(init_key, reg_in_tr[:1], train=False)["params"]
    reg_tx = optax.adamw(learning_rate=1e-3, weight_decay=1e-3)
    reg_opt_state = reg_tx.init(reg_params)

    @jax.jit
    def _reg_step(params, opt_state, x_b, y_b, step_key):
        noise_key, drop_key = jax.random.split(step_key)
        x_noisy = x_b + 0.05 * jax.random.normal(noise_key, x_b.shape)

        def loss_fn(p):
            pred = obj_reg_def.apply({"params": p}, x_noisy, train=True, rngs={"dropout": drop_key})
            return jnp.mean((pred - y_b) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = reg_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def _reg_eval(params, x_v, y_v):
        pred = obj_reg_def.apply({"params": params}, x_v, train=False)
        return pred, jnp.mean((pred - y_v) ** 2)

    best_reg_mse = float("inf")
    best_reg_params = reg_params
    n_tr = reg_in_tr.shape[0]
    bs = min(int(batch_size), n_tr)
    for step_i in range(1, int(obj_steps) + 1):
        rng, b_key, s_key = jax.random.split(rng, 3)
        idx = jax.random.randint(b_key, (bs,), 0, n_tr)
        reg_params, reg_opt_state, _ = _reg_step(
            reg_params, reg_opt_state, reg_in_tr[idx], reg_y_tr[idx], s_key
        )
        if step_i % 50 == 0 or step_i == obj_steps:
            _, val_mse = _reg_eval(reg_params, reg_in_va, reg_y_va)
            val_mse_f = float(val_mse)
            if val_mse_f < best_reg_mse:
                best_reg_mse = val_mse_f
                best_reg_params = reg_params

    hat_obj_tr = np.asarray(_reg_eval(best_reg_params, reg_in_tr, reg_y_tr)[0], dtype=np.float32)
    hat_obj_va = np.asarray(_reg_eval(best_reg_params, reg_in_va, reg_y_va)[0], dtype=np.float32)
    reg_val_r2 = float(1.0 - best_reg_mse / (float(np.var(va_obj)) + 1e-8))

    # ---- Stage 2: Train 5-MLP Ensemble with floored state_std + raw bounded actions ----
    obj_mean = hat_obj_tr.mean(axis=0, keepdims=True).astype(np.float32)
    obj_std = np.maximum(hat_obj_tr.std(axis=0, keepdims=True), float(state_min_std)).astype(np.float32)
    state_mean = tr_st.mean(axis=0, keepdims=True).astype(np.float32)
    state_std = np.maximum(tr_st.std(axis=0, keepdims=True), float(state_min_std)).astype(np.float32)

    dobj_tr = tr_obj_n - tr_obj
    dobj_va = va_obj_n - va_obj
    dobj_std = np.maximum(dobj_tr.std(axis=0, keepdims=True), 1e-4).astype(np.float32)
    Y_tr = jnp.asarray(dobj_tr / dobj_std, dtype=jnp.float32)
    Y_va = jnp.asarray(dobj_va / dobj_std, dtype=jnp.float32)

    S_tr = np.concatenate(
        [
            np.clip((hat_obj_tr - obj_mean) / obj_std, -state_clip, state_clip),
            np.clip((tr_st - state_mean) / state_std, -state_clip, state_clip),
        ],
        axis=1,
    )
    S_va = np.concatenate(
        [
            np.clip((hat_obj_va - obj_mean) / obj_std, -state_clip, state_clip),
            np.clip((va_st - state_mean) / state_std, -state_clip, state_clip),
        ],
        axis=1,
    )
    # Raw actions in [-1, 1] — NO demo-std amplification!
    X_tr = jnp.asarray(np.concatenate([S_tr, tr_act], axis=1), dtype=jnp.float32)
    X_va = jnp.asarray(np.concatenate([S_va, va_act], axis=1), dtype=jnp.float32)

    dyn_def = SingleDynamicsMLP(hidden_dims=dyn_hidden_dims, out_dim=obj_dim)
    dyn_tx = optax.chain(
        optax.clip_by_global_norm(5.0),
        optax.adamw(learning_rate=8e-4, weight_decay=2e-4),
    )

    @jax.jit
    def _dyn_step(params, opt_state, x_b, y_b):
        def loss_fn(p):
            pred = dyn_def.apply({"params": p}, x_b)
            return jnp.mean((pred - y_b) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(params)
        updates, opt_state = dyn_tx.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)
        return params, opt_state, loss

    @jax.jit
    def _dyn_eval(params, x_v, y_v):
        pred = dyn_def.apply({"params": params}, x_v)
        return jnp.mean((pred - y_v) ** 2)

    ens_params_list = []
    ens_r2s = []
    va_var = float(np.var(np.asarray(Y_va))) + 1e-8
    for _ in range(int(ensemble_size)):
        rng, m_init_key = jax.random.split(rng)
        m_params = dyn_def.init(m_init_key, X_tr[:1])["params"]
        m_opt_state = dyn_tx.init(m_params)
        best_m_mse = float("inf")
        best_m_params = m_params
        for step_i in range(1, int(dyn_steps) + 1):
            rng, b_key = jax.random.split(rng)
            idx = jax.random.randint(b_key, (bs,), 0, n_tr)
            m_params, m_opt_state, _ = _dyn_step(m_params, m_opt_state, X_tr[idx], Y_tr[idx])
            if step_i % 100 == 0 or step_i == dyn_steps:
                v_mse = float(_dyn_eval(m_params, X_va, Y_va))
                if v_mse < best_m_mse:
                    best_m_mse = v_mse
                    best_m_params = m_params
        ens_params_list.append(best_m_params)
        ens_r2s.append(1.0 - best_m_mse / va_var)

    dyn_ens_params = jax.tree.map(lambda *xs: jnp.stack(xs, axis=0), *ens_params_list)
    dyn_val_r2 = float(np.mean(ens_r2s))

    # Build fixed deterministic action bank
    ad = tr_act.shape[1]
    if action_sampling_mode == "support":
        action_low = np.percentile(tr_act, 2, axis=0).astype(np.float32)
        action_high = np.percentile(tr_act, 98, axis=0).astype(np.float32)
    elif action_sampling_mode == "uniform":
        action_low = np.full((ad,), -1.0, dtype=np.float32)
        action_high = np.full((ad,), 1.0, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported action_sampling_mode: {action_sampling_mode}")

    fixed_bank = make_fixed_action_bank(
        n_action_samples, ad, action_low=action_low, action_high=action_high, seed=12345
    )

    bundle = CIDCausalModelBundle(
        image_keys=image_keys,
        state_dim=tr_st.shape[1],
        action_dim=ad,
        obj_dim=obj_dim,
        ensemble_size=int(ensemble_size),
        obj_hidden_dims=tuple(obj_hidden_dims),
        dyn_hidden_dims=tuple(dyn_hidden_dims),
        resnet_params=resnet_params,
        pca_ext=pca_ext,
        pca_wrist=pca_wri,
        obj_reg_params=best_reg_params,
        dyn_ensemble_params=dyn_ens_params,
        obj_mean=obj_mean,
        obj_std=obj_std,
        state_mean=state_mean,
        state_std=state_std,
        dobj_std=dobj_std,
        baseline_score=0.0,
        action_low=action_low,
        action_high=action_high,
        fixed_action_bank=fixed_bank,
        use_spatial_features=bool(use_spatial_features),
        state_clip=float(state_clip),
        sigma0_sq=float(sigma0_sq),
        ood_beta=float(ood_beta),
        n_action_samples=int(n_action_samples),
        metrics={},
    )

    scorer = OnlineCIDScorer(bundle, jit_batch_size=32)
    tr_obs_dict = {
        ext_key: _cat(train_trajs, "ext_img"),
        wrist_key: _cat(train_trajs, "wri_img"),
        "state": tr_st,
    }
    va_obs_dict = {
        ext_key: _cat(val_trajs, "ext_img"),
        wrist_key: _cat(val_trajs, "wri_img"),
        "state": va_st,
    }
    tr_scores = scorer.score_observations(tr_obs_dict)
    va_scores, va_epist, va_hat_obj = scorer.score_observations(
        va_obs_dict, return_details=True
    )

    # Independent evaluation label: ground-truth object motion ||Delta s_o|| (never uses state[0]!)
    tr_dnorm = np.linalg.norm(dobj_tr, axis=1)
    va_dnorm = np.linalg.norm(dobj_va, axis=1)
    move_thr = float(np.percentile(tr_dnorm, 50))
    tr_moving = tr_dnorm > move_thr
    va_moving = va_dnorm > move_thr

    c_base = float(np.median(tr_scores[~tr_moving])) if np.any(~tr_moving) else float(np.percentile(tr_scores, 30))
    bundle.baseline_score = c_base

    va_auc_obj_move = compute_roc_auc_np(va_moving, va_scores)
    fold_metrics = {
        "obj_reg_val_r2": reg_val_r2,
        "dyn_ensemble_val_r2": dyn_val_r2,
        "baseline_score": c_base,
        "val_auc_obj_move": va_auc_obj_move,
        "pca_ext_explained": float(pca_ext["explained_variance_ratio"]),
        "pca_wrist_explained": float(pca_wri["explained_variance_ratio"]),
    }
    diagnostics = {
        "val_traj0": val_trajs[0],
        "val_scores": va_scores,
        "val_epist": va_epist,
        "val_hat_obj": va_hat_obj,
        "val_inter": va_moving,
        "val_dobj_norm": va_dnorm,
    }
    return bundle, fold_metrics, diagnostics


def train_cid_bundle_from_demos(
    demo_paths: Sequence[str],
    *,
    obj_extractor: str = "hsv_banana",
    external_coords_path: Optional[str] = None,
    image_keys: Tuple[str, str] = ("external", "wrist"),
    target_action_dim: Optional[int] = None,
    pca_dim: int = 32,
    obj_hidden_dims: Tuple[int, ...] = (128, 64),
    dyn_hidden_dims: Tuple[int, ...] = (256, 256),
    ensemble_size: int = 5,
    obj_steps: int = 800,
    dyn_steps: int = 1200,
    batch_size: int = 128,
    n_action_samples: int = 128,
    action_sampling_mode: str = "uniform",
    state_min_std: float = 0.05,
    state_clip: float = 5.0,
    sigma0_sq: float = 0.25,
    ood_beta: float = 2.0,
    use_spatial_features: bool = True,
    cv_folds: int = 5,
    seed: int = 0,
    resnet_weights_path: Optional[str] = None,
    return_diagnostics: bool = False,
):
    """Offline training with K-fold trajectory CV evaluation & saved CIDCausalModelBundle."""
    _require_jax()
    ext_key, wrist_key = image_keys

    raw_transitions: List[Dict[str, Any]] = []
    for path in demo_paths:
        with open(path, "rb") as f:
            raw_transitions.extend(pkl.load(f))
    raw_trajs = split_flat_transitions(raw_transitions)
    trajs, kept_raw_indices, start_indices = clean_demo_trajectories(
        raw_trajs, return_start_indices=True
    )
    if len(trajs) < 2:
        raise ValueError(f"Need at least 2 cleaned demo trajectories, got {len(trajs)}.")

    external_coords_all = None
    if external_coords_path is not None:
        external_coords_all = np.load(external_coords_path, allow_pickle=True)

    resnet_def, resnet_params = init_frozen_resnet10(
        image_size=(128, 128), public=True, weights_path=resnet_weights_path
    )

    per_traj: List[Dict[str, np.ndarray]] = []
    for tr, raw_idx, start_idx in zip(trajs, kept_raw_indices, start_indices):
        ext_imgs = np.concatenate(
            [_strip_obs_stack_image(t["observations"][ext_key]) for t in tr], axis=0
        )
        ext_next_last = _strip_obs_stack_image(tr[-1]["next_observations"][ext_key])
        wri_imgs = np.concatenate(
            [_strip_obs_stack_image(t["observations"][wrist_key]) for t in tr], axis=0
        )
        states = np.concatenate(
            [_flatten_obs_state(t["observations"]["state"]) for t in tr], axis=0
        )
        actions = np.stack(
            [np.asarray(t["actions"], dtype=np.float32).reshape(-1) for t in tr], axis=0
        )
        if target_action_dim is not None and actions.shape[1] != int(target_action_dim):
            tdim = int(target_action_dim)
            if actions.shape[1] == 7 and tdim == 4:
                actions = actions[:, [0, 1, 2, 6]]
            elif actions.shape[1] > tdim:
                actions = actions[:, :tdim]
            else:
                pad = np.zeros((actions.shape[0], tdim - actions.shape[1]), dtype=np.float32)
                actions = np.concatenate([actions, pad], axis=1)

        seq_ext = np.concatenate([ext_imgs, ext_next_last], axis=0)
        ext_c = None
        if external_coords_all is not None:
            ext_c = slice_aligned_external_coords(
                external_coords_all, raw_trajs, raw_idx, start_idx, len(seq_ext)
            )

        obj_seq, _ = extract_object_trajectory(
            seq_ext, extractor_name=obj_extractor, external_coords=ext_c
        )
        z_ext = extract_frozen_resnet10_features_batched(
            resnet_def, resnet_params, ext_imgs, use_spatial=use_spatial_features
        )
        z_wri = extract_frozen_resnet10_features_batched(
            resnet_def, resnet_params, wri_imgs, use_spatial=use_spatial_features
        )
        per_traj.append(
            {
                "ext_img": ext_imgs,
                "wri_img": wri_imgs,
                "z_ext": z_ext,
                "z_wri": z_wri,
                "state": states,
                "action": actions,
                "obj": obj_seq[:-1],
                "obj_next": obj_seq[1:],
            }
        )

    # K-fold trajectory cross-validation (robust against single-split noise)
    n_trajs = len(per_traj)
    n_folds = min(max(1, int(cv_folds)), n_trajs) if n_trajs >= 2 else 1
    fold_results = []
    final_bundle = None
    final_diag = None

    if n_folds >= 2:
        fold_splits = np.array_split(np.arange(n_trajs), n_folds)
        for fold_i, val_idxs in enumerate(fold_splits):
            val_set = set(int(x) for x in val_idxs)
            tr_list = [per_traj[i] for i in range(n_trajs) if i not in val_set]
            va_list = [per_traj[i] for i in range(n_trajs) if i in val_set]
            b_i, m_i, d_i = _fit_one_split(
                tr_list,
                va_list,
                resnet_params=resnet_params,
                image_keys=image_keys,
                pca_dim=pca_dim,
                obj_hidden_dims=obj_hidden_dims,
                dyn_hidden_dims=dyn_hidden_dims,
                ensemble_size=ensemble_size,
                obj_steps=obj_steps,
                dyn_steps=dyn_steps,
                batch_size=batch_size,
                n_action_samples=n_action_samples,
                action_sampling_mode=action_sampling_mode,
                state_min_std=state_min_std,
                state_clip=state_clip,
                sigma0_sq=sigma0_sq,
                ood_beta=ood_beta,
                use_spatial_features=use_spatial_features,
                seed=seed + fold_i * 100,
            )
            fold_results.append(m_i)
            # Keep the last fold's bundle/diagnostics for export (or pick median-performing fold)
            if fold_i == n_folds - 1:
                final_bundle = b_i
                final_diag = d_i
    else:
        final_bundle, m_0, final_diag = _fit_one_split(
            per_traj[:-1],
            per_traj[-1:],
            resnet_params=resnet_params,
            image_keys=image_keys,
            pca_dim=pca_dim,
            obj_hidden_dims=obj_hidden_dims,
            dyn_hidden_dims=dyn_hidden_dims,
            ensemble_size=ensemble_size,
            obj_steps=obj_steps,
            dyn_steps=dyn_steps,
            batch_size=batch_size,
            n_action_samples=n_action_samples,
            action_sampling_mode=action_sampling_mode,
            state_min_std=state_min_std,
            state_clip=state_clip,
            sigma0_sq=sigma0_sq,
            ood_beta=ood_beta,
            use_spatial_features=use_spatial_features,
            seed=seed,
        )
        fold_results.append(m_0)

    def _agg(key: str) -> Tuple[float, float]:
        vals = [float(r[key]) for r in fold_results if np.isfinite(r[key])]
        if not vals:
            return float("nan"), float("nan")
        return float(np.mean(vals)), float(np.std(vals))

    reg_r2_m, reg_r2_s = _agg("obj_reg_val_r2")
    dyn_r2_m, dyn_r2_s = _agg("dyn_ensemble_val_r2")
    auc_m, auc_s = _agg("val_auc_obj_move")
    base_m, _ = _agg("baseline_score")

    final_bundle.metrics = {
        "num_cleaned_trajs": float(n_trajs),
        "cv_folds": float(len(fold_results)),
        "cv_obj_reg_r2_mean": reg_r2_m,
        "cv_obj_reg_r2_std": reg_r2_s,
        "cv_dyn_r2_mean": dyn_r2_m,
        "cv_dyn_r2_std": dyn_r2_s,
        "cv_auc_obj_move_mean": auc_m,
        "cv_auc_obj_move_std": auc_s,
        "baseline_score": float(final_bundle.baseline_score),
        "cv_baseline_mean": base_m,
    }

    if not return_diagnostics:
        return final_bundle
    return final_bundle, final_diag
