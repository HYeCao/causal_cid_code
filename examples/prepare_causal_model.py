#!/usr/bin/env python3
"""Offline preparation script for Distilled-Object CID Causal Reward (5-fold CV).

Usage:
    python examples/prepare_causal_model.py \
        --exp_name pick_place_banana \
        --demo_path ./demo_data/pick_place_banana_demos.pkl \
        --obj_extractor hsv_banana \
        --output_dir ./cid_causal_ckpt/pick_place_banana
"""

import json
import os
import sys

from absl import app, flags
import numpy as np

# Prefer this checkout even when the active environment has a stale editable install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_SERL_LAUNCHER = os.path.join(REPO_ROOT, "serl_launcher")
LOCAL_ROBOT_INFRA = os.path.join(REPO_ROOT, "serl_robot_infra")
for source_root in (REPO_ROOT, LOCAL_ROBOT_INFRA, LOCAL_SERL_LAUNCHER):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

from serl_launcher.utils.cid_reward_utils import (
    compute_shaped_cid_rewards,
    save_cid_bundle,
    train_cid_bundle_from_demos,
)

FLAGS = flags.FLAGS
flags.DEFINE_string("exp_name", "pick_place_banana", "Experiment name (for logging/output naming).")
flags.DEFINE_multi_string("demo_path", None, "One or more demo .pkl files.")
flags.DEFINE_string(
    "obj_extractor",
    "hsv_banana",
    "Offline object extractor preset: hsv_banana | hsv_red_ring | hsv_eggplant | npy_coords.",
)
flags.DEFINE_string(
    "external_coords_path",
    None,
    "Optional .npy object coordinates file (raw demo frame indexing) when --obj_extractor=npy_coords.",
)
flags.DEFINE_string("external_key", "external", "Observation dict key for fixed third-person camera.")
flags.DEFINE_string("wrist_key", "wrist", "Observation dict key for wrist camera.")
flags.DEFINE_string("output_dir", None, "Directory to save cid_bundle.pkl and diagnostic plot.")
flags.DEFINE_integer("pca_dim", 32, "PCA dimension for frozen ResNet-10 spatial features.")
flags.DEFINE_integer("ensemble_size", 5, "Number of MLPs in the dynamics ensemble.")
flags.DEFINE_integer("obj_steps", 800, "Training steps for distilled visual object locator g_phi.")
flags.DEFINE_integer("dyn_steps", 1200, "Training steps per ensemble member for dynamics MLP.")
flags.DEFINE_integer("batch_size", 128, "Training minibatch size.")
flags.DEFINE_integer("n_action_samples", 128, "Size of fixed deterministic counterfactual action bank.")
flags.DEFINE_string(
    "action_sampling_mode",
    "uniform",
    "Counterfactual action bank range: uniform ([-1, 1]) | support (demo 2nd-98th pct).",
)
flags.DEFINE_float("state_min_std", 0.05, "Minimum std floor for state standardization.")
flags.DEFINE_float("state_clip", 5.0, "Clip bound for standardized state inputs.")
flags.DEFINE_float("sigma0_sq", 0.25, "Homoscedastic base variance for CID KL.")
flags.DEFINE_float("ood_beta", 2.0, "Weight on ensemble epistemic variance.")
flags.DEFINE_boolean("use_spatial_features", True, "Use SpatialSoftmax + avg pool on frozen ResNet-10.")
flags.DEFINE_integer("cv_folds", 5, "Number of trajectory folds for cross-validation.")
flags.DEFINE_integer("seed", 0, "Random seed.")
flags.DEFINE_string("resnet_weights_path", None, "Optional local path to resnet10_params.pkl.")
flags.DEFINE_boolean("save_plot", True, "Whether to render a held-out diagnostic PNG plot.")


def save_diagnostic_plot(output_dir: str, exp_name: str, bundle, diagnostics: dict) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    tr0 = diagnostics["val_traj0"]
    L0 = len(tr0["state"])
    t_ax = np.arange(L0)
    scores0 = diagnostics["val_scores"][:L0]
    epist0 = diagnostics["val_epist"][:L0]
    hat_obj0 = diagnostics["val_hat_obj"][:L0]
    inter0 = diagnostics["val_inter"][:L0]
    dobj0 = diagnostics["val_dobj_norm"][:L0]
    gt_obj0 = tr0["obj"]

    next_scores0 = np.concatenate([scores0[1:], scores0[-1:]])
    dones0 = np.zeros((L0,), dtype=bool)
    dones0[-1] = True
    _, pbrs0, _ = compute_shaped_cid_rewards(
        np.zeros((L0,), dtype=np.float32),
        scores0,
        dones0,
        next_cid_scores=next_scores0,
        discount=0.97,
        step=0,
        baseline_score=bundle.baseline_score,
        weight=0.05,
        max_score=0.5,
        decay_steps=15000,
        mode="potential",
        xp=np,
    )
    phi0 = 0.05 * np.clip(np.maximum(0.0, scores0 - bundle.baseline_score), 0.0, 0.5)

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(5, 8, height_ratios=[1.3, 1.3, 0.9, 1.05, 1.0], hspace=0.45, wspace=0.1)

    fidxs = np.linspace(0, L0 - 1, 8, dtype=int)
    for ci, fi in enumerate(fidxs):
        ax0 = fig.add_subplot(gs[0, ci])
        ax0.imshow(tr0["ext_img"][fi])
        gx = (gt_obj0[fi, 0] + 1.0) / 2.0 * (tr0["ext_img"].shape[2] - 1)
        gy = (gt_obj0[fi, 1] + 1.0) / 2.0 * (tr0["ext_img"].shape[1] - 1)
        px = (hat_obj0[fi, 0] + 1.0) / 2.0 * (tr0["ext_img"].shape[2] - 1)
        py = (hat_obj0[fi, 1] + 1.0) / 2.0 * (tr0["ext_img"].shape[1] - 1)
        ax0.plot(gx, gy, "m+", markersize=9, markeredgewidth=2, label="GT obj" if ci == 0 else None)
        ax0.plot(px, py, "co", markersize=6, markerfacecolor="none", markeredgewidth=1.8, label="Distill hat_s_o" if ci == 0 else None)
        ax0.set_title(f"t={fi}", fontsize=9)
        ax0.axis("off")
        if ci == 0:
            ax0.legend(loc="lower left", fontsize=6.5, framealpha=0.75)

        ax1 = fig.add_subplot(gs[1, ci])
        ax1.imshow(tr0["wri_img"][fi])
        ax1.axis("off")

    def shade(ax):
        for t in range(L0):
            if inter0[t]:
                ax.axvspan(t - 0.5, t + 0.5, color="orange", alpha=0.18, lw=0)

    ax2 = fig.add_subplot(gs[2, :])
    shade(ax2)
    ax2.plot(t_ax, dobj0, color="magenta", lw=1.8, label="GT ||dObj|| (independent motion label)")
    m = bundle.metrics or {}
    ax2.set_title(
        f"{exp_name} ({int(m.get('cv_folds', 1))}-Fold CV) — "
        f"Obj R2={m.get('cv_obj_reg_r2_mean', 0):.3f}±{m.get('cv_obj_reg_r2_std', 0):.3f}, "
        f"Dyn R2={m.get('cv_dyn_r2_mean', 0):.3f}±{m.get('cv_dyn_r2_std', 0):.3f}, "
        f"ObjMove AUC={m.get('cv_auc_obj_move_mean', 0):.3f}±{m.get('cv_auc_obj_move_std', 0):.3f}",
        fontsize=10,
        fontweight="bold",
    )
    ax2.legend(loc="upper left", fontsize=8.5)
    ax2.set_xlim(0, L0 - 1)

    ax3 = fig.add_subplot(gs[3, :])
    shade(ax3)
    ax3.plot(t_ax, scores0, color="tab:blue", lw=2.2, label="Deterministic CID C(o_t) [fixed K bank]")
    ax3.axhline(bundle.baseline_score, color="gray", ls=":", lw=1.5, label=f"c_base={bundle.baseline_score:.3f}")
    ax3_r = ax3.twinx()
    ax3_r.plot(t_ax, epist0, color="tab:purple", lw=1.3, ls="--", alpha=0.75, label="Epistemic Var (right)")
    ax3.set_title("Deterministic CID Score & Ensemble Epistemic Variance", fontsize=10)
    ax3.legend(loc="upper left", fontsize=8.5)
    ax3_r.legend(loc="upper right", fontsize=8.5)
    ax3.set_xlim(0, L0 - 1)

    ax4 = fig.add_subplot(gs[4, :])
    shade(ax4)
    ax4.plot(t_ax, phi0, color="tab:cyan", lw=1.8, ls="--", label="Potential Phi(s)")
    ax4.plot(t_ax, pbrs0, color="tab:blue", lw=2.0, label="PBRS F(s,s') = gamma*Phi(s') - Phi(s)")
    ax4.axhline(0.0, color="black", lw=0.8, alpha=0.5)
    ax4.set_title("Potential-Based Reward Shaping (PBRS — Policy-Invariant)", fontsize=10)
    ax4.set_xlabel("Cleaned Timestep t (orange shade = GT object moving)")
    ax4.legend(loc="upper left", fontsize=8.5, ncol=2)
    ax4.set_xlim(0, L0 - 1)

    plot_path = os.path.join(output_dir, f"{exp_name}_cid_validation.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return plot_path


def main(_):
    if not FLAGS.demo_path:
        raise ValueError("--demo_path is required.")

    output_dir = FLAGS.output_dir or os.path.join(REPO_ROOT, "cid_causal_ckpt", FLAGS.exp_name)
    os.makedirs(output_dir, exist_ok=True)

    print(
        f"[prepare_causal_model] exp={FLAGS.exp_name}, demos={FLAGS.demo_path}, "
        f"extractor={FLAGS.obj_extractor}, cv_folds={FLAGS.cv_folds}, "
        f"K={FLAGS.n_action_samples} ({FLAGS.action_sampling_mode})"
    )
    bundle, diagnostics = train_cid_bundle_from_demos(
        FLAGS.demo_path,
        obj_extractor=FLAGS.obj_extractor,
        external_coords_path=FLAGS.external_coords_path,
        image_keys=(FLAGS.external_key, FLAGS.wrist_key),
        pca_dim=FLAGS.pca_dim,
        ensemble_size=FLAGS.ensemble_size,
        obj_steps=FLAGS.obj_steps,
        dyn_steps=FLAGS.dyn_steps,
        batch_size=FLAGS.batch_size,
        n_action_samples=FLAGS.n_action_samples,
        action_sampling_mode=FLAGS.action_sampling_mode,
        state_min_std=FLAGS.state_min_std,
        state_clip=FLAGS.state_clip,
        sigma0_sq=FLAGS.sigma0_sq,
        ood_beta=FLAGS.ood_beta,
        use_spatial_features=FLAGS.use_spatial_features,
        cv_folds=FLAGS.cv_folds,
        seed=FLAGS.seed,
        resnet_weights_path=FLAGS.resnet_weights_path,
        return_diagnostics=True,
    )

    bundle_path = save_cid_bundle(os.path.join(output_dir, "cid_bundle.pkl"), bundle)
    metrics_path = os.path.join(output_dir, "cid_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(bundle.metrics, f, indent=2)

    print("\n=== Offline CID Causal Model Cross-Validation Summary ===")
    for k, v in (bundle.metrics or {}).items():
        print(f"  {k:28s}: {v:.4f}")
    print(f"Saved bundle : {bundle_path}")
    print(f"Saved metrics: {metrics_path}")

    if FLAGS.save_plot:
        try:
            plot_path = save_diagnostic_plot(output_dir, FLAGS.exp_name, bundle, diagnostics)
            print(f"Saved plot   : {plot_path}")
        except Exception as exc:
            print(f"[prepare_causal_model] Warning: plot generation skipped ({exc})")


if __name__ == "__main__":
    app.run(main)
