"""Automated verification tests for causal_code (Distilled-Object CID Causal Reward).

Runs two tiers of tests:
1. Pure-NumPy / OpenCV / Buffer / Reward-shaping tests (work on any machine, even without JAX).
2. Full JAX / Flax tests (run automatically when JAX & Flax are installed, e.g. on Linux GPU robot machine).
"""

import glob
import os
import pickle as pkl
import sys
import tempfile
import unittest

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (
    REPO_ROOT,
    os.path.join(REPO_ROOT, "serl_launcher"),
    os.path.join(REPO_ROOT, "serl_robot_infra"),
):
    if p not in sys.path:
        sys.path.insert(0, p)

from serl_launcher.utils.cid_reward_utils import (
    CIDCausalModelBundle,
    _JAX_AVAILABLE,
    apply_pca_np,
    clean_demo_trajectories,
    compute_cid_lambda,
    compute_roc_auc_np,
    compute_shaped_cid_rewards,
    extract_object_trajectory,
    fit_pca_params,
    load_cid_bundle,
    make_fixed_action_bank,
    save_cid_bundle,
    slice_aligned_external_coords,
    split_flat_transitions,
)


class TestCIDUtilitiesNoJAX(unittest.TestCase):
    def test_clean_demo_trajectories_trims_leading_zero_actions(self):
        fake_traj = []
        for t in range(15):
            act = (
                np.zeros((4,), dtype=np.float32)
                if t < 10
                else np.array([0.1, 0.0, -0.2, 0.0], dtype=np.float32)
            )
            fake_traj.append({"actions": act, "dones": t == 14})
        cleaned, kept_raw, start_indices = clean_demo_trajectories(
            [fake_traj], return_start_indices=True
        )
        self.assertEqual(len(cleaned), 1)
        self.assertEqual(kept_raw, [0])
        # First moving frame is t=10 -> keeps t=9..14 (start=9, len=6)
        self.assertEqual(start_indices, [9])
        self.assertEqual(len(cleaned[0]), 6)

    def test_slice_aligned_external_coords_matches_raw_frames(self):
        # Two raw trajectories of lengths 12 and 10, with leading stationary frames 6 and 4
        raw_trajs = []
        for raw_len, lead in [(12, 6), (10, 4)]:
            traj = []
            for t in range(raw_len):
                act = (
                    np.zeros((4,), dtype=np.float32)
                    if t < lead
                    else np.array([0.2, 0.0, 0.0, 0.0], dtype=np.float32)
                )
                traj.append({"actions": act, "dones": t == raw_len - 1})
            raw_trajs.append(traj)

        cleaned, kept_raw, starts = clean_demo_trajectories(
            raw_trajs, return_start_indices=True
        )
        self.assertEqual(kept_raw, [0, 1])
        self.assertEqual(starts, [5, 3])

        # Case A: Per-trajectory object array of length len(raw_traj) + 1 (raw frames)
        per_traj_frames = [
            np.arange((len(tr) + 1) * 2, dtype=np.float32).reshape(-1, 2) + 100 * idx
            for idx, tr in enumerate(raw_trajs)
        ]
        for i, (r_idx, st, c_tr) in enumerate(zip(kept_raw, starts, cleaned)):
            seq_len = len(c_tr) + 1
            sliced = slice_aligned_external_coords(
                per_traj_frames, raw_trajs, r_idx, st, seq_len
            )
            expected = per_traj_frames[r_idx][st : st + seq_len]
            self.assertEqual(sliced.shape, (seq_len, 2))
            self.assertTrue(np.allclose(sliced, expected))

        # Case B: Flat 2D array aligned to all raw transitions (sum(len(t)))
        flat_trans = np.concatenate(
            [
                np.arange(len(tr) * 2, dtype=np.float32).reshape(-1, 2) + 100 * idx
                for idx, tr in enumerate(raw_trajs)
            ],
            axis=0,
        )
        for r_idx, st, c_tr in zip(kept_raw, starts, cleaned):
            seq_len = len(c_tr) + 1
            sliced = slice_aligned_external_coords(
                flat_trans, raw_trajs, r_idx, st, seq_len
            )
            self.assertEqual(sliced.shape, (seq_len, 2))
            # First kept transition frame must equal raw frame `st`
            raw_offset = sum(len(raw_trajs[k]) for k in range(r_idx)) + st
            self.assertTrue(np.allclose(sliced[0], flat_trans[raw_offset]))

        # Case C: Flat 2D array aligned to all raw frames (sum(len(t) + 1))
        flat_frames = np.concatenate(per_traj_frames, axis=0)
        for r_idx, st, c_tr in zip(kept_raw, starts, cleaned):
            seq_len = len(c_tr) + 1
            sliced = slice_aligned_external_coords(
                flat_frames, raw_trajs, r_idx, st, seq_len
            )
            expected = per_traj_frames[r_idx][st : st + seq_len]
            self.assertTrue(np.allclose(sliced, expected))

    def test_make_fixed_action_bank_deterministic_and_bounded(self):
        low = np.array([-0.5, -0.2, -1.0, -1.0], dtype=np.float32)
        high = np.array([0.5, 0.3, 1.0, 1.0], dtype=np.float32)
        bank1 = make_fixed_action_bank(128, 4, low, high, seed=42)
        bank2 = make_fixed_action_bank(128, 4, low, high, seed=42)
        self.assertEqual(bank1.shape, (128, 4))
        self.assertTrue(np.array_equal(bank1, bank2))
        self.assertTrue(np.all(bank1 >= low[None, :] - 1e-6))
        self.assertTrue(np.all(bank1 <= high[None, :] + 1e-6))

    def test_pca_roundtrip_and_standardization(self):
        rng = np.random.RandomState(0)
        X = rng.randn(200, 64).astype(np.float32)
        pca = fit_pca_params(X, n_components=16)
        Z = apply_pca_np(X, pca)
        self.assertEqual(Z.shape, (200, 16))
        self.assertTrue(np.allclose(Z.mean(axis=0), 0.0, atol=1e-4))
        self.assertTrue(np.allclose(Z.std(axis=0), 1.0, atol=1e-2))

    def test_reward_shaping_potential_pbrs_invariance_and_edge_cases(self):
        # Transitions:
        # 0: enter high-CID (0.4 -> 0.8), non-terminal -> positive shaping
        # 1: linger at high-CID (0.8 -> 0.8), non-terminal -> negative (gamma - 1) * Phi
        # 2: leave high-CID (0.8 -> 0.4), non-terminal -> negative -Phi
        # 3: terminal success (0.8 -> 0.8), mask=0.0 -> Phi(s') zeroed, shaped = 1.0 - Phi(s)
        # 4: NaN scores safely treated as baseline -> shaping = 0.0
        rewards = np.array([0.0, 0.0, 0.0, 1.0, 0.0], dtype=np.float32)
        dones = np.array([False, False, False, True, False], dtype=bool)
        masks = np.array([1.0, 1.0, 1.0, 0.0, 1.0], dtype=np.float32)
        cid_scores = np.array([0.4, 0.8, 0.8, 0.8, np.nan], dtype=np.float32)
        next_cid_scores = np.array([0.8, 0.8, 0.4, 0.8, np.nan], dtype=np.float32)

        gamma = 0.97
        weight = 0.05
        baseline = 0.4
        # Phi(0.4) = 0.0, Phi(0.8) = 0.05 * (0.8 - 0.4) = 0.02
        phi_hi = weight * (0.8 - baseline)

        shaped, shaping, lam = compute_shaped_cid_rewards(
            rewards,
            cid_scores,
            dones,
            next_cid_scores=next_cid_scores,
            masks=masks,
            discount=gamma,
            step=0,
            baseline_score=baseline,
            weight=weight,
            min_weight=0.0,
            max_score=0.5,
            decay_steps=10000,
            mode="potential",
            xp=np,
        )
        self.assertAlmostEqual(lam, weight, places=6)
        self.assertAlmostEqual(float(shaping[0]), gamma * phi_hi - 0.0, places=6)
        self.assertGreater(float(shaping[0]), 0.0)
        self.assertAlmostEqual(float(shaping[1]), (gamma - 1.0) * phi_hi, places=6)
        self.assertLess(float(shaping[1]), 0.0)  # Lingering cannot farm bonus!
        self.assertAlmostEqual(float(shaping[2]), -phi_hi, places=6)
        self.assertAlmostEqual(float(shaping[3]), -phi_hi, places=6)  # mask=0 zeroes Phi(s')
        self.assertAlmostEqual(float(shaped[3]), 1.0 - phi_hi, places=6)
        self.assertAlmostEqual(float(shaping[4]), 0.0, places=6)

        # Telescoping PBRS property across an entire 10-step episode ending with mask=0:
        # sum_{t=0}^{T-1} gamma^t F(s_t, s_{t+1}) == -Phi(s_0)
        ep_c = np.array([0.4, 0.5, 0.8, 0.9, 0.9, 0.9, 0.85, 0.6, 0.7, 0.9], dtype=np.float32)
        ep_nc = np.concatenate([ep_c[1:], np.array([0.9], dtype=np.float32)])
        ep_r = np.zeros((10,), dtype=np.float32)
        ep_r[-1] = 1.0
        ep_d = np.zeros((10,), dtype=bool)
        ep_d[-1] = True
        ep_m = np.ones((10,), dtype=np.float32)
        ep_m[-1] = 0.0
        _, ep_F, _ = compute_shaped_cid_rewards(
            ep_r,
            ep_c,
            ep_d,
            next_cid_scores=ep_nc,
            masks=ep_m,
            discount=gamma,
            step=0,
            baseline_score=baseline,
            weight=weight,
            max_score=0.5,
            decay_steps=10000,
            mode="potential",
            xp=np,
        )
        discounts = gamma ** np.arange(10, dtype=np.float32)
        discounted_F_sum = float(np.sum(discounts * ep_F))
        # Since Phi(s_0) = Phi(0.4) = 0.0, discounted sum must equal 0.0!
        self.assertAlmostEqual(discounted_F_sum, 0.0, places=5)

        # After decay_steps, shaping decays completely to 0
        shaped_end, shaping_end, lam_end = compute_shaped_cid_rewards(
            rewards,
            cid_scores,
            dones,
            next_cid_scores=next_cid_scores,
            masks=masks,
            discount=gamma,
            step=15000,
            baseline_score=baseline,
            weight=weight,
            min_weight=0.0,
            max_score=0.5,
            decay_steps=10000,
            mode="potential",
            xp=np,
        )
        self.assertAlmostEqual(lam_end, 0.0, places=6)
        self.assertTrue(np.allclose(shaping_end, 0.0))
        self.assertTrue(np.allclose(shaped_end, rewards))

    def test_reward_shaping_additive_decay_and_bounds(self):
        rewards = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
        dones = np.array([False, False, False, True], dtype=bool)
        cid_scores = np.array([0.2, 0.5, 2.5, np.nan], dtype=np.float32)

        shaped0, bonus0, lam0 = compute_shaped_cid_rewards(
            rewards,
            cid_scores,
            dones,
            step=0,
            baseline_score=0.4,
            weight=0.02,
            min_weight=0.0,
            max_score=0.5,
            decay_steps=10000,
            mode="additive_decay",
            xp=np,
        )
        self.assertAlmostEqual(lam0, 0.02, places=6)
        self.assertAlmostEqual(float(bonus0[0]), 0.0, places=6)
        self.assertAlmostEqual(float(bonus0[1]), 0.002, places=6)
        self.assertAlmostEqual(float(bonus0[2]), 0.010, places=6)
        self.assertAlmostEqual(float(bonus0[3]), 0.0, places=6)
        self.assertAlmostEqual(float(shaped0[3]), 1.0, places=6)

        shaped_end, bonus_end, lam_end = compute_shaped_cid_rewards(
            rewards,
            cid_scores,
            dones,
            step=15000,
            baseline_score=0.4,
            weight=0.02,
            min_weight=0.0,
            max_score=0.5,
            decay_steps=10000,
            mode="additive_decay",
            xp=np,
        )
        self.assertAlmostEqual(lam_end, 0.0, places=6)
        self.assertTrue(np.allclose(bonus_end, 0.0))
        self.assertTrue(np.allclose(shaped_end, rewards))

    def test_reward_shaping_step_penalty_capped(self):
        rewards = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        dones = np.array([False, False, True], dtype=bool)
        cid_scores = np.array([0.0, 5.0, 5.0], dtype=np.float32)
        shaped, _, _ = compute_shaped_cid_rewards(
            rewards,
            cid_scores,
            dones,
            step=0,
            baseline_score=0.0,
            weight=0.1,
            max_score=0.5,
            decay_steps=10000,
            mode="step_penalty_capped",
            step_penalty=0.02,
            xp=np,
        )
        self.assertAlmostEqual(float(shaped[0]), -0.02, places=6)
        self.assertAlmostEqual(float(shaped[1]), 0.0, places=6)
        self.assertAlmostEqual(float(shaped[2]), 1.0, places=6)

    def test_bundle_save_and_load_roundtrip(self):
        fixed_bank = make_fixed_action_bank(128, 4, seed=7)
        bundle = CIDCausalModelBundle(
            image_keys=("external", "wrist"),
            state_dim=11,
            action_dim=4,
            obj_dim=2,
            ensemble_size=5,
            obj_hidden_dims=(128, 64),
            dyn_hidden_dims=(256, 256),
            resnet_params={"w": np.ones((2, 2), dtype=np.float32)},
            pca_ext={
                "mean": np.zeros((1, 4), dtype=np.float32),
                "components": np.eye(4, 2, dtype=np.float32),
                "std": np.ones((1, 2), dtype=np.float32),
            },
            pca_wrist={
                "mean": np.zeros((1, 4), dtype=np.float32),
                "components": np.eye(4, 2, dtype=np.float32),
                "std": np.ones((1, 2), dtype=np.float32),
            },
            obj_reg_params={"Dense_0": {"kernel": np.ones((4, 2), dtype=np.float32)}},
            dyn_ensemble_params={"Dense_0": {"kernel": np.ones((5, 17, 2), dtype=np.float32)}},
            obj_mean=np.zeros((1, 2), dtype=np.float32),
            obj_std=np.ones((1, 2), dtype=np.float32),
            state_mean=np.zeros((1, 11), dtype=np.float32),
            state_std=np.ones((1, 11), dtype=np.float32),
            dobj_std=np.ones((1, 2), dtype=np.float32),
            baseline_score=0.45,
            action_low=np.full((4,), -1.0, dtype=np.float32),
            action_high=np.full((4,), 1.0, dtype=np.float32),
            fixed_action_bank=fixed_bank,
            use_spatial_features=True,
            state_clip=5.0,
            n_action_samples=128,
            metrics={"cv_val_auc_mean": 0.87},
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_cid_bundle(os.path.join(tmpdir, "cid_bundle.pkl"), bundle)
            loaded = load_cid_bundle(path)
            self.assertEqual(loaded.state_dim, 11)
            self.assertEqual(loaded.ensemble_size, 5)
            self.assertTrue(loaded.use_spatial_features)
            self.assertAlmostEqual(loaded.state_clip, 5.0, places=6)
            self.assertEqual(loaded.fixed_action_bank.shape, (128, 4))
            self.assertTrue(np.allclose(loaded.fixed_action_bank, fixed_bank))
            self.assertAlmostEqual(loaded.baseline_score, 0.45, places=6)
            self.assertAlmostEqual(loaded.metrics["cv_val_auc_mean"], 0.87, places=6)

    def test_real_cirl_demo_cleaning_and_hsv_extraction_if_present(self):
        demo_dir = os.path.join(os.path.dirname(REPO_ROOT), "CIRL_demodata")
        banana_files = glob.glob(os.path.join(demo_dir, "pick_place_banana_*.pkl"))
        if not banana_files:
            self.skipTest("CIRL_demodata not found next to causal_code")
        with open(banana_files[0], "rb") as f:
            raw = pkl.load(f)
        trajs = split_flat_transitions(raw)
        cleaned, kept_raw, starts = clean_demo_trajectories(
            trajs, return_start_indices=True
        )
        self.assertEqual(len(cleaned), 10)
        self.assertEqual(len(starts), 10)
        imgs = np.concatenate([t["observations"]["external"] for t in cleaned[0]], axis=0)
        coords, areas = extract_object_trajectory(imgs, extractor_name="hsv_banana")
        self.assertEqual(coords.shape, (len(cleaned[0]), 2))
        self.assertTrue(np.all(np.isfinite(coords)))
        self.assertGreater(float(np.median(areas)), 60.0)
        # Verify early frames (t=0..15) of traj 0 stay smoothly on the banana (no jump to pan rim)
        early_step_jumps = np.linalg.norm(np.diff(coords[: min(20, len(coords))], axis=0), axis=1)
        self.assertLess(float(np.max(early_step_jumps)), 0.15)


@unittest.skipUnless(_JAX_AVAILABLE, "JAX/Flax not installed in this environment")
class TestCIDJAXEndToEnd(unittest.TestCase):
    def test_kl_var_prod_approx_jax_properties(self):
        import jax
        import jax.numpy as jnp
        from serl_launcher.utils.cid_reward_utils import kl_var_prod_approx_jax

        # Identical distributions -> KL == 0
        m = jnp.zeros((4, 16, 2), dtype=jnp.float32)
        v = jnp.ones((4, 16, 2), dtype=jnp.float32) * 0.25
        kl_zero = kl_var_prod_approx_jax(m, v, m[:, None, :, :], v[:, None, :, :])
        self.assertEqual(kl_zero.shape, (4, 16))
        self.assertTrue(bool(jnp.allclose(kl_zero, 0.0, atol=1e-4)))

        # Separated means -> strictly positive KL
        rng = jax.random.PRNGKey(0)
        m_sep = jax.random.normal(rng, (4, 16, 2)) * 2.0
        kl_pos = kl_var_prod_approx_jax(m_sep, v, m_sep[:, None, :, :], v[:, None, :, :])
        self.assertTrue(bool(jnp.all(jnp.isfinite(kl_pos))))
        self.assertGreater(float(jnp.mean(kl_pos)), 0.5)

    def test_pool_spatial_features_jax_shape_and_range(self):
        import jax.numpy as jnp
        from serl_launcher.utils.cid_reward_utils import pool_spatial_features_jax

        fmap = jnp.zeros((3, 4, 4, 8), dtype=jnp.float32)
        fmap = fmap.at[:, 0, 3, :].set(10.0)  # peak at top-right (y=-1, x=+1)
        out_spatial = pool_spatial_features_jax(fmap, use_spatial=True)
        out_avg = pool_spatial_features_jax(fmap, use_spatial=False)
        self.assertEqual(out_spatial.shape, (3, 24))
        self.assertEqual(out_avg.shape, (3, 8))
        # SpatialSoftmax expected x ~ +1, expected y ~ -1
        self.assertGreater(float(jnp.mean(out_spatial[:, 8:16])), 0.9)
        self.assertLess(float(jnp.mean(out_spatial[:, 16:24])), -0.9)

    def test_memory_efficient_buffer_lazy_cid_alignment(self):
        import gymnasium as gym
        from serl_launcher.data.memory_efficient_replay_buffer import (
            MemoryEfficientReplayBuffer,
        )

        obs_space = gym.spaces.Dict(
            {
                "external": gym.spaces.Box(0, 255, shape=(1, 16, 16, 3), dtype=np.uint8),
                "wrist": gym.spaces.Box(0, 255, shape=(1, 16, 16, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-10, 10, shape=(1, 11), dtype=np.float32),
            }
        )
        act_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)
        buf = MemoryEfficientReplayBuffer(
            obs_space,
            act_space,
            capacity=20,
            pixel_keys=("external", "wrist"),
            include_cid_scores=True,
        )

        # Insert >2x capacity episodes + non-contiguous intervention with _reset_frame_stack=True
        for ep in range(7):
            for t in range(5):
                val = ep * 10 + t
                img_t = np.full((1, 16, 16, 3), val % 250, dtype=np.uint8)
                img_n = np.full((1, 16, 16, 3), (val + 1) % 250, dtype=np.uint8)
                buf.insert(
                    {
                        "observations": {
                            "external": img_t,
                            "wrist": img_t,
                            "state": np.full((1, 11), float(val % 250), dtype=np.float32),
                        },
                        "next_observations": {
                            "external": img_n,
                            "wrist": img_n,
                            "state": np.full((1, 11), float((val + 1) % 250), dtype=np.float32),
                        },
                        "actions": np.zeros((4,), dtype=np.float32),
                        "rewards": np.float32(1.0 if t == 4 else 0.0),
                        "masks": np.float32(0.0 if t == 4 else 1.0),
                        "dones": bool(t == 4),
                    }
                )

        # Non-contiguous intervention sample
        buf.insert(
            {
                "_reset_frame_stack": True,
                "observations": {
                    "external": np.full((1, 16, 16, 3), 201, dtype=np.uint8),
                    "wrist": np.full((1, 16, 16, 3), 201, dtype=np.uint8),
                    "state": np.full((1, 11), 201.0, dtype=np.float32),
                },
                "next_observations": {
                    "external": np.full((1, 16, 16, 3), 202, dtype=np.uint8),
                    "wrist": np.full((1, 16, 16, 3), 202, dtype=np.uint8),
                    "state": np.full((1, 11), 202.0, dtype=np.float32),
                },
                "actions": np.zeros((4,), dtype=np.float32),
                "rewards": np.float32(0.0),
                "masks": np.float32(1.0),
                "dones": False,
            }
        )

        unscored = buf.get_unscored_cid_indices()
        expected_valid = int(np.sum(buf._is_correct_index[: buf._size]))
        self.assertGreater(expected_valid, 0)
        self.assertEqual(len(unscored), expected_valid)
        obs, next_obs = buf.get_observations_by_indices(unscored, include_next=True)
        for i in range(len(unscored)):
            st_val = int(round(float(obs["state"][i, 0, 0])))
            px_val = int(obs["external"][i, 0, 0, 0, 0])
            nst_val = int(round(float(next_obs["state"][i, 0, 0])))
            npx_val = int(next_obs["external"][i, 0, 0, 0, 0])
            self.assertEqual(px_val, st_val)
            self.assertEqual(npx_val, nst_val)

        # Write BOTH cid_scores and next_cid_scores and verify sample() alignment
        buf.set_cid_scores(
            unscored,
            obs["state"][:, 0, 0],
            next_obs["state"][:, 0, 0],
        )
        self.assertEqual(len(buf.get_unscored_cid_indices()), 0)

        batch = buf.sample(batch_size=32, pack_obs_and_next_obs=False)
        for i in range(32):
            self.assertAlmostEqual(
                float(batch["cid_scores"][i]),
                float(batch["observations"]["state"][i, 0, 0]),
                places=4,
            )
            self.assertAlmostEqual(
                float(batch["next_cid_scores"][i]),
                float(batch["next_observations"]["state"][i, 0, 0]),
                places=4,
            )

    def test_memory_efficient_buffer_cid_prioritized_replay(self):
        import gymnasium as gym
        from serl_launcher.data.memory_efficient_replay_buffer import (
            MemoryEfficientReplayBuffer,
        )

        obs_space = gym.spaces.Dict(
            {
                "external": gym.spaces.Box(0, 255, shape=(1, 8, 8, 3), dtype=np.uint8),
                "wrist": gym.spaces.Box(0, 255, shape=(1, 8, 8, 3), dtype=np.uint8),
                "state": gym.spaces.Box(-10, 10, shape=(1, 4), dtype=np.float32),
            }
        )
        act_space = gym.spaces.Box(-1, 1, shape=(4,), dtype=np.float32)

        for mode in ("ranking", "softmax", "proportional"):
            buf = MemoryEfficientReplayBuffer(
                obs_space,
                act_space,
                capacity=50,
                pixel_keys=("external", "wrist"),
                include_cid_scores=True,
                cid_prioritized_replay=True,
                cid_replay_mode=mode,
                cid_replay_alpha=1.0,
                cid_replay_uniform_mix=0.1,
            )
            buf.seed(0)
            for t in range(20):
                img = np.full((1, 8, 8, 3), t, dtype=np.uint8)
                buf.insert(
                    {
                        "observations": {
                            "external": img,
                            "wrist": img,
                            "state": np.full((1, 4), float(t), dtype=np.float32),
                        },
                        "next_observations": {
                            "external": img,
                            "wrist": img,
                            "state": np.full((1, 4), float(t + 1), dtype=np.float32),
                        },
                        "actions": np.zeros((4,), dtype=np.float32),
                        "rewards": np.float32(0.0),
                        "masks": np.float32(1.0),
                        "dones": bool(t == 19),
                    }
                )
            # Before scoring (all NaN), sampling must safely fall back to uniform without crashing
            pre_batch = buf.sample(batch_size=16, pack_obs_and_next_obs=False)
            self.assertEqual(pre_batch["cid_scores"].shape, (16,))

            # Assign strongly bimodal CID scores: top 2 slots get 5.0, rest get 0.0
            valid = buf.get_unscored_cid_indices()
            scores = np.zeros((len(valid),), dtype=np.float32)
            scores[-2:] = 5.0
            buf.set_cid_scores(valid, scores, scores)

            stats = buf.get_cid_replay_stats()
            self.assertEqual(stats["enabled"], 1)
            self.assertGreater(stats["p_max_over_uniform"], 1.5)
            self.assertLess(stats["effective_sample_frac"], 0.95)

            # High-CID slots (2 out of ~19 -> ~10.5% under uniform) should be heavily enriched
            batch = buf.sample(batch_size=1000, pack_obs_and_next_obs=False)
            hi_frac = float(np.mean(batch["cid_scores"] > 2.5))
            self.assertGreater(hi_frac, 0.25, f"mode={mode} did not enrich high-CID transitions")


if __name__ == "__main__":
    unittest.main()
