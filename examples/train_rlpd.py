#!/usr/bin/env python3

import copy
import glob
import math
import os
import pickle as pkl
import select
import sys
import termios
import threading
import time
import tty
from collections.abc import Mapping

from absl import app, flags
from flax.training import checkpoints
from gymnasium.wrappers.record_episode_statistics import RecordEpisodeStatistics
import jax
import jax.numpy as jnp
from natsort import natsorted
import numpy as np
import tqdm

# Prefer this checkout even when the active environment has a stale editable install.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCAL_SERL_LAUNCHER = os.path.join(REPO_ROOT, "serl_launcher")
LOCAL_ROBOT_INFRA = os.path.join(REPO_ROOT, "serl_robot_infra")
for source_root in (REPO_ROOT, LOCAL_ROBOT_INFRA, LOCAL_SERL_LAUNCHER):
    if source_root not in sys.path:
        sys.path.insert(0, source_root)

from agentlace.data.data_store import QueuedDataStore
from agentlace.trainer import TrainerClient, TrainerServer

from experiments.mappings import CONFIG_MAPPING
from serl_launcher.agents.continuous.sac import SACAgent
from serl_launcher.agents.continuous.sac_hybrid_dual import SACAgentHybridDualArm
from serl_launcher.agents.continuous.sac_hybrid_single import SACAgentHybridSingleArm
from serl_launcher.data.data_store import MemoryEfficientReplayBufferDataStore
from serl_launcher.utils.cid_reward_utils import (
    OnlineCIDScorer,
    compute_shaped_cid_rewards,
    load_cid_bundle,
    save_cid_bundle,
    train_cid_bundle_from_demos,
)
from serl_launcher.utils.launcher import (
    make_sac_pixel_agent,
    make_sac_pixel_agent_hybrid_dual_arm,
    make_sac_pixel_agent_hybrid_single_arm,
    make_trainer_config,
    make_wandb_logger,
)
from serl_launcher.utils.timer_utils import Timer
from serl_launcher.utils.train_utils import concat_batches

FLAGS = flags.FLAGS

flags.DEFINE_string("exp_name", "pick_place_banana", "Name of experiment corresponding to folder.")
flags.DEFINE_integer("seed", 42, "Random seed.")
flags.DEFINE_boolean("learner", False, "Whether this is a learner.")
flags.DEFINE_boolean("actor", False, "Whether this is an actor.")
flags.DEFINE_string("ip", "localhost", "IP address of the learner.")
flags.DEFINE_multi_string("demo_path", None, "Path to the demo data.")
flags.DEFINE_string("checkpoint_path", None, "Path to save checkpoints.")
flags.DEFINE_boolean(
    "resume_training",
    False,
    "Resume from an existing checkpoint_path. Defaults to starting a new run.",
)
flags.DEFINE_boolean(
    "allow_existing_checkpoint_path",
    False,
    "Allow actor to attach to an active run directory without restoring from disk.",
)
flags.DEFINE_integer("eval_checkpoint_step", 0, "Step to evaluate the checkpoint.")
flags.DEFINE_integer("eval_n_trajs", 0, "Number of trajectories to evaluate.")
flags.DEFINE_boolean("save_video", False, "Save video.")
flags.DEFINE_string(
    "cid_model_path",
    None,
    "Optional path to pre-trained cid_bundle.pkl. If omitted and use_cid_reward=True, "
    "learner loads <checkpoint_path>/cid_causal_model/cid_bundle.pkl or pretrains from --demo_path.",
)
flags.DEFINE_boolean(
    "use_cid_reward",
    None,
    "Override config.use_cid_reward (True to enable Distilled-Object CID causal reward).",
)
flags.DEFINE_boolean(
    "use_cid_replay",
    None,
    "Override config.use_cid_replay (True to enable CID prioritized experience replay).",
)
flags.DEFINE_boolean(
    "debug", False, "Debug mode (disables WandB logging)."
)


devices = jax.local_devices()
num_devices = len(devices)
sharding = jax.sharding.PositionalSharding(devices)


def print_green(x):
    return print("\033[92m {}\033[00m".format(x))


class ManualEpisodeLabeler:
    def __init__(self):
        self.success = False
        self.failure = False
        self.pressed_keys = set()
        self.listener = None
        self.keyboard = None
        self.stdin_fd = None
        self.stdin_settings = None

    def start(self):
        try:
            if sys.stdin.isatty():
                self.stdin_fd = sys.stdin.fileno()
                self.stdin_settings = termios.tcgetattr(self.stdin_fd)
                tty.setcbreak(self.stdin_fd)
        except Exception as exc:
            print(f"Manual stdin labels disabled: {exc}")

        try:
            from pynput import keyboard as pynput_keyboard

            self.keyboard = pynput_keyboard
            self.listener = self.keyboard.Listener(
                on_press=self._on_press, on_release=self._on_release
            )
            self.listener.start()
        except Exception as exc:
            print(f"Manual pynput labels disabled: {exc}")

        print("Manual actor labels: SPACE/s = success and reset, f/ESC = failure and reset.")

    def stop(self):
        if self.listener is not None:
            self.listener.stop()
        if self.stdin_fd is not None and self.stdin_settings is not None:
            try:
                termios.tcsetattr(self.stdin_fd, termios.TCSADRAIN, self.stdin_settings)
            except Exception:
                pass

    def consume(self):
        self._poll_stdin()
        if self.success:
            self.success = False
            self.failure = False
            return "success"
        if self.failure:
            self.failure = False
            return "failure"
        return None

    def _mark_char(self, char):
        if char in (" ", "s", "S"):
            self.success = True
        elif char in ("f", "F", "\x1b"):
            self.failure = True

    def _poll_stdin(self):
        if self.stdin_fd is None:
            return
        try:
            while select.select([sys.stdin], [], [], 0)[0]:
                self._mark_char(sys.stdin.read(1))
        except Exception:
            pass

    def _on_press(self, key):
        if self.keyboard is not None and key == self.keyboard.Key.space:
            token = "space"
            if token not in self.pressed_keys:
                self.pressed_keys.add(token)
                self._mark_char(" ")
            return
        if self.keyboard is not None and key == self.keyboard.Key.esc:
            token = "esc"
            if token not in self.pressed_keys:
                self.pressed_keys.add(token)
                self._mark_char("\x1b")
            return
        try:
            token = key.char
            if token not in self.pressed_keys:
                self.pressed_keys.add(token)
                self._mark_char(token)
        except AttributeError:
            pass

    def _on_release(self, key):
        if self.keyboard is not None and key == self.keyboard.Key.space:
            self.pressed_keys.discard("space")
            return
        if self.keyboard is not None and key == self.keyboard.Key.esc:
            self.pressed_keys.discard("esc")
            return
        try:
            self.pressed_keys.discard(key.char)
        except AttributeError:
            pass


def _iter_env_chain(env):
    current = env
    seen = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        yield current
        current = getattr(current, "env", None)


def _call_env_chain(env, method_name):
    for current in _iter_env_chain(env):
        method = getattr(current, method_name, None)
        if callable(method):
            method()


def prepare_manual_reset(env, action_filter):
    action_filter.reset()
    _call_env_chain(env, "clear_intervention")


def align_transition_action(transition, action_space):
    if "actions" not in transition:
        return transition

    target_shape = tuple(action_space.shape)
    target_dim = int(np.prod(target_shape))
    action = np.asarray(transition["actions"], dtype=np.float32).reshape(-1)

    if action.shape[0] == 7 and target_dim == 4:
        action = action[[0, 1, 2, 6]]
    elif action.shape[0] > target_dim:
        action = action[:target_dim]
    elif action.shape[0] < target_dim:
        action = np.pad(action, (0, target_dim - action.shape[0]), constant_values=0.0)

    transition["actions"] = action.reshape(target_shape).astype(np.float32)
    return transition


def validate_value_shape(value, space, field_path):
    if hasattr(space, "spaces"):
        if not isinstance(value, Mapping):
            raise ValueError(f"{field_path} must be a mapping, got {type(value).__name__}")
        missing_keys = sorted(set(space.spaces) - set(value))
        if missing_keys:
            raise ValueError(f"{field_path} is missing keys: {missing_keys}")
        for key, child_space in space.spaces.items():
            validate_value_shape(value[key], child_space, f"{field_path}.{key}")
        return

    expected_shape = tuple(space.shape)
    actual_shape = tuple(np.asarray(value).shape)
    if actual_shape != expected_shape:
        raise ValueError(
            f"{field_path} has shape {actual_shape}, expected {expected_shape}"
        )


def validate_loaded_transition_schema(transition, env, source):
    try:
        validate_value_shape(transition["observations"], env.observation_space, "observations")
        validate_value_shape(
            transition["next_observations"], env.observation_space, "next_observations"
        )
        validate_value_shape(transition["actions"], env.action_space, "actions")
    except (KeyError, ValueError) as exc:
        raise ValueError(
            f"Incompatible transition schema in {source}: {exc}. "
            "Use demos and resume buffers collected with the current environment schema."
        ) from exc


def prepare_loaded_transition(transition, env, include_grasp_penalty, source="loaded data"):
    transition = copy.deepcopy(transition)
    transition = align_transition_action(transition, env.action_space)
    if "infos" in transition and "grasp_penalty" in transition["infos"]:
        transition["grasp_penalty"] = transition["infos"]["grasp_penalty"]
    elif include_grasp_penalty and "grasp_penalty" not in transition:
        transition["grasp_penalty"] = 0.0
    validate_loaded_transition_schema(transition, env, source)
    return transition


def latest_step_from_path(path, prefix):
    if not path:
        return None
    name = os.path.basename(path.rstrip(os.sep))
    name = os.path.splitext(name)[0]
    if not name.startswith(prefix):
        return None
    suffix = name[len(prefix):]
    digits = []
    for ch in suffix:
        if not ch.isdigit():
            break
        digits.append(ch)
    if not digits:
        return None
    return int("".join(digits))


def get_resume_step(checkpoint_path):
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        return 0

    resume_steps = []
    buffer_files = natsorted(glob.glob(os.path.join(checkpoint_path, "buffer", "transitions_*.pkl")))
    for buffer_file in buffer_files:
        step = latest_step_from_path(buffer_file, "transitions_")
        if step is not None:
            resume_steps.append(step + 1)

    latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(checkpoint_path))
    step = latest_step_from_path(latest_ckpt, "checkpoint_")
    if step is not None:
        resume_steps.append(step + 1)

    return max(resume_steps, default=0)


def atomic_pickle_dump(obj, path):
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    tmp_path = f"{path}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        pkl.dump(obj, f, protocol=pkl.HIGHEST_PROTOCOL)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def load_transition_file(path):
    try:
        with open(path, "rb") as f:
            obj = pkl.load(f)
    except EOFError:
        with open(path, "rb") as f:
            data = f.read()
        obj = pkl.loads(data + b"ue.")
        print(f"Recovered truncated pickle while loading {path}")
    required = {"observations", "actions", "next_observations", "rewards", "masks", "dones"}
    if isinstance(obj, dict) and required.issubset(obj.keys()):
        return [obj]
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict) and required.issubset(x.keys())]
    return []


##############################################################################


class EMAActionFilter:
    def __init__(self, hz: float, cutoff_hz: float = 2.0, filter_rot=True, filter_gripper=True):
        self.dt = 1.0 / float(hz)
        tau = 1.0 / (2.0 * math.pi * float(cutoff_hz))
        self.alpha = self.dt / (tau + self.dt)
        self.filter_rot = filter_rot
        self.filter_gripper = filter_gripper
        self.prev = None

    def __call__(self, a: np.ndarray) -> np.ndarray:
        a = np.asarray(a, dtype=np.float32).copy()
        if self.prev is None:
            self.prev = a.copy()
            return a

        if a.shape[0] == 4:
            idx_end = 3
            gripper_idx = 3
        else:
            idx_end = min(6, a.shape[0])
            gripper_idx = 6 if a.shape[0] > 6 else None
        self.prev[:idx_end] = self.prev[:idx_end] + self.alpha * (a[:idx_end] - self.prev[:idx_end])

        if gripper_idx is not None:
            if self.filter_gripper:
                self.prev[gripper_idx] = self.prev[gripper_idx] + self.alpha * (
                    a[gripper_idx] - self.prev[gripper_idx]
                )
            else:
                self.prev[gripper_idx] = a[gripper_idx]

        return np.clip(self.prev, -1.0, 1.0).astype(np.float32)

    def reset(self):
        self.prev = None


##############################################################################


def actor(agent, data_store, intvn_data_store, env, sampling_rng):
    """The actor loop (10Hz real-robot control + SpaceMouse HIL intervention)."""
    action_filter = EMAActionFilter(hz=10, cutoff_hz=2)
    if FLAGS.eval_checkpoint_step:
        success_counter = 0
        time_list = []

        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
            step=FLAGS.eval_checkpoint_step,
        )
        agent = agent.replace(state=ckpt)

        manual_labeler = ManualEpisodeLabeler()
        manual_labeler.start()
        try:
            for episode in range(FLAGS.eval_n_trajs):
                obs, _ = env.reset()
                done = False
                truncated = False
                start_time = time.time()
                while not (done or truncated):
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        argmax=True,
                        seed=key,
                    )
                    actions = np.asarray(jax.device_get(actions))
                    actions = action_filter(actions)
                    next_obs, reward, done, truncated, info = env.step(actions)
                    obs = next_obs

                    manual_label = manual_labeler.consume()
                    if manual_label == "success":
                        reward = 1.0
                        done = True
                        truncated = False
                        print("manual eval success marked; resetting environment")
                    elif manual_label == "failure":
                        reward = 0.0
                        done = True
                        truncated = False
                        print("manual eval failure marked; resetting environment")

                    if done or truncated:
                        if reward:
                            dt = time.time() - start_time
                            time_list.append(dt)
                            print(dt)
                        action_filter.reset()
                        success_counter += reward
                        print(reward)
                        print(f"{success_counter}/{episode + 1}")

            print(f"success rate: {success_counter / FLAGS.eval_n_trajs}")
            print(f"average time: {np.mean(time_list) if time_list else float('nan')}")
        finally:
            try:
                manual_labeler.stop()
            except Exception:
                pass
            try:
                env.close()
            except Exception as exc:
                print(f"eval cleanup failed: {exc}")
        return

    start_step = (
        get_resume_step(FLAGS.checkpoint_path)
        if (FLAGS.resume_training or FLAGS.allow_existing_checkpoint_path)
        else 0
    )

    datastore_dict = {
        "actor_env": data_store,
        "actor_env_intvn": intvn_data_store,
    }

    client = TrainerClient(
        "actor_env",
        FLAGS.ip,
        make_trainer_config(),
        data_stores=datastore_dict,
        wait_for_server=True,
        timeout_ms=500,
    )

    pending_network = {"params": None, "count": 0, "applied": 0}
    pending_network_lock = threading.Lock()

    def update_params(params):
        with pending_network_lock:
            pending_network["params"] = params
            pending_network["count"] += 1

    def apply_pending_network(force=False):
        nonlocal agent
        if not force and step % config.steps_per_update != 0:
            return
        with pending_network_lock:
            params = pending_network["params"]
            count = pending_network["count"]
            pending_network["params"] = None
        if params is not None:
            agent = agent.replace(state=agent.state.replace(params=params))
            skipped = max(0, count - pending_network["applied"] - 1)
            pending_network["applied"] = count
            if skipped:
                print_green(f"Applied latest learner params; skipped {skipped} stale updates")
            else:
                print_green("Applied latest learner params")

    client.recv_network_callback(update_params)

    current_trajectory = []
    trajectory_index = 0
    actor_stats = {
        "saved_trajectories": 0,
        "saved_success_trajectories": 0,
        "saved_failure_trajectories": 0,
        "saved_intervention_trajectories": 0,
        "saved_policy_trajectories": 0,
        "saved_transitions": 0,
        "saved_intervention_samples": 0,
        "saved_intervention_time_s": 0.0,
    }
    last_buffer_dump_step = start_step
    last_seen_step = start_step
    actor_sync_period = max(1, int(os.getenv("ACTOR_SYNC_PERIOD_STEPS", "10")))
    last_actor_sync_step = start_step

    def mark_transition_intervention(transition, intervention, trajectory_had_intervention=False):
        transition["intervention"] = bool(intervention)
        transition["trajectory_had_intervention"] = bool(trajectory_had_intervention)
        info = copy.deepcopy(transition.get("infos", {}))
        info["intervention"] = bool(intervention)
        info["trajectory_had_intervention"] = bool(trajectory_had_intervention)
        transition["infos"] = info
        return transition

    def dump_trajectory(trajectory, step):
        nonlocal trajectory_index
        if FLAGS.checkpoint_path is None or not trajectory:
            return 0
        trajectory_had_intvn = any(bool(t.get("intervention", False)) for t in trajectory)
        reward_value = float(np.asarray(trajectory[-1].get("rewards", 0.0)).reshape(()))
        done_value = bool(trajectory[-1].get("dones", False))
        success = bool(reward_value > 0.5 and done_value)
        label = "success" if success else "failure"
        source = "intvn" if trajectory_had_intvn else "policy"
        trajectory = [
            mark_transition_intervention(
                copy.deepcopy(t), t.get("intervention", False), trajectory_had_intvn
            )
            for t in trajectory
        ]
        filename = f"transitions_{step}_{trajectory_index:06d}_{label}_{source}.pkl"
        buffer_path = os.path.join(FLAGS.checkpoint_path, "buffer", filename)
        atomic_pickle_dump(trajectory, buffer_path)

        intervention_samples = [
            copy.deepcopy(t) for t in trajectory if bool(t.get("intervention", False))
        ]
        if intervention_samples:
            for transition in intervention_samples:
                transition["_reset_frame_stack"] = True
            demo_filename = f"interventions_{step}_{trajectory_index:06d}_{label}.pkl"
            demo_path = os.path.join(FLAGS.checkpoint_path, "demo_buffer", demo_filename)
            atomic_pickle_dump(intervention_samples, demo_path)
            for transition in intervention_samples:
                intvn_data_store.insert(copy.deepcopy(transition))

        actor_stats["saved_trajectories"] += 1
        actor_stats["saved_success_trajectories"] += int(success)
        actor_stats["saved_failure_trajectories"] += int(not success)
        actor_stats["saved_intervention_trajectories"] += int(trajectory_had_intvn)
        actor_stats["saved_policy_trajectories"] += int(not trajectory_had_intvn)
        actor_stats["saved_transitions"] += len(trajectory)
        actor_stats["saved_intervention_samples"] += len(intervention_samples)
        trajectory_intervention_time_s = 0.0
        for transition in trajectory:
            info = transition.get("infos", {})
            if isinstance(info, dict):
                trajectory_intervention_time_s += float(info.get("intervention_duration_s", 0.0))
        actor_stats["saved_intervention_time_s"] += trajectory_intervention_time_s

        stats_payload = {
            "buffer/actor_saved_trajectories": actor_stats["saved_trajectories"],
            "buffer/actor_saved_success_trajectories": actor_stats["saved_success_trajectories"],
            "buffer/actor_saved_failure_trajectories": actor_stats["saved_failure_trajectories"],
            "buffer/actor_saved_intervention_trajectories": actor_stats["saved_intervention_trajectories"],
            "buffer/actor_saved_policy_trajectories": actor_stats["saved_policy_trajectories"],
            "buffer/actor_saved_transitions": actor_stats["saved_transitions"],
            "buffer/actor_saved_intervention_samples": actor_stats["saved_intervention_samples"],
            "buffer/actor_saved_intervention_time_s": actor_stats["saved_intervention_time_s"],
            "human_intervention/total_time_s": actor_stats["saved_intervention_time_s"],
            "buffer/last_trajectory_length": len(trajectory),
            "buffer/last_trajectory_success": int(success),
            "buffer/last_trajectory_intervention": int(trajectory_had_intvn),
            "buffer/last_trajectory_intervention_samples": len(intervention_samples),
            "buffer/last_trajectory_intervention_time_s": trajectory_intervention_time_s,
            "human_intervention/last_trajectory_time_s": trajectory_intervention_time_s,
        }
        try:
            client.request("send-stats", stats_payload)
        except Exception as exc:
            print(f"actor stats log failed: {exc}", flush=True)
        trajectory_index += 1
        print(
            f"saved trajectory step={step} len={len(trajectory)} label={label} "
            f"intervention={trajectory_had_intvn}",
            flush=True,
        )
        return len(trajectory)

    def sync_actor_data(step, force=False):
        nonlocal last_actor_sync_step
        if not force and step - last_actor_sync_step < actor_sync_period:
            return
        try:
            if client.update():
                last_actor_sync_step = step
        except Exception as exc:
            print(f"actor datastore sync failed: {exc}", flush=True)

    manual_labeler = ManualEpisodeLabeler()
    manual_labeler.start()

    obs, _ = env.reset()
    action_filter.reset()

    timer = Timer()
    running_return = 0.0
    already_intervened = False
    trajectory_had_intervention = False
    intervention_count = 0
    intervention_steps = 0
    intervention_time_s = 0.0

    pbar = tqdm.tqdm(range(start_step, config.max_steps), dynamic_ncols=True)
    try:
        for step in pbar:
            last_seen_step = step
            timer.tick("total")
            apply_pending_network()

            manual_label = manual_labeler.consume()
            if manual_label is not None:
                prepare_manual_reset(env, action_filter)
                reward = 1.0 if manual_label == "success" else 0.0
                info = {
                    "succeed": manual_label == "success",
                    "episode": {
                        "intervention_count": intervention_count,
                        "intervention_steps": intervention_steps,
                        "intervention_time_s": intervention_time_s,
                    },
                }
                zero_action = np.zeros(env.action_space.shape, dtype=np.float32)
                transition = dict(
                    observations=obs,
                    actions=zero_action,
                    next_observations=obs,
                    rewards=reward,
                    masks=0.0,
                    dones=True,
                    infos=copy.deepcopy(info),
                )
                if config.setup_mode in ("single-arm-learned-gripper", "dual-arm-learned-gripper"):
                    transition["grasp_penalty"] = 0.0
                transition = mark_transition_intervention(transition, False, trajectory_had_intervention)
                data_store.insert(transition)
                current_trajectory.append(copy.deepcopy(transition))
                dump_trajectory(current_trajectory, step)
                current_trajectory = []
                sync_actor_data(step, force=True)
                running_return += reward
                print(f"manual {manual_label} marked; resetting environment", flush=True)
                obs, _ = env.reset()
                pbar.set_description(f"last return: {running_return}")
                running_return = 0.0
                intervention_count = 0
                intervention_steps = 0
                intervention_time_s = 0.0
                already_intervened = False
                trajectory_had_intervention = False
                action_filter.reset()
                timer.tock("total")
                continue

            with timer.context("sample_actions"):
                if step < config.random_steps:
                    actions = env.action_space.sample()
                else:
                    sampling_rng, key = jax.random.split(sampling_rng)
                    actions = agent.sample_actions(
                        observations=jax.device_put(obs),
                        seed=key,
                        argmax=False,
                    )
                    actions = np.asarray(jax.device_get(actions))
                    actions = action_filter(actions)

            with timer.context("step_env"):
                step_env_start_time = time.time()
                next_obs, reward, done, truncated, info = env.step(actions)
                step_env_time_s = time.time() - step_env_start_time
                info.pop("left", None)
                info.pop("right", None)

                if "intervene_action" in info:
                    actions = info.pop("intervene_action")
                    print("\n\n INTERVENING: ", actions)
                    intervention_steps += 1
                    intervention_time_s += step_env_time_s
                    if not already_intervened:
                        intervention_count += 1
                    already_intervened = True
                    current_intervention = True
                    trajectory_had_intervention = True
                else:
                    actions = info.pop("executed_action", actions)
                    already_intervened = False
                    current_intervention = False

                manual_label = manual_labeler.consume()
                if manual_label == "success":
                    prepare_manual_reset(env, action_filter)
                    reward = 1.0
                    done = True
                    truncated = False
                    info["succeed"] = True
                    info.setdefault("episode", {})
                    print("manual success marked; resetting environment")
                elif manual_label == "failure":
                    prepare_manual_reset(env, action_filter)
                    reward = 0.0
                    done = True
                    truncated = False
                    info["succeed"] = False
                    info.setdefault("episode", {})
                    print("manual failure marked; resetting environment")

                terminal = bool(done or truncated)
                info["intervention_duration_s"] = step_env_time_s if current_intervention else 0.0
                info["episode_intervention_time_s"] = intervention_time_s
                if terminal:
                    info.setdefault("succeed", bool(reward > 0.0 and not truncated))
                    info.setdefault("episode", {})
                    info["episode"]["intervention_count"] = intervention_count
                    info["episode"]["intervention_steps"] = intervention_steps
                    info["episode"]["intervention_time_s"] = intervention_time_s
                running_return += reward
                transition = dict(
                    observations=obs,
                    actions=actions,
                    next_observations=next_obs,
                    rewards=reward,
                    masks=1.0 - float(done),
                    dones=terminal,
                    infos=copy.deepcopy(info),
                )
                if "grasp_penalty" in info:
                    transition["grasp_penalty"] = info["grasp_penalty"]

                transition = mark_transition_intervention(
                    transition, current_intervention, trajectory_had_intervention
                )
                data_store.insert(transition)
                current_trajectory.append(copy.deepcopy(transition))
                obs = next_obs
                sync_actor_data(step)

                if done or truncated:
                    pbar.set_description(f"last return: {running_return}")
                    dump_trajectory(current_trajectory, step)
                    current_trajectory = []
                    sync_actor_data(step, force=True)
                    running_return = 0.0
                    intervention_count = 0
                    intervention_steps = 0
                    intervention_time_s = 0.0
                    already_intervened = False
                    trajectory_had_intervention = False
                    prepare_manual_reset(env, action_filter)
                    obs, _ = env.reset()
                    action_filter.reset()

            if config.buffer_period > 0 and step - last_buffer_dump_step >= config.buffer_period:
                last_buffer_dump_step = step

            timer.tock("total")
    finally:
        try:
            pbar.close()
        except Exception:
            pass
        try:
            sync_actor_data(last_seen_step, force=True)
        except Exception as exc:
            print(f"final actor sync failed: {exc}")
        try:
            manual_labeler.stop()
        except Exception:
            pass
        try:
            env.close()
        except Exception as exc:
            print(f"actor cleanup failed: {exc}")


##############################################################################


def _resolve_or_train_cid_bundle(use_cid_reward: bool):
    """Load pre-trained cid_bundle.pkl or pretrain once from --demo_path if missing."""
    if not use_cid_reward:
        return None

    candidate_paths = []
    if FLAGS.cid_model_path:
        candidate_paths.append(FLAGS.cid_model_path)
    cfg_path = getattr(config, "cid_model_path", "")
    if cfg_path:
        candidate_paths.append(cfg_path)
    if FLAGS.checkpoint_path is not None:
        candidate_paths.append(
            os.path.join(
                os.path.abspath(FLAGS.checkpoint_path),
                "cid_causal_model",
                "cid_bundle.pkl",
            )
        )

    for path in candidate_paths:
        if path and os.path.exists(path):
            print_green(f"[CID] Loaded frozen CID causal model bundle from: {path}")
            return load_cid_bundle(path)

    # Auto-pretrain once from --demo_path if no bundle exists yet
    if not FLAGS.demo_path:
        raise FileNotFoundError(
            "use_cid_reward=True but no --cid_model_path found and --demo_path is empty."
        )
    obj_extractor = str(getattr(config, "cid_obj_extractor", "hsv_banana"))
    ext_key = str(getattr(config, "cid_external_key", "external"))
    wrist_key = str(getattr(config, "cid_wrist_key", "wrist"))
    print_green(
        f"[CID] No pre-built cid_bundle.pkl found. Auto-pretraining from {FLAGS.demo_path} "
        f"(extractor={obj_extractor})..."
    )
    bundle = train_cid_bundle_from_demos(
        FLAGS.demo_path,
        obj_extractor=obj_extractor,
        external_coords_path=getattr(config, "cid_external_coords_path", None) or None,
        image_keys=(ext_key, wrist_key),
        target_action_dim=int(getattr(config, "policy_action_dim", 0)) or None,
        pca_dim=int(getattr(config, "cid_pca_dim", 32)),
        ensemble_size=int(getattr(config, "cid_ensemble_size", 5)),
        obj_steps=int(getattr(config, "cid_obj_steps", 800)),
        dyn_steps=int(getattr(config, "cid_dyn_steps", 1200)),
        n_action_samples=int(getattr(config, "cid_action_samples", 128)),
        action_sampling_mode=str(getattr(config, "cid_action_sampling_mode", "uniform")),
        state_min_std=float(getattr(config, "cid_state_min_std", 0.05)),
        state_clip=float(getattr(config, "cid_state_clip", 5.0)),
        sigma0_sq=float(getattr(config, "cid_sigma0_sq", 0.25)),
        ood_beta=float(getattr(config, "cid_ood_beta", 2.0)),
        use_spatial_features=bool(getattr(config, "cid_use_spatial_features", True)),
        cv_folds=int(getattr(config, "cid_cv_folds", 5)),
        seed=FLAGS.seed,
    )
    if FLAGS.checkpoint_path is not None:
        save_path = os.path.join(
            os.path.abspath(FLAGS.checkpoint_path), "cid_causal_model", "cid_bundle.pkl"
        )
        save_cid_bundle(save_path, bundle)
        print_green(f"[CID] Saved auto-pretrained bundle to: {save_path}")
    print_green(f"[CID] Validation metrics: {bundle.metrics}")
    return bundle


def learner(
    rng,
    agent,
    replay_buffer,
    demo_buffer,
    use_cid_reward=True,
    use_cid_replay=True,
    wandb_logger=None,
):
    """The learner loop: RLPD / SAC + lazy-cached CID Causal Reward & Experience Replay."""
    start_step = get_resume_step(FLAGS.checkpoint_path) if FLAGS.resume_training else 0
    step = start_step
    use_cid_model = bool(use_cid_reward or use_cid_replay)

    def stats_callback(type: str, payload: dict) -> dict:
        assert type == "send-stats", f"Invalid request type: {type}"
        if wandb_logger is not None:
            wandb_logger.log(dict(payload), step=step)
        return {}

    # ---- T0: Load frozen CID causal model & pre-score initial buffers ----
    cid_bundle = _resolve_or_train_cid_bundle(use_cid_model)
    cid_scorer = (
        OnlineCIDScorer(
            cid_bundle,
            jit_batch_size=int(getattr(config, "cid_jit_batch_size", 32)),
            sigma0_sq=float(getattr(config, "cid_sigma0_sq", cid_bundle.sigma0_sq)),
            ood_beta=float(getattr(config, "cid_ood_beta", cid_bundle.ood_beta)),
        )
        if cid_bundle is not None
        else None
    )
    cid_baseline = float(cid_bundle.baseline_score) if cid_bundle is not None else 0.0
    cid_bonus_weight = float(getattr(config, "cid_bonus_weight", 0.05))
    cid_bonus_min_weight = float(getattr(config, "cid_bonus_min_weight", 0.0))
    cid_bonus_max = float(getattr(config, "cid_bonus_max", 0.5))
    cid_bonus_decay_steps = int(getattr(config, "cid_bonus_decay_steps", 15000))
    cid_reward_mode = str(getattr(config, "cid_reward_mode", "potential"))
    cid_step_penalty = float(getattr(config, "cid_step_penalty", 0.02))

    cid_stats = {
        "scored_new_online": 0,
        "scored_new_demo": 0,
        "last_online_epistemic_var": 0.0,
        "last_lambda_t": 0.0,
        "last_bonus_mean": 0.0,
        "last_bonus_max": 0.0,
        "last_bonus_nonzero_frac": 0.0,
        "last_online_score_mean": 0.0,
        "last_demo_score_mean": 0.0,
    }

    def score_unscored_buffer(buffer_store, tag="online", max_items=None):
        """Main-thread lazy incremental CID scoring (fixed-batch JIT, 0 recompiles)."""
        if cid_scorer is None:
            return 0
        indices, obs_batch, next_obs_batch = buffer_store.snapshot_unscored_cid_observations(
            max_items=max_items, include_next=True
        )
        if obs_batch is None or len(indices) == 0:
            return 0
        scores, epist_var, _ = cid_scorer.score_observations(obs_batch, return_details=True)
        next_scores = cid_scorer.score_observations(next_obs_batch, return_details=False)
        buffer_store.set_cid_scores(indices, scores, next_scores=next_scores)
        if tag == "online":
            cid_stats["scored_new_online"] += len(indices)
            cid_stats["last_online_epistemic_var"] = float(np.mean(epist_var))
        else:
            cid_stats["scored_new_demo"] += len(indices)
        return int(len(indices))

    if cid_scorer is not None:
        n_demo_scored = score_unscored_buffer(demo_buffer, tag="demo")
        n_online_scored = score_unscored_buffer(replay_buffer, tag="online")
        print_green(
            f"[CID] T0 pre-scored buffers: demo={n_demo_scored}, online={n_online_scored}, "
            f"reward={'ON' if use_cid_reward else 'OFF'} (mode={cid_reward_mode}, w0={cid_bonus_weight}), "
            f"replay={'ON' if use_cid_replay else 'OFF'} "
            f"(mode={getattr(config, 'cid_replay_mode', 'ranking')}, "
            f"alpha={getattr(config, 'cid_replay_alpha', 0.6)}), "
            f"c_base={cid_baseline:.4f}"
        )

    # Start actor communication only after T0 CID initialization
    server = TrainerServer(make_trainer_config(), request_callback=stats_callback)
    server.register_data_store("actor_env", replay_buffer)
    server.register_data_store("actor_env_intvn", demo_buffer)
    server.start(threaded=True)
    server.publish_network(agent.state.params)
    print_green("sent initial network to actor")

    # RLPD symmetric sampling setup (50% replay + 50% demo once online_ready)
    demo_frac = 0.5
    demo_bs = max(1, min(config.batch_size - 1, int(round(config.batch_size * demo_frac))))
    online_bs = config.batch_size - demo_bs
    print_green(
        f"Sampling schedule: warmup 100% demo ({config.batch_size}); "
        f"after {config.training_starts} online transitions -> demo={demo_bs}, online={online_bs}"
    )

    replay_iterator = None
    using_online_replay = False

    demo_only_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": config.batch_size, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )
    demo_iterator = demo_buffer.get_iterator(
        sample_args={"batch_size": demo_bs, "pack_obs_and_next_obs": True},
        device=sharding.replicate(),
    )

    def _apply_cid_reward_shaping(batch, current_step: int, n_online: int, record_metrics: bool = False):
        if cid_scorer is None or "cid_scores" not in batch:
            return batch
        if not use_cid_reward:
            if record_metrics:
                cid_scores = batch["cid_scores"]
                safe_scores = jnp.where(jnp.isfinite(cid_scores), cid_scores, float(cid_baseline))
                cid_stats["last_lambda_t"] = 0.0
                cid_stats["last_bonus_mean"] = 0.0
                cid_stats["last_bonus_max"] = 0.0
                cid_stats["last_bonus_nonzero_frac"] = 0.0
                if n_online > 0:
                    cid_stats["last_online_score_mean"] = float(jnp.mean(safe_scores[:n_online]))
                    cid_stats["last_demo_score_mean"] = float(jnp.mean(safe_scores[n_online:]))
                else:
                    cid_stats["last_demo_score_mean"] = float(jnp.mean(safe_scores))
            return batch
        shaped_rewards, bonus, lambda_t = compute_shaped_cid_rewards(
            rewards=batch["rewards"],
            cid_scores=batch["cid_scores"],
            next_cid_scores=batch.get("next_cid_scores", None),
            masks=batch.get("masks", None),
            dones=batch["dones"],
            discount=float(config.discount),
            step=current_step,
            baseline_score=cid_baseline,
            weight=cid_bonus_weight,
            min_weight=cid_bonus_min_weight,
            max_score=cid_bonus_max,
            decay_steps=cid_bonus_decay_steps,
            mode=cid_reward_mode,
            step_penalty=cid_step_penalty,
            xp=jnp,
        )
        if record_metrics:
            cid_scores = batch["cid_scores"]
            safe_scores = jnp.where(jnp.isfinite(cid_scores), cid_scores, float(cid_baseline))
            cid_stats["last_lambda_t"] = float(lambda_t)
            cid_stats["last_bonus_mean"] = float(jnp.mean(bonus))
            cid_stats["last_bonus_max"] = float(jnp.max(jnp.abs(bonus)))
            cid_stats["last_bonus_nonzero_frac"] = float(jnp.mean((jnp.abs(bonus) > 1e-6).astype(jnp.float32)))
            if n_online > 0:
                cid_stats["last_online_score_mean"] = float(jnp.mean(safe_scores[:n_online]))
                cid_stats["last_demo_score_mean"] = float(jnp.mean(safe_scores[n_online:]))
            else:
                cid_stats["last_demo_score_mean"] = float(jnp.mean(safe_scores))

        return batch.copy(add_or_replace={"rewards": shaped_rewards})

    def next_train_batch(current_step: int, record_metrics: bool = False):
        nonlocal replay_iterator, using_online_replay
        online_ready = len(replay_buffer) >= config.training_starts and online_bs > 0
        if online_ready:
            if replay_iterator is None:
                replay_iterator = replay_buffer.get_iterator(
                    sample_args={"batch_size": online_bs, "pack_obs_and_next_obs": True},
                    device=sharding.replicate(),
                )
            if not using_online_replay:
                using_online_replay = True
                print_green(
                    f"Online replay ready: {len(replay_buffer)} transitions. Switching to mixed demo/online batches."
                )
            batch_online = next(replay_iterator)
            batch_demo = next(demo_iterator)
            batch = concat_batches(batch_online, batch_demo, axis=0)
            return _apply_cid_reward_shaping(
                batch, current_step, n_online=online_bs, record_metrics=record_metrics
            )
        batch = next(demo_only_iterator)
        return _apply_cid_reward_shaping(
            batch, current_step, n_online=0, record_metrics=record_metrics
        )

    if isinstance(agent, SACAgent):
        train_critic_networks_to_update = frozenset({"critic"})
        train_networks_to_update = frozenset({"critic", "actor", "temperature"})
    else:
        train_critic_networks_to_update = frozenset({"critic", "grasp_critic"})
        train_networks_to_update = frozenset({"critic", "grasp_critic", "actor", "temperature"})

    timer = Timer()
    for step in tqdm.tqdm(
        range(start_step, config.max_steps), dynamic_ncols=True, desc="learner"
    ):
        # ---- T2: Main-thread lazy incremental CID scoring for newly arrived transitions ----
        if cid_scorer is not None:
            with timer.context("cid_lazy_scoring"):
                score_unscored_buffer(replay_buffer, tag="online", max_items=256)
                score_unscored_buffer(demo_buffer, tag="demo", max_items=256)

        # ---- T3: High-UTD critic + actor updates with dynamically shaped CID rewards ----
        for _ in range(config.cta_ratio - 1):
            with timer.context("sample_train_batch"):
                batch = next_train_batch(step, record_metrics=False)
            with timer.context("train_critics"):
                agent, _ = agent.update(
                    batch,
                    networks_to_update=train_critic_networks_to_update,
                )

        should_log = (step % config.log_period == 0) and (wandb_logger is not None)
        with timer.context("train"):
            batch = next_train_batch(step, record_metrics=should_log)
            agent, update_info = agent.update(
                batch,
                networks_to_update=train_networks_to_update,
            )

        if step > 0 and step % config.steps_per_update == 0:
            agent = jax.block_until_ready(agent)
            server.publish_network(agent.state.params)

        # ---- T4: WandB logging ----
        if step % config.log_period == 0 and wandb_logger:
            replay_prio_stats = replay_buffer.get_cid_replay_stats()
            wandb_logger.log(update_info, step=step)
            wandb_logger.log({"timer": timer.get_average_times()}, step=step)
            wandb_logger.log(
                {
                    "cid/enabled": int(cid_scorer is not None),
                    "cid/reward_enabled": int(bool(use_cid_reward) and cid_scorer is not None),
                    "cid/replay_enabled": int(bool(use_cid_replay) and cid_scorer is not None),
                    "cid/replay_p_max_over_uniform": replay_prio_stats["p_max_over_uniform"],
                    "cid/replay_effective_sample_frac": replay_prio_stats["effective_sample_frac"],
                    "cid/baseline_score": cid_baseline,
                    "cid/lambda_t": cid_stats["last_lambda_t"],
                    "cid/online_score_mean": cid_stats["last_online_score_mean"],
                    "cid/demo_score_mean": cid_stats["last_demo_score_mean"],
                    "cid/bonus_mean": cid_stats["last_bonus_mean"],
                    "cid/bonus_max": cid_stats["last_bonus_max"],
                    "cid/bonus_nonzero_frac": cid_stats["last_bonus_nonzero_frac"],
                    "cid/epistemic_var_online_mean": cid_stats["last_online_epistemic_var"],
                    "cid/scored_online_total": cid_stats["scored_new_online"],
                    "cid/scored_demo_total": cid_stats["scored_new_demo"],
                    "buffer/online_size": len(replay_buffer),
                    "buffer/demo_size": len(demo_buffer),
                    "buffer/online_ready": int(len(replay_buffer) >= config.training_starts),
                    "buffer/using_online_replay": int(using_online_replay),
                    "buffer/training_starts": config.training_starts,
                },
                step=step,
            )

        if (
            step > 0
            and config.checkpoint_period
            and step % config.checkpoint_period == 0
        ):
            checkpoints.save_checkpoint(
                os.path.abspath(FLAGS.checkpoint_path), agent.state, step=step, keep=100
            )


##############################################################################


def main(_):
    global config
    assert FLAGS.exp_name in CONFIG_MAPPING, f"Experiment '{FLAGS.exp_name}' not found."
    config = CONFIG_MAPPING[FLAGS.exp_name]()

    assert config.batch_size % num_devices == 0
    rng = jax.random.PRNGKey(FLAGS.seed)
    rng, sampling_rng = jax.random.split(rng)

    env = config.get_environment(
        fake_env=FLAGS.learner,
        save_video=FLAGS.save_video,
        classifier=True,
    )
    env = RecordEpisodeStatistics(env)
    config.policy_action_dim = int(np.prod(env.action_space.shape))

    rng, sampling_rng = jax.random.split(rng)

    if config.setup_mode in ("single-arm-fixed-gripper", "dual-arm-fixed-gripper"):
        agent: SACAgent = make_sac_pixel_agent(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = False
    elif config.setup_mode == "single-arm-learned-gripper":
        agent: SACAgentHybridSingleArm = make_sac_pixel_agent_hybrid_single_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    elif config.setup_mode == "dual-arm-learned-gripper":
        agent: SACAgentHybridDualArm = make_sac_pixel_agent_hybrid_dual_arm(
            seed=FLAGS.seed,
            sample_obs=env.observation_space.sample(),
            sample_action=env.action_space.sample(),
            image_keys=config.image_keys,
            encoder_type=config.encoder_type,
            discount=config.discount,
        )
        include_grasp_penalty = True
    else:
        raise NotImplementedError(f"Unknown setup mode: {config.setup_mode}")

    agent = jax.device_put(jax.tree.map(jnp.array, agent), sharding.replicate())

    if FLAGS.resume_training:
        if FLAGS.checkpoint_path is None or not os.path.exists(FLAGS.checkpoint_path):
            raise FileNotFoundError(
                "--resume_training requires an existing --checkpoint_path."
            )
        ckpt = checkpoints.restore_checkpoint(
            os.path.abspath(FLAGS.checkpoint_path),
            agent.state,
        )
        agent = agent.replace(state=ckpt)
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        ckpt_step = latest_step_from_path(latest_ckpt, "checkpoint_")
        print_green(f"Loaded previous checkpoint at step {ckpt_step}.")
    elif (
        FLAGS.checkpoint_path is not None
        and os.path.exists(FLAGS.checkpoint_path)
        and not FLAGS.allow_existing_checkpoint_path
    ):
        latest_ckpt = checkpoints.latest_checkpoint(os.path.abspath(FLAGS.checkpoint_path))
        buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer", "*.pkl"))
        demo_buffer_files = glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer", "*.pkl"))
        if latest_ckpt or buffer_files or demo_buffer_files:
            raise FileExistsError(
                f"Checkpoint path already has training state: {FLAGS.checkpoint_path}. "
                "Use --resume_training to continue it, or use a new --checkpoint_path."
            )

    use_cid_reward = (
        bool(FLAGS.use_cid_reward)
        if FLAGS.use_cid_reward is not None
        else bool(getattr(config, "use_cid_reward", True))
    )
    use_cid_replay = (
        bool(FLAGS.use_cid_replay)
        if FLAGS.use_cid_replay is not None
        else bool(getattr(config, "use_cid_replay", True))
    )
    use_cid_model = bool(use_cid_reward or use_cid_replay)
    cid_replay_kwargs = dict(
        cid_replay_mode=str(getattr(config, "cid_replay_mode", "ranking")),
        cid_replay_alpha=float(getattr(config, "cid_replay_alpha", 0.6)),
        cid_replay_eps=float(getattr(config, "cid_replay_eps", 1e-3)),
        cid_replay_uniform_mix=float(getattr(config, "cid_replay_uniform_mix", 0.2)),
    )
    cid_replay_on_demo = bool(use_cid_replay and getattr(config, "cid_replay_on_demo", False))

    def create_replay_buffer_and_wandb_logger():
        replay_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_cid_scores=use_cid_model,
            cid_prioritized_replay=use_cid_replay,
            **cid_replay_kwargs,
        )
        wandb_output_dir = None
        if FLAGS.checkpoint_path is not None:
            wandb_output_dir = os.path.abspath(FLAGS.checkpoint_path)
            os.makedirs(wandb_output_dir, exist_ok=True)

        wandb_logger = make_wandb_logger(
            project="hil-serl",
            description=(
                f"{FLAGS.exp_name}_{os.path.basename(os.path.abspath(FLAGS.checkpoint_path))}"
                if FLAGS.checkpoint_path is not None
                else FLAGS.exp_name
            ),
            output_dir=wandb_output_dir,
            debug=FLAGS.debug,
        )
        return replay_buffer, wandb_logger

    if FLAGS.learner:
        sampling_rng = jax.device_put(sampling_rng, device=sharding.replicate())
        replay_buffer, wandb_logger = create_replay_buffer_and_wandb_logger()
        demo_buffer = MemoryEfficientReplayBufferDataStore(
            env.observation_space,
            env.action_space,
            capacity=config.replay_buffer_capacity,
            image_keys=config.image_keys,
            include_grasp_penalty=include_grasp_penalty,
            include_cid_scores=use_cid_model,
            cid_prioritized_replay=cid_replay_on_demo,
            **cid_replay_kwargs,
        )

        assert FLAGS.demo_path is not None
        for path in FLAGS.demo_path:
            print("Demo path is: ", path, "Current working directory is: ", os.getcwd())
            with open(path, "rb") as f:
                transitions = pkl.load(f)
                for index, transition in enumerate(transitions):
                    transition = prepare_loaded_transition(
                        transition,
                        env,
                        include_grasp_penalty,
                        source=f"{path} transition {index}",
                    )
                    demo_buffer.insert(transition)
        print_green(f"demo buffer size: {len(demo_buffer)}")
        print_green(f"online buffer size: {len(replay_buffer)}")

        if FLAGS.resume_training and FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "buffer")
        ):
            for file in natsorted(glob.glob(os.path.join(FLAGS.checkpoint_path, "buffer/*.pkl"))):
                try:
                    transitions = load_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable buffer file {file}: {exc}")
                    continue
                for index, transition in enumerate(transitions):
                    transition = prepare_loaded_transition(
                        transition,
                        env,
                        include_grasp_penalty,
                        source=f"{file} transition {index}",
                    )
                    replay_buffer.insert(transition)
            print_green(f"Loaded previous buffer data. Replay buffer size: {len(replay_buffer)}")

        if FLAGS.resume_training and FLAGS.checkpoint_path is not None and os.path.exists(
            os.path.join(FLAGS.checkpoint_path, "demo_buffer")
        ):
            for file in natsorted(
                glob.glob(os.path.join(FLAGS.checkpoint_path, "demo_buffer/*.pkl"))
            ):
                try:
                    transitions = load_transition_file(file)
                except Exception as exc:
                    print(f"Skipping unreadable demo_buffer file {file}: {exc}")
                    continue
                for index, transition in enumerate(transitions):
                    transition = prepare_loaded_transition(
                        transition,
                        env,
                        include_grasp_penalty,
                        source=f"{file} transition {index}",
                    )
                    transition["_reset_frame_stack"] = True
                    demo_buffer.insert(transition)
            print_green(f"Loaded previous demo buffer data. Demo buffer size: {len(demo_buffer)}")

        print_green("starting learner loop")
        try:
            learner(
                sampling_rng,
                agent,
                replay_buffer,
                demo_buffer=demo_buffer,
                use_cid_reward=use_cid_reward,
                use_cid_replay=use_cid_replay,
                wandb_logger=wandb_logger,
            )
        finally:
            if wandb_logger is not None:
                wandb_logger.finish()

    elif FLAGS.actor:
        sampling_rng = jax.device_put(sampling_rng, sharding.replicate())
        data_store = QueuedDataStore(50000)
        intvn_data_store = QueuedDataStore(50000)

        print_green("starting actor loop")
        actor(
            agent,
            data_store,
            intvn_data_store,
            env,
            sampling_rng,
        )

    else:
        raise NotImplementedError("Must be either a learner or an actor")


if __name__ == "__main__":
    print(os.getcwd())
    app.run(main)
