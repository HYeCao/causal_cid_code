import datetime
import os
import tempfile
from copy import copy
from socket import gethostname

import absl.flags as flags
import ml_collections
import wandb

def _load_wandb_api_key() -> str:
    """Resolve WANDB_API_KEY from environment or gitignored repo-root .env.local/.env."""
    key = os.environ.get("WANDB_API_KEY", "").strip()
    if key:
        return key
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
    for fname in (".env.local", ".env"):
        fpath = os.path.join(repo_root, fname)
        if not os.path.isfile(fpath):
            continue
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    if k.strip() == "WANDB_API_KEY":
                        return v.strip().strip('"').strip("'")
        except OSError:
            pass
    return ""


def _recursive_flatten_dict(d: dict):
    keys, values = [], []
    for key, value in d.items():
        if isinstance(value, dict):
            sub_keys, sub_values = _recursive_flatten_dict(value)
            keys += [f"{key}/{k}" for k in sub_keys]
            values += sub_values
        else:
            keys.append(key)
            values.append(value)
    return keys, values


class WandBLogger(object):
    @staticmethod
    def get_default_config():
        config = ml_collections.ConfigDict()
        config.project = "serl_launcher"  # WandB Project Name
        config.entity = ml_collections.config_dict.FieldReference(None, field_type=str)
        # Which entity to log as (default: your own user)
        config.exp_descriptor = ""  # Run name (doesn't have to be unique)
        # Unique identifier for run (will be automatically generated unless
        # provided)
        config.unique_identifier = ""
        config.group = None
        return config

    def __init__(
        self,
        wandb_config,
        variant,
        wandb_output_dir=None,
        debug=False,
    ):
        self.config = wandb_config
        if self.config.unique_identifier == "":
            self.config.unique_identifier = datetime.datetime.now().strftime(
                "%Y%m%d_%H%M%S"
            )

        self.config.experiment_id = (
            self.experiment_id
        ) = f"{self.config.exp_descriptor}_{self.config.unique_identifier}"  # NOQA

        print(self.config)

        if wandb_output_dir is None:
            wandb_output_dir = tempfile.mkdtemp()

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if debug:
            mode = "disabled"
        else:
            mode = "online"
            api_key = _load_wandb_api_key()
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
                wandb.login(key=api_key, relogin=True)

        self.run = wandb.init(
            config=self._variant,
            project=self.config.project,
            entity=self.config.entity,
            group=self.config.group,
            tags=self.config.tag,
            dir=wandb_output_dir,
            id=self.config.experiment_id,
            save_code=True,
            mode=mode,
        )

        if flags.FLAGS.is_parsed():
            flag_dict = {k: getattr(flags.FLAGS, k) for k in flags.FLAGS}
        else:
            flag_dict = {}
        for k in flag_dict:
            if isinstance(flag_dict[k], ml_collections.ConfigDict):
                flag_dict[k] = flag_dict[k].to_dict()
        wandb.config.update(flag_dict)

    def log(self, data: dict, step: int = None):
        data_flat = _recursive_flatten_dict(data)
        data = {k: v for k, v in zip(*data_flat)}
        wandb.log(data, step=step)

    def finish(self):
        wandb.finish()
