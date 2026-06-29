# Fine-tuning on a New Robot: Setup Guide

Steps for setting up openpi π₀.₅ fine-tuning on a fresh GPU instance with a new robot dataset.

---

## 1. Instance Setup

### NVIDIA Drivers
```bash
sudo apt-get update
sudo apt-get install -y ubuntu-drivers-common build-essential linux-libc-dev ffmpeg
sudo ubuntu-drivers autoinstall
sudo reboot
# After reboot:
nvidia-smi  # verify all GPUs visible
```

### Install uv
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env
# Add to shell permanently:
echo 'source ~/.local/bin/env' >> ~/.bashrc
```

---

## 2. Clone the Repo

```bash
ssh-keyscan github.com >> ~/.ssh/known_hosts
GIT_LFS_SKIP_SMUDGE=1 git clone --recurse-submodules \
    git@github.com:XiaoweiLinXL/openpi-finetuning.git ~/openpi-finetuning
cd ~/openpi-finetuning
GIT_LFS_SKIP_SMUDGE=1 uv sync
GIT_LFS_SKIP_SMUDGE=1 uv pip install -e .
```

Verify GPUs are visible to JAX:
```bash
uv run python -c "import jax; print(jax.devices())"
```

---

## 3. Transfer the Dataset

The dataset must be in **LeRobot v2.1 format** (parquet + hevc video files) under:
```
~/.cache/huggingface/lerobot/local/<dataset_name>/
├── data/chunk-000/*.parquet
├── videos/chunk-000/<camera_key>/episode_*.mp4
└── meta/info.json
```

If the dataset lives on another instance, transfer via a single pipe (avoid parallel transfers to the same files):
```bash
# From data-upload instance directly to training instance (VPC-internal):
ssh -A <training_instance> \
  "ssh <data_instance_internal_ip> 'cat /path/to/dataset.tgz' \
   | tar -xz -C ~/.cache/huggingface/lerobot/local/"
```

---

## 4. Write the Policy File

Create `src/openpi/policies/<robot_name>_policy.py`. Use an existing policy as a reference (e.g., `unitree_g1_policy.py`).

Key things to define:
- `ACTUAL_ACTION_DIM`: the real action dimension (may differ from model's padded dimension)
- `<Robot>Inputs`: maps dataset keys to model format (`state`, `image`, `image_mask`, optionally `actions`, `prompt`)
- `<Robot>Outputs`: slices model output back to `ACTUAL_ACTION_DIM`

Camera images should be `(H, W, 3)` uint8. Use `image_mask` to disable unused camera slots (`np.False_`).

---

## 5. Add Training Config

In `src/openpi/training/config.py`:

**Add import** alongside other policy imports:
```python
import openpi.policies.<robot_name>_policy as <robot_name>_policy
```

**Add a `DataConfigFactory` subclass** (model after `LeRobotUnitreeG1DataConfig`):
- Define `repack_transform`: maps dataset column names to policy input keys
- Define `data_transforms`: wraps `<Robot>Inputs` and `<Robot>Outputs`
- Set `action_sequence_keys=("action",)`
- Optionally add `DeltaActions`/`AbsoluteActions` if actions are absolute joint positions

**Add a `TrainConfig` entry** in `_CONFIGS`:
```python
TrainConfig(
    name="pi05_<robot_name>",
    model=pi0_config.Pi0Config(pi05=True, action_horizon=50),
    data=LeRobot<Robot>DataConfig(
        repo_id="local/<dataset_name>",
        default_prompt="<task description>",
        base_config=DataConfig(action_sequence_keys=("action",)),
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader(
        "gs://openpi-assets/checkpoints/pi05_base/params"
    ),
    batch_size=64,
    num_train_steps=50000,
    save_interval=5000,
    keep_period=5000,
    fsdp_devices=4,   # match number of GPUs
    ema_decay=0.999,
    wandb_enabled=False,  # set True if W&B is configured
),
```

The `repo_id` key names in `repack_transform` must match the column names in the dataset's parquet files. Check with:
```bash
uv run python -c "
import polars as pl, pathlib
df = pl.read_parquet(next(pathlib.Path('~/.cache/huggingface/lerobot/local/<dataset_name>/data').expanduser().rglob('*.parquet')))
print(df.columns)
"
```

---

## 6. Compute Norm Stats

The fast path reads parquet directly (no video decoding):
```bash
uv run scripts/compute_norm_stats.py --config-name pi05_<robot_name>
```

Stats are written to `assets/pi05_<robot_name>/local/<dataset_name>/norm_stats.json`.

If this is slow (it decodes video to count frames), use a custom script that reads parquet directly — see `scripts/compute_norm_stats.py` for the expected output format (`mean`, `std`, `q01`, `q99` per key).

---

## 7. Start Training

```bash
cd ~/openpi-finetuning
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 uv run scripts/train.py pi05_<robot_name> \
    --exp-name=<run_name> --overwrite
```

Monitor GPU usage:
```bash
watch -n 1 nvidia-smi
```

Checkpoints are saved to `checkpoints/pi05_<robot_name>/<run_name>/` every `save_interval` steps.

---

## Common Errors

| Error | Cause | Fix |
|-------|-------|-----|
| `command not found: uv` | uv not on PATH | `source ~/.local/bin/env` |
| `PermissionError: info.json` | Dataset owned by another user | `chmod -R o+rX <dataset_dir>` and `chmod o+x ~/.cache` |
| `ValueError: Normalization stats not found` | Wrong stats path | Stats must be at `assets/<config_name>/<repo_id>/norm_stats.json` |
| `RuntimeError: Could not open input file` | Corrupted video (parallel writes) | Re-transfer dataset with a single transfer |
| `evdev build failure` | Missing C headers | `sudo apt-get install -y build-essential linux-libc-dev` |
| `wandb` login error | W&B not configured | Add `wandb_enabled=False` to `TrainConfig` |
