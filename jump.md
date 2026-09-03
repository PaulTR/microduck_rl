# Microduck Jump Task — Training & Deployment Guide

This guide contains everything needed to train, inspect, export, and deploy the **Microduck Jump** policy (`Mjlab-Jump-Flat-MicroDuck`) on an NVIDIA GPU machine.

---

## 1. Setup & Git Pull (on your GPU machine)

Fetch the `jump` branch from GitHub:

```bash
git fetch paul
git checkout jump
git pull paul jump
```

Sync Python dependencies with `uv`:

```bash
uv sync
```

---

## 2. Fast Smoke Test (Always Run First)

Before running an hours-long training run, run this 5-iteration smoke test on 64 parallel robots. It validates that CUDA kernels compile, observation buffers match 61D, reward terms compute without NaNs, and memory is allocated cleanly:

```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

---

## 3. Full Training Run (4,096 parallel robots)

Launch the full PPO training run on your GPU:

```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
```

### Backlash Variant (Optional A/B comparison)
To train directly in the environment that simulates $\pm 1^\circ$ gearbox gear slop:

```bash
uv run train Mjlab-Jump-Flat-Backlash-MicroDuck --env.scene.num-envs 4096
```

### Hugging Face Jobs (Optional Remote Cloud Run)
If running on remote Hugging Face GPU clusters:

```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

---

## 4. Monitoring the Training Run (WandB)

Logs are saved under the project `mjlab_microduck` and in the local folder `logs/jump/`.

### What to check in the curves:
1. **Total Reward / `Episode_Reward/jump_air_time`:** Should steadily climb as the policy learns to launch into the air.
2. **`Episode_Reward/jump_height`:** Should rise as apex reaches target $z \approx 0.150\text{ m}$.
3. **Penalties Check (The Golden Rule):** Every penalty metric (`jump_drift_penalty`, `jump_foot_impact`, `action_rate_l2`, `body_ang_vel`, `self_collisions`, `head_pitch_limit`, `jump_rebound_penalty`) **must evaluate to $\le 0$**.
4. **Episode Length:** Stays stable around 1.0 s (50 steps at 50 Hz).

---

## 5. Visualizing the Policy in the 3D Viewer

View the trained policy live in the interactive MuJoCo viewer:

```bash
uv run play Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id>
```

Or load directly from a local checkpoint file:

```bash
uv run play Mjlab-Jump-Flat-MicroDuck --checkpoint logs/jump/<run_folder>/model_XXXX.pt
```

---

## 6. Export to ONNX (Standalone Deployment Binary)

Export the trained model to an ONNX file. This automatically bakes in the running observation normalizer:

```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id>
```

This creates an ONNX binary (e.g. `microduck_jump.onnx`).

---

## 7. CPU Deployment Rehearsal

Test the exported ONNX model on your local CPU (including your Mac!) before copying it to the physical robot's onboard computer.

### Option A: Run Jump Once and Return to Standing (Standalone Jump Policy)
To run the jump policy by itself, execute the jump once on startup, and automatically settle and hold the standing landing posture (rather than continuously jumping in an infinite loop):

```bash
uv run scripts/infer_policy.py --jump <jump.onnx> --jump-once --jump-duration 1.0 --new-cmd-obs
```

* The robot will execute the jump maneuver once for 1.0 second.
* Upon landing, it automatically locks and holds the standing posture.
* You can press **`J`** in the terminal at any time to execute another jump and return to standing!

### Option B: Trigger Jump as a hot-swapped trick (with Standing/Walking base)
If you have a trained standing or walking ONNX model, load it as the base policy alongside `--jump`. Microduck will stand/walk until you press **`J`**, at which point it jumps once and seamlessly hands control back to standing:

```bash
uv run scripts/infer_policy.py --standing <standing.onnx> --jump <jump.onnx> --jump-duration 1.0 --new-cmd-obs
```

* Starts in standing mode.
* Press **`J`** in the terminal: runs the jump policy for 1.0s, lands, and hands control back to standing.

### Option C: Run Jump policy continuously
If you want to evaluate repeated jumps in a continuous loop:

```bash
uv run scripts/infer_policy.py --walking <jump.onnx> --new-cmd-obs
```

---

## 8. Summary of Task Files

- **MDP Scoring Functions:** `src/mjlab_microduck/tasks/mdp.py` (contains `jump_air_time_reward`, `jump_height_target`, `jump_launch_velocity`, `jump_landing_composite`, `jump_drift_penalty`, `jump_foot_impact_penalty`, `head_pitch_limit_penalty`)
- **Task Configuration:** `src/mjlab_microduck/tasks/microduck_jump_env_cfg.py`
- **Task Registry:** `src/mjlab_microduck/tasks/__init__.py`
- **Unit Tests:** `tests/test_jump_cfg.py` (run anytime with `uv run --with pytest pytest tests/test_jump_cfg.py`)
