# Microduck Jump-4: Clean-Slate Two-Legged Vertical Jump Policy

## Overview & Background

The previous attempts suffered from two compounding issues:
1. **The Spawn Height Trap**: `make_velocity_env_cfg()` inherited a base reset with `z ∈ (0.01, 0.05)` (spawning 2–5 cm mid-air) and `yaw ∈ (-π, π)`. When spawned mid-air without an active walking gait, the robot fell onto the floor, tilted, and immediately tripped the 16° tilt termination threshold at step 0 before any policy could act.
2. **Boolean Gate Dead Zones**: Multiple consecutive boolean gates (`tilt < 0.035`, `head_err < 0.03`, `horiz_vel < 0.04`) wiped out all policy gradients across >98% of the exploration space. With zero gradient, the policy defaulted to passive collapses and random twitching.

**Jump-4 fixes both issues from first principles:**
- **Zero-Drop Spawn**: Spawns Microduck standing stably with feet flat on the ground (`z ∈ (-0.003, 0.003)`, `yaw ∈ (-0.05, 0.05)`).
- **Dense Continuous Gradient (Kinematic Reference Trajectory $q^*(\phi)$)**: Dense Gaussian joint-space tracking gives every policy action an immediate, non-zero gradient from step 0.
- **Biomechanical 4-Phase Cycle**:
  1. $\phi \in [0.00, 0.22]$: **Crouch Dip** — knees flex smoothly to $-0.35$ rad, hips pitch forward, lowering trunk by ~15 mm to load leg springs.
  2. $\phi \in [0.20, 0.42]$: **Explosive Push-off** — rapid knee extension to $0.0$, ankles drive down, generating vertical velocity $v_z > 0$ while feet are in contact with the floor.
  3. $\phi \in [0.38, 0.72]$: **Airborne Flight** — both feet leave the ground, rewarded by apex lift height ($z \ge 0.115$ m) with legs extended in preparation for landing.
  4. $\phi \in [0.65, 1.00]$: **Touchdown & Stand Recovery** — touchdown with feet flat, stabilizing into the nominal upright HOME stand pose.
- **Head & Body Verticality Locks**:
  - Continuous L2 penalty on head servos 5–8 (`neck_pitch`, `head_pitch`, `head_yaw`, `head_roll`) prevents head curls, chin tucks, and nodding.
  - Continuous tilt penalty ($g_x^2 + g_y^2$) and yaw rate penalty ($\omega_z^2$) enforce strictly vertical, twist-free vertical motion.
- **Bilateral Symmetry**: Mirror-loss enabled across left and right leg actions to prevent asymmetric hopping or turning.

---

## Commands for Linux Training Machine

### 1. Pull branch `jump-4`
```bash
git checkout jump-4
git pull paul jump-4
```

### 2. Smoke Test (5 iterations, 64 envs)
Run a quick CPU/GPU smoke test to confirm environment construction and stepping:
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

### 3. Training Run (4096 envs, 1200 iterations)
Train the jump policy with 4096 parallel environments (configured for 1200 iterations with checkpoints every 100 iterations):
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
```

### 4. Early Checkpoint Testing (Don't wait for all iterations!)
Checkpoints are saved every 100 iterations (`logs/microduck_jump/<date_time>/checkpoints/model_*.pt`).
You can export and test an early checkpoint (e.g. after 200–300 iterations, ~10–15 min) while training continues:
```bash
# Export specific checkpoint (e.g. model_300.pt) from local logs:
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --checkpoint logs/microduck_jump/<run_dir>/checkpoints/model_300.pt

# Or export from wandb:
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id> --checkpoint 300
```

### 5. What to Check in WandB at Iteration 100–200
Open the wandb run to verify the policy is discovering the jump:
- `Episode_Reward/jump_trajectory_tracking`: climbing steadily towards ~4–5.
- `Episode_Reward/jump_push_velocity`: rising above 0.5 (indicates pushing floor during extension).
- `Episode_Reward/jump_airborne`: rising above 0.5 (indicates feet leaving ground).
- `Episode_Reward/jump_yaw_rate`: near 0.0 (confirms NO spinning).
- `Episode_Reward/head_posture`: near 0.0 (confirms NO chin-tucking or ball-curling).

### 6. Deployment Rehearsal & Testing
Run the jump policy standalone with the BAM M6 actuator model:
```bash
uv run scripts/infer_policy.py --jump <path_to_exported_policy.onnx> --new-cmd-obs
```
**Interactive Controls:**
- `J`: Trigger the vertical jump cycle from standing.
- `X`: Reset the robot to default standing pose and trigger the jump.
- `Space`: Zero commands.
- `Q`: Quit viewer.

---

## Architecture Summary

| Component | Setting / Implementation |
|---|---|
| **Task ID** | `Mjlab-Jump-Flat-MicroDuck` (and `-Rough-`, `-Backlash-` variants) |
| **Observation Dim** | 61D unified layout: 48D proprioception + 13D command `[twist(3), head(4), body(6)]` |
| **Command Encoding** | `[cos(2πφ), sin(2πφ), 0]` in twist slot, representing jump cycle phase $\phi \in [0, 1)$ |
| **Episode Length** | 1.2 s (60 control steps @ 50 Hz) |
| **Actuator Model** | BAM M6 voltage-controlled Dynamixel XL330 |
| **Bilateral Symmetry** | Enabled (`PpoWithSymmetryCfg` with 61D observation swap table) |
| **Terminations** | `fell_over` (tilt > 23°), `non_foot_contact` (any body part touches floor), `nan_state` |
| **Spawn Range** | $z \in [-3, +3]$ mm, $\text{yaw} \in [-0.05, +0.05]$ rad (no mid-air drop) |
