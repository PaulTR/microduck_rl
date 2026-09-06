# Microduck Two-Legged Vertical Jump Policy (Jump-3)

A clean, minimal, two-legged vertical jump in place designed from scratch.
The robot starts standing, crouches and launches straight up into the air (target apex $z \approx 0.170$ m), and lands cleanly on both feet directly into a vertical standing posture ($z \approx 0.115$ m) at $(x=0, y=0)$ without spinning or drifting.

---

## 1. Why Jump-3 Succeeded Where Jump-2 Failed

1. **Eliminated Yaw Spinning**:
   - Checkpoint 3999 in jump-2 spun rapidly in circles because yaw rotation at $(0, 0)$ produced zero linear drift and was under-penalized.
   - **Jump-3 Fix**: 
     - Active sagittal symmetry (`ENABLE_SYMMETRY = True` with `mirror_loss_coeff = 0.5`) mathematically enforces identical left-right leg actions.
     - `jump_yaw_rate` strictly penalizes $\omega_{z,b}^2$ with weight `-3.0`.

2. **Eliminated Forward Shuffling and Butt-Landings**:
   - Strict world-frame horizontal drift penalty `jump_horizontal_drift` (weight `-4.0`) and velocity penalty `jump_horizontal_vel` (weight `-3.0`).
   - `jump_stand_composite` directly rewards nominal standing posture at `STAND_Z = 0.115 m` during the landing half ($\phi \in [0.45, 1.00]$). No crouched landings.
   - `non_foot_contact` termination immediately aborts the episode if trunk, hips, knees, or face touch the terrain.

3. **Radical Simplification**:
   - Replaced 23 fragmented reward terms with 4 clean task objectives and 4 anti-exploit penalties.
   - Phase $\phi \in [0, 1)$ over 2.0 s:
     - $\phi \in [0.00, 0.45]$: Dense Gaussian height reward toward apex ($z \to 0.170$ m) + airborne bonus.
     - $\phi \in [0.45, 1.00]$: Landing directly on both feet into HOME upright stand ($z \to 0.115$ m).

---

## 2. Training Commands

### Smoke Test (Run first — takes ~10 seconds)
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

### Full Training (4096 envs, GPU or HF Jobs)
```bash
# Local GPU:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096

# Or on Hugging Face Jobs:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

---

## 3. Export to ONNX

Once trained (typically 2000–3000 iterations), export with baked observation normalizer:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id> -o jump.onnx
```

---

## 4. Playback and Resettable Testing

Run the jump policy standalone with the BAM M6 actuator model:
```bash
uv run scripts/infer_policy.py --jump jump.onnx --new-cmd-obs
```

### Interactive Controls (Type in terminal):
- **`J`**: Trigger vertical jump cycle. The robot crouches, pushes straight up, lands on both feet, and stands.
- **`X`**: Instant reset to spawn standing state at origin $(0, 0, 0.125)$ with zeroed velocity.
- **`T`**: Pause / unpause policy inference (motors hold last position).
- **`Q`**: Quit simulation.
