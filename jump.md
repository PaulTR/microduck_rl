# Microduck Jump-5: Crouch-Push-Tuck Vertical Jump Policy

## Overview & The 4-Stage Biomechanical Jump Cycle

Following the working baseline from `jump-4` (commit `5cb80b4`), **Option A** introduces the **4-Stage Crouch-Push-Tuck Jump Cycle** on branch `jump-5`. This resolves the "falling-push" momentum loss and multiplies effective ground clearance without demanding physically impossible actuator speeds.

```
Stage 1: Pre-Crouch             Stage 2: Vertical Thrust        Stage 3: Mid-Air Tuck           Stage 4: Landing & Handover
(phi in [0.08, 0.24], vz -> 0)  (phi in [0.22, 0.34], thrust)   (phi in [0.30, 0.52], tuck)     (phi >= 0.50, stand)

      [ Head ]                        [ Head ]                        [ Head ]                        [ Head ]
      [ Body ]                        [ Body ]                        [ Body ]                        [ Body ]
       /    \                           |    |                         <    >                          /    \
      /      \                          |    |                                                        /      \
   --[Sole]--[Sole]--                --[Sole]--[Sole]--               --[Sole]--[Sole]--           --[Sole]--[Sole]--
─────────────────────────         ─────────────────────────        ─────────────────────────    ─────────────────────────
      (Grounded)                      (Push into Floor)             (High Ground Clearance!)        (Stable Landing)
```

### The 4 Stages in Detail:

1. **Stage 1: Controlled Pre-Crouch ($\phi \in [0.08, 0.24]$, $t \approx 0.10 - 0.28$ s)**:
   - Knees flex smoothly to $+0.36$ rad, hips pitch $+0.18$ rad, ankles pitch $+0.18$ rad ($\Delta z \approx 9$ mm dip).
   - Sagittal balance condition ($\Delta \text{hip} + \Delta \text{ankle} = \Delta \text{knee}$) holds horizontal CoM shift to $|dx| < 0.3$ mm (zero backward tipping).
   - **Crucial Advantage**: At the bottom of the crouch ($\phi \approx 0.20 - 0.22$), downward velocity settles to $v_z \approx 0$. 100% of motor work in Stage 2 goes directly into upward acceleration instead of braking a downward fall.

2. **Stage 2: Explosive Vertical Thrust ($\phi \in [0.22, 0.34]$, $t \approx 0.26 - 0.40$ s)**:
   - Knees extend to $-0.12$ rad, driving the feet hard down into the floor.
   - Ground reaction force vector passes strictly through the CoM, producing pure vertical thrust with zero pitch torque (preventing face-plants).
   - `jump_push_velocity`: rewards $v_z > 0$ with `target_vz = 0.45` m/s (clamp range $[0, 2.5]$, weight 12.0).

3. **Stage 3: Mid-Air Leg Tuck ($\phi \in [0.30, 0.52]$, $t \approx 0.36 - 0.62$ s)**:
   - Once airborne, the knees flex up to $+0.40$ rad, retracting the feet **2.5 cm upward toward the pelvis**.
   - With ~1.5 cm of trunk liftoff plus 2.5 cm of leg retraction, **total foot ground clearance reaches 4.0 cm**!
   - `jump_airborne`: strictly requires `both_airborne > 0` ($R_{\text{air}} = \text{both\_airborne} \cdot (1.0 + 1.5 \cdot \text{lift}) \cdot u_{\text{score}}$, weight 12.0).
   - Latches `_jump_achieved_airborne = True` to unlock the landing stand reward.

4. **Stage 4: Landing Reach & Handover ($\phi \in [0.50, 1.00]$, $t \approx 0.60 - 1.20$ s)**:
   - At $\phi \approx 0.50$, as the robot descends, the reference trajectory smoothly returns to the nominal HOME standing pose to make flat-footed contact and absorb impact.
   - `jump_stand`: active from $\phi = 0.50$ onwards (weight 6.0), strictly gated on having achieved airborne flight.
   - At the end of the jump cycle, control seamlessly hands over to `alpha_stand.onnx`.

---

## Commands for Linux Training Machine

### 1. Fetch & Force Checkout Branch `jump-5`
```bash
git fetch paul
git checkout jump-5
git reset --hard paul/jump-5
```

### 2. Smoke Test (5 iterations, 64 envs)
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```

### 3. Training Run (4096 envs, 1200 iterations)
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
```
*(Runs with experiment name `microduck_jump_tuck`)*

### 4. Early Checkpoint Export & Testing
Checkpoints are saved every 100 iterations in `logs/microduck_jump_tuck/<date_time>/checkpoints/model_*.pt`.

Export and test checkpoints while training runs:
```bash
# Export from local logs:
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --checkpoint logs/microduck_jump_tuck/<run_dir>/checkpoints/model_300.pt

# Or export via wandb:
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity>/mjlab_microduck/<run_id> --checkpoint 300
```

### 5. WandB Metrics to Monitor
- `Episode_Reward/jump_push_velocity`: Target $\ge 3.0$ (strong vertical launch).
- `Episode_Reward/jump_airborne`: Target $\ge 4.0$ (both feet leaving the ground + mid-air tuck).
- `Episode_Reward/jump_stand`: Unlocks once airborne flight is achieved and guides touchdown landing.
- `Episode_Reward/jump_yaw_rate` and `jump_horizontal_vel`: Near 0 (straight vertical leap without drift or spin).
- `Episode_Reward/head_posture`: Near 0 (head remains fixed and upright).

### 6. Deployment Rehearsal in MuJoCo
Download the official standing policy (if not already cached):
```bash
curl -L -o alpha_stand.onnx https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/alpha_stand.onnx
```

Rehearse the jump with automatic standing handover:
```bash
uv run scripts/infer_policy.py \
    --jump <path_to_exported_policy.onnx> \
    --standing alpha_stand.onnx \
    --new-cmd-obs
```

**Interactive Controls:**
- Spawns in `--standing` policy (`alpha_stand.onnx`).
- Press `J`: Triggers Crouch-Push-Tuck jump. Control hands over to `--jump`.
- At touchdown / end of jump cycle (1.2 s), control hands back to `--standing` to stick the landing.
- Press `X`: Reset robot to standing pose.
- Press `Space`: Zero commands.
- Press `Q`: Quit viewer.

---

## Architecture Summary

| Component | Setting / Implementation |
|---|---|
| **Branch** | `jump-5` (clean separation from `jump-4`) |
| **Task ID** | `Mjlab-Jump-Flat-MicroDuck` (and `-Rough-`, `-Backlash-` variants) |
| **Experiment Name** | `microduck_jump_tuck` |
| **Observation Dim** | 61D unified layout: 48D proprioception + 13D command `[twist(3), head(4), body(6)]` |
| **Command Encoding** | `[cos(2πφ), sin(2πφ), 0]` in twist slot, representing jump cycle phase $\phi \in [0, 1)$ |
| **Episode Length** | 1.2 s (60 control steps @ 50 Hz) |
| **Pre-Crouch Dip** | $\Delta z \approx 9$ mm ($z_{\text{crouch}} \approx 0.104$ m, knees $+0.36$ rad, $dx < 0.3$ mm, settled $v_z \to 0$) |
| **Push Velocity Target** | $v_z = 0.45$ m/s (clamp $[0, 2.5]$, weight 12.0) |
| **Mid-Air Tuck** | Knees $+0.40$ rad flex mid-air, pulling feet up by 2.5 cm ($\approx 4$ cm total ground clearance) |
| **Apex Target** | $z = 0.145$ m (weight 12.0) |
| **Flight Gating** | Strictly requires `both_airborne > 0`; grounded robot earns 0.0 |
| **Stand Gating** | Latch on `_jump_achieved_airborne`; grounded robot earns 0.0 |
| **Stand Handover** | $\phi_{\text{start}} = 0.50$ (step 25) |
| **Actuator Model** | BAM M6 voltage-controlled Dynamixel XL330 |
| **Bilateral Symmetry** | Enabled (`microduck_jump_symmetry`, phase-invariant) |
| **Terminations** | `fell_over` (tilt limit 0.85 rad / ~48.7°), `nan_state` |
