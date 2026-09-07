# Microduck Strictly Vertical Two-Legged Jump (Jump-3)

A physics-grounded, two-legged vertical jump in place designed to keep Microduck strictly vertical throughout all phases: stable countermovement dip, explosive vertical push-off, upright ballistic flight, and balanced two-foot landing.

---

## 1. Root Causes & Physics Solutions

### A. Preventing Head Curling into a Ball
- **Root Cause**: Microduck's head is **38% of total robot mass**. When the legs crouched, the policy discovered it could tuck its chin into its chest (servos 5–8: `neck_pitch`, `head_pitch`) to shift mass forward without extending its legs.
- **Physics Solution**:
  1. **Rigid Head Lock**: `head_posture_penalty` weight boosted to **`-10.0`** (penalizes $\sum (q_{\text{head}} - q_{\text{head, home}})^2$).
  2. **Head Lock Gate**: `jump_crouch`, `jump_push_velocity`, and `jump_airborne` now include a strict head gate ($\sum (q_{\text{head}} - q_{\text{head, default}})^2 < 0.03$). Any head movement immediately zeros out the positive rewards. The head remains locked upright in the HOME posture.

### B. Preventing Backward Falls onto the Back
- **Root Cause**:
  1. `CROUCH_Z = 0.080` m was physically too deep for Microduck's small 5.4 cm feet (heels at $-20$ mm, toes at $+34$ mm). Dropping 35 mm shifted the Center of Mass behind the ankle axis, making backward toppling unavoidable.
  2. The previous pitch rate penalty (`jump_pitch_rate = -2.5` on $\omega_y^2$) taxed rapid leg extension during push-off, leaving the robot off-balance without vertical thrust.
- **Physics Solution**:
  1. **Stable Countermovement Dip ($z = 0.100$ m)**: A 15 mm dip from stand ($0.115$ m down to $0.100$ m). Keeps feet completely flat, CoM centered over the ankle axis, trunk vertical ($0.2^\circ$), and joints within their maximum torque band.
  2. **Removal of Pitch Rate Penalty**: Leg push-off is no longer taxed for rotational speed.
  3. **Strict Verticality Penalty & Gate**: `jump_verticality` boosted to **`-8.0`** ($g_x^2 + g_y^2$). All positive rewards strictly require tilt $< 10.7^\circ$ ($g_x^2 + g_y^2 < 0.035$).
  4. **Strict Orientation Termination**: `fell_over` `limit_angle` tightened from $0.35$ (~20°) to **`0.28`** (~16°). Any forward or backward topple terminates the episode immediately.

### C. Purely Vertical Push-Off ($v_z > 0$)
- In `jump_push_velocity`, upward velocity $v_z$ is rewarded only when:
  - Both feet are pushing against the ground.
  - The trunk is strictly upright (tilt $< 10.7^\circ$).
  - The head is locked in HOME posture.
  - Horizontal velocity is minimal ($v_x^2 + v_y^2 < 0.04$, no lunging or spinning).

---

## 2. 5-Phase Jump Timeline ($T = 1.2$ s, 60 steps @ 50 Hz)

| Phase | Phase $\phi$ | Duration | Objective | Key Rewards / Gates |
|---|---|---|---|---|
| **1. Ready Stand** | $[0.00, 0.08]$ | $0.10$ s | Stable upright stand in HOME pose ($z = 0.115$ m) | `jump_stand` (6.0), `jump_feet_grounded` (2.5) |
| **2. Crouch Dip** | $[0.08, 0.28]$ | $0.24$ s | Lower CoM to $z = 0.100$ m with flat feet & upright trunk | `jump_crouch` (4.0, head gate, tilt gate) |
| **3. Push-Off** | $[0.25, 0.48]$ | $0.28$ s | Explosively drive straight up ($v_z > 0$) | `jump_push_velocity` (8.0, vertical gate, head gate) |
| **4. Flight** | $[0.45, 0.70]$ | $0.30$ s | Reach apex $z \approx 0.150$ m upright in the air | `jump_airborne` (6.0, lift-scaled), `jump_apex_height` (6.0) |
| **5. Land & Settle** | $[0.68, 1.00]$ | $0.38$ s | Touch down on two feet into HOME stand ($z = 0.115$ m) | `jump_stand` (6.0), `jump_landing_damping` (-2.0) |

---

## 3. Training & Playback

### Pull Updates & Train (on Linux GPU or HF Jobs)
```bash
# On your Linux GPU machine:
git pull
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
# Or on Hugging Face Jobs:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

### Export to ONNX
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id> -o jump.onnx
```

### Playback in MuJoCo Viewer
```bash
# Standalone execution:
uv run scripts/infer_policy.py --jump jump.onnx --new-cmd-obs
```
The policy completes the 1.2 s vertical jump cycle and automatically holds balance in the upright standing pose at $\phi = 1.0$.
