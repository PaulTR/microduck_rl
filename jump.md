# Microduck Two-Legged Vertical Jump Policy (Jump-3)

A physics-grounded, two-legged vertical jump in place.
The robot starts standing, crouches to load its legs ($z = 0.085$ m), drives upward explosively ($v_z > 0$), achieves ballistic flight (apex $z \approx 0.155$ m), and lands cleanly on both feet directly into a vertical standing posture ($z \approx 0.115$ m) at $(x=0, y=0)$ without spinning, drifting, or falling.

---

## 1. Biomechanics of a Realistic Jump

A bipedal jump cannot be learned by simply asking for apex height from step 0:
1. **The Countermovement / Crouch Requirement**:
   - To jump, an 800g robot with low-gear-ratio XL330 servos must first crouch down ($\Delta z \approx 3$ cm) to create stroke length for acceleration.
   - An apex height reward active from $t=0$ actually *penalizes* crouching (increasing height error), trapping the policy in standing or falling.
   - **Fix**: Phase 1 ($\phi \in [0.00, 0.25]$) explicitly rewards lowering CoM to `CROUCH_Z = 0.085 m` while keeping feet flat and trunk upright.

2. **The Push-Off Velocity Gradient**:
   - A ballistic flight reward only fires *after* leaving the ground.
   - **Fix**: Phase 2 ($\phi \in [0.20, 0.45]$) rewards upward vertical velocity ($v_z > 0$) while feet are pushing against the ground. This gives a dense, monotonic gradient from the very first step of leg extension.

3. **Physics-Consistent Flight Duration ($T = 1.2$ s)**:
   - Microduck achieves $\approx 4$ cm of vertical lift under realistic XL330 torque and friction.
   - Under gravity ($g = 9.81\text{ m/s}^2$), a 4 cm jump has a ballistic flight duration of only:
     $$t_{\text{flight}} = 2 \sqrt{\frac{2 \cdot 0.04}{9.81}} \approx 0.18\text{ s}$$
   - A 0.6 s flight requires launching 44 cm into the air—physically impossible for this robot. In past runs, the only way the agent could keep feet off the ground for 0.6 s was to collapse onto its back!
   - **Fix**: Cycle duration is $T = 1.2$ s (60 steps @ 50 Hz) with a tight $0.20$ s flight window ($\phi \in [0.45, 0.65]$).

4. **Eliminating the Back-Flop Exploit**:
   - **Anti-Flop Flight Gate**: `jump_airborne` is strictly gated on upright trunk orientation (tilt $< 15^\circ$) and trunk height $z > 0.120$ m. Lying on the back yields exactly $0.0$.
   - **Tilt Termination**: `fell_over` terminates immediately at tilt $> 20^\circ$ (`limit_angle = 0.35`).
   - **Non-Foot Contact Termination**: Every non-foot geom (`^(?!.*foot_collision).*$`) terminates the episode if it touches the floor.

5. **Eliminating Head Bobbing**:
   - The head and neck represent 38% of Microduck's total mass.
   - Without constraints, the policy discovered it could oscillate the head like an inverted pendulum to rock and drag itself.
   - **Fix**: `head_posture_penalty` (weight `-2.0`) locks servos 5–8 to the nominal HOME pose.

6. **Eliminating Spinning & Drift**:
   - **Bilateral Symmetry**: Active sagittal symmetry (`ENABLE_SYMMETRY = True`, `mirror_loss_coeff = 0.5`) mathematically forces left and right leg actions to match.
   - **Anti-Spin Penalty**: `jump_yaw_rate` penalizes $\omega_{z,b}^2$ with weight `-3.0`.
   - **In-Place Penalties**: `jump_horizontal_drift` (weight `-4.0`) and `jump_horizontal_vel` (weight `-3.0`).

---

## 2. 4-Phase Jump Timeline ($T = 1.2$ s, 60 steps @ 50 Hz)

| Phase | Phase $\phi$ | Duration | Objective | Key Rewards |
|---|---|---|---|---|
| **1. Crouch** | $[0.00, 0.25]$ | $0.30$ s | Lower CoM to $z = 0.085$ m with flat feet | `jump_crouch` (4.0) |
| **2. Push-Off** | $[0.20, 0.45]$ | $0.30$ s | Explosively drive legs straight up ($v_z > 0$) | `jump_push_velocity` (5.0) |
| **3. Flight** | $[0.45, 0.65]$ | $0.24$ s | Reach apex $z \approx 0.155$ m in the air upright | `jump_airborne` (4.0), `jump_apex_height` (5.0) |
| **4. Land & Stand** | $[0.65, 1.00]$ | $0.42$ s | Touch down on two feet into HOME stand ($z = 0.115$ m) | `jump_stand` (5.0), `jump_feet_grounded` (2.0) |

---

## 3. Training Commands

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

## 4. Export to ONNX

Once trained (typically 2000–3000 iterations), export with baked observation normalizer:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id> -o jump.onnx
```

---

## 5. Playback and Resettable Testing

Run the jump policy standalone with the BAM M6 actuator model:
```bash
uv run scripts/infer_policy.py --jump jump.onnx --new-cmd-obs
```

### Interactive Controls (Type in terminal):
- **`J`**: Trigger vertical jump cycle. The robot crouches, pushes straight up, lands on both feet, and stands.
- **`X`**: Instant reset to spawn standing state at origin $(0, 0, 0.125)$ with zeroed velocity.
- **`T`**: Pause / unpause policy inference (motors hold last position).
- **`Q`**: Quit simulation.
