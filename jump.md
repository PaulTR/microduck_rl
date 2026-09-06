# Microduck Vertical Jump (`Mjlab-Jump-Flat-MicroDuck`)

A reinforcement learning task teaching Microduck (~800 g, ~25 cm bipedal robot with 14 Dynamixel XL330 servos) to perform a clean, two-footed vertical jump straight up in place with air-time gated landing on its feet (in a crouch) and stable upright standing without landing on its butt or shuffling.

---

## 1. Overview & Motion Progression

The jump is parameterized over a 3.0 s period ($T = 3.0$ s) with phase $\phi \in [0, 1)$ encoded in the `twist` observation slot as $[\cos(2\pi\phi), \sin(2\pi\phi), 0]$. Each episode starts standing at $\phi = 0$:

1. **Initial Stand & Settle ($\phi \in [0.00, 0.10]$ | $t \in [0.0, 0.3]\text{ s}$)**:
   - Starts standing at nominal height (`STAND_Z = 0.115 m`) in the `HOME` joint pose.
   - Body and head strictly vertical ($g_x \approx 0, g_y \approx 0$).
2. **Crouch Loading ($\phi \in [0.10, 0.32]$ | $t \in [0.3, 1.0]\text{ s}$)**:
   - Symmetrically lowers trunk height to `CROUCH_Z = 0.065 m` by flexing knees and hips directly beneath the torso.
   - Both feet remain firmly planted on the ground (`jump_crouch_feet_grounded`).
   - Anti-pre-hop penalty (`jump_crouch_feet_lift`) prevents premature skipping.
3. **Explosive Vertical Takeoff ($\phi \in [0.30, 0.48]$ | $t \in [0.9, 1.4]\text{ s}$)**:
   - Demands rapid vertical upward velocity ($v_z \approx 1.0\text{ m/s}$) straight up in **world frame** ($+z_w$).
   - **Zero Forward/Horizontal Translation**: Strictly locked in place at $(x=0, y=0)$ via `jump_stay_in_place` reward and `jump_horizontal_drift` penalty.
   - **Anti-Kickback Constraint**: `jump_hip_pitch_extension_penalty` prevents extending hips backward (kicking legs behind the torso), eliminating the forward rotational torque that causes pitch dives.
   - Body verticality penalty (`jump_verticality_penalty`, weight -4.0) prevents pitch lean or lateral roll ($g_x^2 + g_y^2$).
   - Both feet push off simultaneously.
4. **Airborne Flight ($\phi \in [0.35, 0.65]$ | $t \in [1.0, 1.9]\text{ s}$)**:
   - Both feet airborne simultaneously (`torch.minimum(air_l, air_r)`).
   - Flight qualification gate (`jump_flight_gate`) unlocks landing rewards only once $\ge 80\text{ ms}$ continuous air time is achieved.
5. **Two-Footed Landing in Crouch ($\phi \in [0.55, 0.75]$ | $t \in [1.65, 2.25]\text{ s}$)**:
   - **Crouch Shock Absorption on Feet**: Rewards landing in the crouched position (`CROUCH_Z = 0.065 m`) with both feet flat on the ground and trunk upright.
   - **Strict Non-Foot Ground Contact Termination**: If head/beak, butt/battery, hips, or knees touch the ground at ANY point, the episode **terminates instantly** (`non_foot_contact` termination). Falling on the face or butt and standing back up is physically impossible because the episode immediately ends!
   - **Tilt Fall Termination**: If trunk tilts $> 37^\circ$ (`fell_over` termination with `limit_angle=0.65`), the episode terminates immediately.
6. **Return to Vertical Stand & Stillness ($\phi \in [0.72, 1.00]$ | $t \in [2.16, 3.0]\text{ s}$)**:
   - Recovers trunk from crouch to `STAND_Z = 0.115 m`, body and head upright, HOME joint pose.
   - **Clean-Landing Gate**: Only pays if the robot completed flight and landed on its feet without falling.
   - Stillness penalty (`jump_post_landing_stillness_penalty`) damps linear and angular velocity once landed to prevent shuffling.
   - Post-landing hop penalty (`jump_post_landing_hop`) strictly taxes lifting feet after landing.

---

## 2. Standard Workflow (Consistent with Repo README)

All commands use the standard tooling documented in `README.md` and `AGENTS.md`.

### Step 1: Synchronize Dependencies
```bash
uv sync
```
*(On `linux-aarch64` / DGX Spark, uv automatically routes PyPI's CPU wheel to the cu129 index as pinned in `pyproject.toml`).*

### Step 2: Smoke Test First (5 Iterations)
Always run a quick smoke test before launching long runs to verify config and shapes:
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```
*Expected: builds scenes, steps without NaNs, confirms 61D observation shape, and exits cleanly.*

### Step 3: Train the Jump Policy
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 4000
```
- If running on Hugging Face Jobs, simply add `--hf-jobs`:
  ```bash
  uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
  ```
- Logs to Weights & Biases under project `mjlab_microduck` (`experiment_name="microduck_jump"`).
- Checkpoints are saved to `logs/microduck_jump/<run_id>/model_XXXX.pt`.

### Step 4: Watch Trained Policy in the Viewer
Watch a checkpoint directly in the viewer using mjlab's standard `play` command:
```bash
uv run play Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```
Or view a local checkpoint directly:
```bash
uv run play Mjlab-Jump-Flat-MicroDuck --checkpoint-file logs/microduck_jump/<run_id>/model_3999.pt
```

### Step 5: Export to ONNX (with Baked Observation Normalizer)
Export the trained checkpoint using `scripts/export.py`:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```
Or export from a local checkpoint file directly:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --checkpoint-file logs/microduck_jump/<run_id>/model_3999.pt --onnx-file output.onnx
```

### Step 6: Standalone Policy Playback in MuJoCo Viewer
You do not need a `walk.onnx` policy — the jump policy runs completely standalone:
```bash
uv run scripts/infer_policy.py --jump output.onnx --new-cmd-obs
```
- **How it works**:
  - The robot starts standing upright in place (phase 0).
  - Press **`J`** or **`SPACE`** in the terminal to trigger the jump!
  - Microduck crouches, launches straight up, stays airborne, lands on two feet in a clean crouch, and recovers to an upright standing posture.
  - Once finished, it remains standing in place, ready for you to press **`J`** or **`SPACE`** to jump again.

---

## 3. Monitoring Training in Weights & Biases

Key metrics to watch per iteration:
- `Episode_Reward/jump_takeoff_velocity`: Rises as vertical upward velocity reaches ~1.0 m/s.
- `Episode_Reward/jump_flight_air_time`: Tracks airborne flight progression.
- `Episode_Reward/jump_two_foot_landing`: Pays for landing on feet in crouch height (`CROUCH_Z = 0.065 m`) without butt contact.
- `Episode_Reward/jump_return_stand`: Dominant annuity once full jump sequence succeeds without butt contact.
- `Episode_Reward/jump_non_foot_contact`: Tracks ground contact by trunk/butt/hips. Must remain **$\le 0$** and trend toward 0.
- All penalties (`jump_horizontal_vel`, `jump_verticality`, `jump_stillness`, `jump_crouch_feet_lift`, `jump_post_landing_hop`) must remain **$\le 0$**.

---

## 4. Key Files

- `src/mjlab_microduck/tasks/mdp.py`: Jump phase command, non-foot contact detection, crouch takeoff/landing rewards, and butt-contact lockout.
- `src/mjlab_microduck/tasks/microduck_jump_env_cfg.py`: Jump environment configuration (`non_foot_ground_contact` sensor, landing crouch rewards, 61D layout).
- `src/mjlab_microduck/tasks/__init__.py`: Task registration (`Mjlab-Jump-Flat-MicroDuck` and backlash variants).
- `scripts/infer_policy.py`: Deployment rehearsal viewer with standalone `--jump` support and `J` / `SPACE` keys.
- `tests/test_jump_cfg.py`: Regression test suite locking in invariants, non-foot contact sensors, and reward signs.
