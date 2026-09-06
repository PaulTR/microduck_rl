# Microduck Vertical Jump (`Mjlab-Jump-Flat-MicroDuck`)

A reinforcement learning task teaching Microduck (~800 g, ~25 cm bipedal robot with 14 Dynamixel XL330 servos) to perform a clean, two-footed vertical jump straight up in place with air-time gated landing and stable upright standing (no forward shuffle or drift).

---

## 1. Overview & Motion Progression

The jump is parameterized over a 3.0 s period ($T = 3.0$ s) with phase $\phi \in [0, 1)$ encoded in the `twist` observation slot as $[\cos(2\pi\phi), \sin(2\pi\phi), 0]$. Each episode starts standing at $\phi = 0$:

1. **Initial Stand & Settle ($\phi \in [0.00, 0.10]$ | $t \in [0.0, 0.3]\text{ s}$)**:
   - Starts standing at nominal height (`STAND_Z = 0.115 m`) in the `HOME` joint pose.
   - Body and head strictly vertical ($g_x \approx 0, g_y \approx 0$).
2. **Crouch Loading ($\phi \in [0.10, 0.32]$ | $t \in [0.3, 1.0]\text{ s}$)**:
   - Symmetrically lowers trunk height to `CROUCH_Z = 0.065 m` by flexing knees and hips.
   - Both feet remain firmly planted on the ground (`jump_crouch_feet_grounded`).
   - Anti-pre-hop penalty (`jump_crouch_feet_lift`) prevents premature skipping.
3. **Explosive Vertical Takeoff ($\phi \in [0.30, 0.48]$ | $t \in [0.9, 1.4]\text{ s}$)**:
   - Demands rapid vertical upward velocity ($v_z \approx 1.0\text{ m/s}$) straight up.
   - Horizontal velocity penalty (`jump_horizontal_velocity_penalty`) strictly taxes any horizontal movement ($v_x^2 + v_y^2$).
   - Body verticality penalty (`jump_verticality_penalty`) prevents forward pitch lean or lateral roll ($g_x^2 + g_y^2$).
   - Both feet push off simultaneously.
4. **Airborne Flight ($\phi \in [0.35, 0.65]$ | $t \in [1.0, 1.9]\text{ s}$)**:
   - Both feet airborne simultaneously (`torch.minimum(air_l, air_r)`).
   - Body remains vertical in the air without pitching forward.
5. **Two-Footed Landing in Place ($\phi \in [0.58, 0.78]$ | $t \in [1.7, 2.3]\text{ s}$)**:
   - **Air-time gated**: Landing annuity is multiplied by `jump_flight_gate`. If the robot never achieves $\ge 80\text{ ms}$ continuous flight, landing pays **zero**.
   - Both feet touch down simultaneously in place with vertical shock absorption ($|a_z|$ penalty).
6. **Return to Vertical Stand & Stillness ($\phi \in [0.72, 1.00]$ | $t \in [2.1, 3.0]\text{ s}$)**:
   - Recovers trunk to `STAND_Z = 0.115 m`, body and head upright, HOME joint pose.
   - Stillness penalty (`jump_post_landing_stillness_penalty`) damps linear and angular velocity once landed to prevent shuffling, walking forward, or toppling.
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
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
```
- Or train with an explicit iteration count:
  ```bash
  uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 4000
  ```
- If running on Hugging Face Jobs, simply add `--hf-jobs`:
  ```bash
  uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
  ```
- Logs to Weights & Biases under project `mjlab_microduck` (`experiment_name="microduck_jump"`).
- Checkpoints are saved to `logs/microduck_jump/<run_id>/model_XXXX.pt`.

### Step 4: Watch Trained Policy in the Viewer
Watch the checkpoint in the viewer using mjlab's standard `play` command:
```bash
uv run play Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```
Or view a local checkpoint directly:
```bash
uv run play Mjlab-Jump-Flat-MicroDuck --checkpoint-file logs/microduck_jump/<run_id>/model_4000.pt
```

### Step 5: Export to ONNX (with Baked Observation Normalizer)
Always export using the repo's exporter (`scripts/export.py`) so the observation normalizer is baked into the ONNX graph:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
```
Or export from a local checkpoint:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --checkpoint-file logs/microduck_jump/<run_id>/model_4000.pt --onnx-file output.onnx
```

### Step 6: Deployment Rehearsal in MuJoCo Viewer
Rehearse the exported policy in CPU MuJoCo with BAM M6 actuators:
```bash
uv run scripts/infer_policy.py --walking walk.onnx --jump output.onnx --new-cmd-obs
```
- In the viewer:
  - Press **`J`** to trigger the jump!
  - Microduck crouches, launches straight up, stays airborne, lands on two feet in place, and returns to a steady vertical stand.

---

## 3. Monitoring Training in Weights & Biases

Key metrics to watch per iteration:
- `Episode_Reward/jump_takeoff_velocity`: Should rise steadily as vertical upward velocity reaches ~1.0 m/s.
- `Episode_Reward/jump_flight_air_time`: Tracks flight air-time progression.
- `Episode_Reward/jump_two_foot_landing`: Starts paying once flight gate opens ($\ge 80\text{ ms}$ flight achieved).
- `Episode_Reward/jump_return_stand`: Dominant annuity once full jump sequence succeeds.
- Penalties (`jump_horizontal_vel`, `jump_verticality`, `jump_stillness`, `jump_crouch_feet_lift`, `jump_post_landing_hop`) must all remain **$\le 0$**.

---

## 4. Key Files

- `src/mjlab_microduck/tasks/mdp.py`: Jump phase command, flight gate, takeoff/flight/landing rewards, horizontal/tilt/stillness penalties.
- `src/mjlab_microduck/tasks/microduck_jump_env_cfg.py`: Jump environment config (curricula, 61D obs layout, BAM friction DR).
- `src/mjlab_microduck/tasks/__init__.py`: Task registration (`Mjlab-Jump-Flat-MicroDuck` and backlash variants).
- `scripts/infer_policy.py`: Deployment rehearsal viewer with `--jump` and `J` key shortcut.
- `tests/test_jump_cfg.py`: Regression test suite locking in invariants, reward signs, and 61D layout.
