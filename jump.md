# Microduck Forward Jump (`Mjlab-Jump-Flat-MicroDuck`)

A reinforcement learning task teaching Microduck (~800 g, ~25 cm bipedal robot with 14 Dynamixel XL330 servos) to perform a clean, two-footed forward jump with air-time gated landing.

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
3. **Explosive Takeoff ($\phi \in [0.30, 0.48]$ | $t \in [0.9, 1.4]\text{ s}$)**:
   - Demands rapid upward velocity ($v_z \approx 1.0\text{ m/s}$) and forward velocity ($v_x \approx 0.8\text{ m/s}$).
   - Both feet push off simultaneously.
4. **Airborne Flight & Feet Swing ($\phi \in [0.35, 0.65]$ | $t \in [1.0, 1.9]\text{ s}$)**:
   - Both feet airborne simultaneously (`torch.minimum(air_l, air_r)`).
   - Flexes hip pitch forward (`jump_feet_swing_forward`) to swing feet forward under/in front of the CoM in preparation for touchdown.
   - Controlled forward pitch lean ($\sim 0\text{--}20^\circ$) is permitted; lateral roll ($g_y$) is strictly penalized.
5. **Two-Footed Landing ($\phi \in [0.58, 0.78]$ | $t \in [1.7, 2.3]\text{ s}$)**:
   - **Air-time gated**: Landing annuity is multiplied by `jump_flight_gate`. If the robot never achieves $\ge 80\text{ ms}$ continuous flight, landing pays **zero**.
   - Both feet touch down simultaneously with vertical shock absorption ($|a_z|$ penalty).
6. **Return to Vertical Stand ($\phi \in [0.72, 1.00]$ | $t \in [2.1, 3.0]\text{ s}$)**:
   - Recovers trunk to `STAND_Z = 0.115 m`, body and head upright, HOME joint pose.
   - Post-landing hop penalty (`jump_post_landing_hop`) strictly taxes lifting feet after landing, preventing double-jumping or skipping.

---

## 2. Linux Machine (with NVIDIA GPU) Workflow

### Step 1: Synchronize Dependencies
```bash
uv sync
```
*(On `linux-aarch64` / DGX Spark, uv automatically routes PyPI's CPU wheel to the cu129 index as pinned in `pyproject.toml`).*

### Step 2: Run the 5-Iteration Smoke Test First
Always run a quick smoke test before launching long runs:
```bash
uv run python scripts/train_jump.py --smoke-test
# or directly:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```
*Expected: builds scenes, steps without NaNs, confirms 61D observation shape, and exits with code 0.*

### Step 3: Launch Full Training
```bash
uv run python scripts/train_jump.py --train
# or directly:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 4000
```
- Logs to weights & biases under project `mjlab_microduck` (`experiment_name="microduck_jump"`).
- Checkpoints are saved every 250 iterations to `logs/microduck_jump/<run_id>/`.
- If training on Hugging Face Jobs, simply append `--hf-jobs`:
  ```bash
  uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.max_iterations 4000 --hf-jobs
  ```

### Step 4: Resume Training (if needed)
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --agent.load_checkpoint logs/microduck_jump/<run_id>/model_XXXX.pt --agent.resume True
```

### Step 5: Export to ONNX (with Baked Observation Normalizer)
Once training converges:
```bash
uv run python scripts/train_jump.py --export --checkpoint logs/microduck_jump/<run_id>/model_4000.pt --output-onnx microduck_jump.onnx
# or directly:
uv run python scripts/export.py Mjlab-Jump-Flat-MicroDuck --checkpoint logs/microduck_jump/<run_id>/model_4000.pt --output microduck_jump.onnx
```

### Step 6: Deployment Rehearsal in MuJoCo Viewer
Test the exported ONNX policy with BAM M6 actuators in the CPU viewer:
```bash
uv run python scripts/infer_policy.py --jump microduck_jump.onnx --new-cmd-obs
```
- In the viewer:
  - Press **`J`** to trigger the two-footed forward jump!
  - Microduck crouches, launches forward, stays airborne, lands on two feet, and returns to standing.

---

## 3. Monitoring Training in Weights & Biases

Key metrics to watch per iteration:
- `Episode_Reward/jump_takeoff_velocity`: Should steadily rise as the duck learns explosive extension.
- `Episode_Reward/jump_flight_air_time`: Tracks air-time progress.
- `Episode_Reward/jump_two_foot_landing`: Starts paying once flight gate opens.
- `Episode_Reward/jump_return_stand`: Dominant annuity once full jump sequence succeeds.
- Every penalty (`Episode_Reward/jump_crouch_feet_lift`, `jump_post_landing_hop`, `jump_sagittal`, `jump_orientation`) must remain **$\le 0$**.

---

## 4. Key Files

- `src/mjlab_microduck/tasks/mdp.py`: Jump phase command, jump state tracker, flight gate, and reward/penalty functions.
- `src/mjlab_microduck/tasks/microduck_jump_env_cfg.py`: Environment configuration, 61D observation setup, symmetry loss, and curricula.
- `src/mjlab_microduck/tasks/__init__.py`: Task registration (`Mjlab-Jump-Flat-MicroDuck` and backlash variants).
- `scripts/train_jump.py`: Convenience CLI runner for smoke tests, training, export, and rehearsal.
- `scripts/infer_policy.py`: Deployment rehearsal viewer with `--jump` and `J` key shortcut.
- `tests/test_jump_cfg.py`: Unit test suite ensuring invariants, reward signs, and 61D layout stay intact.
