# Microduck Jump-4: Autonomous Two-Legged Vertical Jump Policy

A complete technical specification, replication guide, and architectural walkthrough for software engineers.

---

## 1. Executive Summary

- **Objective**: Teach **Microduck**—an 800-gram, 25-cm-tall miniature bipedal robot with 14 Dynamixel XL330 hobby micro-servos—to perform a stable two-legged vertical jump from an upright standing pose and cleanly land back on its feet.
- **Method**: Model-Free Reinforcement Learning (PPO via `rsl_rl`) trained inside a GPU-accelerated physics engine ([mjlab](https://github.com/mujocolab/mjlab) / MuJoCo Warp) at **50 Hz** (20 ms timestep).
- **Key Result (`jump-4`, commit `5cb80b4`)**: A clean, repeatable, balanced vertical hop (~1.5 cm liftoff) that adheres strictly to the physical power limits of the hardware, sticks the landing, and automatically hands over control to a pre-trained standing policy (`alpha_stand.onnx`).

```
  Standing (t = 0.0s)       Pre-Crouch (t = 0.2s)      Explosive Push (t = 0.35s)     Apex Liftoff (t = 0.4s)     Touchdown & Stand (t = 0.6s+)
     [  Head  ]                  [  Head  ]                   [  Head  ]                   [  Head  ]                  [  Head  ]
     [  Body  ]                  [  Body  ]                   [  Body  ]                   [  Body  ]                  [  Body  ]
      /      \                    /      \                     |      |                     |      |                    /      \
     /        \                  /        \                    |      |                     |      |                   /        \
  --[Sole]--[Sole]--          --[Sole]--[Sole]--            --[Sole]--[Sole]--                                    --[Sole]--[Sole]--
───────────────────────     ───────────────────────       ───────────────────────     ───────────────────────     ───────────────────────
    Flat Ground                 Dip (-10 mm)               Drive into Floor                ~1.5 cm Apex Liftoff          Stick Landing!
```

---

## 2. Core Concepts: RL for Software Engineers

If you are a software engineer with little robotics background, think of this system as an **asynchronous closed-loop optimization system**:

1. **The Controller (Neural Network Policy)**:
   - A multi-layer perceptron (MLP) running as an inference loop at **50 Hz** (every 20 ms).
   - **Input (Observation Vector, 61 floats)**: Current joint positions, joint velocities, gravity vector (from the IMU tilt sensor), angular velocity, past actions, and a **13-float command block**.
   - **Output (Action Vector, 14 floats)**: Target position offsets for the 14 Dynamixel servo motors. An on-chip PID controller converts these into electrical voltages.

2. **The "Phase Clock" ($\phi \in [0, 1)$)**:
   - Feedforward neural networks have no internal state or clock. To tell the policy *what phase of the jump it is in*, we inject a cyclic clock signal into the 61D input vector:
     $$\text{command} = [\cos(2\pi\phi),\, \sin(2\pi\phi),\, 0.0]$$
   - An episode lasts $1.2\text{ seconds}$ (60 steps). At step 0, $\phi = 0.0$. At step 60, $\phi = 1.0$. The neural network conditions its behavior directly on this clock.

3. **The Objective Function (Reward Engineering)**:
   - In RL, you do not write the control algorithm; you define an **objective function** (rewards and penalty taxes). An optimizer (PPO) simulates millions of attempts across 4,096 parallel worlds, mutating network weights to maximize the score.
   - **Reward Hacking**: RL agents are ruthless optimizers. If there is a mathematical loophole in the reward function, the agent will exploit it instead of learning the desired behavior (e.g., falling over backwards to get "airborne" points without jumping).

---

## 3. The Hardware Reality: The "Power Budget"

Robotics is bound by physics. Unlike virtual video-game characters, physical actuators have hard limits:

| Component | Microduck Spec | Engineering Constraint |
|---|---|---|
| **Total Weight ($m$)** | ~800 g ($mg = 7.85\text{ N}$) | Every gram must be accelerated upward against gravity. |
| **Actuators** | 14× Dynamixel XL330-M288 (7.4 V) | Miniature 18-gram plastic-geared hobby servos (288:1 reduction). |
| **Stall Torque ($\tau_{\text{stall}}$)** | $0.963\text{ Nm}$ | Maximum holding torque before the motor stalls. |
| **No-Load Speed ($\omega_0$)** | $20.2\text{ rad/s}$ ($193\text{ RPM}$) | Maximum speed with zero load. |
| **Peak Mechanical Power** | $\approx \mathbf{4.86\text{ W}}$ per servo | $P_{\text{max}} = \frac{1}{4} \tau_{\text{stall}} \omega_0$. |
| **Leg Pitch Budget (4 servos)** | $\approx \mathbf{19.4\text{ W}}$ theoretical total | Across 2 knees and 2 hips. Real-world thermal/gear efficiency yields ~10–12 W. |
| **Back-EMF Velocity Cap** | $v_z \approx 0.50 - 0.52\text{ m/s}$ | As the motor spins faster, back-EMF counteracts battery voltage. Available torque drops near zero. |

### Why ~1.5 cm is the Hard Physical Limit:
$$\text{Max Ballistic Apex: } h = \frac{v_z^2}{2g} \approx \frac{0.52^2}{2 \times 9.81} \approx \mathbf{1.38\text{ cm} - 1.8\text{ cm}}$$
$$\text{Total Flight Duration: } T_{\text{flight}} = \frac{2 v_z}{g} \approx \frac{2 \times 0.52}{9.81} \approx \mathbf{0.106\text{ seconds (approx 5 steps @ 50 Hz)}}$$

**Key Takeaway**: Any reward shaping that asks Microduck to jump 5 cm forces the optimizer to exploit unnatural behaviors (like whipping its 300g head like a catapult or falling over), because the motors physically cannot deliver the required kinetic energy.

---

## 4. What Was Changed: The Journey of Bugs and Exploits

The final working system (`jump-4`, commit `5cb80b4`) was achieved by systematically diagnosing and eliminating 6 failure modes:

### Issue 1: The Mid-Air Spawn Drop Crash
- **Bug**: Inherited locomotion config spawned the robot 2–5 cm above the ground.
- **Symptom**: When dropped, the uncoordinated robot hit the floor, tilted >16°, and crashed the simulation at step 0 before the policy could act.
- **Fix**: Spawns flat on the ground (`z ∈ [-0.003, +0.003]`, `yaw ∈ [-0.05, +0.05]`).

### Issue 2: The "Boolean Cliff" Zero-Gradient Dead Zone
- **Bug**: Early attempts gated rewards behind boolean thresholds (`if tilt < 0.035 and horiz_vel < 0.04: reward += 1.0`).
- **Symptom**: Random initial exploration never met all strict conditions at once. Reward was 0 everywhere $\implies$ zero gradient $\implies$ policy learned to go limp.
- **Fix**: Replaced boolean gates with a continuous Gaussian kinematic reference trajectory $q^*(\phi)$ (`jump_trajectory_tracking`), providing dense gradient guidance on every millisecond of the motion.

### Issue 3: The "Standing Still" Reward Farm
- **Bug**: The policy was rewarded for standing stably at the end of the episode.
- **Symptom**: Standing still collected safe points with zero risk of falling. The policy refused to jump.
- **Fix**: Gated the landing reward strictly on achieving genuine flight (`_jump_achieved_airborne = True` and lift threshold). If both feet never leave the ground, landing pays exactly 0.0.

### Issue 4: The "Butt Landing" Fake Flight Hack
- **Bug**: Rewarding "both feet off ground".
- **Symptom**: The robot learned to fall backward onto its rear. While sitting on its butt, both feet were lifted into the air! The agent farmed max airborne points without ever jumping.
- **Fix**: Gated all jump rewards on an **upright trunk score**:
  $$u_{\text{score}} = \exp\left(-\frac{\text{tilt}^2}{0.25^2}\right)$$
  If the torso tilts into a fall, the reward instantly collapses to zero.

### Issue 5: Asymmetric Leg Twitches and Spinning
- **Bug**: Left and right legs independently explored different actions, causing the robot to twist or hop on one foot.
- **Fix**: Enabled **Bilateral Symmetry Loss** (`PpoWithSymmetryCfg` via `JUMP_SYMMETRY_CFG`). This enforces that mirroring the robot's state must produce exactly mirrored leg actions while preserving the jump clock phase.

### Issue 6: Sagittal Balance (Zero Backward Drift)
- **Bug**: Flexing knees without counter-adjusting hips moved the center of mass (CoM) behind the heels, toppling the robot backward during pre-crouch.
- **Fix**: Closed kinematic chain balance:
  $$\Delta \text{hip\_pitch} + \Delta \text{ankle\_pitch} = \Delta \text{knee}$$
  This restricts horizontal CoM displacement to $|dx| < 0.6\text{ mm}$, keeping the robot perfectly centered over its feet.

---

## 5. The 4-Stage Biomechanical Jump Cycle

The reference trajectory $q^*(\phi)$ coordinates 6 sagittal pitch joints:
- Left Leg: `hip_pitch` (idx 2), `knee` (idx 3), `ankle` (idx 4)
- Right Leg: `hip_pitch` (idx 11), `knee` (idx 12), `ankle` (idx 13)

```
Stage 1: Pre-Crouch              Stage 2: Explosive Push         Stage 3: Flight Lift            Stage 4: Touchdown & Handover
phi in [0.08, 0.24]              phi in [0.22, 0.34]             phi in [0.34, 0.50]             phi in [0.50, 1.00]
-------------------              -------------------             -------------------             -------------------
Knees flex to +0.36 rad          Knees extend to -0.12 rad       Legs track nominal extension    Feet touch down
Dips ~10 mm; CoM centered        Drives feet down into floor     Apex liftoff (~1.5 cm)          Absorbs impact
Upward velocity vz -> 0          Peak vertical thrust (vz > 0)   Latches _achieved_airborne      Hands over to alpha_stand.onnx
```

---

## 6. Replication Guide: Step-by-Step

### Prerequisites
- Python 3.12+
- `uv` package manager (`curl -LsSf https://astral.sh/uv/install.sh | sh`)
- Linux machine with NVIDIA GPU (recommended for 4096-env training) or Apple Silicon / CPU for verification and evaluation.

### Step 1: Clone and Checkout Branch `jump-4`
```bash
git clone https://github.com/pollen-robotics/microduck_rl.git
cd microduck_rl
git checkout jump-4
uv sync
```

### Step 2: Run the Smoke Test (5 iterations, 64 environments)
Always run a quick smoke test first to confirm environment construction and CUDA/CPU stepping:
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 64 --agent.max_iterations 5
```
*Expected: Finishes in ~5 seconds with 0 NaNs and exit code 0.*

### Step 3: Run Full Training (4,096 environments, 1,200 iterations)
```bash
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
```
- **Wall Time**: ~45–50 minutes on a modern GPU (RTX 3090, 4090, or A100).
- **Checkpoints**: Saved every 100 iterations in `logs/microduck_jump/<timestamp>/checkpoints/model_*.pt`.

### Step 4: What to Monitor in Weights & Biases (WandB)
- `Episode_Reward/jump_trajectory_tracking`: Climbs smoothly to ~3.0–4.0 (following the jump pattern).
- `Episode_Reward/jump_push_velocity`: Reaches $\ge 0.5$ (pushing the floor during extension).
- `Episode_Reward/jump_airborne`: Reaches $\ge 1.0$ (both feet leaving the floor).
- `Episode_Reward/jump_stand`: Reaches $\ge 1.5$ (touchdown landing unlocked by flight).
- `Episode_Termination/fell_over`: Should remain low (<5%).
- `Episode_Metrics/mean_action_acc`: Stays smooth (<3.0; values >8.0 indicate violent twitching).

### Step 5: Export to ONNX (Baking the Observation Normalizer)
The environment normalizes observations with a running mean/variance. The export script bakes this normalizer directly into the ONNX compute graph so the exported model can run anywhere without external state:
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck \
    --checkpoint logs/microduck_jump/<timestamp>/checkpoints/model_1199.pt
```
*Output: `out.onnx`*

### Step 6: Interactive Rehearsal & Standing Handover
Download the official standing policy to catch the landing:
```bash
curl -L -o alpha_stand.onnx https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/alpha_stand.onnx
```

Launch the interactive MuJoCo simulator:
```bash
uv run scripts/infer_policy.py \
    --jump out.onnx \
    --standing alpha_stand.onnx \
    --new-cmd-obs
```

**Interactive Controls:**
- The robot starts balancing stably in the standing policy (`alpha_stand.onnx`).
- Press **`J`**: Triggers the vertical jump. Control transfers to your exported jump policy.
- At touchdown ($\phi = 1.0$, $1.2\text{ s}$), control seamlessly hands back to `alpha_stand.onnx` to hold the upright stance.
- Press **`X`**: Reset robot to the starting pose.
- Press **`Q`**: Exit viewer.

---

## 7. Architecture Reference Table

| Parameter | Configuration | Technical Rationale |
|---|---|---|
| **Branch** | `jump-4` | Working reference baseline (commit `5cb80b4`). |
| **Task ID** | `Mjlab-Jump-Flat-MicroDuck` | Registered in `src/mjlab_microduck/tasks/__init__.py`. |
| **Observation Dim** | 61D | 48D proprioception + 13D command `[twist(3), head(4), body(6)]`. Unified across all Microduck policies for hot-swapping. |
| **Phase Clock Slot** | `twist[0:2]` = `[cos(2πφ), sin(2πφ)]` | Provides cyclic continuous time signal without requiring recurrent neural network (RNN). |
| **Episode Length** | 1.2 s (60 steps @ 50 Hz) | Allows 0.3 s crouch, 0.1 s push, 0.1 s flight, and 0.7 s landing settle. |
| **Actuator Model** | BAM M6 | Realistic Dynamixel voltage-torque-friction physics model. |
| **Symmetry Loss** | Enabled (`PpoWithSymmetryCfg`) | Prevents unilateral hopping and rotation without artificial constraints. |
| **Termination Tilt** | 0.85 rad (~48.7°) | Permits dynamic crouching without premature termination. |
| **Spawn Range** | Grounded: $z \in [-3, +3]$ mm, $\text{yaw} \in [-0.05, +0.05]$ rad | Eliminates step-0 drop crash. |
