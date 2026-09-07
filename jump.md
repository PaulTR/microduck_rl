# Microduck Two-Legged Vertical Jump Policy (Jump-3)

A physics-grounded, two-legged vertical jump in place with high explosive lift and balanced landing.
The robot starts in an upright stand ($z = 0.115$ m), dips into a crouch ($z = 0.080$ m), drives upward explosively ($v_z > 0$, weight 8.0), achieves high ballistic flight (target apex $z \approx 0.170$ m, lift-scaled airborne bonus), and lands cleanly on both feet directly into a vertical standing posture ($z \approx 0.115$ m) at $(x=0, y=0)$ without falling backwards or spinning.

---

## 1. Key Physics & Biomechanical Upgrades

### A. Maximizing Jump Height & Air Time
1. **Continuous Lift-Scaled Airborne Reward**:
   - The previous binary 0/1 airborne bonus paid 100% of the reward for a 2 mm hop. PPO naturally settled on a tiny hop to minimize energy and avoid tilt risk.
   - **Fix**: The airborne bonus is now scaled continuously by lift height:
     $$\text{score} = \text{both\_feet\_airborne} \times \text{upright\_gate} \times \text{clamp}\left(\frac{z - 0.120}{0.050}, 0.0, 1.5\right)$$
     A 5 mm hop earns only $10\%$, while a 55 mm leap earns $100\%$ (up to $150\%$ for higher). Higher jumps earn up to **15× more reward**.
2. **Aggressive Push-Off Velocity ($v_z > 0$)**:
   - `jump_push_velocity` weight boosted to **8.0** with reward scaling up to $1.2\text{ m/s}$. The harder the legs drive into the floor, the larger the payout.
3. **Higher Apex Target**:
   - `APEX_Z` increased from $0.155\text{ m}$ to **$0.170\text{ m}$** (a $5.5\text{ cm}$ vertical lift above standing).
4. **Low Action-Rate Attempt Tax**:
   - `action_rate_l2` starts at `-0.003` so explosive acceleration is not throttled during push-off.

### B. Eliminating Backward Landing Falls
1. **Seamless Circular Phase Wrap**:
   - Twist command is $[ \cos(2\pi\phi), \sin(2\pi\phi), 0 ]$. At $\phi = 1.0$, the embedding is $[1.0, 0.0]$, which is identical to $\phi = 0.0$.
   - When $\phi = 0$ was crouch and $\phi = 1$ was stand, the policy received $[1.0, 0.0]$ right at landing and immediately bent its knees to crouch again, collapsing backwards onto its back.
   - **Fix**: Re-centered the cycle so **both $\phi = 0.0$ and $\phi = 1.0$ command nominal STAND**! Holding $\phi = 1.0$ after the jump now commands the policy to stand upright and balance stably.
2. **Anti-Pitch Rate Penalty ($\omega_y^2$)**:
   - `jump_pitch_rate` (weight `-2.5`) heavily penalizes body pitch angular velocity during push-off and flight. This prevents the robot from rotating backward in mid-air.
3. **Landing Damping**:
   - `jump_landing_damping` (weight `-2.0`) in $\phi \in [0.78, 0.05]$ penalizes residual linear and angular velocity upon touchdown, bringing the body to a stable rest.

---

## 2. 5-Phase Jump Timeline ($T = 1.2$ s, 60 steps @ 50 Hz)

| Phase | Phase $\phi$ | Duration | Objective | Key Rewards |
|---|---|---|---|---|
| **1. Ready Stand** | $[0.00, 0.08]$ | $0.10$ s | Stable upright stand in HOME pose ($z = 0.115$ m) | `jump_stand` (6.0), `jump_feet_grounded` (2.5) |
| **2. Crouch** | $[0.08, 0.28]$ | $0.24$ s | Lower CoM to $z = 0.080$ m with flat feet | `jump_crouch` (4.0) |
| **3. Push-Off** | $[0.25, 0.48]$ | $0.28$ s | Explosively drive legs straight up ($v_z > 0$) | `jump_push_velocity` (8.0) |
| **4. Flight** | $[0.45, 0.70]$ | $0.30$ s | Reach apex $z \approx 0.170$ m in the air upright | `jump_airborne` (6.0, lift-scaled), `jump_apex_height` (6.0) |
| **5. Land & Settle** | $[0.68, 1.00]$ | $0.38$ s | Touch down on two feet into HOME stand ($z = 0.115$ m) | `jump_stand` (6.0), `jump_landing_damping` (-2.0) |

---

## 3. Training & Playback

### Full Training (4096 envs on Linux GPU or HF Jobs)
```bash
# On your Linux training machine:
git pull
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096
# Or on Hugging Face Jobs:
uv run train Mjlab-Jump-Flat-MicroDuck --env.scene.num-envs 4096 --hf-jobs
```

### Export to ONNX
```bash
uv run scripts/export.py Mjlab-Jump-Flat-MicroDuck --wandb-run-path <entity/project/run_id> -o jump.onnx
```

### Playback & Stand Options
You can run the jump policy in two ways:

1. **Standalone (Jump Policy Balances Itself)**:
   ```bash
   uv run scripts/infer_policy.py --jump jump.onnx --new-cmd-obs
   ```
   The policy completes the jump and automatically settles and balances in the upright standing pose at $\phi = 1.0$.

2. **Chained Handover to a Separate Standing Policy**:
   ```bash
   uv run scripts/infer_policy.py --jump jump.onnx --standing stand.onnx --new-cmd-obs
   ```
   At the end of the jump cycle ($1.2$ s), `infer_policy.py` automatically hands control over to your standing policy to maintain balance.
