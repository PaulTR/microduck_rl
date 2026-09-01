"""Microduck Jump task — explosive vertical jump and compliant landing recovery.

Episodic policy: the robot starts STANDING (HOME pose + noise). The goal is to
compress the legs (squat), push off explosively with positive vertical velocity
(vz > 0), achieve flight phase with both feet breaking ground contact and
reaching target apex height (z ≈ 0.16 m vs standing 0.115 m), then land compliantly
on both feet and re-stabilize back into the neutral HOME standing pose.

Key design points:
  - 61D Unified Observation Contract: keeps the full proprioception (48) +
    13D command block [twist(3), head_pose(4), body_pose(6)]. Commands are zero-padded/
    near-zero sampled so the input struct is 100% hot-swappable in the runtime daemon.
  - Rebased on make_microduck_velocity_env_cfg: inherits the full domain randomization
    (CoM, mass/inertia, BAM friction, IMU misalignment, encoder bias, NaN guard) for
    reliable sim2real transfer.
  - Hardware Protection: foot impact penalty to discourage slamming the Dynamixel XL330
    gears and 3D-printed ankle parts upon landing.
"""

import math
from copy import deepcopy

# Left/Right symmetry enabled to ensure bilateral symmetric push-off
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to velocity / standup for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # Disabled for jumping to keep trajectory vertical
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges (matched to velocity recipe) ───────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003
HEAD_COM_RANDOMIZATION_RANGE        = 0.003
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# ── Task constants ────────────────────────────────────────────────────────────
# 1.2 seconds at 50 Hz = 60 control steps (squat, launch, flight, and landing).
EPISODE_LENGTH_S = 1.2

# Trunk heights (m)
STAND_Z = 0.115
JUMP_TARGET_APEX_Z = 0.145

# Servo indices
_LEG_JOINTS  = [0, 1, 2, 3, 4, 9, 10, 11, 12, 13]
_NECK_JOINTS = [5, 6, 7, 8]

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlModelCfg,
)
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.velocity import mdp
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
    make_microduck_velocity_env_cfg,
    NUM_STEPS_PER_ENV,
)
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_jump_env_cfg(
    play: bool = False,
    rough: bool = False,
) -> ManagerBasedRlEnvCfg:
    """Create Microduck Jump environment configuration.

    Inherits base DR, actuators, observation pipeline, and NaN guards from the
    proven velocity recipe, re-wiring rewards for explosive jump and landing recovery.
    """
    cfg = make_microduck_velocity_env_cfg(play=play, rough=rough)

    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Drop walking & passive tracking rewards ───────────────────────────────
    # Standing still must not earn an annuity from unused tracking tasks.
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "air_time",
        "pose",
        "head_pose_tracking",
        "head_pose_bias",
        "body_pose_tracking",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Drop walking & pose curricula inherited from velocity ─────────────────
    for name in [
        "head_pose_bias_weight",
        "standing_envs",
        "head_pose_range",
        "body_pose_range",
        "action_rate_weight",
    ]:
        if name in cfg.curriculum:
            del cfg.curriculum[name]

    # ── Tune general posture & smoothness stabilizers (keep LOW during skill discovery)
    # Low weights ensure "standing still" does NOT beat attempting the jump.
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 0.2
        cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.02

    if "angular_momentum" in cfg.rewards:
        cfg.rewards["angular_momentum"].weight = -0.01

    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.05

    # ── Add Jump Task Rewards (Dominant Mass) ──────────────────────────────────
    # 1. Initial explosive upward push-off velocity near the floor (first 0.35s only)
    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=microduck_mdp.jump_launch_velocity,
        weight=10.0,
        params={
            "max_height": 0.135,
            "max_step": 18,
            "upright_std": 0.3,
        },
    )

    # 2. Airborne reward: both feet fully in the air while upright
    cfg.rewards["jump_air_time"] = RewardTermCfg(
        func=microduck_mdp.jump_air_time_reward,
        weight=15.0,
        params={
            "sensor_name": "feet_ground_contact",
            "min_height": 0.125,
            "upright_std": 0.25,
        },
    )

    # 3. Peak height progress: potential-based reward paying for every mm gained
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.jump_peak_height_progress,
        weight=15.0,
        params={
            "target_apex_height": JUMP_TARGET_APEX_Z,
            "stand_z": STAND_Z,
            "upright_std": 0.25,
        },
    )

    # 4. Landing recovery: strictly gated on having achieved flight (z >= 0.130m)
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=microduck_mdp.jump_landing_composite,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.02,
            "upright_std": 0.25,
            "pose_std": 0.3,
            "joint_indices": _LEG_JOINTS,
            "require_airborne_latch": True,
        },
    )

    # 5. Lateral & forward drift penalty: strictly penalize horizontal stepping (negative weight)
    cfg.rewards["jump_drift_penalty"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-2.0,
    )

    # 6. Yaw spin rate penalty: strictly penalize clockwise/counter-clockwise rotation
    cfg.rewards["jump_yaw_rate"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_rate_penalty,
        weight=-1.0,
    )

    # 7. Heading anchor: small guidance reward to keep initial spawn heading
    cfg.rewards["heading_anchor"] = RewardTermCfg(
        func=microduck_mdp.heading_hold_reward,
        weight=0.2,
        params={"std": 0.25},
    )

    # 8. Bilateral leg symmetry: enforce identical mirrored left/right push-off
    cfg.rewards["leg_symmetry"] = RewardTermCfg(
        func=microduck_mdp.leg_symmetry_reward,
        weight=0.2,
    )

    # 9. Touchdown impact penalty: protect XL330 gears from extreme landing force
    cfg.rewards["jump_foot_impact"] = RewardTermCfg(
        func=microduck_mdp.jump_foot_impact_penalty,
        weight=-0.1,
        params={
            "sensor_name": "feet_ground_contact",
            "force_threshold": 25.0,
        },
    )

    # ── Events ────────────────────────────────────────────────────────────────
    # Reset the airborne latch and max height on every episode reset
    cfg.events["reset_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
        params={"stand_z": STAND_Z},
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    # Early reset if robot tips over beyond 60 degrees tilt
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(60.0), "asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Commands: clamp velocity commands to zero for stationary jumping ──────
    # Keeps the 13D command block active in observation space without requesting walking
    if "twist" in cfg.commands:
        cfg.commands["twist"].ranges.lin_vel_x = (0.0, 0.0)
        cfg.commands["twist"].ranges.lin_vel_y = (0.0, 0.0)
        cfg.commands["twist"].ranges.ang_vel_z = (0.0, 0.0)

    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    actor=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
        distribution_cfg={
            "class_name": "GaussianDistribution",
            "init_std": 1.0,
            "std_type": "scalar",
        },
    ),
    critic=RslRlModelCfg(
        hidden_dims=(512, 256, 128),
        activation="elu",
        obs_normalization=True,
    ),
    algorithm=PpoWithSymmetryCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
        symmetry_cfg=SYMMETRY_CFG if ENABLE_SYMMETRY else None,
    ),
    wandb_project="mjlab_microduck",
    experiment_name="jump",
    run_name="jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=2000,
)
