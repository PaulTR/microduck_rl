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

# Left/Right symmetry enabled for bilateral forward broad jump
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to velocity / standup for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # Disabled for jumping to keep trajectory clean
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
# 3.0 seconds at 50 Hz = 150 control steps:
#   - 0.0 - 0.4s: squat & bilateral push-off
#   - 0.4 - 0.7s: parabolic flight
#   - 0.7 - 1.0s: touchdown & knee crouch absorption
#   - 1.0 - 1.6s: extension from crouch to standing height (0.115m)
#   - 1.6 - 3.0s: rock-solid standing hold (1.4s of dense standing annuity!)
EPISODE_LENGTH_S = 3.0

# Trunk heights (m)
STAND_Z = 0.115
JUMP_TARGET_APEX_Z = 0.148

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

    # ── Drop walking-specific locomotion and head-look rewards ────────────────
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
        "upright",  # Always-on upright opposes forward takeoff lean; orientation is gated by jump terms
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Drop conflicting walking-specific curricula ───────────────────────────
    # Prevents action rate penalty from ramping up to -1.0 (which blocks explosive launch),
    # and eliminates wide head flailing commands that contradict head pitch limits.
    for c_name in [
        "action_rate_weight",
        "standing_envs",
        "head_pose_range",
        "head_pose_bias_weight",
    ]:
        if c_name in cfg.curriculum:
            del cfg.curriculum[c_name]

    # ── Tune general posture & smoothness stabilizers ─────────────────────────
    # Damping roll/pitch angular wobble
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.05

    # Action smoothness: gentle -0.05 damps servo jitter without taxing explosive impulse
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.05

    # ── Add Jump Task Rewards ─────────────────────────────────────────────────
    # 1. Bilateral takeoff launch velocity: rewards both vx > 0 and vz > 0 with both feet grounded
    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=microduck_mdp.jump_launch_velocity,
        weight=4.0,
        params={
            "sensor_name": "feet_ground_contact",
            "require_both_feet_ground": True,
            "max_height": 0.135,
            "upright_std": 0.35,
            "target_vx": 0.4,
            "target_vz": 0.6,
        },
    )

    # 2. Bilateral leg similarity: guides left and right legs to push symmetrically
    cfg.rewards["leg_similarity"] = RewardTermCfg(
        func=microduck_mdp.leg_similarity_reward,
        weight=1.5,
    )

    # 3. Airborne reward: both feet fully in the air while upright (active until touchdown)
    cfg.rewards["jump_air_time"] = RewardTermCfg(
        func=microduck_mdp.jump_air_time_reward,
        weight=5.0,
        params={
            "sensor_name": "feet_ground_contact",
            "min_height": 0.125,
            "upright_std": 0.35,
        },
    )

    # 4. Apex height target: Gaussian reward for reaching apex z ≈ 0.148 m
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.jump_height_target,
        weight=4.0,
        params={
            "target_height": JUMP_TARGET_APEX_Z,
            "height_std": 0.025,
            "upright_std": 0.35,
        },
    )

    # 5. Forward distance reward: rewards forward displacement (x - x_spawn) achieved through flight
    cfg.rewards["jump_forward_dist"] = RewardTermCfg(
        func=microduck_mdp.jump_forward_distance_reward,
        weight=3.0,
        params={"target_distance": 0.15},
    )

    # 6. Compliant landing & standing recovery: absorbs impact with a knee crouch upon touchdown,
    # then smoothly extends back into an erect standing posture and holds still for the rest of the episode.
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=microduck_mdp.jump_compliant_landing,
        weight=5.0,
        params={
            "crouch_height": 0.100,
            "stand_height": STAND_Z,
            "crouch_steps": 10,
            "settle_steps": 30,
            "height_std": 0.02,
            "upright_std": 0.20,
            "pose_std": 0.35,
        },
    )

    # 7. Lateral velocity penalty: prevent sideways drift (vy^2) while permitting forward jump (vx > 0)
    cfg.rewards["jump_lateral_vel"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_velocity_penalty,
        weight=-0.5,
    )

    # 8. Yaw rate penalty: suppress spinning in the air
    cfg.rewards["jump_yaw_rate"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_rate_penalty,
        weight=-0.3,
    )

    # 9. Head pitch constraint: allows free movement within +/-30 deg for balance,
    # charging a quadratic barrier penalty only beyond 30 deg forward or backward.
    cfg.rewards["head_pitch_limit"] = RewardTermCfg(
        func=microduck_mdp.head_pitch_limit_penalty,
        weight=-2.0,
        params={"max_angle_rad": math.radians(30.0)},
    )

    # 10. Touchdown impact penalty: protect XL330 gears from extreme landing force
    cfg.rewards["jump_foot_impact"] = RewardTermCfg(
        func=microduck_mdp.jump_foot_impact_penalty,
        weight=-0.1,
        params={
            "sensor_name": "feet_ground_contact",
            "force_threshold": 25.0,
        },
    )

    # ── Events ────────────────────────────────────────────────────────────────
    # Reset the airborne latch on every episode reset
    cfg.events["reset_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
    )

    # Disable random pushes during jumping (incoherent for jump/landing discovery)
    if not ENABLE_VELOCITY_PUSHES and "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    # ── Terminations ──────────────────────────────────────────────────────────
    # Early reset if robot falls over beyond 55 degrees tilt (allows forward crouch & landing absorption)
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(55.0), "asset_cfg": SceneEntityCfg("robot")},
    )

    # Terminate immediately if robot bounces airborne a second time after touchdown (kills skipping)
    cfg.terminations["double_bounce"] = TerminationTermCfg(
        func=microduck_mdp.jump_double_bounce,
        params={"sensor_name": "feet_ground_contact"},
    )

    # Terminate immediately if head pitches beyond 40 degrees forward or backward
    cfg.terminations["head_pitch_exceeded"] = TerminationTermCfg(
        func=microduck_mdp.head_pitch_exceeded,
        params={"max_angle_rad": math.radians(40.0)},
    )

    # ── Commands: near-zero ranges keep 61D obs neurons alive without commanding walking ──
    if "twist" in cfg.commands:
        cfg.commands["twist"].ranges.lin_vel_x = (-0.01, 0.01)
        cfg.commands["twist"].ranges.lin_vel_y = (-0.01, 0.01)
        cfg.commands["twist"].ranges.ang_vel_z = (-0.01, 0.01)
        cfg.commands["twist"].heading_command = False
        cfg.commands["twist"].ranges.heading = None
        cfg.commands["twist"].rel_standing_envs = 0.0
        cfg.commands["twist"].rel_heading_envs = 0.0

    # Keep small non-zero sampling near zero to preserve 61D obs neurons alive without commanding flailing
    if "head_pose" in cfg.commands:
        cfg.commands["head_pose"].ranges = ((-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01))
    if "body_pose" in cfg.commands:
        cfg.commands["body_pose"].ranges = (
            (-0.005, 0.005), (-0.005, 0.005), (-0.005, 0.005),
            (-0.01, 0.01), (-0.01, 0.01), (-0.01, 0.01),
        )

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
    max_iterations=5000,
)
