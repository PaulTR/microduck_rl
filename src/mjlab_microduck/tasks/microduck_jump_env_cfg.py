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

# Left/Right symmetry can be enabled if desired (symmetric task). Left OFF by default.
ENABLE_SYMMETRY = False

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
# 2.0 seconds at 50 Hz = 100 control steps. Enough for squat, launch, flight, landing.
EPISODE_LENGTH_S = 2.0

# Trunk heights (m)
STAND_Z = 0.115
JUMP_TARGET_APEX_Z = 0.160

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

    # ── Drop walking-specific locomotion rewards ──────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "air_time",
        "pose",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Tune general posture & smoothness stabilizers ─────────────────────────
    # Upright reward to keep torso vertical (weight 1.0 allows slight lean for launch)
    if "upright" in cfg.rewards:
        cfg.rewards["upright"].weight = 1.0
        cfg.rewards["upright"].params["std"] = math.sqrt(0.05)

    # Damping roll/pitch angular wobble
    if "body_ang_vel" in cfg.rewards:
        cfg.rewards["body_ang_vel"].weight = -0.05

    # Action smoothness (damp high-frequency servo jitter without blocking big launch)
    if "action_rate_l2" in cfg.rewards:
        cfg.rewards["action_rate_l2"].weight = -0.1

    # ── Add Jump Task Rewards ─────────────────────────────────────────────────
    # 1. Initial explosive upward push-off velocity near the floor (boosted)
    cfg.rewards["jump_launch"] = RewardTermCfg(
        func=microduck_mdp.jump_launch_velocity,
        weight=4.0,
        params={
            "max_height": 0.135,
            "upright_std": 0.3,
        },
    )

    # 2. Airborne reward: both feet fully in the air while upright (boosted)
    cfg.rewards["jump_air_time"] = RewardTermCfg(
        func=microduck_mdp.jump_air_time_reward,
        weight=6.0,
        params={
            "sensor_name": "feet_ground_contact",
            "min_height": 0.125,
            "upright_std": 0.25,
        },
    )

    # 3. Apex height target: Gaussian reward for reaching apex z ≈ 0.16 m
    cfg.rewards["jump_height"] = RewardTermCfg(
        func=microduck_mdp.jump_height_target,
        weight=4.0,
        params={
            "target_height": JUMP_TARGET_APEX_Z,
            "height_std": 0.025,
            "upright_std": 0.25,
        },
    )

    # 4. Compliant landing recovery: absorbs impact with a knee crouch upon touchdown,
    # then smoothly extends back into an erect standing posture with upright head.
    cfg.rewards["jump_landing"] = RewardTermCfg(
        func=microduck_mdp.jump_compliant_landing,
        weight=3.0,
        params={
            "crouch_height": 0.102,
            "stand_height": STAND_Z,
            "crouch_steps": 8,
            "settle_steps": 16,
            "height_std": 0.02,
            "upright_std": 0.20,
            "pose_std": 0.35,
        },
    )

    # 5. Lateral drift penalty: keep jump strictly vertical in-place (negative weight)
    cfg.rewards["jump_drift_penalty"] = RewardTermCfg(
        func=microduck_mdp.jump_lateral_drift_penalty,
        weight=-0.5,
    )

    # 6. Yaw rate penalty: suppress spinning in the air without forcing leg symmetry
    cfg.rewards["jump_yaw_rate"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_rate_penalty,
        weight=-0.3,
    )

    # 7. Head pitch constraint: allows free movement within +/-30 deg for balance,
    # charging a quadratic barrier penalty only beyond 30 deg forward or backward.
    cfg.rewards["head_pitch_limit"] = RewardTermCfg(
        func=microduck_mdp.head_pitch_limit_penalty,
        weight=-2.0,
        params={"max_angle_rad": math.radians(30.0)},
    )

    # 8. Touchdown impact penalty: protect XL330 gears from extreme landing force
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

    # ── Terminations ──────────────────────────────────────────────────────────
    # Early reset if robot tips over beyond 35 degrees tilt
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=mdp.bad_orientation,
        params={"limit_angle": math.radians(35.0), "asset_cfg": SceneEntityCfg("robot")},
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
