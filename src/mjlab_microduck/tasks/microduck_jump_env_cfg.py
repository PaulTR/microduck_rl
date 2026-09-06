"""Microduck two-footed vertical jump task.

Episodic policy: the robot starts standing at HOME pose, crouches by lowering
its trunk with knees and hips flexed, performs an explosive two-footed push-off
launching straight upward (in place), maintains airborne flight, lands squarely
on two feet with impact damping, and returns to a stable vertical stand without
shuffling or drifting forward.

Key design decisions:
  • Phase-driven cyclic command: command = [cos(2π·phase), sin(2π·phase), 0]
    over a 3.0 s period (randomize_phase=False → starts at phase 0 from stand).
    Provides the MLP actor with unambiguous phase awareness so crouch, takeoff,
    flight, landing, and settle are cleanly separated in time.
  • In-place vertical motion (no forward shuffle):
    - Takeoff rewards upward velocity vz, with no forward push target.
    - Horizontal linear velocity (vx² + vy²) is strictly penalized across all phases.
    - Body tilt is strictly penalized throughout (vertical body gx ≈ 0, gy ≈ 0).
    - Post-landing stillness penalty damps residual motion to stick the landing.
  • Air-time gated landing: landing rewards and the post-jump standing annuity
    are multiplied by a continuous smoothstep flight gate (jump_flight_gate).
    If the robot never achieves a minimum airborne duration (>= 80 ms), the
    landing annuity pays zero — grounding/walking/shuffling cannot farm the goal.
  • Single large jump (anti-skipping / anti-double-jump):
    - Crouch phase requires feet grounded (lifting feet is penalized).
    - Post-landing stand phase strictly penalizes foot lifting (anti-hop penalty).
  • Sagittal symmetry: Bilateral mirror-loss (PpoWithSymmetryCfg) keeps left
    and right legs coordinated for a straight symmetric jump.
  • Obs layout: Unified 61D actor layout (48 proprio + 13 command slots) for
    seamless runtime hot-swapping.
"""

import math
from copy import deepcopy

# Bilateral symmetry — sagittal jump is left-right symmetric
ENABLE_SYMMETRY = True

# ── Domain randomisation (matched to standup/velocity for sim2real parity) ───
ENABLE_COM_RANDOMIZATION             = True
ENABLE_HEAD_COM_RANDOMIZATION        = True
ENABLE_KP_RANDOMIZATION              = False
ENABLE_KD_RANDOMIZATION              = False
ENABLE_MASS_INERTIA_RANDOMIZATION    = True
ENABLE_JOINT_FRICTION_RANDOMIZATION  = True
ENABLE_ARMATURE_RANDOMIZATION        = True
ENABLE_VELOCITY_PUSHES               = False  # pushes mid-air/mid-jump are incoherent
ENABLE_IMU_ORIENTATION_RANDOMIZATION = True
ENABLE_ENCODER_BIAS                  = True

# ── Ranges ───────────────────────────────────────────────────────────────────
COM_RANDOMIZATION_RANGE             = 0.003   # ramped via curriculum
HEAD_COM_RANDOMIZATION_RANGE        = 0.003   # ramped via curriculum
MASS_INERTIA_RANDOMIZATION_RANGE    = (0.95, 1.05)
ARMATURE_RANDOMIZATION_RANGE        = (0.9, 1.1)
JOINT_FRICTION_RANDOMIZATION_RANGE  = (0.9, 1.1)
ENCODER_BIAS_RANGE                  = (-0.015, 0.015)
KP_RANDOMIZATION_RANGE              = (0.85, 1.15)
KD_RANDOMIZATION_RANGE              = (0.9, 1.1)
IMU_ORIENTATION_RANDOMIZATION_ANGLE = 6.0

# Episode duration and targets
EPISODE_LENGTH_S = 3.0
STAND_Z          = 0.115
CROUCH_Z         = 0.065

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
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
from mjlab.tasks.velocity.velocity_env_cfg import make_velocity_env_cfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from mjlab_microduck.robot.microduck_constants import MICRODUCK_STANDUP_ROBOT_CFG
from mjlab_microduck.tasks import mdp as microduck_mdp
from mjlab_microduck.tasks.microduck_velocity_env_cfg import HEAD_BODY_NAMES
from mjlab_microduck.tasks.symmetry import PpoWithSymmetryCfg, SYMMETRY_CFG


def make_microduck_jump_env_cfg(play: bool = False, rough: bool = False) -> ManagerBasedRlEnvCfg:
    """Create Microduck forward jump environment configuration."""

    feet_ground_cfg = ContactSensorCfg(
        name="feet_ground_contact",
        primary=ContactMatch(
            mode="geom",
            pattern=r"^(left_foot_collision|right_foot_collision)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found", "force"),
        reduce="netforce",
        num_slots=1,
        track_air_time=True,
    )

    self_collision_cfg = ContactSensorCfg(
        name="self_collision",
        primary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        secondary=ContactMatch(mode="subtree", pattern="trunk_base", entity="robot"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific terms ──────────────────────────────────
    for name in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
        "pose",
        "upright",
    ]:
        if name in cfg.rewards:
            del cfg.rewards[name]

    # ── Rewards: Jump task objectives ─────────────────────────────────────────

    # 1. Crouch loading: lower trunk to CROUCH_Z while keeping feet grounded
    cfg.rewards["jump_crouch_height"] = RewardTermCfg(
        func=microduck_mdp.jump_crouch_height,
        weight=2.5,
        params={
            "target_height": CROUCH_Z,
            "std": 0.02,
            "crouch_start": 0.10,
            "crouch_end": 0.32,
            "command_name": "twist",
        },
    )
    cfg.rewards["jump_crouch_feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.jump_crouch_feet_grounded,
        weight=2.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "crouch_start": 0.00,
            "crouch_end": 0.30,
            "command_name": "twist",
        },
    )
    cfg.rewards["jump_crouch_feet_lift"] = RewardTermCfg(
        func=microduck_mdp.jump_crouch_feet_lift_penalty,
        weight=-2.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "crouch_start": 0.00,
            "crouch_end": 0.30,
            "command_name": "twist",
        },
    )

    # 2. Explosive takeoff: upward velocity vz (pure vertical jump, no forward push)
    cfg.rewards["jump_takeoff_velocity"] = RewardTermCfg(
        func=microduck_mdp.jump_takeoff_velocity,
        weight=6.0,
        params={
            "target_vz": 1.0,
            "std_vz": 0.35,
            "takeoff_start": 0.30,
            "takeoff_end": 0.48,
            "command_name": "twist",
        },
    )

    # 3. Airborne flight: both feet in air (jumping in place)
    cfg.rewards["jump_flight_air_time"] = RewardTermCfg(
        func=microduck_mdp.jump_flight_air_time,
        weight=5.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "target_air_time": microduck_mdp.JUMP_TARGET_AIR_TIME,
            "flight_start": 0.35,
            "flight_end": 0.65,
            "command_name": "twist",
        },
    )

    # 4. Anti-shuffle / in-place constraint: strictly penalize any horizontal velocity (vx² + vy²)
    cfg.rewards["jump_horizontal_vel"] = RewardTermCfg(
        func=microduck_mdp.jump_horizontal_velocity_penalty,
        weight=-2.0,
    )

    # 5. Flight-gated landing on two feet
    cfg.rewards["jump_two_foot_landing"] = RewardTermCfg(
        func=microduck_mdp.jump_two_foot_landing,
        weight=4.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "landing_start": 0.58,
            "landing_end": 0.78,
            "command_name": "twist",
        },
    )

    # 6. Gentle landing shock penalty (|a_z|) — self-negating, positive weight
    cfg.rewards["gentle_landing"] = RewardTermCfg(
        func=microduck_mdp.trunk_vertical_accel_penalty,
        weight=0.002,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",))},
    )

    # 7. Flight-gated return to stand annuity
    cfg.rewards["jump_return_stand"] = RewardTermCfg(
        func=microduck_mdp.jump_return_stand_composite,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.03,
            "upright_std": 0.35,
            "pose_std": 0.35,
            "stand_start": 0.72,
            "stand_end": 1.00,
            "sensor_name": feet_ground_cfg.name,
            "command_name": "twist",
        },
    )

    # 8. Post-landing anti-hop penalty: penalize lifting feet after landing
    cfg.rewards["jump_post_landing_hop"] = RewardTermCfg(
        func=microduck_mdp.jump_post_landing_hop_penalty,
        weight=-3.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "stand_start": 0.72,
            "stand_end": 1.00,
            "command_name": "twist",
        },
    )

    # 9. Post-landing stillness penalty: damp out residual motion once landed to stick the landing
    cfg.rewards["jump_stillness"] = RewardTermCfg(
        func=microduck_mdp.jump_post_landing_stillness_penalty,
        weight=-1.5,
        params={
            "stand_start": 0.72,
            "stand_end": 1.00,
            "command_name": "twist",
            "sensor_name": feet_ground_cfg.name,
        },
    )

    # 10. Orientation and straightness penalties (strictly vertical throughout)
    cfg.rewards["jump_sagittal"] = RewardTermCfg(
        func=microduck_mdp.jump_sagittal_penalty,
        weight=-0.5,
    )
    cfg.rewards["jump_verticality"] = RewardTermCfg(
        func=microduck_mdp.jump_verticality_penalty,
        weight=-2.0,
    )
    cfg.rewards["jump_neck_posture"] = RewardTermCfg(
        func=microduck_mdp.jump_neck_posture_penalty,
        weight=-0.5,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    cfg.rewards["action_rate_l2"] = RewardTermCfg(func=mdp.action_rate_l2, weight=-0.1)
    cfg.rewards["joint_torque_rate_l2"] = RewardTermCfg(
        func=microduck_mdp.joint_torque_rate_l2, weight=0.0
    )
    cfg.rewards["body_ang_vel"].params["asset_cfg"].body_names = ("trunk_base",)
    cfg.rewards["body_ang_vel"].weight = -0.001
    cfg.rewards["angular_momentum"].weight = -0.001
    cfg.rewards.pop("soft_landing", None)

    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # ── Observations (unified 61D layout) ─────────────────────────────────────
    del cfg.observations["actor"].terms["base_lin_vel"]

    cfg.observations["critic"].terms["base_lin_vel"] = ObservationTermCfg(
        func=mdp.base_lin_vel, scale=1.0,
    )
    del cfg.observations["critic"].terms["foot_height"]
    del cfg.observations["actor"].terms["height_scan"]
    del cfg.observations["critic"].terms["height_scan"]

    gravity_term_name = "projected_gravity"
    cfg.observations["actor"].terms[gravity_term_name] = deepcopy(
        cfg.observations["actor"].terms[gravity_term_name]
    )
    cfg.observations["actor"].terms["base_ang_vel"] = deepcopy(
        cfg.observations["actor"].terms["base_ang_vel"]
    )

    cfg.observations["actor"].terms["base_ang_vel"].delay_min_lag = 0
    cfg.observations["actor"].terms["base_ang_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["base_ang_vel"].delay_update_period = 64
    cfg.observations["actor"].terms[gravity_term_name].delay_min_lag = 0
    cfg.observations["actor"].terms[gravity_term_name].delay_max_lag = 1
    cfg.observations["actor"].terms[gravity_term_name].delay_update_period = 64

    cfg.observations["actor"].terms["base_ang_vel"].noise    = Unoise(n_min=-0.03, n_max=0.03)
    cfg.observations["actor"].terms[gravity_term_name].noise = Unoise(n_min=-0.01, n_max=0.01)
    cfg.observations["actor"].terms["joint_pos"].noise       = Unoise(n_min=-0.001, n_max=0.001)
    cfg.observations["actor"].terms["joint_vel"].noise       = Unoise(n_min=-0.25, n_max=0.25)

    if ENABLE_IMU_ORIENTATION_RANDOMIZATION:
        av = cfg.observations["actor"].terms["base_ang_vel"]
        av.func = microduck_mdp.base_ang_vel_imu_misaligned
        av.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}
        g = cfg.observations["actor"].terms[gravity_term_name]
        g.func = microduck_mdp.projected_gravity_imu_misaligned
        g.params = {"max_angle_deg": IMU_ORIENTATION_RANDOMIZATION_ANGLE}

    cfg.observations["actor"].terms["joint_vel"] = deepcopy(
        cfg.observations["actor"].terms["joint_vel"]
    )
    cfg.observations["actor"].terms["joint_vel"].delay_min_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_max_lag = 1
    cfg.observations["actor"].terms["joint_vel"].delay_update_period = 0

    passive_excluded = SceneEntityCfg("robot", joint_names=(r"^(?!passive_).*",))
    for grp in ("actor", "critic"):
        for term in ("joint_pos", "joint_vel"):
            cfg.observations[grp].terms[term] = deepcopy(cfg.observations[grp].terms[term])
            cfg.observations[grp].terms[term].params["asset_cfg"] = deepcopy(passive_excluded)

    if ENABLE_ENCODER_BIAS:
        cfg.events["encoder_bias"].params["bias_range"] = ENCODER_BIAS_RANGE
        cfg.observations["actor"].terms["joint_pos"].params["biased"] = True
        cfg.observations["critic"].terms["joint_pos"].params["biased"] = False
    else:
        cfg.events.pop("encoder_bias", None)

    # 13D command block (twist phase + zero-padded head/body)
    for group in ("actor", "critic"):
        cfg.observations[group].terms["head_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 4},
        )
        cfg.observations[group].terms["body_command"] = ObservationTermCfg(
            func=microduck_mdp.zero_command_padding, params={"dim": 6},
        )

    # Twist slot: JumpPhaseCommand [cos(2πφ), sin(2πφ), 0]
    twist_cmd = cfg.commands["twist"]
    cfg.commands["twist"] = microduck_mdp.JumpPhaseCommandCfg(
        **{
            **vars(twist_cmd),
            "class_type": microduck_mdp.JumpPhaseCommand,
            "period": EPISODE_LENGTH_S,
            "randomize_phase": False,
        }
    )

    # ── Terminations ──────────────────────────────────────────────────────────
    if "fell_over" in cfg.terminations:
        del cfg.terminations["fell_over"]
    cfg.terminations["nan_state"] = TerminationTermCfg(
        func=microduck_mdp.robot_state_is_nan,
        time_out=False,
    )

    # ── Events ────────────────────────────────────────────────────────────────
    cfg.events["expand_bam_friction_fields"] = EventTermCfg(
        func=microduck_mdp.expand_bam_friction_fields,
        mode="startup",
    )
    cfg.events["reset_action_history"] = EventTermCfg(
        func=microduck_mdp.reset_action_history,
        mode="reset",
    )
    cfg.events["reset_jump_state"] = EventTermCfg(
        func=microduck_mdp.reset_jump_state,
        mode="reset",
    )
    cfg.events["foot_friction"].params["asset_cfg"].geom_names = foot_frictions_geom_names
    cfg.events["foot_friction"].params["ranges"] = (0.7, 1.3)

    if "push_robot" in cfg.events:
        del cfg.events["push_robot"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.events["randomize_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "operation": "add",
                "ranges": (-COM_RANDOMIZATION_RANGE, COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.events["randomize_head_com"] = EventTermCfg(
            func=dr.body_ipos,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=HEAD_BODY_NAMES),
                "operation": "add",
                "ranges": (-HEAD_COM_RANDOMIZATION_RANGE, HEAD_COM_RANDOMIZATION_RANGE),
            },
        )

    if ENABLE_ARMATURE_RANDOMIZATION:
        cfg.events["randomize_armature"] = EventTermCfg(
            func=dr.joint_armature,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=(r".*",)),
                "operation": "scale",
                "ranges": ARMATURE_RANDOMIZATION_RANGE,
            },
        )

    if ENABLE_MASS_INERTIA_RANDOMIZATION:
        _mi_lo, _mi_hi = MASS_INERTIA_RANDOMIZATION_RANGE
        cfg.events["randomize_mass_inertia"] = EventTermCfg(
            func=dr.pseudo_inertia,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
                "alpha_range": (math.log(_mi_lo) / 2.0, math.log(_mi_hi) / 2.0),
            },
        )

    if ENABLE_JOINT_FRICTION_RANDOMIZATION:
        cfg.events["randomize_joint_friction"] = EventTermCfg(
            func=microduck_mdp.randomize_bam_friction,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "scale_range": JOINT_FRICTION_RANDOMIZATION_RANGE,
            },
        )

    # ── Terrain ───────────────────────────────────────────────────────────────
    cfg.scene.terrain.terrain_type = "plane"
    cfg.scene.terrain.terrain_generator = None

    # ── Curricula ────────────────────────────────────────────────────────────
    if "terrain_levels" in cfg.curriculum:
        del cfg.curriculum["terrain_levels"]
    if "command_vel" in cfg.curriculum:
        del cfg.curriculum["command_vel"]

    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.006},
                    {"step": 1000 * 24, "range": 0.010},
                    {"step": 1500 * 24, "range": 0.015},
                ],
            },
        )

    if ENABLE_HEAD_COM_RANDOMIZATION:
        cfg.curriculum["head_com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_head_com",
                "range_stages": [
                    {"step": 0,         "range": 0.003},
                    {"step": 500 * 24,  "range": 0.006},
                    {"step": 1000 * 24, "range": 0.010},
                ],
            },
        )

    # action_rate ramp — gentle at start to enable explosive discovery, tightened later
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0,         "weight": -0.05},
                {"step": 1000 * 24, "weight": -0.15},
                {"step": 2000 * 24, "weight": -0.30},
            ],
        },
    )

    # Torque rate smoothness introduced after skill discovery
    cfg.curriculum["torque_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "joint_torque_rate_l2",
            "weight_stages": [
                {"step": 0,         "weight": 0.0},
                {"step": 1500 * 24, "weight": -2e-4},
                {"step": 2500 * 24, "weight": -5e-4},
            ],
        },
    )

    cfg.curriculum["gentle_landing_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "gentle_landing",
            "weight_stages": [
                {"step": 0,         "weight": 0.002},
                {"step": 1500 * 24, "weight": 0.005},
            ],
        },
    )

    return cfg


# ── RL runner config ──────────────────────────────────────────────────────────

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
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    save_interval=250,
    num_steps_per_env=24,
    max_iterations=4000,
)
