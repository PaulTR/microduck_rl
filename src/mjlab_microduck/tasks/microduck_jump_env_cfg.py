"""Microduck Jump Task — Two-legged vertical jump in place.

Episodic policy that starts from standing, crouches and launches straight up
into the air (apex z ≈ 0.17 m), and lands cleanly on two feet, recovering
to a vertical stand at nominal height (STAND_Z = 0.115 m) without horizontal
drift or yaw spinning.

Episode duration: 2.0 s (100 steps @ 50 Hz).
Normalized phase: phi ∈ [0, 1) encoded in the twist slot as [cos(2π·phase), sin(2π·phase), 0].
  • phi ∈ [0.00, 0.45]: Jump half — dense upward height reward toward apex + airborne bonus.
  • phi ∈ [0.45, 1.00]: Landing & stand half — land on two feet directly into upright HOME posture.

Anti-Exploit Invariants:
  • Yaw rate penalty: strictly crushes spinning in place.
  • Bilateral symmetry: Left-right mirror loss enforces identical leg behavior.
  • Strict terminations: Falling (>40° tilt) or any non-foot contact terminates immediately.
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
EPISODE_LENGTH_S = 2.0
STAND_Z          = 0.115
APEX_Z           = 0.170

from mjlab.envs import ManagerBasedRlEnvCfg
import mjlab.envs.mdp as base_mdp
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
    """Create Microduck vertical jump environment configuration."""

    # ── Sensors ───────────────────────────────────────────────────────────────
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

    non_foot_ground_cfg = ContactSensorCfg(
        name="non_foot_ground_contact",
        primary=ContactMatch(
            mode="body",
            pattern=r"^(trunk_base|hip_l.*|leg.*|jaw_soft)$",
            entity="robot",
        ),
        secondary=ContactMatch(mode="body", pattern="terrain"),
        fields=("found",),
        reduce="none",
        num_slots=1,
    )

    foot_frictions_geom_names = ("left_foot_collision", "right_foot_collision")

    # ── Base config ───────────────────────────────────────────────────────────
    cfg = make_velocity_env_cfg()

    cfg.scene.entities = {"robot": MICRODUCK_STANDUP_ROBOT_CFG}
    cfg.scene.sensors  = (feet_ground_cfg, self_collision_cfg, non_foot_ground_cfg)
    cfg.viewer.body_name = "trunk_base"
    cfg.episode_length_s = EPISODE_LENGTH_S

    # ── Actions ───────────────────────────────────────────────────────────────
    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg)
    joint_pos_action.scale = 1.0

    # ── Rewards: drop walking-specific velocity tracking terms ────────────────
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

    # ── Rewards: Clean Two-Phase Jump Objectives ──────────────────────────────

    # 1. Apex height: dense Gaussian reward pulling trunk height up to APEX_Z (phi ∈ [0.0, 0.45])
    cfg.rewards["jump_apex_height"] = RewardTermCfg(
        func=microduck_mdp.jump_apex_height,
        weight=5.0,
        params={
            "target_height": APEX_Z,
            "std": 0.035,
            "jump_start": 0.00,
            "jump_end": 0.45,
            "command_name": "twist",
        },
    )

    # 2. Airborne bonus: reward when both feet leave the ground (phi ∈ [0.15, 0.45])
    cfg.rewards["jump_airborne"] = RewardTermCfg(
        func=microduck_mdp.jump_airborne,
        weight=4.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "flight_start": 0.15,
            "flight_end": 0.45,
            "command_name": "twist",
        },
    )

    # 3. Landing & standing: dense composite reward for upright HOME pose at STAND_Z (phi ∈ [0.45, 1.00])
    cfg.rewards["jump_stand"] = RewardTermCfg(
        func=microduck_mdp.jump_stand_composite,
        weight=5.0,
        params={
            "target_height": STAND_Z,
            "height_std": 0.025,
            "upright_std": 0.30,
            "pose_std": 0.35,
            "stand_start": 0.45,
            "stand_end": 1.00,
            "command_name": "twist",
        },
    )

    # 4. Grounded feet bonus after landing (phi ∈ [0.50, 1.00])
    cfg.rewards["jump_feet_grounded"] = RewardTermCfg(
        func=microduck_mdp.jump_feet_grounded,
        weight=2.0,
        params={
            "sensor_name": feet_ground_cfg.name,
            "stand_start": 0.50,
            "stand_end": 1.00,
            "command_name": "twist",
        },
    )

    # 5. Anti-spin penalty: heavily penalize yaw angular velocity (ω_z²)
    cfg.rewards["jump_yaw_rate"] = RewardTermCfg(
        func=microduck_mdp.jump_yaw_rate_penalty,
        weight=-3.0,
    )

    # 6. In-place constraints: strictly penalize horizontal velocity and drift
    cfg.rewards["jump_horizontal_vel"] = RewardTermCfg(
        func=microduck_mdp.jump_horizontal_velocity_penalty,
        weight=-3.0,
    )
    cfg.rewards["jump_horizontal_drift"] = RewardTermCfg(
        func=microduck_mdp.jump_horizontal_drift_penalty,
        weight=-4.0,
    )

    # 7. Verticality penalty: keep body vertical (gx² + gy²)
    cfg.rewards["jump_verticality"] = RewardTermCfg(
        func=microduck_mdp.jump_verticality_penalty,
        weight=-3.0,
    )

    # ── Sim2real regularisers ─────────────────────────────────────────────────
    # Start action_rate_l2 low so dynamic push-off is not penalized during exploration
    cfg.rewards["action_rate_l2"] = RewardTermCfg(
        func=mdp.action_rate_l2,
        weight=-0.01,
    )
    cfg.rewards["self_collisions"] = RewardTermCfg(
        func=mdp.self_collision_cost,
        weight=-0.1,
        params={"sensor_name": self_collision_cfg.name},
    )

    # Clean up unused base terms
    cfg.rewards.pop("soft_landing", None)
    cfg.rewards.pop("body_ang_vel", None)
    cfg.rewards.pop("angular_momentum", None)

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
    cfg.terminations["fell_over"] = TerminationTermCfg(
        func=base_mdp.bad_orientation,
        params={
            "limit_angle": 0.70,  # ~40 deg tilt limit
            "asset_cfg": SceneEntityCfg("robot", body_names=("trunk_base",)),
        },
    )
    cfg.terminations["non_foot_contact"] = TerminationTermCfg(
        func=microduck_mdp.non_foot_ground_contact_termination,
        params={"sensor_name": non_foot_ground_cfg.name},
    )
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

    # ── Curricula ─────────────────────────────────────────────────────────────
    if ENABLE_COM_RANDOMIZATION:
        cfg.curriculum["com_range"] = CurriculumTermCfg(
            func=microduck_mdp.com_range_curriculum,
            params={
                "event_name": "randomize_com",
                "range_stages": [
                    {"step": 0, "range": 0.003},
                    {"step": 12000, "range": 0.006},
                    {"step": 24000, "range": 0.010},
                    {"step": 36000, "range": 0.015},
                ],
            },
        )

    # Gentle action rate ramp after initial dynamic skill discovery
    cfg.curriculum["action_rate_weight"] = CurriculumTermCfg(
        func=microduck_mdp.reward_weight,
        params={
            "reward_name": "action_rate_l2",
            "weight_stages": [
                {"step": 0, "weight": -0.01},
                {"step": 20000, "weight": -0.03},
                {"step": 40000, "weight": -0.05},
            ],
        },
    )

    return cfg


MicroduckJumpRlCfg = RslRlOnPolicyRunnerCfg(
    num_steps_per_env=24,
    max_iterations=3000,
    save_interval=250,
    experiment_name="microduck_jump",
    run_name="microduck_jump",
    wandb_project="mjlab_microduck",
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
)
