"""Invariant and regression tests for the Microduck Jump task configuration."""

import math
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    make_microduck_jump_env_cfg,
    MicroduckJumpRlCfg,
    EPISODE_LENGTH_S,
    STAND_Z,
    JUMP_TARGET_APEX_Z,
)


def test_jump_cfg_instantiation():
    """Verify that the Jump environment configuration can be instantiated cleanly."""
    cfg = make_microduck_jump_env_cfg()
    assert cfg is not None
    assert cfg.episode_length_s == EPISODE_LENGTH_S


def test_jump_rewards_present_and_signs():
    """Verify that all Jump rewards are wired and follow strict sign conventions."""
    cfg = make_microduck_jump_env_cfg()

    # Positive goal / shaping rewards
    assert "jump_launch" in cfg.rewards
    assert cfg.rewards["jump_launch"].weight > 0.0
    assert cfg.rewards["jump_launch"].params["require_both_feet_ground"] is True

    assert "leg_similarity" in cfg.rewards
    assert cfg.rewards["leg_similarity"].weight > 0.0

    assert "jump_air_time" in cfg.rewards
    assert cfg.rewards["jump_air_time"].weight > 0.0

    assert "jump_height" in cfg.rewards
    assert cfg.rewards["jump_height"].weight > 0.0
    assert cfg.rewards["jump_height"].params["target_height"] == JUMP_TARGET_APEX_Z

    assert "jump_forward_dist" in cfg.rewards
    assert cfg.rewards["jump_forward_dist"].weight > 0.0
    assert cfg.rewards["jump_forward_dist"].params["target_distance"] == 0.15

    assert "jump_landing" in cfg.rewards
    assert cfg.rewards["jump_landing"].weight > 0.0
    assert cfg.rewards["jump_landing"].params["stand_height"] == STAND_Z
    assert cfg.rewards["jump_landing"].params["crouch_height"] == 0.100

    # Negative penalty / cost terms
    assert "jump_lateral_vel" in cfg.rewards
    assert cfg.rewards["jump_lateral_vel"].weight < 0.0

    assert "jump_yaw_rate" in cfg.rewards
    assert cfg.rewards["jump_yaw_rate"].weight < 0.0

    assert "head_pitch_limit" in cfg.rewards
    assert cfg.rewards["head_pitch_limit"].weight < 0.0
    assert cfg.rewards["head_pitch_limit"].params["max_angle_rad"] == math.radians(30.0)

    assert "jump_foot_impact" in cfg.rewards
    assert cfg.rewards["jump_foot_impact"].weight < 0.0

    assert "action_rate_l2" in cfg.rewards
    assert cfg.rewards["action_rate_l2"].weight < 0.0


def test_jump_symmetry_enabled():
    """Verify that bilateral symmetry is enabled in the PPO runner config."""
    assert MicroduckJumpRlCfg.algorithm.symmetry_cfg is not None


def test_walking_rewards_dropped():
    """Ensure walking locomotion and head-look tracking rewards are dropped for the jump task."""
    cfg = make_microduck_jump_env_cfg()
    for walking_term in [
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
        assert walking_term not in cfg.rewards, f"Unexpected walking term {walking_term} found in jump rewards"


def test_conflicting_curricula_dropped():
    """Ensure action rate ramp and wide head pose curricula are dropped."""
    cfg = make_microduck_jump_env_cfg()
    for c_name in [
        "action_rate_weight",
        "standing_envs",
        "head_pose_range",
        "head_pose_bias_weight",
    ]:
        assert c_name not in cfg.curriculum, f"Unexpected curriculum {c_name} found in jump curriculum"


def test_actor_observation_keeps_61d_contract():
    """Verify that actor observations preserve the unified 61D slot layout and parity with velocity."""
    from mjlab_microduck.tasks.microduck_velocity_env_cfg import (
        make_microduck_velocity_env_cfg,
    )
    jump = make_microduck_jump_env_cfg()
    vel = make_microduck_velocity_env_cfg()

    # The actor and critic observation term keys must match the velocity baseline exactly
    for grp in ("actor", "critic"):
        assert list(jump.observations[grp].terms.keys()) == list(
            vel.observations[grp].terms.keys()
        ), f"Observation layout mismatch on group {grp}"


def test_terminations():
    """Verify early termination on extreme tilt, double bounce, and head pitch excursion."""
    cfg = make_microduck_jump_env_cfg()
    assert "fell_over" in cfg.terminations
    assert cfg.terminations["fell_over"].params["limit_angle"] == math.radians(35.0)

    assert "double_bounce" in cfg.terminations
    assert "head_pitch_exceeded" in cfg.terminations
    assert cfg.terminations["head_pitch_exceeded"].params["max_angle_rad"] == math.radians(40.0)


def test_jump_task_registered():
    """Verify that Mjlab-Jump-Flat-MicroDuck is registered in the task registry."""
    tasks = list_tasks()
    assert "Mjlab-Jump-Flat-MicroDuck" in tasks
    assert "Mjlab-Jump-Flat-Backlash-MicroDuck" in tasks


def test_landing_is_gated_on_airborne_latch():
    """Verify that the landing composite reward uses compliant landing and resets."""
    from mjlab_microduck.tasks import mdp as microduck_mdp
    cfg = make_microduck_jump_env_cfg()
    assert cfg.rewards["jump_landing"].func == microduck_mdp.jump_compliant_landing
    assert "reset_jump_state" in cfg.events


def test_twist_command_heading_none():
    """Verify heading_command is False and ranges.heading is None to satisfy mjlab validator."""
    cfg = make_microduck_jump_env_cfg()
    assert cfg.commands["twist"].heading_command is False
    assert cfg.commands["twist"].ranges.heading is None


def test_lateral_velocity_penalty_ignores_forward_speed():
    """Verify that jump_lateral_velocity_penalty only taxes vy and ignores vx."""
    import torch
    from unittest.mock import MagicMock
    from mjlab_microduck.tasks.mdp import jump_lateral_velocity_penalty

    mock_env = MagicMock()
    mock_robot = MagicMock()
    lin_vel_b = torch.tensor([[0.5, 0.0, 0.0], [0.5, 0.3, 0.0]])
    mock_robot.data.root_link_lin_vel_b = lin_vel_b
    mock_env.scene = {"robot": mock_robot}

    penalty = jump_lateral_velocity_penalty(mock_env)
    assert penalty[0].item() == 0.0
    assert abs(penalty[1].item() - 0.09) < 1e-5


