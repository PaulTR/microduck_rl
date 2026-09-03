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

    assert "jump_air_time" in cfg.rewards
    assert cfg.rewards["jump_air_time"].weight > 0.0

    assert "jump_height" in cfg.rewards
    assert cfg.rewards["jump_height"].weight > 0.0
    assert cfg.rewards["jump_height"].params["target_height"] == JUMP_TARGET_APEX_Z

    assert "jump_landing" in cfg.rewards
    assert cfg.rewards["jump_landing"].weight > 0.0
    assert cfg.rewards["jump_landing"].params["target_height"] == STAND_Z
    assert cfg.rewards["jump_landing"].params["min_landing_step"] == 16
    assert cfg.rewards["jump_landing"].params["target_overrides"] == {5: 0.0, 6: 0.0, 7: 0.0, 8: 0.0}

    assert "leg_similarity" in cfg.rewards
    assert cfg.rewards["leg_similarity"].weight > 0.0

    # Negative penalty / cost terms
    assert "jump_drift_penalty" in cfg.rewards
    assert cfg.rewards["jump_drift_penalty"].weight < 0.0

    assert "jump_yaw_rate" in cfg.rewards
    assert cfg.rewards["jump_yaw_rate"].weight < 0.0

    assert "jump_foot_impact" in cfg.rewards
    assert cfg.rewards["jump_foot_impact"].weight < 0.0

    assert "action_rate_l2" in cfg.rewards
    assert cfg.rewards["action_rate_l2"].weight < 0.0

    assert "head_pitch_limit" in cfg.rewards
    assert cfg.rewards["head_pitch_limit"].weight < 0.0
    assert cfg.rewards["head_pitch_limit"].params["max_angle_rad"] == math.radians(30.0)

    assert "jump_rebound_penalty" in cfg.rewards
    assert cfg.rewards["jump_rebound_penalty"].weight < 0.0


def test_walking_rewards_dropped():
    """Ensure walking locomotion tracking rewards are dropped for the jump task."""
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
    assert cfg.terminations["fell_over"].params["limit_angle"] == math.radians(25.0)

    assert "double_bounce" in cfg.terminations
    assert "head_pitch_exceeded" in cfg.terminations
    assert cfg.terminations["head_pitch_exceeded"].params["max_angle_rad"] == math.radians(40.0)


def test_jump_task_registered():
    """Verify that Mjlab-Jump-Flat-MicroDuck is registered in the task registry."""
    tasks = list_tasks()
    assert "Mjlab-Jump-Flat-MicroDuck" in tasks
    assert "Mjlab-Jump-Flat-Backlash-MicroDuck" in tasks


def test_landing_is_gated_on_airborne_latch():
    """Verify that the landing composite reward requires the airborne latch."""
    cfg = make_microduck_jump_env_cfg()
    assert cfg.rewards["jump_landing"].params["require_airborne_latch"] is True
    assert "reset_jump_state" in cfg.events


def test_curricula_clean():
    """Verify that curricula do not reference deleted reward terms."""
    cfg = make_microduck_jump_env_cfg()
    assert "head_pose_bias_weight" not in cfg.curriculum
    assert "standing_envs" not in cfg.curriculum
    assert "head_pose_range" not in cfg.curriculum
    assert "body_pose_range" not in cfg.curriculum


