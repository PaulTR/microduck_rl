"""Tests for Microduck vertical jump environment configuration and invariants."""

import torch
from mjlab.tasks.registry import list_tasks

from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    EPISODE_LENGTH_S,
    make_microduck_jump_env_cfg,
    MicroduckJumpRlCfg,
)
from mjlab_microduck.tasks.mdp import (
    JumpPhaseCommand,
    jump_reference_qpos,
)


def test_jump_task_registration():
    """Verify that all four jump task variants are registered."""
    tasks = list_tasks()
    expected = [
        "Mjlab-Jump-Flat-MicroDuck",
        "Mjlab-Jump-Rough-MicroDuck",
        "Mjlab-Jump-Flat-Backlash-MicroDuck",
        "Mjlab-Jump-Rough-Backlash-MicroDuck",
    ]
    for task_name in expected:
        assert task_name in tasks, f"Task {task_name} missing from registry"


def test_jump_env_cfg_episode_length_and_command():
    """Verify episode duration and JumpPhaseCommand configuration."""
    cfg = make_microduck_jump_env_cfg()
    assert cfg.episode_length_s == EPISODE_LENGTH_S
    assert cfg.episode_length_s == 1.2

    cmd = cfg.commands["twist"]
    assert cmd.class_type is JumpPhaseCommand
    assert cmd.period == 1.2
    assert cmd.randomize_phase is False


def test_jump_spawn_pose_range():
    """Verify spawn fix: z set to nominal standing height (0.12, 0.13) above terrain origin."""
    cfg = make_microduck_jump_env_cfg()
    pose_range = cfg.events["reset_base"].params["pose_range"]
    assert pose_range["z"] == (0.12, 0.13), f"z range unexpected: {pose_range['z']}"
    assert pose_range["yaw"] == (-0.05, 0.05), f"yaw range unexpected: {pose_range['yaw']}"
    assert pose_range["x"] == (-0.02, 0.02)
    assert pose_range["y"] == (-0.02, 0.02)
    assert "reset_jump_state" in cfg.events


def test_jump_rewards_and_penalties_signs():
    """Verify reward sign conventions: task rewards > 0, penalties < 0."""
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards

    # Task progression rewards (must be positive)
    assert "jump_crouch" in r and r["jump_crouch"].weight > 0
    assert "jump_trajectory_tracking" in r and r["jump_trajectory_tracking"].weight > 0
    assert "jump_push_velocity" in r and r["jump_push_velocity"].weight > 0
    assert "jump_airborne" in r and r["jump_airborne"].weight > 0
    assert "jump_stand" in r and r["jump_stand"].weight > 0
    assert r["jump_stand"].params["stand_start"] == 0.50

    # Penalties (must be negative)
    assert "action_rate_l2" in r and r["action_rate_l2"].weight < 0
    assert "head_posture" in r and r["head_posture"].weight < 0
    assert "jump_verticality" in r and r["jump_verticality"].weight < 0
    assert "jump_yaw_rate" in r and r["jump_yaw_rate"].weight < 0
    assert "jump_horizontal_vel" in r and r["jump_horizontal_vel"].weight < 0


def test_jump_terminations():
    """Verify terminations prevent fallen or exploding states."""
    cfg = make_microduck_jump_env_cfg()
    terms = cfg.terminations

    assert "time_out" in terms
    assert "fell_over" in terms
    assert terms["fell_over"].params["limit_angle"] == 1.0  # ~57 degrees
    assert "nan_state" in terms
    assert terms["nan_state"].params.get("sensor_names") == ("feet_ground_contact",)


def test_jump_obs_nan_sanitization():
    """Verify observation groups are configured with nan_policy='sanitize' and safe critic terms."""
    cfg = make_microduck_jump_env_cfg()
    assert cfg.observations["actor"].nan_policy == "sanitize"
    assert cfg.observations["critic"].nan_policy == "sanitize"

    critic_terms = cfg.observations["critic"].terms
    assert critic_terms["foot_contact_forces"].func.__name__ == "foot_contact_forces_safe"
    assert critic_terms["foot_air_time"].func.__name__ == "foot_air_time_safe"


def test_jump_reference_trajectory_continuity():
    """Verify kinematic reference trajectory: smooth crouch, rapid extension, extended landing/stand."""
    phases = torch.tensor([0.0, 0.18, 0.30, 0.40, 0.60, 1.0], dtype=torch.float32)
    default_pos = torch.zeros((len(phases), 14), dtype=torch.float32)
    # Set nominal head angles
    default_pos[:, 5] = 0.3491  # neck_pitch
    default_pos[:, 6] = 0.3491  # head_pitch

    q_ref = jump_reference_qpos(phases, default_pos)

    # 14 joint positions: [0..4 left leg, 5..8 neck/head, 9..13 right leg]
    assert q_ref.shape == (6, 14)

    # Knees are index 3 (left) and index 12 (right)
    # At phase 0.0: knees near default (within 1e-3)
    assert torch.isclose(q_ref[0, 3], default_pos[0, 3], atol=1e-3)
    assert torch.isclose(q_ref[0, 12], default_pos[0, 12], atol=1e-3)

    # At phase 0.18: knees fully crouched (-0.35 rad left, +0.35 rad right)
    assert torch.isclose(q_ref[1, 3], torch.tensor(-0.35), atol=1e-2)
    assert torch.isclose(q_ref[1, 12], torch.tensor(0.35), atol=1e-2)

    # At phase 0.40 to 1.0: knees fully extended in default pose
    for i in [3, 4, 5]:
        assert torch.isclose(q_ref[i, 3], default_pos[i, 3], atol=1e-3)
        assert torch.isclose(q_ref[i, 12], default_pos[i, 12], atol=1e-3)

    # Head and neck (indices 5..8) must stay fixed at default_pos across all phases
    for p in range(6):
        assert torch.isclose(q_ref[p, 5], default_pos[p, 5], atol=1e-4)
        assert torch.isclose(q_ref[p, 6], default_pos[p, 6], atol=1e-4)
        assert torch.isclose(q_ref[p, 7], default_pos[p, 7], atol=1e-4)
        assert torch.isclose(q_ref[p, 8], default_pos[p, 8], atol=1e-4)


def test_jump_variants_build():
    """Verify rough and play variants instantiate successfully."""
    cfg_rough = make_microduck_jump_env_cfg(rough=True)
    assert cfg_rough.scene.terrain is not None
    assert "jump_trajectory_tracking" in cfg_rough.rewards

    cfg_play = make_microduck_jump_env_cfg(play=True)
    assert "jump_trajectory_tracking" in cfg_play.rewards


def test_jump_runner_cfg_symmetry():
    """Verify runner config enables bilateral symmetry mirror loss."""
    assert MicroduckJumpRlCfg.experiment_name == "microduck_jump"
    assert MicroduckJumpRlCfg.algorithm.symmetry_cfg is not None
