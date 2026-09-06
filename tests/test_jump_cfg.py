from mjlab.tasks.registry import load_env_cfg, load_rl_cfg
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    make_microduck_jump_env_cfg,
    MicroduckJumpRlCfg,
)
from mjlab_microduck.tasks.mdp import JumpPhaseCommand


def test_jump_task_registration():
    """All jump task variants must be registered in mjlab."""
    task_ids = [
        "Mjlab-Jump-Flat-MicroDuck",
        "Mjlab-Jump-Rough-MicroDuck",
        "Mjlab-Jump-Flat-Backlash-MicroDuck",
        "Mjlab-Jump-Rough-Backlash-MicroDuck",
    ]
    for task_id in task_ids:
        env_cfg = load_env_cfg(task_id)
        assert env_cfg is not None
        rl_cfg = load_rl_cfg(task_id)
        assert rl_cfg is not None


def test_jump_cfg_rewards():
    """Jump reward structure: 4 positive task objectives + 4 anti-exploit penalties."""
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards

    # 1. Apex height: dense Gaussian reward pulling trunk height up to APEX_Z
    assert "jump_apex_height" in r
    assert r["jump_apex_height"].weight == 5.0
    assert r["jump_apex_height"].params["target_height"] == 0.170

    # 2. Airborne bonus: reward when both feet leave the ground
    assert "jump_airborne" in r
    assert r["jump_airborne"].weight == 4.0

    # 3. Landing & standing: dense composite reward for upright HOME pose at STAND_Z
    assert "jump_stand" in r
    assert r["jump_stand"].weight == 5.0
    assert r["jump_stand"].params["target_height"] == 0.115

    # 4. Grounded feet bonus after landing
    assert "jump_feet_grounded" in r
    assert r["jump_feet_grounded"].weight == 2.0

    # 5. Anti-spin penalty: heavily penalize yaw angular velocity (ω_z²)
    assert "jump_yaw_rate" in r
    assert r["jump_yaw_rate"].weight == -3.0

    # 6. In-place constraints: strictly penalize horizontal velocity and drift
    assert "jump_horizontal_vel" in r
    assert r["jump_horizontal_vel"].weight == -3.0
    assert "jump_horizontal_drift" in r
    assert r["jump_horizontal_drift"].weight == -4.0

    # 7. Verticality penalty: keep body vertical (gx² + gy²)
    assert "jump_verticality" in r
    assert r["jump_verticality"].weight == -3.0

    # 8. Regularizers: low attempt tax on action rate
    assert "action_rate_l2" in r
    assert r["action_rate_l2"].weight == -0.01

    # Invariant: No walking tracking rewards
    for unwanted in [
        "track_linear_velocity",
        "track_angular_velocity",
        "air_time",
        "foot_clearance",
        "foot_swing_height",
        "foot_slip",
    ]:
        assert unwanted not in r


def test_jump_cfg_terminations():
    """Strict terminations: fell_over (>40° tilt), non_foot_contact, nan_state."""
    cfg = make_microduck_jump_env_cfg()
    terms = cfg.terminations

    assert "fell_over" in terms
    assert terms["fell_over"].params["limit_angle"] == 0.70

    assert "non_foot_contact" in terms
    assert terms["non_foot_contact"].params["sensor_name"] == "non_foot_ground_contact"

    assert "nan_state" in terms


def test_jump_command_is_jump_phase():
    """Twist command slot must encode JumpPhaseCommand."""
    cfg = make_microduck_jump_env_cfg()
    cmd = cfg.commands["twist"]
    assert cmd.class_type is JumpPhaseCommand
    assert cmd.period == 2.0

    # 13D command layout: 4D head + 6D body padding
    for group in ("actor", "critic"):
        assert "head_command" in cfg.observations[group].terms
        assert "body_command" in cfg.observations[group].terms


def test_jump_symmetry_configured():
    """Jump runner must have bilateral symmetry mirror loss enabled."""
    rl_cfg = MicroduckJumpRlCfg
    assert rl_cfg.algorithm.symmetry_cfg is not None


def test_jump_variants_build():
    """Rough and play variants must build cleanly."""
    cfg_rough = make_microduck_jump_env_cfg(rough=True)
    assert "jump_apex_height" in cfg_rough.rewards

    cfg_play = make_microduck_jump_env_cfg(play=True)
    assert "jump_apex_height" in cfg_play.rewards
