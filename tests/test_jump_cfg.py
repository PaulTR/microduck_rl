from mjlab.tasks.registry import list_tasks
from mjlab_microduck.tasks.microduck_jump_env_cfg import (
    make_microduck_jump_env_cfg,
    MicroduckJumpRlCfg,
)
from mjlab_microduck.tasks import mdp as microduck_mdp


def test_jump_cfg_command_is_phase():
    cfg = make_microduck_jump_env_cfg()
    cmd = cfg.commands["twist"]
    assert isinstance(cmd, microduck_mdp.JumpPhaseCommandCfg)
    assert cmd.period == 3.0
    assert cmd.randomize_phase is False


def test_jump_cfg_contact_sensor_air_time():
    cfg = make_microduck_jump_env_cfg()
    sensors = {s.name: s for s in cfg.scene.sensors}
    assert "feet_ground_contact" in sensors
    assert sensors["feet_ground_contact"].track_air_time is True
    assert "non_foot_ground_contact" in sensors


def test_jump_cfg_rewards_present_and_signs():
    cfg = make_microduck_jump_env_cfg()
    r = cfg.rewards

    # Crouch rewards
    assert "jump_crouch_height" in r and r["jump_crouch_height"].weight > 0
    assert "jump_crouch_feet_grounded" in r and r["jump_crouch_feet_grounded"].weight > 0
    assert "jump_crouch_feet_lift" in r and r["jump_crouch_feet_lift"].weight < 0

    # Takeoff & flight rewards
    assert "jump_takeoff_velocity" in r and r["jump_takeoff_velocity"].weight > 0
    assert "jump_flight_air_time" in r and r["jump_flight_air_time"].weight > 0
    assert "jump_horizontal_vel" in r and r["jump_horizontal_vel"].weight < 0
    assert "jump_horizontal_drift" in r and r["jump_horizontal_drift"].weight < 0
    assert "jump_stay_in_place" in r and r["jump_stay_in_place"].weight > 0
    assert "jump_hip_pitch_extension" in r and r["jump_hip_pitch_extension"].weight < 0
    assert "jump_upright" in r and r["jump_upright"].weight > 0

    # Landing & return stand rewards
    assert "jump_two_foot_landing" in r and r["jump_two_foot_landing"].weight > 0
    assert "jump_non_foot_contact" in r and r["jump_non_foot_contact"].weight < 0
    assert "gentle_landing" in r and r["gentle_landing"].weight > 0  # self-negating (|a_z|)
    assert "jump_return_stand" in r and r["jump_return_stand"].weight > 0
    assert "jump_post_landing_hop" in r and r["jump_post_landing_hop"].weight < 0
    assert "jump_stillness" in r and r["jump_stillness"].weight < 0

    # Orientation & sagittal penalties
    assert "jump_sagittal" in r and r["jump_sagittal"].weight < 0
    assert "jump_verticality" in r and r["jump_verticality"].weight < 0
    assert "jump_neck_posture" in r and r["jump_neck_posture"].weight < 0

    # Regularizers
    assert "action_rate_l2" in r and r["action_rate_l2"].weight < 0
    assert "self_collisions" in r and r["self_collisions"].weight < 0

    # Walking-specific terms removed
    for gone in ("track_linear_velocity", "track_angular_velocity", "air_time", "foot_clearance", "pose", "upright"):
        assert gone not in r


def test_jump_cfg_terminations():
    cfg = make_microduck_jump_env_cfg()
    assert "fell_over" in cfg.terminations
    assert cfg.terminations["fell_over"].params["limit_angle"] <= 0.70
    assert "non_foot_contact" in cfg.terminations
    assert "nan_state" in cfg.terminations


def test_jump_cfg_events_wired():
    cfg = make_microduck_jump_env_cfg()
    assert "expand_bam_friction_fields" in cfg.events
    assert "reset_jump_state" in cfg.events
    assert cfg.events["reset_jump_state"].mode == "reset"


def test_jump_cfg_obs_layout():
    cfg = make_microduck_jump_env_cfg()
    actor_terms = cfg.observations["actor"].terms
    # Check key terms exist
    assert "base_ang_vel" in actor_terms
    assert "projected_gravity" in actor_terms
    assert "joint_pos" in actor_terms
    assert "joint_vel" in actor_terms
    assert "actions" in actor_terms
    assert "command" in actor_terms
    assert "head_command" in actor_terms
    assert "body_command" in actor_terms


def test_jump_task_registration():
    tasks = list_tasks()
    assert "Mjlab-Jump-Flat-MicroDuck" in tasks
    assert "Mjlab-Jump-Rough-MicroDuck" in tasks
    assert "Mjlab-Jump-Flat-Backlash-MicroDuck" in tasks


def test_jump_variants_build():
    play_cfg = make_microduck_jump_env_cfg(play=True)
    assert play_cfg is not None
    rough_cfg = make_microduck_jump_env_cfg(rough=True)
    assert rough_cfg is not None


def test_jump_phase_window():
    import torch
    # Window from 0.20 to 0.50
    phase = torch.tensor([0.0, 0.10, 0.20, 0.35, 0.50, 0.60, 1.0])
    w = microduck_mdp._jump_phase_window(phase, 0.20, 0.50, blend=0.05)
    assert w[0] == 0.0
    assert w[1] == 0.0
    assert w[3] == 1.0
    assert w[5] == 0.0
    assert w[6] == 0.0


def test_jump_flight_gate_logic():
    import torch

    class _MockEnv:
        def __init__(self, air_times):
            self.device = "cpu"
            self.num_envs = len(air_times)
            self._jump_max_air_time = air_times.clone()
            self._jump_has_flown = torch.zeros(self.num_envs, dtype=torch.bool)
            self._jump_has_landed = torch.zeros(self.num_envs, dtype=torch.bool)
            self._jump_last_update_step = 0
            self.common_step_counter = 0
            self.scene = {"sensors": {}}

    air_times = torch.tensor([0.0, 0.05, 0.08, 0.12, 0.16, 0.25])
    env = _MockEnv(air_times)
    gate = microduck_mdp.jump_flight_gate(env, min_air_time=0.08, target_air_time=0.16)
    assert gate[0] == 0.0
    assert gate[1] == 0.0
    assert gate[2] == 0.0
    assert 0.0 < gate[3] < 1.0
    assert gate[4] == 1.0
    assert gate[5] == 1.0


def test_jump_phase_command_stepping():
    import torch

    cfg = make_microduck_jump_env_cfg()
    cmd_cfg = cfg.commands["twist"]

    class _MockRobot:
        pass

    class _MockEnv:
        def __init__(self, num_envs=4):
            self.device = "cpu"
            self.num_envs = num_envs
            self.step_dt = 0.02
            self.scene = {"robot": _MockRobot()}

    env = _MockEnv()
    cmd = microduck_mdp.JumpPhaseCommand(cmd_cfg, env)

    # Initial phase is 0
    assert torch.all(cmd._jump_phase == 0.0)
    assert torch.allclose(cmd.command[:, 0], torch.tensor(1.0))
    assert torch.allclose(cmd.command[:, 1], torch.tensor(0.0))

    # Step by 0.75s (0.25 of 3.0s period)
    cmd.compute(0.75)
    assert torch.allclose(cmd._jump_phase, torch.tensor(0.25))
    assert torch.allclose(cmd.command[:, 0], torch.tensor(0.0), atol=1e-5)
    assert torch.allclose(cmd.command[:, 1], torch.tensor(1.0), atol=1e-5)


def test_jump_butt_contact_gating():
    import torch

    class _MockSensor:
        def __init__(self, found):
            class _Data:
                pass
            self.data = _Data()
            self.data.found = found

    class _MockRobotData:
        def __init__(self, num_envs):
            self.root_link_pos_w = torch.tensor([[0.0, 0.0, 0.065]], dtype=torch.float32).repeat(num_envs, 1)
            self.root_link_quat_w = torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32).repeat(num_envs, 1)
            self.joint_pos = torch.zeros(num_envs, 14)

    class _MockRobot:
        def __init__(self, num_envs):
            self.data = _MockRobotData(num_envs)

    class _MockCmdManager:
        def get_command(self, name):
            # Phase 0.65 (landing) -> [cos(2pi*0.65), sin(2pi*0.65), 0]
            phase = 0.65
            c = torch.tensor([[math.cos(2 * math.pi * phase), math.sin(2 * math.pi * phase), 0.0]])
            return c.repeat(2, 1)

    import math
    class _MockScene:
        def __init__(self, robot, sensors):
            self._robot = robot
            self.sensors = sensors

        def __getitem__(self, key):
            return self._robot

    class _MockEnv:
        def __init__(self):
            self.device = "cpu"
            self.num_envs = 2
            self.common_step_counter = 1
            self.command_manager = _MockCmdManager()
            self._jump_max_air_time = torch.tensor([0.20, 0.20])
            self._jump_has_flown = torch.tensor([True, True])
            self._jump_has_landed = torch.tensor([True, True])
            self._jump_has_butt_contact = torch.tensor([False, True])  # env 0: clean landing; env 1: butt contact
            self._jump_last_update_step = 1
            self.scene = _MockScene(
                robot=_MockRobot(2),
                sensors={
                    "feet_ground_contact": _MockSensor(torch.tensor([[1, 1], [1, 1]])),
                    "non_foot_ground_contact": _MockSensor(torch.tensor([[0], [1]])),
                },
            )

    env = _MockEnv()
    reward = microduck_mdp.jump_two_foot_landing(
        env,
        sensor_name="feet_ground_contact",
        non_foot_sensor_name="non_foot_ground_contact",
        landing_start=0.55,
        landing_end=0.75,
        crouch_z=0.065,
    )
    # Env 0 (clean landing on feet) should earn positive reward
    assert reward[0] > 0.0
    # Env 1 (butt contact) must be zeroed out
    assert reward[1] == 0.0
