import time
import pickle

import numpy as np
from mini_bdx_runtime.rustypot_position_hwi import HWI
from mini_bdx_runtime.onnx_infer import OnnxInfer

from mini_bdx_runtime.raw_imu import Imu
from mini_bdx_runtime.poly_reference_motion import PolyReferenceMotion
from mini_bdx_runtime.controller import Controller
from mini_bdx_runtime.feet_contacts import FeetContacts
from mini_bdx_runtime.eyes import Eyes
from mini_bdx_runtime.sounds import Sounds
from mini_bdx_runtime.antennas import Antennas
from mini_bdx_runtime.projector import Projector
from mini_bdx_runtime.rl_utils import make_action_dict, LowPassActionFilter
from mini_bdx_runtime.duck_config import DuckConfig

import os
import sys

HOME_DIR = os.path.expanduser("~")
USE_MOTOR_SPEED_LIMITS = True

class RLWalk:
    def __init__(
        self,
        onnx_model_path: str,
        duck_config_path: str = f"{HOME_DIR}/duck_config.json",
        serial_port: str = "/dev/ttyACM0",
        control_freq: float = 50,
        pid=[30, 0, 0],
        action_scale=None,
        save_obs=False,
        replay_obs=None
    ):

        self.duck_config = DuckConfig(config_json_path=duck_config_path)

        self.onnx_model_path = onnx_model_path
        self.policy = OnnxInfer(self.onnx_model_path, awd=True)
        self.ctrl_dt = self.policy.ctrl_dt

        self.quat = self.policy.quat
        self.contacts = self.policy.contacts

        self.num_dofs = 14
        self.max_motor_velocity = 5.24  # rad/s

        # Control
        self.control_freq = control_freq
        self.pid = pid

        self.nojoystick = False

        self.save_obs = save_obs
        if self.save_obs:
            self.saved_obs = []

        self.replay_obs = replay_obs
        if self.replay_obs is not None:
            self.replay_obs = pickle.load(open(self.replay_obs, "rb"))

        self.hwi = HWI(self.duck_config, serial_port)

        self.start()

        self.imu = Imu(
            sampling_freq=int(self.control_freq),
            upside_down=self.duck_config.imu_upside_down,
        )

        self.feet_contacts = FeetContacts()

        # Scales
        self.action_scale = self.policy.action_scale if action_scale is None else action_scale
        self.dof_vel_scale = self.policy.dof_vel_scale

        self.last_action = np.zeros(self.num_dofs)
        self.last_last_action = np.zeros(self.num_dofs)
        self.last_last_last_action = np.zeros(self.num_dofs)

        # Update our init_pos from the policy
        for j, off in zip(self.policy.joints, self.policy.action_offset):
            if j in self.hwi.init_pos:
                self.hwi.init_pos[j] = float(off)

        self.init_pos = list(self.hwi.init_pos.values())

        self.motor_targets = np.array(self.init_pos.copy())
        self.prev_motor_targets = np.array(self.init_pos.copy())

        self.last_commands = [
            0.0,    # lin_vel_x
            0.0,    # lin_vel_y
            0.0,    # ang_vel
            0.0,    # height

            0.0,    # neck pitch
            0.0,    # head pitch
            0.0,    # head yaw
            0.0,    # head roll

            0.0,    # trunk pitch
            0.0,    # trunk yaw
            0.0     # trunk roll
        ]

        self.paused = self.duck_config.start_paused

        self.controller = None
        self.command_freq = 20  # hz
        if not self.nojoystick:
            self.controller = Controller(self.command_freq)

        self.imitation_i = 0
        self.imitation_phase = np.array([0, 0])
        self.phase_frequency_factor = 1.0
        self.phase_frequency_factor_offset = (
            0
        )

        # Optional expression features
        if self.duck_config.eyes:
            self.eyes = Eyes()
        if self.duck_config.projector:
            self.projector = Projector()
        if self.duck_config.speaker:
            self.sounds = Sounds(
                volume=1.0, sound_directory="../mini_bdx_runtime/assets/"
            )
        if self.duck_config.antennas:
            self.antennas = Antennas()

        self.phase_gate_alpha = 0.0
        self.alpha_attack = 0.9
        self.alpha_decay = 0.1
        self.alpha_cmd_scale = 0.05
        self.alpha_rate_on   = np.deg2rad(5.0)
        self.alpha_rate_full = np.deg2rad(30.0)

    def _update_phase_gate_alpha(self, imu_data):
        # Command magnitude gate
        if len(self.last_commands) >= 4:
            cmd_mag = (abs(self.last_commands[0]) + abs(self.last_commands[1]) +
                       abs(self.last_commands[2]) + abs(self.last_commands[3]))
        else:
            cmd_mag = float(np.sum(np.abs(self.last_commands)))
        alpha_cmd = float(np.clip(cmd_mag / max(1e-6, self.alpha_cmd_scale), 0.0, 1.0))

        # Gyro rate gate
        max_rate = float(np.max(np.abs(imu_data["gyro"])))
        if max_rate <= self.alpha_rate_on:
            alpha_rate = 0.0
        elif max_rate >= self.alpha_rate_full:
            alpha_rate = 1.0
        else:
            alpha_rate = (max_rate - self.alpha_rate_on) / (self.alpha_rate_full - self.alpha_rate_on)

        alpha_raw = max(alpha_cmd, alpha_rate)

        k = self.alpha_attack if alpha_raw > self.phase_gate_alpha else self.alpha_decay
        self.phase_gate_alpha += k * (alpha_raw - self.phase_gate_alpha)
        self.phase_gate_alpha = float(np.clip(self.phase_gate_alpha, 0.0, 1.0))
        return self.phase_gate_alpha

    def get_obs(self):

        imu_data = self.imu.get_data()

        dof_pos = self.hwi.get_present_positions(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad

        dof_vel = self.hwi.get_present_velocities(
            ignore=[
                "left_antenna",
                "right_antenna",
            ]
        )  # rad/s

        if dof_pos is None or dof_vel is None:
            return None

        if len(dof_pos) != self.num_dofs:
            print(f"ERROR len(dof_pos) != {self.num_dofs}")
            return None

        if len(dof_vel) != self.num_dofs:
            print(f"ERROR len(dof_vel) != {self.num_dofs}")
            return None

        cmds = self.last_commands

        alpha = self._update_phase_gate_alpha(imu_data)
        gated_phase = self.imitation_phase * alpha

        optional_contacts = self.feet_contacts.get() if self.contacts else np.array([])
        optional_quat = imu_data["quat"] if self.quat else np.array([])
        obs_parts = [
            imu_data["gyro"],
            imu_data["accelero"],
            optional_quat,
            cmds,
            dof_pos - self.init_pos,
            dof_vel * self.dof_vel_scale,
            self.last_action,
            self.last_last_action,
            self.last_last_last_action,
            self.motor_targets,
            optional_contacts,
            gated_phase,
        ]
        return np.concatenate(obs_parts)

    def start(self):
        kps = [self.pid[0]] * 14
        kds = [self.pid[2]] * 14

        # lower head kps
        kps[5:9] = [8, 8, 8, 8]

        self.hwi.set_kps(kps)
        self.hwi.set_kds(kds)
        self.hwi.turn_on()

        time.sleep(2)

    def run(self):
        i = 0
        try:
            print("Starting")
            start_t = time.time()
            while True:
                t = time.time()

                if self.controller:
                    left_trigger = 0
                    right_trigger = 0
                    self.last_commands, self.buttons, left_trigger, right_trigger = (
                        self.controller.get_last_command()
                    )
                    if self.buttons.X.triggered:
                        if self.duck_config.projector:
                            self.projector.switch()

                    if self.buttons.B.triggered:
                        if self.duck_config.speaker:
                            self.sounds.play_random_sound()

                    if self.duck_config.antennas:
                        self.antennas.set_position_left(right_trigger)
                        self.antennas.set_position_right(left_trigger)

                    if self.buttons.A.triggered:
                        self.paused = not self.paused
                        if self.paused:
                            print("PAUSE")
                        else:
                            print("UNPAUSE")

                if self.paused:
                    time.sleep(0.1)
                    continue

                obs = self.get_obs()
                if obs is None:
                    continue

                self.imitation_i += 1.0 * self.phase_frequency_factor
                self.imitation_i = (
                    self.imitation_i % self.policy.nb_steps_in_period
                )
                self.imitation_phase = np.array(
                    [
                        np.cos(
                            self.imitation_i / self.policy.nb_steps_in_period * 2 * np.pi
                        ),
                        np.sin(
                            self.imitation_i / self.policy.nb_steps_in_period * 2 * np.pi
                        ),
                    ]
                )

                if self.save_obs:
                    self.saved_obs.append(obs)

                if self.replay_obs is not None:
                    if i < len(self.replay_obs):
                        obs = self.replay_obs[i]
                    else:
                        print("BREAKING ")
                        break

                action = self.policy.infer(obs)

                self.last_last_last_action = self.last_last_action.copy()
                self.last_last_action = self.last_action.copy()
                self.last_action = action.copy()

                self.motor_targets = self.init_pos + action * self.action_scale
                if USE_MOTOR_SPEED_LIMITS:
                    max_delta = self.max_motor_velocity * self.ctrl_dt
                    self.motor_targets = np.clip(
                        self.motor_targets,
                        self.prev_motor_targets - max_delta,
                        self.prev_motor_targets + max_delta,
                    )
                self.prev_motor_targets = self.motor_targets.copy()

                action_dict = make_action_dict(
                    self.motor_targets, list(self.hwi.joints.keys())
                )

                self.hwi.set_position_all(action_dict)

                i += 1

                took = time.time() - t
                # print("Full loop took", took, "fps : ", np.around(1 / took, 2))
                if (1 / self.control_freq - took) < 0:
                    print(
                        "Policy control budget exceeded by",
                        np.around(took - 1 / self.control_freq, 3),
                    )
                time.sleep(max(0, 1 / self.control_freq - took))

        except KeyboardInterrupt:
            if self.duck_config.antennas:
                self.antennas.stop()
            if self.duck_config.eyes:
                self.eyes.stop()
            if self.duck_config.projector:
                self.projector.stop()
            self.feet_contacts.stop()

        if self.save_obs:
            pickle.dump(self.saved_obs, open("robot_saved_obs.pkl", "wb"))
        print("TURNING OFF")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--onnx_model_path", type=str, required=True)
    parser.add_argument(
        "--duck_config_path",
        type=str,
        required=False,
        default=f"{HOME_DIR}/duck_config.json",
    )
    parser.add_argument("-a", "--action_scale", type=float, default=None)
    parser.add_argument("-p", type=int, default=30)
    parser.add_argument("-i", type=int, default=0)
    parser.add_argument("-d", type=int, default=0)
    parser.add_argument("-c", "--control_freq", type=int, default=50)
    parser.add_argument("-s", "--save_obs", action="store_true", default=False, help="Save observations")
    parser.add_argument(
        "--replay_obs",
        type=str,
        required=False,
        default=None,
        help="replay the observations from a previous run (can be from the robot or from mujoco)",
    )
    args = parser.parse_args()
    pid = [args.p, args.i, args.d]

    rl_walk = RLWalk(
        args.onnx_model_path,
        duck_config_path=args.duck_config_path,
        action_scale=args.action_scale,
        pid=pid,
        control_freq=args.control_freq,
        save_obs=args.save_obs,
        replay_obs=args.replay_obs
    )
    print("Done instantiating RLWalk")
    rl_walk.run()
