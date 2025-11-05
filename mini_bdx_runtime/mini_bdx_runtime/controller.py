import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'
import pygame
from threading import Thread
from queue import Queue
import time
import numpy as np
from mini_bdx_runtime.buttons import Buttons

TORSO_BLEND = 0.35

def deadzone_axis(v: float, dz: float = 0.06, rescale: bool = True) -> float:
    if abs(v) <= dz:
        return 0.0
    if not rescale:
        return v
    return (v - dz) / (1.0 - dz) if v > 0 else (v + dz) / (1.0 - dz)

class Controller:
    def __init__(self, command_freq, only_head_control=False):
        self.command_freq = command_freq
        self.only_head_control = only_head_control  # kept for signature compatibility

        # [0] lin_vel_x, [1] lin_vel_y, [2] ang_vel, [3] height
        # [4] neck_pitch, [5] head_pitch, [6] head_yaw, [7] head_roll
        # [8] trunk_pitch, [9] trunk_yaw, [10] trunk_roll
        self.last_commands = [0.0] * 11

        self.last_left_trigger = 0.0
        self.last_right_trigger = 0.0

        pygame.init()
        self.p1 = pygame.joystick.Joystick(0)
        self.p1.init()
        self.num_axes = self.p1.get_numaxes()
        print(f"Loaded joystick with {self.num_axes} axes.")
        self.cmd_queue = Queue(maxsize=1)

        self.A_pressed = False
        self.B_pressed = False
        self.X_pressed = False
        self.Y_pressed = False
        self.LB_pressed = False
        self.RB_pressed = False

        self.buttons = Buttons()
        Thread(target=self.commands_worker, daemon=True).start()

    def commands_worker(self):
        while True:
            self.cmd_queue.put(self.get_commands())
            time.sleep(1 / self.command_freq)

    def get_commands(self):
        last_commands = self.last_commands
        left_trigger = self.last_left_trigger
        right_trigger = self.last_right_trigger

        # Axes (typical pygame): 0=LX, 1=LY, 2=RX, 3=RY, 4=RT, 5=LT (sometimes swapped)
        lx = -self.p1.get_axis(0)
        ly = -self.p1.get_axis(1)
        rx = -self.p1.get_axis(2)
        ry = -self.p1.get_axis(3)

        lx = deadzone_axis(lx, dz=0.06, rescale=True)
        ly = deadzone_axis(ly, dz=0.06, rescale=True)
        rx = deadzone_axis(rx, dz=0.06, rescale=True)
        ry = deadzone_axis(ry, dz=0.06, rescale=True)

        # Triggers as analogs in [0,1]; synthesize if only 4 axes
        if self.num_axes >= 6:
            lt_raw = self.p1.get_axis(4)
            rt_raw = self.p1.get_axis(5)
            right_trigger = (rt_raw + 1) / 2
            left_trigger  = (lt_raw + 1) / 2
            if left_trigger < 0.1:  left_trigger = 0.0
            if right_trigger < 0.1: right_trigger = 0.0
        else:
            left_trigger = 1.0 if self.p1.get_button(7) else 0.0
            right_trigger = 1.0 if self.p1.get_button(8) else 0.0

        # Left stick walking
        lin_vel_x = ly
        ang_vel   = lx

        # Shoulder triggers strafing
        lin_vel_y = (right_trigger - left_trigger)

        # Right stick drives head (gaze) + small torso blend (posture)
        head_yaw   = rx
        head_pitch = -ry
        head_roll  = rx * 0.5
        neck_pitch = -ry

        trunk_yaw   = TORSO_BLEND * rx
        trunk_pitch = TORSO_BLEND * (-ry)
        trunk_roll  = 0.0

        def clamp(v):
            return float(np.clip(v, -1.0, 1.0))

        last_commands[0]  = clamp(lin_vel_x)
        last_commands[1]  = clamp(lin_vel_y)
        last_commands[2]  = clamp(ang_vel)
        last_commands[3]  = 0.0  # walk height unused

        last_commands[4]  = clamp(neck_pitch)
        last_commands[5]  = clamp(head_pitch)
        last_commands[6]  = clamp(head_yaw)
        last_commands[7]  = clamp(head_roll)
        last_commands[8]  = clamp(trunk_pitch)
        last_commands[9]  = clamp(trunk_yaw)
        last_commands[10] = clamp(trunk_roll)

        # Buttons
        for event in pygame.event.get():
            if event.type == pygame.JOYBUTTONDOWN:
                if self.p1.get_button(1): self.A_pressed = True  # A
                if self.p1.get_button(0): self.B_pressed = True  # B
                if self.p1.get_button(2): self.X_pressed = True  # X
                if self.p1.get_button(3): self.Y_pressed = True  # Y (free now)
                if self.p1.get_button(6): self.LB_pressed = True
                if self.p1.get_button(7): self.RB_pressed = True
            if event.type == pygame.JOYBUTTONUP:
                self.A_pressed = self.B_pressed = self.X_pressed = self.Y_pressed = False
                self.LB_pressed = self.RB_pressed = False

        up_down = self.p1.get_hat(0)[1]
        pygame.event.pump()

        return (
            np.around(last_commands, 3),
            self.A_pressed,
            self.B_pressed,
            self.X_pressed,
            self.Y_pressed,
            self.LB_pressed,
            self.RB_pressed,
            left_trigger,
            right_trigger,
            up_down,
        )

    def get_last_command(self):
        A_pressed = B_pressed = X_pressed = Y_pressed = False
        LB_pressed = RB_pressed = False
        up_down = 0
        try:
            (
                self.last_commands,
                A_pressed,
                B_pressed,
                X_pressed,
                Y_pressed,
                LB_pressed,
                RB_pressed,
                self.last_left_trigger,
                self.last_right_trigger,
                up_down,
            ) = self.cmd_queue.get(False)
        except Exception:
            pass

        self.buttons.update(
            A_pressed, B_pressed, X_pressed, Y_pressed, LB_pressed, RB_pressed,
            up_down == 1, up_down == -1,
        )

        return (
            self.last_commands,
            self.buttons,
            self.last_left_trigger,
            self.last_right_trigger,
        )

if __name__ == "__main__":
    controller = Controller(20)
    while True:
        print(controller.get_last_command())
        time.sleep(0.05)
