#!/usr/bin/env python3
"""
save_arm_pose.py (manipulation_interfaces)

Records arm poses: the pose of the hand in the body frame (TF body -> hand), the same
representation used by the MoveArmToPose action and by the *_detection_pose parameters
of autonomous_manipulation_fsm.

Keys (in the terminal the node was started from):
  Enter   save the current pose (asks for a name, empty = pose_<n>)
  Esc     exit
  Ctrl+C  exit

Every saved pose is appended to output_file in the fsm_params.yaml format, ready to be
copied under fsm_node/ros__parameters:

  far_detection_pose:
    position:
      x: 0.6020
      ...

Parameters:
  base_frame   (default "body")
  hand_frame   (default "hand")
  output_file  (default "arm_poses.yaml" in the current directory)

Usage:
  ros2 run manipulation_interfaces save_arm_pose.py
  ros2 run manipulation_interfaces save_arm_pose.py --ros-args -p output_file:=/home/spot_ws/arm_poses.yaml
"""
import os
import select
import sys
import termios
import time
import tty

import rclpy
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
import tf2_ros

ENTER_KEYS = ("\n", "\r")
ESC_KEY = "\x1b"


class SaveArmPoseNode(Node):

    def __init__(self):
        super().__init__("save_arm_pose")
        self.declare_parameter("base_frame", "body")
        self.declare_parameter("hand_frame", "hand")
        self.declare_parameter("output_file", "arm_poses.yaml")
        self.base_frame = self.get_parameter("base_frame").value
        self.hand_frame = self.get_parameter("hand_frame").value
        self.output_file = os.path.abspath(os.path.expanduser(self.get_parameter("output_file").value))

        self.tf_buffer = tf2_ros.Buffer()
        # spin_thread: the TF listener receives /tf in its own thread, the main thread reads the keyboard
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self, spin_thread=True)
        self.saved = 0

    def current_pose(self):
        """Pose of hand_frame in base_frame, None if the TF is not available."""
        try:
            tf = self.tf_buffer.lookup_transform(self.base_frame, self.hand_frame, Time(),
                                                 timeout=Duration(seconds=1.0))
        except tf2_ros.TransformException as e:
            self.get_logger().error(f"TF {self.base_frame} -> {self.hand_frame} not available: {e}")
            return None
        return tf.transform.translation, tf.transform.rotation

    def save(self, name, pose):
        t, q = pose
        # always with a decimal point: the FSM declares these parameters as double
        lines = [
            f"# {self.hand_frame} in {self.base_frame}, saved {time.strftime('%Y-%m-%d %H:%M:%S')}",
            f"{name}:",
            "  position:",
            f"    x: {t.x:.4f}",
            f"    y: {t.y:.4f}",
            f"    z: {t.z:.4f}",
            "  orientation:",
            f"    x: {q.x:.4f}",
            f"    y: {q.y:.4f}",
            f"    z: {q.z:.4f}",
            f"    w: {q.w:.4f}",
            "",
        ]
        with open(self.output_file, "a") as f:
            f.write("\n".join(lines) + "\n")
        self.saved += 1
        print("\n".join(lines[1:]), flush=True)
        print(f"Saved '{name}' to {self.output_file}\n", flush=True)


class Keyboard:
    """Single-key reading from the terminal (cbreak mode), restored on exit."""

    def __init__(self):
        self.tty = open("/dev/tty")
        self.fd = self.tty.fileno()
        self.old = termios.tcgetattr(self.fd)

    def __enter__(self):
        tty.setcbreak(self.fd)
        return self

    def __exit__(self, *exc):
        self.restore()
        self.tty.close()

    def restore(self):
        termios.tcsetattr(self.fd, termios.TCSADRAIN, self.old)

    def read_key(self, timeout=0.1):
        """Returns the key pressed, ESC_KEY for a lone Esc, None on timeout or other escape sequences."""
        if not select.select([self.fd], [], [], timeout)[0]:
            return None
        key = os.read(self.fd, 1).decode(errors="ignore")
        if key == ESC_KEY:
            # arrow keys & co. send Esc + more bytes: only a lone Esc means exit
            if select.select([self.fd], [], [], 0.05)[0]:
                os.read(self.fd, 16)
                return None
        return key

    def read_line(self, prompt):
        """Normal line input (echo on) for the pose name."""
        self.restore()
        try:
            sys.stdout.write(prompt)
            sys.stdout.flush()
            return self.tty.readline().strip()
        finally:
            tty.setcbreak(self.fd)


def main(args=None):
    rclpy.init(args=args)
    node = SaveArmPoseNode()
    print(f"Saving {node.hand_frame} in {node.base_frame} to {node.output_file}", flush=True)
    print("Enter = save the current pose, Esc = exit", flush=True)

    try:
        with Keyboard() as keyboard:
            while rclpy.ok():
                key = keyboard.read_key()
                if key is None:
                    continue
                if key == ESC_KEY:
                    break
                if key in ENTER_KEYS:
                    pose = node.current_pose()
                    if pose is None:
                        continue
                    default = f"pose_{node.saved + 1}"
                    name = keyboard.read_line(f"Pose name [{default}]: ") or default
                    node.save(name, pose)
    except KeyboardInterrupt:
        pass
    finally:
        print(f"\n{node.saved} pose(s) saved to {node.output_file}", flush=True)
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
