#!/usr/bin/env python3
"""
nav2_bridge.py (spot_motion)

Collega la percezione (TargetPose3D, pubblicata da Pose3DEstimationNode) a
Nav2:

  1. Trasforma il target nel frame 'odom' (via TF — non piu' SDK bosdyn,
     spot_driver pubblica gia' tutto l'albero TF che serve).
  2. Lo smussa con lo STESSO filtro di Kalman gia' scritto per l'approccio
     SDK diretto (_ConstantVelocityKalman1D) — matematica pura, riusata
     senza modifiche, lavora gia' in un frame fisso (odom), qui non serve
     nemmeno il giro di andata/ritorno per il frame body che faceva
     motion_command_node.py.
  3. Al PRIMO target confermato: innesca una NavigateToPose. Il behavior
     tree NON va specificato qui — usa gia' il default (follow_point.xml,
     impostato in navigation_launch.py).
  4. Sui successivi: pubblica su /goal_update, che il nodo GoalUpdater
     dentro l'albero legge per aggiornare il goal SENZA dover rimandare
     una nuova azione ogni volta (evita di ricominciare la navigazione da
     capo ad ogni target).

PERDITA DEL TARGET: a differenza dei comandi SDK diretti (che avevano un
end_time_secs e scadevano da soli), una NavigateToPose non si ferma da
sola — bisogna CANCELLARLA esplicitamente. Se non arriva un dato vero da
piu' di MAX_TARGET_LOSS_SEC, la navigazione in corso viene cancellata.

NESSUN cambiamento a tracking_fsm.py — stessa decisione presa per
l'approccio precedente: ci si basa solo sul tempo reale trascorso
dall'ultimo dato vero, non sullo stato SEARCH/TRACKING/RECOVERY.
"""
import time

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient

import tf2_ros
from tf2_geometry_msgs import do_transform_pose_stamped

from geometry_msgs.msg import PoseStamped
from nav2_msgs.action import NavigateToPose

from demo_interfaces.msg import TargetPose3D

# ============================================================
GLOBAL_FRAME = 'odom'       # deve combaciare con global_frame in nav2_params_spot_real.yaml
KF_PROCESS_VAR = 0.05       # stessi valori di motion_command_node.py — stesso significato
KF_MEASUREMENT_VAR = 0.05
KF_RESET_GAP_SEC = 1.0
MAX_TARGET_LOSS_SEC = 5.0   # oltre questo senza un dato vero, la navigazione in corso
                              # viene cancellata esplicitamente — regolabile, non ancora
                              # tarato su test reali
# ============================================================


class _ConstantVelocityKalman1D:
    """Identica a quella in motion_command_node.py — matematica pura,
    nessuna dipendenza da bosdyn/Nav2, riusata cosi' com'e'."""

    def __init__(self, process_var, measurement_var):
        self.pos = 0.0
        self.vel = 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]
        self.q = process_var
        self.r = measurement_var

    def reset(self, pos):
        self.pos = pos
        self.vel = 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]

    def predict(self, dt):
        self.pos = self.pos + self.vel * dt
        p00, p01, p10, p11 = self.P[0][0], self.P[0][1], self.P[1][0], self.P[1][1]
        self.P = [
            [p00 + dt * (p10 + p01) + dt * dt * p11 + self.q * dt, p01 + dt * p11],
            [p10 + dt * p11, p11 + self.q * dt],
        ]

    def update(self, measurement):
        innovation = measurement - self.pos
        s = self.P[0][0] + self.r
        k_pos = self.P[0][0] / s
        k_vel = self.P[1][0] / s
        self.pos = self.pos + k_pos * innovation
        self.vel = self.vel + k_vel * innovation
        p00, p01 = self.P[0][0], self.P[0][1]
        self.P = [
            [self.P[0][0] - k_pos * p00, self.P[0][1] - k_pos * p01],
            [self.P[1][0] - k_vel * p00, self.P[1][1] - k_vel * p01],
        ]


class Nav2Bridge(Node):
    def __init__(self):
        super().__init__('nav2_bridge')

        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        self._kf_x = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_y = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_last_update_time = None

        self._goal_active = False
        self._goal_handle = None

        self.goal_update_pub = self.create_publisher(PoseStamped, 'goal_update', 5)
        self._nav_client = ActionClient(self, NavigateToPose, 'navigate_to_pose')

        self.create_subscription(TargetPose3D, 'target_3d', self._on_target_pose, 5)
        self.create_timer(1.0, self._check_target_loss)

        self.get_logger().info(
            f"nav2_bridge pronto. global_frame={GLOBAL_FRAME}, "
            f"max_target_loss={MAX_TARGET_LOSS_SEC:.1f}s.")

    def _on_target_pose(self, msg: TargetPose3D):
        try:
            transform = self.tf_buffer.lookup_transform(
                GLOBAL_FRAME, msg.header.frame_id, rclpy.time.Time())
        except tf2_ros.TransformException as ex:
            self.get_logger().warn(
                f"TF non disponibile ({msg.header.frame_id}->{GLOBAL_FRAME}): {ex}",
                throttle_duration_sec=2.0)
            return

        raw_pose = PoseStamped()
        raw_pose.header = msg.header
        raw_pose.pose.position = msg.position
        raw_pose.pose.orientation.w = 1.0
        world_pose = do_transform_pose_stamped(raw_pose, transform)

        # --- Filtro di Kalman, gia' nel frame odom (fisso) — nessun giro
        # di andata/ritorno per il frame body, a differenza di
        # motion_command_node.py: qui il goal serve gia' nel frame mondo.
        now = time.monotonic()
        if self._kf_last_update_time is None or (now - self._kf_last_update_time) > KF_RESET_GAP_SEC:
            self._kf_x.reset(world_pose.pose.position.x)
            self._kf_y.reset(world_pose.pose.position.y)
        else:
            dt = now - self._kf_last_update_time
            self._kf_x.predict(dt)
            self._kf_y.predict(dt)
            self._kf_x.update(world_pose.pose.position.x)
            self._kf_y.update(world_pose.pose.position.y)
        self._kf_last_update_time = now
        # --- fine filtro ---

        goal = PoseStamped()
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.header.frame_id = GLOBAL_FRAME
        goal.pose.position.x = self._kf_x.pos
        goal.pose.position.y = self._kf_y.pos
        goal.pose.orientation.w = 1.0  # non critico: TruncatePath+RotateToGoal (gia' attivi)
                                          # gestiscono l'avvicinamento finale, non l'orientamento
                                          # del goal lontano

        if not self._goal_active:
            self._send_initial_goal(goal)
        else:
            self.goal_update_pub.publish(goal)

    def _send_initial_goal(self, goal: PoseStamped):
        if not self._nav_client.wait_for_server(timeout_sec=2.0):
            self.get_logger().warn("Action navigate_to_pose non disponibile.")
            return
        req = NavigateToPose.Goal()
        req.pose = goal
        self._goal_active = True
        future = self._nav_client.send_goal_async(req)
        future.add_done_callback(self._on_goal_response)
        self.get_logger().info(
            f"NavigateToPose inviata: ({goal.pose.position.x:.2f}, {goal.pose.position.y:.2f})")

    def _on_goal_response(self, future):
        handle = future.result()
        if not handle.accepted:
            self.get_logger().warn("Goal NavigateToPose rifiutato.")
            self._goal_active = False
            return
        self._goal_handle = handle
        handle.get_result_async().add_done_callback(self._on_goal_result)

    def _on_goal_result(self, future):
        result = future.result()
        self.get_logger().warn(
            f"[DEBUG] Goal concluso — status={result.status} "
            f"(4=SUCCEEDED, 5=CANCELED, 6=ABORTED)")
        self._goal_active = False
        self._goal_handle = None

    def _check_target_loss(self):
        if not self._goal_active or self._kf_last_update_time is None:
            return
        elapsed = time.monotonic() - self._kf_last_update_time
        if elapsed > MAX_TARGET_LOSS_SEC:
            self.get_logger().warn(f"Target perso da {elapsed:.1f}s — cancello la navigazione.")
            if self._goal_handle is not None:
                self._goal_handle.cancel_goal_async()
            self._goal_active = False


def main():
    rclpy.init()
    node = Nav2Bridge()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()