#!/usr/bin/env python3
"""
costmap_refresher.py (spot_motion)

Periodically clears one or more costmaps by calling the Nav2
ClearEntireCostmap service, ALWAYS: even with the robot standing still, in
HOLD (rotating in place), with a goal from RViz or with no navigation active.

WHY A SEPARATE NODE (and not in the Behavior Tree)
  The BT only runs during a NavigateToPose. When the robot is still it is
  not executed, so a refresh placed there would stop exactly when it is
  needed. This node is independent of navigation and following.

WHAT IT COVERS
  Obstacles left OUTSIDE the field of view of the front cameras (e.g. a
  person who walked past on the side): no ray can clear them, so without a
  refresh they would stay until they leave the costmap window.
  Inside the field of view the continuous refresh is already done by the
  clear_frontleft / clear_frontright sources in the YAML, on every cloud.

AFTER EACH REFRESH the real obstacles in front of the robot reappear with the
next cloud (~0.3-0.5 s); the lateral ones when the cameras see them again.

PARAMETERS
  period    seconds between two refreshes (default 2.0)
  services  list of the services to call
            (default: only the local costmap; the planner needs the global
             one to remember obstacles already passed, better not to clear it often)

Usage:
  ros2 run spot_motion costmap_refresher
  ros2 run spot_motion costmap_refresher --ros-args -p period:=1.0
"""
import rclpy
from rclpy.node import Node
from nav2_msgs.srv import ClearEntireCostmap


class CostmapRefresher(Node):
    def __init__(self):
        super().__init__('costmap_refresher')
        self.declare_parameter('period', 2.0)
        self.declare_parameter('services', ['/local_costmap/clear_entirely_local_costmap'])

        self.period = float(self.get_parameter('period').value)
        names = list(self.get_parameter('services').value)

        self.clients_ = {n: self.create_client(ClearEntireCostmap, n) for n in names}
        self.pending = {n: None for n in names}      # only one request in flight per service
        self.ok_logged = {n: False for n in names}

        self.create_timer(self.period, self._refresh)
        self.get_logger().info(f"Refresh ogni {self.period:.1f}s di: {', '.join(names)}")

    def _refresh(self):
        for name, client in self.clients_.items():
            if not client.service_is_ready():
                self.get_logger().warn(f"Servizio {name} non disponibile (Nav2 attivo?)",
                                       throttle_duration_sec=10.0)
                continue
            fut = self.pending[name]
            if fut is not None and not fut.done():
                continue  # the previous request has not returned yet: do not queue more
            fut = client.call_async(ClearEntireCostmap.Request())
            fut.add_done_callback(lambda f, n=name: self._on_done(n, f))
            self.pending[name] = fut

    def _on_done(self, name, fut):
        if fut.exception() is not None:
            self.get_logger().warn(f"Refresh {name} fallito: {fut.exception()}", throttle_duration_sec=10.0)
        elif not self.ok_logged[name]:
            self.get_logger().info(f"Primo refresh di {name} eseguito.")
            self.ok_logged[name] = True


def main():
    rclpy.init()
    node = CostmapRefresher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    main()