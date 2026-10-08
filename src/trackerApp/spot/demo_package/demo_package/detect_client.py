#!/usr/bin/env python3
"""
detect_client.py

Thin wrapper around a ROS 2 client of the detector_interfaces/Detect
service (exposed by the DetectorNode in the yolo container).

SYNCHRONOUS/BLOCKING version: call_sync() does not return until the
response arrives (or the timeout expires). This means that while waiting
the node does not process other frames — a deliberate choice, accepted
knowing that it reduces the number of frames actually processed.

HOW the blocking wait is implemented, and why NOT with
rclpy.spin_until_future_complete: unless you explicitly pass it the right
executor, that function may use/create one different from the one already
spinning the node (our MultiThreadedExecutor) — a node attached to two
executors at the same time is a condition rclpy does not handle safely,
and it can silently block everything after the first successful call.
Instead we use a plain `threading.Event`: the MultiThreadedExecutor, which
is already spinning on its own, processes the response via
`future.add_done_callback` (on the group dedicated to the client, see
tracking_fsm.py) and sets the event; the calling thread only waits for
that event — no second spin involved.

WARNING — requirement to avoid a deadlock, not optional: this client must
be built with a `callback_group` DIFFERENT from that of the callback that
calls call_sync() (typically the image one), and the node must run on a
MultiThreadedExecutor (not the default SingleThreadedExecutor of
rclpy.spin()). If client and caller were in the same group, the response
could never be processed while the thread is blocked waiting for it. See
tracking_fsm.py (main()) for the complete setup.
"""

import threading
import time

from detector_interfaces.srv import Detect


class DetectClient:
    def __init__(self, node, service_name='detect', wait_timeout_sec=2.0, callback_group=None):
        self.node = node
        self.client = node.create_client(Detect, service_name, callback_group=callback_group)
        if not self.client.wait_for_service(timeout_sec=wait_timeout_sec):
            node.get_logger().warn(
                f"Servizio '{service_name}' non ancora disponibile dopo {wait_timeout_sec}s "
                f"(il container yolo e' su? ROS_DOMAIN_ID combacia tra i due container?) — "
                f"riprovera' alle prossime chiamate.")

    def call_sync(self, image_msg, target_classes=None, timeout_sec=10.0, reset_tracker=False):
        """BLOCKING call: does not return until the response arrives (or
        `timeout_sec` expires).

        `reset_tracker=True`: asks the DetectorNode to reset the internal
        tracker state (BoT-SORT) BEFORE processing this request — use it
        when tracking_fsm goes back to SEARCH after losing the target, so
        that an old track_id does not resurface in a new session.

        Returns the response (detector_interfaces.srv.Detect.Response,
        `detections` field), or None if: the service is not ready, the
        call times out, or it fails for any other reason."""
        if not self.client.service_is_ready():
            return None

        request = Detect.Request()
        request.image = image_msg
        if target_classes is not None:
            request.target_classes = list(target_classes)
        request.reset_tracker = reset_tracker

        done_event = threading.Event()
        result = {}

        def _on_done(future):
            try:
                result['response'] = future.result()
            except Exception as ex:  # any rclpy/rmw error on the call
                result['error'] = ex
            done_event.set()

        t0 = time.monotonic()
        future = self.client.call_async(request)
        future.add_done_callback(_on_done)

        completed = done_event.wait(timeout=timeout_sec)
        elapsed_ms = (time.monotonic() - t0) * 1000.0

        if not completed:
            self.node.get_logger().warn(
                f"Timeout ({timeout_sec:.0f}s) in attesa della risposta dal servizio detect.")
            return None

        if 'error' in result:
            self.node.get_logger().warn(f"Chiamata al servizio detect fallita: {result['error']}")
            return None

        self.node.get_logger().warn(f"[DetectClient] round-trip (sincrono): {elapsed_ms:.0f} ms")
        return result.get('response')
