#!/usr/bin/env python3
"""
detect_client.py

Wrapper leggero attorno a un client ROS2 del servizio
detector_interfaces/Detect (esposto dal DetectorNode nel container yolo).

Versione SINCRONA/BLOCCANTE: call_sync() non ritorna finche' non arriva la
risposta (o scade il timeout). Significa che durante l'attesa il nodo non
elabora altri frame — scelta consapevole, accettata sapendo che riduce il
numero di frame effettivamente processati.

COME e' implementata l'attesa bloccante, e perche' NON con
rclpy.spin_until_future_complete: quella funzione, se non le passi
esplicitamente l'executor giusto, puo' usarne/crearne uno diverso da quello
che sta gia' facendo girare il nodo (il nostro MultiThreadedExecutor) — un
nodo associato a due executor contemporaneamente e' una condizione che
rclpy non gestisce in modo sicuro, e puo' bloccare tutto silenziosamente
dopo la prima chiamata riuscita. Usiamo invece un semplice
`threading.Event`: il MultiThreadedExecutor, che sta gia' girando per
conto suo, elabora la risposta tramite `future.add_done_callback` (sul
gruppo dedicato al client, vedi tracking_fsm.py) e sblocca l'evento; il
thread chiamante aspetta solo quell'evento — nessun secondo spin coinvolto.

ATTENZIONE — requisito per evitare un deadlock, non opzionale: questo
client va costruito con un `callback_group` DIVERSO da quello della
callback che chiama call_sync() (tipicamente quella dell'immagine), e il
nodo deve girare su un MultiThreadedExecutor (non il default
SingleThreadedExecutor di rclpy.spin()). Se client e chiamante fossero
nello stesso gruppo, la risposta non potrebbe mai essere elaborata mentre
il thread e' bloccato ad aspettarla. Vedi tracking_fsm.py (main()) per il
setup completo.
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

    def call_sync(self, image_msg, target_classes=None, timeout_sec=10.0):
        """Chiamata BLOCCANTE: non ritorna finche' non arriva la risposta (o
        scade `timeout_sec`).

        Ritorna la risposta (detector_interfaces.srv.Detect.Response, campo
        `detections`), oppure None se: il servizio non e' pronto, la
        chiamata va in timeout, o fallisce per qualunque altro motivo."""
        if not self.client.service_is_ready():
            return None

        request = Detect.Request()
        request.image = image_msg
        if target_classes is not None:
            request.target_classes = list(target_classes)

        done_event = threading.Event()
        result = {}

        def _on_done(future):
            try:
                result['response'] = future.result()
            except Exception as ex:  # qualunque errore rclpy/rmw sulla chiamata
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

        self.node.get_logger().info(f"[DetectClient] round-trip (sincrono): {elapsed_ms:.0f} ms")
        return result.get('response')