#!/usr/bin/env python3
"""
costmap_refresher.py (spot_motion)

Svuota periodicamente uno o piu' costmap chiamando il servizio
ClearEntireCostmap di Nav2, SEMPRE: anche a robot fermo, in HOLD (rotazione
sul posto), con goal da RViz o senza nessuna navigazione attiva.

PERCHE' UN NODO A PARTE (e non nel Behavior Tree)
  Il BT gira solo durante una NavigateToPose. Quando il robot e' fermo non
  viene eseguito, quindi un refresh messo li' si fermerebbe proprio quando
  serve. Questo nodo e' indipendente da navigazione e following.

COSA COPRE
  Gli ostacoli rimasti FUORI dal campo visivo delle camere frontali (es. una
  persona passata di lato): nessun raggio puo' pulirli, quindi senza refresh
  resterebbero finche' non escono dalla finestra del costmap.
  Dentro il campo visivo il refresh continuo lo fanno gia' le sorgenti
  clear_frontleft / clear_frontright del YAML, a ogni nuvola.

DOPO OGNI REFRESH gli ostacoli reali davanti al robot ricompaiono alla nuvola
successiva (~0.3-0.5 s); quelli laterali quando le camere li rivedono.

PARAMETRI
  period    secondi tra due refresh (default 2.0)
  services  lista dei servizi da chiamare
            (default: solo il costmap locale; il globale serve al planner per
             ricordare gli ostacoli gia' superati, meglio non svuotarlo spesso)

Uso:
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
        self.pending = {n: None for n in names}      # una sola richiesta in volo per servizio
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
                continue  # la richiesta precedente non e' ancora tornata: non accodarne altre
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