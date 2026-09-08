#!/usr/bin/env python3
"""
motion_command_node.py (spot_motion)

Nodo che riceve TargetPose3D (posizione GREZZA del target — non un goal gia'
scalato — nel frame OTTICO della camera, pubblicata da Pose3DEstimationNode)
e comanda Spot a:

  1. Restare SEMPRE rivolto verso il target (rotazione, ad ogni messaggio
     ricevuto).
  2. Camminare per convergere verso TARGET_DISTANCE, solo avvicinandosi (mai
     all'indietro) e solo oltre DISTANCE_TOLERANCE di scarto.

CATENA DI FRAME — due transform diversi, presi da due fonti diverse:
  - body -> wrist (meccanico): LIVE, da kinematic_state.transforms_snapshot
    (robot_state_client) ad ogni messaggio.
  - wrist -> camera (ottico): FISSO, preso UNA VOLTA SOLA all'avvio da un
    ImageResponse.shot.transforms_snapshot (via ImageClient).
  I due si compongono: body_tform_camera = body_tform_wrist * wrist_tform_camera.

FILTRO DI KALMAN (velocita' costante, 2 assi indipendenti x/y): smussa la
posizione del target nel frame ODOM (fisso — MAI il frame body, che si
muove col robot). Il filtro NON e' l'estrapolazione — l'estrapolazione (vedi
sotto) LEGGE lo stato del filtro, non lo modifica mai al di fuori di un dato
vero ricevuto in _on_target_pose.

ESTRAPOLAZIONE (_on_timer): gira su un timer INDIPENDENTE dall'arrivo dei
messaggi — se non arriva un TargetPose3D fresco, proietta in avanti
l'ultima posizione/velocita' nota del filtro (senza toccare lo stato
persistente) e manda comunque un comando, cosi' Spot non si ferma di colpo
per un buco breve. Oltre MAX_EXTRAPOLATION_SEC dall'ultimo dato vero, si
smette di indovinare — lascia che COMMAND_DURATION fermi Spot. NESSUN
cambiamento allo stato di tracking_fsm: ci si basa solo sul tempo reale
trascorso dall'ultimo dato vero, non sullo stato SEARCH/TRACKING/RECOVERY.

CONCORRENZA — importante, causa di un blocco osservato in una versione
precedente: _on_target_pose e _on_timer fanno ENTRAMBI una chiamata di rete
vera (get_robot_state()). Se girassero sullo stesso thread (l'executor a
thread singolo di rclpy.spin() di default), l'uno bloccherebbe l'altro ogni
volta che capitano vicini nel tempo. Stesso identico problema — e stessa
soluzione — gia' adottata in tracking_fsm.py: due callback group separati
+ un MultiThreadedExecutor (vedi main()), cosi' le due chiamate possono
girare su thread diversi senza aspettarsi a vicenda.

Usa RobotCommandBuilder.synchro_trajectory_command_in_body_frame(): prende
un goal RELATIVO al corpo (dx, dy, dyaw) + uno snapshot delle trasformazioni,
e lo converte lei stessa nel frame mondo non mobile (odom, hardcoded).

VELOCITA': limitata via MobilityParams.vel_limit (MAX_LINEAR_VEL/MAX_ANGULAR_VEL
qui sotto) — pattern confermato dall'esempio ufficiale Boston Dynamics
spot_detect_and_follow.py.

Parla direttamente con l'SDK bosdyn — bypassa Nav2 e cmd_vel di spot_ros2.
spot_driver (se in esecuzione in parallelo) va lanciato con
auto_claim/auto_power_on/auto_stand a false, cosi' non compete per il lease.

CONFIGURAZIONE: costanti qui sotto, non argomenti da riga di comando.
ATTENZIONE — SPOT_USERNAME/SPOT_PASSWORD in chiaro sono un'eccezione
TEMPORANEA per la prova iniziale: vanno spostate su variabile d'ambiente
(BOSDYN_CLIENT_USERNAME/PASSWORD) prima di qualunque commit.
"""
import math
import time

import rclpy
from rclpy.node import Node
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from rclpy.executors import MultiThreadedExecutor

import bosdyn.client
import bosdyn.client.util
from bosdyn.api import geometry_pb2
from bosdyn.api.spot import robot_command_pb2 as spot_command_pb2
from bosdyn.client import math_helpers
from bosdyn.client.frame_helpers import (
    BODY_FRAME_NAME, ODOM_FRAME_NAME, get_a_tform_b, get_se2_a_tform_b)
from bosdyn.client.image import ImageClient
from bosdyn.client.lease import LeaseClient, LeaseKeepAlive
from bosdyn.client.robot_command import RobotCommandBuilder, RobotCommandClient, blocking_stand
from bosdyn.client.robot_state import RobotStateClient

from demo_interfaces.msg import TargetPose3D
from demo_package.common import CONE_MIN_RANGE

# ============================================================
# Configurazione — modifica qui, non da riga di comando.
# ============================================================
SPOT_HOSTNAME = '192.168.80.3'  # <-- metti l'IP vero del tuo Spot
SPOT_USERNAME = 'admin'         # <-- SOLO per la prova iniziale, vedi nota sopra
SPOT_PASSWORD = 'prb4e3wparqx'  #     da spostare su env var prima di qualunque commit
COMMAND_DURATION = 3.0          # secondi — rete di sicurezza end_time_secs
DRY_RUN = False                 # True: calcola e logga SENZA mai inviare comandi al robot
HAND_CAMERA_IMAGE_SOURCE = 'hand_color_image'
WRIST_FRAME_NAME = 'arm0.link_wr1'  # verificato: presente in entrambi gli snapshot
TARGET_DISTANCE = 2.5           # metri — distanza che Spot cerca sempre di mantenere
DISTANCE_TOLERANCE = 0.15       # metri — sotto questo scarto, resta fermo (solo rotazione)
MAX_LINEAR_VEL = 0.6   # m/s — molto sotto il massimo hardware di Spot; abbassa se serve
                         #       ancora piu' lento, il comando non "va veloce" oltre questo
MAX_ANGULAR_VEL = 0.5  # rad/s — stesso principio per la rotazione
KF_PROCESS_VAR = 0.05       # rumore di PROCESSO — quanto ci aspettiamo che la velocita' vera del
                              # target possa cambiare (piu' alto = filtro piu' reattivo a cambi di
                              # direzione/curve, ma smussa meno il rumore)
KF_MEASUREMENT_VAR = 0.05   # rumore di MISURA — quanto ci fidiamo della singola bx,by grezza
                              # (piu' alto = smussa di piu', ma reagisce piu' lentamente ai cambi veri)
KF_RESET_GAP_SEC = 2.0      # se passa piu' di questo dall'ultimo aggiornamento, il filtro si
                              # REINIZIALIZZA sulla nuova misura invece di fonderla con uno stato
                              # ormai troppo vecchio (es. dopo una RECOVERY prolungata)
EXTRAPOLATION_TIMER_PERIOD = 0.2  # secondi tra un controllo e l'altro quando non arrivano dati
                                    # freschi — piu' corto dell'intervallo tipico tra due
                                    # TargetPose3D (~0.3-0.6s), cosi' i buchi si notano in fretta
MAX_EXTRAPOLATION_SEC = 2.0       # oltre questo tempo SENZA un dato vero, si smette di camminare
                                    # "alla cieca" — nessun nuovo comando, lascia che
                                    # COMMAND_DURATION fermi Spot come rete di sicurezza finale.
# ============================================================


def _se2_transform_point(a_tform_b, x, y):
    """Trasforma un punto (x,y) dal frame b al frame a. SE2Pose non ha un
    transform_point diretto (a differenza di Quat/SE3Pose, verificato) —
    lo otteniamo componendo con .mult(), l'unico metodo di composizione
    confermato: un punto e' una SE2Pose con angle=0, il risultato della
    composizione ne eredita la posizione trasformata."""
    result = a_tform_b.mult(math_helpers.SE2Pose(x, y, 0.0))
    return result.x, result.y


class _ConstantVelocityKalman1D:
    """Filtro di Kalman 1D, modello a velocita' costante: stato [pos, vel].
    Usato due volte (assi x e y indipendenti) per smussare la posizione del
    target nel frame ODOM (fisso nel mondo — MAI il frame body, che si
    muove col robot: stimare una velocita' su coordinate che si muovono
    gia' da sole mescolerebbe il moto del target con quello del robot)."""

    def __init__(self, process_var, measurement_var):
        self.pos = 0.0
        self.vel = 0.0
        self.P = [[1e3, 0.0], [0.0, 1e3]]  # covarianza iniziale alta: non ci fidiamo ancora di nulla
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


class MotionCommandNode(Node):
    def __init__(self, robot_state_client, robot_command_client, wrist_tform_camera):
        super().__init__('motion_command_node')
        self.robot_state_client = robot_state_client
        self.robot_command_client = robot_command_client
        self._wrist_tform_camera = wrist_tform_camera  # fisso, calcolato una volta in main()
        self._command_duration = COMMAND_DURATION
        self._dry_run = DRY_RUN
        self._mobility_params = spot_command_pb2.MobilityParams(
            vel_limit=geometry_pb2.SE2VelocityLimit(
                max_vel=geometry_pb2.SE2Velocity(
                    linear=geometry_pb2.Vec2(x=MAX_LINEAR_VEL, y=MAX_LINEAR_VEL),
                    angular=MAX_ANGULAR_VEL)))
        self._kf_x = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_y = _ConstantVelocityKalman1D(KF_PROCESS_VAR, KF_MEASUREMENT_VAR)
        self._kf_last_update_time = None

        # Due callback group separati: senza questo, _on_target_pose e
        # _on_timer girerebbero sullo stesso thread (default rclpy) e si
        # bloccherebbero a vicenda ogni volta che entrambi fanno una
        # chiamata di rete vicine nel tempo — vedi nota in cima al file.
        pose_group = MutuallyExclusiveCallbackGroup()
        timer_group = MutuallyExclusiveCallbackGroup()

        self.create_subscription(TargetPose3D, 'target_3d', self._on_target_pose, 1,
                                  callback_group=pose_group)
        self.create_timer(EXTRAPOLATION_TIMER_PERIOD, self._on_timer,
                           callback_group=timer_group)

        self.get_logger().info(
            f"Pronto. command_duration={self._command_duration:.1f}s, "
            f"target_distance={TARGET_DISTANCE:.2f}m (+/-{DISTANCE_TOLERANCE:.2f}m), "
            f"vel_limit=({MAX_LINEAR_VEL:.2f}m/s, {MAX_ANGULAR_VEL:.2f}rad/s), frame mondo=odom, "
            f"max_extrapolation={MAX_EXTRAPOLATION_SEC:.1f}s. dry_run={self._dry_run} — "
            f"'NESSUN comando verra\\' inviato al robot' if self._dry_run else 'comandi ATTIVI'.")

    def _on_target_pose(self, msg: TargetPose3D):
        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot

        body_tform_wrist = get_a_tform_b(transforms, BODY_FRAME_NAME, WRIST_FRAME_NAME)
        body_tform_camera = body_tform_wrist * self._wrist_tform_camera

        bx, by, bz = body_tform_camera.transform_point(
            msg.position.x, msg.position.y, msg.position.z)

        # --- Filtro di Kalman, nel frame ODOM (fisso), non su bx,by direttamente ---
        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        world_x, world_y = _se2_transform_point(odom_tform_body, bx, by)

        now = time.monotonic()
        if self._kf_last_update_time is None or (now - self._kf_last_update_time) > KF_RESET_GAP_SEC:
            self._kf_x.reset(world_x)
            self._kf_y.reset(world_y)
        else:
            dt = now - self._kf_last_update_time
            self._kf_x.predict(dt)
            self._kf_y.predict(dt)
            self._kf_x.update(world_x)
            self._kf_y.update(world_y)
        self._kf_last_update_time = now

        body_tform_odom = odom_tform_body.inverse()
        bx, by = _se2_transform_point(body_tform_odom, self._kf_x.pos, self._kf_y.pos)
        # --- fine filtro ---

        self._build_and_send_command(bx, by, transforms, source_tag="reale")

    def _on_timer(self):
        """Gira ogni EXTRAPOLATION_TIMER_PERIOD, indipendentemente dai
        messaggi. Legge lo stato del filtro (senza mai modificarlo) per
        proiettare in avanti la posizione durante un buco breve."""
        if self._kf_last_update_time is None:
            return  # nessun dato vero ricevuto ancora

        elapsed = time.monotonic() - self._kf_last_update_time
        if elapsed < EXTRAPOLATION_TIMER_PERIOD or elapsed > MAX_EXTRAPOLATION_SEC:
            return

        extrapolated_world_x = self._kf_x.pos + self._kf_x.vel * elapsed
        extrapolated_world_y = self._kf_y.pos + self._kf_y.vel * elapsed

        transforms = self.robot_state_client.get_robot_state().kinematic_state.transforms_snapshot
        odom_tform_body = get_se2_a_tform_b(transforms, ODOM_FRAME_NAME, BODY_FRAME_NAME)
        body_tform_odom = odom_tform_body.inverse()
        bx, by = _se2_transform_point(body_tform_odom, extrapolated_world_x, extrapolated_world_y)

        self._build_and_send_command(bx, by, transforms, source_tag="estrapolato")

    def _build_and_send_command(self, bx, by, transforms, source_tag):
        goal_heading_rt_body = math.atan2(by, bx)
        r_horizontal = math.sqrt(bx * bx + by * by)

        if r_horizontal > TARGET_DISTANCE + DISTANCE_TOLERANCE:
            zone = "si avvicina"
            scale = (r_horizontal - TARGET_DISTANCE) / r_horizontal
            dx, dy = bx * scale, by * scale
        else:
            zone = "a distanza (fermo)"
            dx, dy = 0.0, 0.0

        mode_tag = f"[SIMULAZIONE-{source_tag}]" if self._dry_run else f"[INVIATO-{source_tag}]"
        self.get_logger().info(
            f"{mode_tag} zona={zone} bx={bx:.2f} by={by:.2f} "
            f"r_orizz={r_horizontal:.2f}m -> dx={dx:.2f}m dy={dy:.2f}m "
            f"yaw={math.degrees(goal_heading_rt_body):.1f}deg", throttle_duration_sec=0.5)

        if self._dry_run:
            return

        robot_cmd = RobotCommandBuilder.synchro_trajectory_command_in_body_frame(
            goal_x_rt_body=dx, goal_y_rt_body=dy, goal_heading_rt_body=goal_heading_rt_body,
            frame_tree_snapshot=transforms, params=self._mobility_params)

        self.robot_command_client.robot_command(
            lease=None, command=robot_cmd,
            end_time_secs=time.time() + self._command_duration)


def main():
    rclpy.init()

    sdk = bosdyn.client.create_standard_sdk('SpotMotionClient')
    robot = sdk.create_robot(SPOT_HOSTNAME)
    robot.authenticate(SPOT_USERNAME, SPOT_PASSWORD)
    robot.time_sync.wait_for_sync()

    assert not robot.is_estopped(), (
        "Robot in e-stop. Serve un client e-stop attivo prima di poter comandare movimento.")

    lease_client = robot.ensure_client(LeaseClient.default_service_name)
    robot_state_client = robot.ensure_client(RobotStateClient.default_service_name)
    robot_command_client = robot.ensure_client(RobotCommandClient.default_service_name)
    image_client = robot.ensure_client(ImageClient.default_service_name)

    image_responses = image_client.get_image_from_sources([HAND_CAMERA_IMAGE_SOURCE])
    camera_snapshot = image_responses[0].shot.transforms_snapshot
    camera_frame_name = image_responses[0].shot.frame_name_image_sensor
    wrist_tform_camera = get_a_tform_b(camera_snapshot, WRIST_FRAME_NAME, camera_frame_name)
    assert wrist_tform_camera is not None, (
        f"get_a_tform_b non ha trovato un percorso tra '{WRIST_FRAME_NAME}' e "
        f"'{camera_frame_name}' nello snapshot immagine.")

    with LeaseKeepAlive(lease_client, must_acquire=True, return_at_exit=True):
        robot.power_on()
        blocking_stand(robot_command_client)

        node = MotionCommandNode(robot_state_client, robot_command_client, wrist_tform_camera)
        # MultiThreadedExecutor, non rclpy.spin(node): _on_target_pose e
        # _on_timer sono su callback group separati apposta per poter girare
        # su thread diversi — con l'executor a thread singolo di default,
        # i due callback group non servirebbero a nulla, resterebbero comunque
        # in coda uno dietro l'altro.
        executor = MultiThreadedExecutor(num_threads=2)
        executor.add_node(node)
        try:
            executor.spin()
        except KeyboardInterrupt:
            pass
        finally:
            robot_command_client.robot_command(RobotCommandBuilder.stop_command())
            node.destroy_node()

    rclpy.shutdown()


if __name__ == '__main__':
    main()