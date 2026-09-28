#!/usr/bin/env python3
"""
list_spot_led_behaviors.py

Elenca i behavior A/V (LED + buzzer) GIA' presenti sul robot, con i colori
di ogni gruppo LED e l'indicazione se contengono una sequenza AUDIO.
Serve a scegliere quali nomi associare agli stati della FSM in
tracking_fsm.py (dizionario STATI_LED), dato che il software del robot
5.0.1 non permette di crearne di nuovi (manca AddOrModifyBehavior).

Uso (le credenziali si passano come variabili d'ambiente, come suggerisce
l'SDK stesso):
    export BOSDYN_CLIENT_USERNAME=admin
    export BOSDYN_CLIENT_PASSWORD=...
    python3 list_spot_led_behaviors.py            # solo elenco
    python3 list_spot_led_behaviors.py NOME 5     # prova il behavior NOME per 5 secondi
"""

import os
import sys
import time

import bosdyn.client
from bosdyn.client.audio_visual import AudioVisualClient

ROBOT_IP = "192.168.80.3"
LED_GROUPS = ["front_center", "front_left", "front_right", "hind_left", "hind_right", "status_lights", "spot_cam"]


def describe_sequence(led_sequence):
    """Tipo di animazione + colore di una LedSequence, in una riga."""
    fields = led_sequence.ListFields()
    if not fields:
        return None
    kind_field, seq = fields[0]
    kind = kind_field.name.replace("_sequence", "")
    color = getattr(seq, "color", None)
    if color is None or not color.ListFields():
        return f"{kind}"
    if color.HasField("rgb"):
        return f"{kind} rgb=({color.rgb.r},{color.rgb.g},{color.rgb.b})"
    return f"{kind} preset={color.preset}"


def describe_behavior(live):
    b = live.behavior
    lines = [f"  NOME: {live.name!r}   permanente={live.permanent}   enabled={b.enabled}   priorita'={b.priority}"]
    has_audio = b.HasField("audio_sequence_group") and bool(b.audio_sequence_group.ListFields())
    lines.append(f"    AUDIO: {'SI  <-- suona il buzzer!' if has_audio else 'no'}")
    group = b.led_sequence_group
    for name in LED_GROUPS:
        if hasattr(group, name) and group.HasField(name):
            desc = describe_sequence(getattr(group, name))
            if desc:
                lines.append(f"    {name:14s} {desc}")
    return "\n".join(lines)


def main():
    username = os.environ.get("BOSDYN_CLIENT_USERNAME")
    password = os.environ.get("BOSDYN_CLIENT_PASSWORD")
    if not username or not password:
        print("Imposta prima BOSDYN_CLIENT_USERNAME e BOSDYN_CLIENT_PASSWORD.")
        sys.exit(1)

    sdk = bosdyn.client.create_standard_sdk("list_led_behaviors")
    sdk.register_service_client(AudioVisualClient)
    robot = sdk.create_robot(ROBOT_IP)
    robot.authenticate(username, password)
    robot.time_sync.wait_for_sync()
    av = robot.ensure_client(AudioVisualClient.default_service_name)

    try:
        params = av.get_system_params()
        print(f"Parametri sistema A/V:\n{params}")
    except Exception as ex:
        print(f"get_system_params fallita: {ex}")

    behaviors = av.list_behaviors()
    print(f"\n{len(behaviors)} behavior presenti sul robot:\n")
    for live in behaviors:
        print(describe_behavior(live))
        print()

    # Prova opzionale di un behavior: python3 list_spot_led_behaviors.py NOME [secondi]
    if len(sys.argv) >= 2:
        name = sys.argv[1]
        seconds = float(sys.argv[2]) if len(sys.argv) >= 3 else 5.0
        print(f"Avvio {name!r} per {seconds:.0f}s...")
        av.run_behavior(name, time.time() + seconds, restart=True)
        time.sleep(seconds)
        av.stop_behavior(name)
        print("Fatto.")


if __name__ == "__main__":
    main()