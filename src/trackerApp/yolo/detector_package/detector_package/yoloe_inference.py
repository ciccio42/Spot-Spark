#!/usr/bin/env python3
"""
yoloe_inference.py

Una sola modalita': detect(frame, classes) — detection per classi di testo
sull'immagine intera.

La classe va istanziata UNA VOLTA (carica il modello una sola volta) e
riusata per tutta la vita del nodo ROS2 che la incapsula.
"""

import dataclasses
from typing import List, Optional, Sequence

import numpy as np
from ultralytics import YOLOE


@dataclasses.dataclass
class Detection:
    box: np.ndarray   # [x1, y1, x2, y2] in pixel (float), stesso formato di box.xyxy()
    score: float
    class_name: str


class YoloEInference:
    def __init__(self, model_path: str, imgsz: int = 640, conf_threshold: float = 0.35):
        """model_path: percorso LOCALE del file .pt. Nessun download a
        runtime: il modello va salvato nell'immagine Docker del container
        yolo (o montato come volume), non scaricato ad ogni avvio."""
        self.model = YOLOE(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self._current_classes: Optional[List[str]] = None

    def set_classes(self, classes: Sequence[str]) -> None:
        classes = list(classes)
        if classes != self._current_classes:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
            self._current_classes = classes

    def detect(
        self,
        frame_bgr: np.ndarray,
        classes: Optional[Sequence[str]] = None,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        """Detection sull'immagine intera.

        - frame_bgr: immagine completa (numpy H x W x 3 BGR).
        - classes: classi di testo da cercare. Se None, riusa le ultime
          impostate con set_classes() (errore se non e' mai stata impostata
          nessuna classe)."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        sz = imgsz if imgsz is not None else self.imgsz

        if classes is not None:
            self.set_classes(classes)
        elif self._current_classes is None:
            raise RuntimeError(
                "Nessuna classe impostata: passa `classes` almeno alla prima chiamata "
                "(o chiama set_classes() prima di detect()).")
        active_classes = list(classes) if classes is not None else self._current_classes

        results = self.model.predict(frame_bgr, conf=conf, imgsz=sz, verbose=False)[0]

        detections = []
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            # guardia extra (ridondante col set_classes, ma innocua): scarta
            # eventuali classi non richieste che il modello restituisse comunque.
            if active_classes and class_name not in active_classes:
                continue
            detections.append(Detection(
                box=box.xyxy[0].cpu().numpy(),
                score=float(box.conf[0]),
                class_name=class_name,
            ))
        return detections
