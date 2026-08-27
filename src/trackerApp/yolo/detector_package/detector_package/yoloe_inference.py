#!/usr/bin/env python3
"""
yoloe_inference.py

Wrapper puro (NESSUNA dipendenza ROS) attorno al modello YOLOE. Testabile da
solo, fuori da un nodo, con un semplice frame numpy.

Due modalita':
- detect(frame, classes) — detection singola, indipendente frame per frame
  (nessuna identita' mantenuta).
- track(frame, classes) — come detect(), ma usa model.track() di
  Ultralytics (tracker BoT-SORT di default) con persist=True: mantiene un
  track_id stabile per lo stesso oggetto tra chiamate CONSECUTIVE, a patto
  che arrivino in una sequenza temporale reale (stesso flusso video, senza
  salti) — persist=True tiene in vita lo stato interno del tracker
  (filtro di Kalman, contatore ID) tra una chiamata e l'altra sulla stessa
  istanza di YoloEInference.

Nessun crop, nessuna ROI, nessun visual-prompt: quelle sono responsabilita'
di chi chiama (demo_package), non di questa classe.

La classe va istanziata UNA VOLTA (carica il modello una sola volta) e
riusata per tutta la vita del nodo ROS2 che la incapsula.
"""

import dataclasses
from typing import List, Optional, Sequence

from torchreid.utils import FeatureExtractor

import cv2
import numpy as np
import torch
from ultralytics import YOLOE


@dataclasses.dataclass
class Detection:
    box: np.ndarray   # [x1, y1, x2, y2] in pixel (float), stesso formato di box.xyxy()
    score: float
    class_name: str
    track_id: int = -1  # -1 = non tracciato (da detect(), o track() senza match) — vedi track()
    embedding: Optional[np.ndarray] = None  # vettore d'aspetto (da un modello di
                                              # PERSON RE-IDENTIFICATION vero — vedi
                                              # reid_model_name/reid_model_path) — None se
                                              # non configurato


class YoloEInference:
    def __init__(self, model_path: str, imgsz: int = 640, conf_threshold: float = 0.35,
                 reid_model_name: Optional[str] = None, reid_model_path: Optional[str] = None):
        """model_path: percorso LOCALE del file .pt. Nessun download a
        runtime: il modello va salvato nell'immagine Docker del container
        yolo (o montato come volume), non scaricato ad ogni avvio.

        reid_model_name/reid_model_path: nome del modello (es. 'osnet_x1_0')
        e percorso LOCALE dei pesi pre-addestrati (es. su Market1501) per
        un modello di PERSON RE-IDENTIFICATION vero (libreria torchreid) —
        indipendente da YOLOE e da BoT-SORT. Serve un file scaricato una
        volta e tenuto in locale (stesso principio degli altri modelli, mai
        scaricato a runtime), NON i soli pesi ImageNet generici. Se uno dei
        due manca, extract_embedding() ritorna sempre None."""
        self.model = YOLOE(model_path)
        self.imgsz = imgsz
        self.conf_threshold = conf_threshold
        self._current_classes: Optional[List[str]] = None

        self.reid_extractor = None
        if reid_model_name and reid_model_path:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            self.reid_extractor = FeatureExtractor(
                model_name=reid_model_name, model_path=reid_model_path, device=device)

    def set_classes(self, classes: Sequence[str]) -> None:
        classes = list(classes)
        if classes != self._current_classes:
            self.model.set_classes(classes, self.model.get_text_pe(classes))
            self._current_classes = classes

    def _resolve_classes(self, classes: Optional[Sequence[str]]) -> List[str]:
        if classes is not None:
            self.set_classes(classes)
        elif self._current_classes is None:
            raise RuntimeError(
                "Nessuna classe impostata: passa `classes` almeno alla prima chiamata "
                "(o chiama set_classes() prima di detect()/track()).")
        return list(classes) if classes is not None else self._current_classes

    def detect(
        self,
        frame_bgr: np.ndarray,
        classes: Optional[Sequence[str]] = None,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> List[Detection]:
        """Detection sull'immagine intera, SENZA identita' tra un frame e
        l'altro (ogni chiamata e' indipendente). Per il tracciamento con
        track_id persistente, vedi track()."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        sz = imgsz if imgsz is not None else self.imgsz
        active_classes = self._resolve_classes(classes)

        results = self.model.predict(frame_bgr, conf=conf, imgsz=sz, verbose=False)[0]

        detections = []
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            if active_classes and class_name not in active_classes:
                continue
            detections.append(Detection(
                box=box.xyxy[0].cpu().numpy(),
                score=float(box.conf[0]),
                class_name=class_name,
            ))
        return detections

    def track(
        self,
        frame_bgr: np.ndarray,
        classes: Optional[Sequence[str]] = None,
        conf_threshold: Optional[float] = None,
        imgsz: Optional[int] = None,
        tracker: str = "oc_sort.yaml",
        persist: bool = True,
    ) -> List[Detection]:
        """Come detect(), ma via model.track() — assegna un track_id
        persistente allo stesso oggetto fisico tra chiamate consecutive.

        IMPORTANTE — persist=True presuppone che le chiamate arrivino in
        sequenza temporale reale sulla STESSA istanza di questa classe
        (stesso processo, nessun riavvio nel mezzo): lo stato del tracker
        (filtro di Kalman, contatore ID) vive dentro self.model tra una
        chiamata e l'altra. Se il chiamante manda frame non consecutivi
        (es. salta molti frame, o interrompe e riprende dopo molto tempo),
        il tracker puo' comportarsi in modo inatteso — non e' un bug
        nostro, e' il comportamento documentato di persist=True.

        ReID disattivato di default nel tracker BoT-SORT (with_reid: False
        nel file botsort.yaml usato internamente da Ultralytics) per
        minimizzare il costo — l'associazione qui e' basata su moto
        (Kalman) + IoU, non su un confronto d'aspetto vero. Se serve
        abilitarlo, va fatto in un file di config del tracker personalizzato,
        non in questo wrapper."""
        conf = conf_threshold if conf_threshold is not None else self.conf_threshold
        sz = imgsz if imgsz is not None else self.imgsz
        active_classes = self._resolve_classes(classes)

        results = self.model.track(
            frame_bgr, conf=conf, imgsz=sz, verbose=False, tracker=tracker, persist=persist)[0]

        detections = []
        for box in results.boxes:
            class_name = results.names[int(box.cls[0])]
            if active_classes and class_name not in active_classes:
                continue
            # box.id e' None se questa specifica detection non e' stata
            # agganciata a nessun track in questo frame (capita, non e'
            # un errore) — usiamo -1 come convenzione per "non tracciato".
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append(Detection(
                box=box.xyxy[0].cpu().numpy(),
                score=float(box.conf[0]),
                class_name=class_name,
                track_id=track_id,
            ))
        return detections

    def extract_embedding(self, frame_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Ritaglia il box e ne estrae un vettore d'aspetto tramite un
        modello di PERSON RE-IDENTIFICATION vero (torchreid/OSNet) — non
        piu' un modello di classificazione generico: questo e' addestrato
        specificamente a separare individui diversi, non categorie di
        oggetti.

        Ritorna None se reid_extractor non e' configurato, se il box e'
        degenere (fuori immagine, area nulla), o se l'estrazione fallisce
        per qualunque motivo — MAI un vettore "finto" o di fallback: se
        qualcosa va storto lo si vede nei log, non si propaga un embedding
        sbagliato che produrrebbe similarita' fuorvianti a valle."""
        if self.reid_extractor is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        # torchreid si aspetta immagini RGB — il nostro frame e' BGR (OpenCV).
        # Il ridimensionamento a 256x128 e la normalizzazione sono gestiti
        # INTERNAMENTE da FeatureExtractor, non li facciamo qui.
        crop_rgb = cv2.cvtColor(frame_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2RGB)
        try:
            features = self.reid_extractor([crop_rgb])
        except Exception as ex:
            print(f"[YoloEInference] estrazione embedding ReID fallita: {ex}")
            return None
        if features is None or len(features) == 0:
            print("[YoloEInference] FeatureExtractor ha ritornato un risultato vuoto per questo box.")
            return None
        vec = features[0].cpu().numpy().flatten()
        if vec.size == 0 or not np.any(vec):
            print("[YoloEInference] embedding ReID estratto ma degenere (vuoto o tutto zero) — scartato.")
            return None
        return vec

    def reset_tracker(self) -> None:
        """Forza un reset dello stato interno del tracker (Kalman, contatore
        ID) — pensato per essere chiamato quando tracking_fsm torna in
        SEARCH dopo aver perso il target, cosi' un vecchio track_id non
        possa riemergere in modo confuso in una sessione di tracciamento
        successiva.

        NON VERIFICATO A FONDO: Ultralytics non espone un metodo pubblico
        di reset esplicito e documentato per model.track(); qui sfruttiamo
        il comportamento documentato di persist=False ("il tracker viene
        azzerato quando si passa persist=False o cambia la sorgente") con
        una chiamata a vuoto su un frame minimale. Da testare per davvero
        prima di fare affidamento su questo in produzione."""
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.model.track(dummy, conf=0.99, imgsz=64, verbose=False, persist=False)
        except Exception:
            pass  # il reset e' un "meglio se funziona", non deve far cadere il nodo se fallisce