#!/usr/bin/env python3
"""
yoloe_inference.py

Pure wrapper (NO ROS dependency) around the YOLOE model. Testable on its
own, outside a node, with a plain numpy frame.

Two modes:
- detect(frame, classes) — single detection, independent frame by frame
  (no identity kept).
- track(frame, classes) — like detect(), but uses Ultralytics
  model.track() (BoT-SORT tracker by default) with persist=True: keeps a
  stable track_id for the same object across CONSECUTIVE calls, provided
  they arrive in a real temporal sequence (same video stream, no jumps) —
  persist=True keeps the internal tracker state (Kalman filter, ID
  counter) alive between calls on the same YoloEInference instance.

No crop, no ROI, no visual prompt: those are the caller's responsibility
(demo_package), not this class's.

The class must be instantiated ONCE (it loads the model only once) and
reused for the whole life of the ROS 2 node wrapping it.
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
    box: np.ndarray   # [x1, y1, x2, y2] in pixels (float), same format as box.xyxy()
    score: float
    class_name: str
    track_id: int = -1  # -1 = not tracked (from detect(), or track() without a match) — see track()
    embedding: Optional[np.ndarray] = None  # appearance vector (from a real
                                              # PERSON RE-IDENTIFICATION model — see
                                              # reid_model_name/reid_model_path) — None if
                                              # not configured


class YoloEInference:
    def __init__(self, model_path: str, imgsz: int = 640, conf_threshold: float = 0.35,
                 reid_model_name: Optional[str] = None, reid_model_path: Optional[str] = None):
        """model_path: LOCAL path of the .pt file. No download at run
        time: the model must be stored in the Docker image of the yolo
        container (or mounted as a volume), not downloaded at every start.

        reid_model_name/reid_model_path: model name (e.g. 'osnet_x1_0')
        and LOCAL path of the pre-trained weights (e.g. on Market1501) for
        a real PERSON RE-IDENTIFICATION model (torchreid library) —
        independent of YOLOE and BoT-SORT. It needs a file downloaded once
        and kept locally (same principle as the other models, never
        downloaded at run time), NOT just the generic ImageNet weights. If
        either is missing, extract_embedding() always returns None."""
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
        """Detection on the whole image, WITHOUT identity between one frame
        and the next (each call is independent). For tracking with a
        persistent track_id, see track()."""
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
        """Like detect(), but via model.track() — assigns a persistent
        track_id to the same physical object across consecutive calls.

        IMPORTANT — persist=True assumes that calls arrive in real temporal
        sequence on the SAME instance of this class (same process, no
        restart in between): the tracker state (Kalman filter, ID counter)
        lives inside self.model between calls. If the caller sends
        non-consecutive frames (e.g. skips many frames, or stops and
        resumes after a long time), the tracker may behave unexpectedly —
        not our bug, it is the documented behaviour of persist=True.

        ReID is disabled by default in the BoT-SORT tracker (with_reid:
        False in the botsort.yaml used internally by Ultralytics) to
        minimise cost — association here is based on motion (Kalman) +
        IoU, not on a real appearance comparison. If you need it, enable it
        in a custom tracker config file, not in this wrapper."""
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
            # box.id is None if this specific detection was not attached
            # to any track in this frame (it happens, it is not an
            # error) — we use -1 as the convention for "not tracked".
            track_id = int(box.id[0]) if box.id is not None else -1
            detections.append(Detection(
                box=box.xyxy[0].cpu().numpy(),
                score=float(box.conf[0]),
                class_name=class_name,
                track_id=track_id,
            ))
        return detections

    def extract_embedding(self, frame_bgr: np.ndarray, box: np.ndarray) -> Optional[np.ndarray]:
        """Crops the box and extracts an appearance vector from it with a
        real PERSON RE-IDENTIFICATION model (torchreid/OSNet) — no longer
        a generic classification model: this one is trained specifically
        to tell different individuals apart, not object categories.

        Returns None if reid_extractor is not configured, if the box is
        degenerate (outside the image, zero area), or if extraction fails
        for any reason — NEVER a "fake" or fallback vector: if something
        goes wrong it shows up in the logs, and a wrong embedding that
        would produce misleading similarities downstream is not propagated."""
        if self.reid_extractor is None:
            return None
        h, w = frame_bgr.shape[:2]
        x1, y1, x2, y2 = [int(v) for v in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        # torchreid expects RGB images — our frame is BGR (OpenCV).
        # Resizing to 256x128 and normalisation are handled INTERNALLY by
        # FeatureExtractor, we do not do them here.
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
        """Forces a reset of the internal tracker state (Kalman, ID
        counter) — meant to be called when tracking_fsm goes back to
        SEARCH after losing the target, so that an old track_id cannot
        confusingly resurface in a later tracking session.

        NOT THOROUGHLY VERIFIED: Ultralytics does not expose an explicit,
        documented public reset method for model.track(); here we rely on
        the documented behaviour of persist=False ("the tracker is reset
        when persist=False is passed or the source changes") with a dummy
        call on a minimal frame. Test it for real before relying on this
        in production."""
        dummy = np.zeros((64, 64, 3), dtype=np.uint8)
        try:
            self.model.track(dummy, conf=0.99, imgsz=64, verbose=False, persist=False)
        except Exception:
            pass  # the reset is best-effort, it must not bring the node down if it fails