#!/usr/bin/env python3
"""
common.py (demo_package)

Funzioni geometriche, di lettura depth e di ReID leggero usate da
tracking_fsm.py (design a camera singola: solo braccio). Nessuna
dipendenza da rclpy NE' da ultralytics/YOLOE: la detection vera passa dal
servizio Detect (vedi detect_client.py) esposto dal DetectorNode nel
container yolo — qui restano solo le funzioni "pure" che non dipendono da
come viene fatta la detection.
"""

import math

import cv2
import numpy as np


# ============================================================
# ROI — UNICA fonte di verita'. Cambia i valori qui, si applicano
# automaticamente sia al crop (compute_roi_crop_rect) sia al controllo di
# distanza post-detection in tracking_fsm.py.
# ============================================================
CONE_FOV_DEG = 25.0        # apertura TOTALE della ROI, in gradi — larghezza del crop (via intrinseci)
CONE_MIN_RANGE = 1.5       # distanza minima, in metri (controllo POST-detection, via depth)
CONE_MAX_RANGE = 3.5       # distanza massima, in metri (controllo POST-detection, via depth)
CROP_TOP_MARGIN_FRAC = 0   # frazione dell'altezza immagine tagliata dall'ALTO nel crop
CROP_BOTTOM_MARGIN_FRAC = 0  # frazione dell'altezza immagine tagliata dal BASSO nel crop
                                  # Nessun significato fisico ("altezza da terra") — puramente
                                  # pixel: il target deve solo cadere DENTRO il crop, la distanza
                                  # vera si verifica DOPO sulla depth del box (box_center_depth),
                                  # non sul crop stesso. Nessuna TF coinvolta.


# ============================================================
# Geometria 2D/3D
# ============================================================

def box_center(box):
    x1, y1, x2, y2 = box
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def distance(p1, p2):
    """Distanza euclidea; funziona sia per punti 2D che 3D (tuple della
    stessa lunghezza)."""
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def deproject_pixel_to_point(u, v, depth, fx, fy, cx, cy):
    """Modello pinhole inverso: pixel (u, v) + profondita' (metri) -> punto
    3D nel frame OTTICO della camera (x destra, y basso, z avanti —
    convenzione standard OpenCV/ROS)."""
    x = (u - cx) * depth / fx
    y = (v - cy) * depth / fy
    z = depth
    return x, y, z


def project_point_to_pixel(x, y, z, fx, fy, cx, cy):
    """Proiezione pinhole diretta: punto 3D nel frame ottico della camera ->
    pixel (u, v). None se il punto e' dietro la camera (z <= 0)."""
    if z <= 1e-6:
        return None
    u = fx * x / z + cx
    v = fy * y / z + cy
    return u, v


class CameraIntrinsics:
    """Estrae fx, fy, cx, cy dalla matrice K (row-major 3x3) di un messaggio
    sensor_msgs/CameraInfo. In ROS2 il campo si chiama 'k' (minuscolo)."""

    __slots__ = ("fx", "fy", "cx", "cy")

    def __init__(self, camera_info_msg):
        k = camera_info_msg.k
        self.fx = k[0]
        self.fy = k[4]
        self.cx = k[2]
        self.cy = k[5]


# ============================================================
# Depth
# ============================================================

def box_center_depth(depth_image, box, patch_frac=0.2, min_patch_px=3, min_valid_pixels=3):
    """Profondita' (mediana, in metri) su una PICCOLA porzione centrata sul
    box — non tutto il box, non un singolo pixel puro.

    Perche' non tutto il box: se il box non e' stretto attorno al soggetto
    (comune con YOLOE), i pixel di sfondo ai bordi contaminano la mediana,
    spostando la distanza calcolata verso quella dello sfondo.

    Perche' non un singolo pixel: i sensori depth hanno spesso buchi in
    punti isolati (rumore, riflessi) — un solo pixel sfortunato darebbe
    None anche con il soggetto perfettamente visibile li'.

    `patch_frac`: frazione della larghezza/altezza del box da campionare,
    centrata. Con patch_frac=0.2 su un box 100x200px, la patch e' 20x40px."""
    h, w = depth_image.shape[:2]
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    box_w, box_h = (x2 - x1), (y2 - y1)
    patch_w = max(box_w * patch_frac, min_patch_px)
    patch_h = max(box_h * patch_frac, min_patch_px)

    px1 = max(0, int(cx - patch_w / 2))
    px2 = min(w, int(cx + patch_w / 2))
    py1 = max(0, int(cy - patch_h / 2))
    py2 = min(h, int(cy + patch_h / 2))
    if px2 <= px1 or py2 <= py1:
        return None

    crop = depth_image[py1:py2, px1:px2].astype(np.float32)
    if depth_image.dtype == np.uint16:
        crop = crop / 1000.0

    valid = crop[(crop > 0.05) & np.isfinite(crop)]
    print(f"Min valid value: {valid.min() if valid.size > 0 else 'N/A'}, Max valid value: {valid.max() if valid.size > 0 else 'N/A'}, Valid pixel count: {valid.size}")
    if valid.size < min_valid_pixels:
        return None
    return float(np.median(valid))


def scale_box_to_depth(box, rgb_shape, depth_shape):
    """Riscala le coordinate di un box (in pixel dell'immagine RGB su cui e'
    girata la detection) alla risoluzione dell'immagine di depth, se le due
    hanno dimensioni diverse — comune con sensori ToF, spesso a
    risoluzione nettamente minore della camera RGB. Se le due risoluzioni
    combaciano, ritorna il box invariato."""
    rgb_h, rgb_w = rgb_shape[:2]
    depth_h, depth_w = depth_shape[:2]
    if (rgb_w, rgb_h) == (depth_w, depth_h):
        return box
    scale_x = depth_w / rgb_w
    scale_y = depth_h / rgb_h
    x1, y1, x2, y2 = box
    return (x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y)


# ============================================================
# Aspetto (ReID leggero) — stessa logica di yoloe_offline_reid.py
# ============================================================

def extract_appearance_embedding(frame_bgr, box, hist_bins=(8, 8, 8)):
    frame_h, frame_w = frame_bgr.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    x1, y1 = max(0, x1), max(0, y1)
    x2, y2 = min(frame_w, x2), min(frame_h, y2)
    if x2 <= x1 or y2 <= y1:
        return None
    crop = frame_bgr[y1:y2, x1:x2]
    hsv_crop = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv_crop], [0, 1, 2], None, list(hist_bins), [0, 180, 0, 256, 0, 256])
    hist = cv2.normalize(hist, None, alpha=1.0, norm_type=cv2.NORM_L1).flatten()
    return hist.astype(np.float32)


def embedding_similarity(e1, e2):
    """Similarita' in [-1, 1] (1 = identico). 0.0 se manca un embedding.
    Per l'istogramma HSV di extract_appearance_embedding — vedi
    neural_embedding_similarity per il nuovo embedding neurale (da
    detector_interfaces/BoxDetection.embedding, campo compilato lato yolo)."""
    if e1 is None or e2 is None:
        return 0.0
    return float(cv2.compareHist(e1, e2, cv2.HISTCMP_CORREL))


def embedding_from_msg(embedding_field):
    """Converte il campo ROS BoxDetection.embedding (float32[], vuoto se il
    DetectorNode non aveva un modello di embedding configurato) in un
    numpy array, o None se e' vuoto — cosi' il chiamante puo' trattare
    'array vuoto dal messaggio' e 'nessun embedding' con lo stesso
    controllo (`is None`), senza dover controllare la lunghezza ogni volta."""
    if not embedding_field:
        return None
    return np.array(embedding_field, dtype=np.float32)


def neural_embedding_similarity(e1, e2):
    """Similarita' coseno in [-1, 1] (1 = identico) tra due embedding
    NEURALI (da un modello di classificazione separato, penultimo layer —
    non l'istogramma HSV, che usa embedding_similarity/HISTCMP_CORREL).
    0.0 se manca un embedding."""
    if e1 is None or e2 is None:
        return 0.0
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    return float(np.dot(e1, e2)/(n1 * n2))

def rich_neural_embedding_similarity(e1, e2, w_cosine=1.0, w_euclidean=1.0 , w_magnitude=1.0, euclidean_scale=10.0 , magnitude_scale=10.0):
    """_Combina la similarità coseno e la distanza euclidea sui vettori grezzi (sensibile anche al modulo non solo alla direzione) in un unico punteggio di similarità. Ritorna 0.0 se manca un embedding."""
    
    if e1 is None or e2 is None:
        return 0.0
    
    n1 = np.linalg.norm(e1)
    n2 = np.linalg.norm(e2)
    
    cosine = float(np.dot(e1, e2) / (n1 * n2)) if n1 > 1e-8 and n2 > 1e-8 else 0.0
    euclidean_dist = np.linalg.norm(e1 - e2)
    euclidean_sim = 1.0 / (1.0 + euclidean_dist / euclidean_scale)  # Normalizza la distanza euclidea in un punteggio di similarità tra 0 e 1
    magnitude_diff = abs(n1 - n2)
    magnitude_sim = 1.0 / (1.0 + magnitude_diff / magnitude_scale)  # Normalizza la differenza di magnitudine in un punteggio di similarità
    total_weight = w_cosine + w_euclidean + w_magnitude
    
    print(f"Cosine: {cosine:.4f}, Euclidean Sim: {euclidean_sim:.4f}, Magnitude Sim: {magnitude_sim:.4f}, Total Weight: {total_weight:.4f}")
    
    
    return (w_cosine * cosine + w_euclidean * euclidean_sim + w_magnitude * magnitude_sim) / total_weight 


# ============================================================
# ROI come crop dell'immagine (design a camera singola: braccio)
# ============================================================

def compute_roi_crop_rect(image_width, image_height, intrinsics, fov_rad,
                           top_margin_frac=CROP_TOP_MARGIN_FRAC, bottom_margin_frac=CROP_BOTTOM_MARGIN_FRAC):
    """Calcola il rettangolo di crop (x1, y1, x2, y2) in pixel. NESSUNA TF
    coinvolta, su nessuno dei due assi:

    - LARGHEZZA: dagli intrinseci della camera (fx, cx) e dal FOV, come
      prima — mezza_larghezza_px = fx * tan(fov_rad / 2).

    - ALTEZZA: frazione FISSA dell'immagine, tagliata dall'alto e dal
      basso. Nessun significato fisico ("altezza da terra") — il target
      deve solo cadere DENTRO il crop; la distanza/range reale si verifica
      DOPO, sulla depth del box rilevato (box_center_depth in
      tracking_fsm.py), non sul crop stesso.

    Sempre calcolabile una volta noti gli intrinseci (a differenza della
    versione precedente, non dipende dalla posa del braccio in quel
    istante) — ritorna None solo se i margini configurati sono degeneri
    (es. margini che si sovrappongono)."""
    half_width_px = intrinsics.fx * math.tan(fov_rad / 2.0)
    x1 = max(0, int(intrinsics.cx - half_width_px))
    x2 = min(image_width, int(intrinsics.cx + half_width_px))

    y1 = max(0, int(image_height * top_margin_frac))
    y2 = min(image_height, int(image_height * (1.0 - bottom_margin_frac)))

    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)
