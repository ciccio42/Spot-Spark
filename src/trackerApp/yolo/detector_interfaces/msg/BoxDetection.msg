# A single box detected by YOLOE, in pixel coordinates of the frame passed
# in the request (xyxy convention: x1,y1 = top-left corner, x2,y2 =
# bottom-right corner), same format as ultralytics box.xyxy().

float32 x1
float32 y1
float32 x2
float32 y2
float32 score
string class_name
int32 track_id   # ID assigned by BoT-SORT (persistent across consecutive calls
                  # with persist=True). -1 if the detection has no associated track
                  # (it happens: not every detection is necessarily attached to
                  # a track in a given frame) or if tracking is not active.

float32[] embedding  # Appearance feature vector (from a separate classification
                      # model, e.g. yolo11n-cls.pt, penultimate layer) — used by
                      # OUR ReID in demo_package (cosine comparison), independent
                      # of BoT-SORT. EMPTY array if embedding_model_path is not
                      # configured on the DetectorNode side — the caller must handle
                      # this case, not assume it is always populated.
