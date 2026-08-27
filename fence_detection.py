"""
Pose-Based Fence / Wall Climb Detection using YOLOv8-Pose
=========================================================

What this does
--------------
Given a camera view of a wall or a fence, this tool does three things:

  1. BOUNDARY          -- you define the wall/fence as a line (the top edge of
                          the wall, or the line of the fence) directly on the
                          video. Everything on the far side of that line is the
                          "restricted" zone.
  2. BOUNDARY CROSSED  -- when a person's feet move onto the restricted side of
                          the line, we raise "BOUNDARY CROSSED".
  3. CLIMB DETECTED    -- when a person is up against the fence/wall and their
                          body posture matches climbing (reaching up with the
                          hands, feet lifted off the ground, body vertical, feet
                          rising above the fence line over time), we raise
                          "CLIMB DETECTED".

Why pose-based (not a plain object detector)?
---------------------------------------------
A "person climbing a fence" box looks almost identical to a person simply
standing in front of the fence, so an appearance-only detector cannot tell them
apart and fires constant false alarms. Climbing is defined by *body geometry*:
the hands reach up to grip, the feet leave the ground and rise, and the whole
body stacks vertically against the surface. Using the body key-points
(skeleton) we can measure that geometry directly.

Model
-----
YOLOv8-Pose (nano `yolov8n-pose.pt` or small `yolov8s-pose.pt`). It outputs the
17 COCO key-points per person:

    0 nose        5 left_shoulder   11 left_hip      15 left_ankle
    1 left_eye    6 right_shoulder  12 right_hip     16 right_ankle
    2 right_eye   7 left_elbow      13 left_knee
    3 left_ear    8 right_elbow     14 right_knee
    4 right_ear   9 left_wrist      10 right_wrist

Defining the boundary
----------------------
On the first frame you draw the fence/wall line:
    * click point A, then click point B          -> the fence/wall line
    * click a third point in the RESTRICTED area  -> which side is off-limits
    * press ENTER to confirm  (press R to redo,  ESC to cancel)
You can also hard-code the line via BOUNDARY_A / BOUNDARY_B / RESTRICTED_POINT
below and set INTERACTIVE_BOUNDARY = False to skip drawing.

Climb logic (per person)
------------------------
  * near_fence      : the person's box overlaps / sits against the fence line.
  * hands_up        : one or both wrists are raised above the shoulders
                      (reaching up to grip the top).
  * feet_lifted     : the ankles are raised well above their normal standing
                      position (knees pulled up / feet off the ground), or the
                      ankles rise above the fence line itself.
  * body_vertical   : the torso is upright (climbing is a vertical effort).
A CLIMB is flagged when the person is against the fence AND shows the climbing
posture (hands up and/or feet lifted with a vertical body).

Temporal confirmation
----------------------
Single frames are noisy, so for video/webcam each person is tracked (YOLO
tracker) and "BOUNDARY CROSSED" / "CLIMB DETECTED" are only declared after the
condition holds for a few consecutive frames. This ignores momentary poses
(e.g. someone briefly reaching up near the fence).

Usage
-----
    python fence_detection.py
        -> interactive menu (webcam / video / image / folder)

    python fence_detection.py --source 0                  # webcam
    python fence_detection.py --source path/to/video.mp4  # video file
    python fence_detection.py --source path/to/image.jpg  # single image
    python fence_detection.py --source data/              # folder of images
    python fence_detection.py --model s                   # use yolov8s-pose
    python fence_detection.py --no-show                   # headless (no window)
"""

import os
import argparse
import shutil
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# PATHS
# ----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")

for _d in (MODELS_DIR, DATA_DIR, OUTPUT_DIR):
    os.makedirs(_d, exist_ok=True)

# ----------------------------------------------------------------------------
# INPUT PATHS  (edit these, then just run `python fence_detection.py`)
# ----------------------------------------------------------------------------
# Set the video or image you want to test. Placing your files in the data/
# folder keeps the project self-contained. The interactive menu uses these as
# the default when you press Enter without typing a path.
VIDEO_PATH = os.path.join(DATA_DIR, "wall.mp4")        # e.g. data/test_video.mp4
IMAGE_PATH = os.path.join(DATA_DIR, "test_image.jpg")   # e.g. data/test_image.jpg

# ----------------------------------------------------------------------------
# BOUNDARY CONFIG
# ----------------------------------------------------------------------------
# If INTERACTIVE_BOUNDARY is True you draw the fence line on the first frame with
# the mouse. If False, the hard-coded line below is used (pixel coordinates in
# the ORIGINAL frame resolution).
INTERACTIVE_BOUNDARY = True

# Fallback / hard-coded boundary (used when INTERACTIVE_BOUNDARY is False, or as
# a default if drawing is skipped). A and B are the two ends of the fence/wall
# line; RESTRICTED_POINT is any point that lies in the off-limits area.
BOUNDARY_A = (100, 300)
BOUNDARY_B = (1180, 300)
RESTRICTED_POINT = (640, 60)     # the region "above"/"beyond" the line

# ----------------------------------------------------------------------------
# CONFIG  (tune these thresholds for your camera angle / scene)
# ----------------------------------------------------------------------------
# Which pose model to use by default: "n" (fast) or "s" (more accurate).
DEFAULT_MODEL_SIZE = "n"

# Detection / key-point confidence.
PERSON_CONF = 0.35          # min person detection confidence
KPT_CONF = 0.30             # min confidence for a key-point to be trusted
IMG_SIZE = 640              # inference resolution

# --- Boundary crossing ------------------------------------------------------
# The person's "feet" reference point (bottom-centre of the box) must sit this
# many pixels onto the restricted side before it counts as a crossing (avoids
# flicker right on the line).
CROSS_MARGIN_PX = 6

# --- Climb geometry thresholds ----------------------------------------------
# How close (in pixels) the person's box must come to the fence line to be
# considered "against the fence".
NEAR_FENCE_PX = 40

# Torso (spine) angle from the vertical axis, in degrees. Climbing keeps the
# body fairly upright, so we require the torso below this angle.
TORSO_VERTICAL_DEG = 45.0

# A wrist counts as "raised" when it is above the shoulder by at least this
# fraction of the person's box height.
HANDS_UP_FRAC = 0.05

# The feet are "lifted" when the ankles rise above their normal standing spot by
# at least this fraction of the box height (knees pulled up / feet off ground).
FEET_LIFTED_FRAC = 0.15

# Knee strongly bent (hip-knee-ankle angle below this) => leg pulled up onto the
# fence, a common climbing cue.
KNEE_BENT_DEG = 110.0

# --- Temporal confirmation (video / webcam only) ----------------------------
CROSS_CONFIRM_FRAMES = 4    # consecutive frames before confirming a crossing
CLIMB_CONFIRM_FRAMES = 5    # consecutive frames before confirming a climb
HISTORY_LEN = 15            # per-person rolling window length

# COCO skeleton links for drawing.
SKELETON = [
    (5, 7), (7, 9), (6, 8), (8, 10),          # arms
    (5, 6), (5, 11), (6, 12), (11, 12),       # torso
    (11, 13), (13, 15), (12, 14), (14, 16),   # legs
    (0, 5), (0, 6),                           # head to shoulders
]

# Colors (BGR)
COLOR_OK = (0, 200, 0)          # person present, no event
COLOR_CROSS = (0, 165, 255)     # boundary crossed
COLOR_CLIMB = (0, 0, 255)       # climb detected
COLOR_FENCE = (255, 0, 255)     # the fence / wall boundary line


# ----------------------------------------------------------------------------
# BOUNDARY GEOMETRY
# ----------------------------------------------------------------------------
class Boundary:
    """A fence/wall line plus a marker for which side is restricted.

    The line is defined by two endpoints A and B. A third reference point that
    lies in the off-limits area fixes the "restricted" side. We then classify
    any query point by the sign of the 2D cross product relative to A->B; if it
    matches the sign of the restricted reference point, the query point is on
    the restricted side.
    """

    def __init__(self, a, b, restricted_point):
        self.a = np.array(a, dtype=float)
        self.b = np.array(b, dtype=float)
        self.restricted_point = np.array(restricted_point, dtype=float)
        # Sign of the restricted reference point defines "inside" the zone.
        self.restricted_sign = np.sign(self._cross(self.restricted_point)) or 1.0

    def _cross(self, p):
        """2D cross product of (B-A) x (P-A). Sign tells us the side of P."""
        ab = self.b - self.a
        ap = np.asarray(p, dtype=float) - self.a
        return ab[0] * ap[1] - ab[1] * ap[0]

    def signed_distance(self, p):
        """Perpendicular distance from p to the line (unsigned pixels)."""
        ab = self.b - self.a
        denom = np.linalg.norm(ab) + 1e-6
        return abs(self._cross(p)) / denom

    def is_restricted(self, p, margin_px=0.0):
        """True if point p is on the restricted side by at least margin_px."""
        ab = self.b - self.a
        denom = np.linalg.norm(ab) + 1e-6
        signed = self._cross(p) / denom            # signed perpendicular dist
        return (np.sign(signed) == self.restricted_sign) and (abs(signed) >= margin_px)

    def y_on_line(self, x):
        """Return the line's y for a given x (used to compare ankle height)."""
        ax, ay = self.a
        bx, by = self.b
        if abs(bx - ax) < 1e-6:
            return None                            # vertical line, undefined y(x)
        t = (x - ax) / (bx - ax)
        return ay + t * (by - ay)

    def draw(self, frame):
        a = tuple(self.a.astype(int))
        b = tuple(self.b.astype(int))
        cv2.line(frame, a, b, COLOR_FENCE, 3)
        cv2.circle(frame, a, 6, COLOR_FENCE, -1)
        cv2.circle(frame, b, 6, COLOR_FENCE, -1)
        # small arrow pointing into the restricted side
        mid = ((self.a + self.b) / 2.0)
        rp = self.restricted_point
        direction = rp - mid
        n = np.linalg.norm(direction)
        if n > 1e-6:
            tip = mid + direction / n * 40.0
            cv2.arrowedLine(frame, tuple(mid.astype(int)), tuple(tip.astype(int)),
                            COLOR_FENCE, 2, tipLength=0.3)
        cv2.putText(frame, "FENCE / WALL", (a[0], max(20, a[1] - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FENCE, 2)


def draw_boundary_interactive(first_frame):
    """Let the user click the fence line (A, B) and the restricted-side point.

    Returns a Boundary, or None if cancelled.
    """
    clicks = []
    display = first_frame.copy()
    win = "Draw fence/wall line  (click A, B, then restricted side)"

    def redraw():
        img = first_frame.copy()
        for i, p in enumerate(clicks):
            cv2.circle(img, p, 6, COLOR_FENCE, -1)
            cv2.putText(img, ["A", "B", "restricted"][i], (p[0] + 8, p[1]),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLOR_FENCE, 2)
        if len(clicks) >= 2:
            cv2.line(img, clicks[0], clicks[1], COLOR_FENCE, 3)
        hint = ("Click A, B (fence line), then a point in the RESTRICTED area. "
                "ENTER=confirm  R=redo  ESC=cancel")
        cv2.putText(img, hint, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (255, 255, 255), 2)
        return img

    def on_mouse(event, x, y, flags, param):
        nonlocal display
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 3:
            clicks.append((x, y))
            display = redraw()

    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    display = redraw()

    while True:
        cv2.imshow(win, display)
        key = cv2.waitKey(20) & 0xFF
        if key == 27:                       # ESC -> cancel
            cv2.destroyWindow(win)
            return None
        if key in (ord("r"), ord("R")):     # redo
            clicks.clear()
            display = redraw()
        if key in (13, 10):                 # ENTER -> confirm
            if len(clicks) >= 2:
                a, b = clicks[0], clicks[1]
                rp = clicks[2] if len(clicks) >= 3 else (
                    # default restricted side: the perpendicular "up" from mid
                    int((a[0] + b[0]) / 2), int(min(a[1], b[1]) - 60))
                cv2.destroyWindow(win)
                return Boundary(a, b, rp)
    # unreachable


# ----------------------------------------------------------------------------
# KEY-POINT GEOMETRY HELPERS
# ----------------------------------------------------------------------------
def _valid(pt_conf):
    """A key-point is usable if its confidence clears the threshold."""
    return pt_conf is not None and pt_conf >= KPT_CONF


def _midpoint(kpts, confs, i, j):
    """Mean of two key-points, using whichever of the pair is confident."""
    a_ok, b_ok = _valid(confs[i]), _valid(confs[j])
    if a_ok and b_ok:
        return (kpts[i] + kpts[j]) / 2.0
    if a_ok:
        return kpts[i].copy()
    if b_ok:
        return kpts[j].copy()
    return None


def _angle_from_vertical(p_low, p_high):
    """Angle (deg) of the vector p_low->p_high from the vertical axis.

    0 -> vertically aligned (upright), 90 -> horizontally aligned (flat).
    """
    if p_low is None or p_high is None:
        return None
    dx = float(p_high[0] - p_low[0])
    dy = float(p_high[1] - p_low[1])
    return float(np.degrees(np.arctan2(abs(dx), abs(dy) + 1e-6)))


def _joint_angle(a, b, c):
    """Interior angle (deg) at joint b formed by segments b->a and b->c."""
    if a is None or b is None or c is None:
        return None
    v1 = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    v2 = np.asarray(c, dtype=float) - np.asarray(b, dtype=float)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return None
    cosang = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(cosang)))


def feet_point(kpts, confs, bbox):
    """Best estimate of where the person's feet are (for crossing tests).

    Prefer the mean of the confident ankles; fall back to the bottom-centre of
    the bounding box.
    """
    ankle = _midpoint(kpts, confs, 15, 16)
    if ankle is not None:
        return ankle
    x1, y1, x2, y2 = bbox
    return np.array([(x1 + x2) / 2.0, y2], dtype=float)


# ----------------------------------------------------------------------------
# CLIMB POSTURE ANALYSIS
# ----------------------------------------------------------------------------
def analyze_climb(kpts, confs, bbox, boundary):
    """Decide whether a single person's posture looks like climbing.

    Returns a dict with:
        is_climb  : bool (raw, single-frame decision)
        near      : bool (person is against the fence line)
        metrics   : dict of computed features (for overlay / debug)
    """
    x1, y1, x2, y2 = bbox
    box_h = max(1.0, y2 - y1)

    shoulder = _midpoint(kpts, confs, 5, 6)
    hip = _midpoint(kpts, confs, 11, 12)
    ankle = _midpoint(kpts, confs, 15, 16)

    # Torso orientation (upright vs tilted).
    torso_angle = _angle_from_vertical(hip, shoulder)
    body_vertical = (torso_angle is not None) and (torso_angle <= TORSO_VERTICAL_DEG)

    # Hands raised: a wrist above the shoulder line by a fraction of box height.
    hands_up = False
    if shoulder is not None:
        for w_idx in (9, 10):
            if _valid(confs[w_idx]) and (shoulder[1] - kpts[w_idx][1]) > HANDS_UP_FRAC * box_h:
                hands_up = True
                break

    # Feet lifted: ankles pulled up toward the hips (small hip->ankle drop), or
    # a strongly bent knee (leg hooked onto the fence).
    feet_lifted = False
    hip_ankle_drop = None
    if hip is not None and ankle is not None:
        hip_ankle_drop = float(ankle[1] - hip[1])         # +ve => ankle below hip
        # When standing this drop is large (~box_h*0.5). When a foot is lifted
        # onto the fence the drop shrinks.
        if hip_ankle_drop < (0.5 - FEET_LIFTED_FRAC) * box_h:
            feet_lifted = True

    knee_l = _joint_angle(kpts[11], kpts[13], kpts[15]) if all(
        _valid(confs[k]) for k in (11, 13, 15)) else None
    knee_r = _joint_angle(kpts[12], kpts[14], kpts[16]) if all(
        _valid(confs[k]) for k in (12, 14, 16)) else None
    knee_angles = [k for k in (knee_l, knee_r) if k is not None]
    knee_angle = min(knee_angles) if knee_angles else None
    if knee_angle is not None and knee_angle < KNEE_BENT_DEG:
        feet_lifted = True

    # Ankles risen above the fence line itself (feet climbed over the top).
    ankle_above_line = False
    if ankle is not None:
        line_y = boundary.y_on_line(float(ankle[0]))
        if line_y is not None and ankle[1] < line_y:
            ankle_above_line = True

    # Is the person against the fence? Distance from the box to the line.
    box_corners = [np.array([x1, y1]), np.array([x2, y1]),
                   np.array([x1, y2]), np.array([x2, y2]),
                   np.array([(x1 + x2) / 2.0, (y1 + y2) / 2.0])]
    min_dist = min(boundary.signed_distance(c) for c in box_corners)
    near = min_dist <= NEAR_FENCE_PX

    metrics = {
        "torso_angle": torso_angle,
        "knee_angle": knee_angle,
        "hip_ankle_drop": hip_ankle_drop,
        "fence_dist": min_dist,
    }

    # --- Decision ------------------------------------------------------------
    # Climbing = up against the fence, body vertical, and either the hands are
    # reaching up to grip or the feet are lifted onto the surface. Feet climbing
    # above the line is an immediate strong cue.
    climbing_posture = body_vertical and (hands_up or feet_lifted)
    is_climb = (near and climbing_posture) or ankle_above_line

    return {
        "is_climb": bool(is_climb),
        "near": bool(near),
        "hands_up": hands_up,
        "feet_lifted": feet_lifted,
        "ankle_above_line": ankle_above_line,
        "metrics": metrics,
    }


# ----------------------------------------------------------------------------
# DRAWING
# ----------------------------------------------------------------------------
def draw_person(frame, kpts, confs, bbox, label, color, track_id=None):
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

    for a, b in SKELETON:
        if _valid(confs[a]) and _valid(confs[b]):
            pa = (int(kpts[a][0]), int(kpts[a][1]))
            pb = (int(kpts[b][0]), int(kpts[b][1]))
            cv2.line(frame, pa, pb, color, 2)
    for i in range(len(kpts)):
        if _valid(confs[i]):
            cv2.circle(frame, (int(kpts[i][0]), int(kpts[i][1])), 3, color, -1)

    tag = f"{label}" if track_id is None else f"ID{track_id} {label}"
    (tw, th), _ = cv2.getTextSize(tag, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (x1, max(0, y1 - th - 8)), (x1 + tw + 6, y1), color, -1)
    cv2.putText(frame, tag, (x1 + 3, max(12, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


def draw_alert_banner(frame, text, color):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 45), color, -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    cv2.putText(frame, text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)


def decide_display(crossed, climbing):
    """Map event flags to the on-screen (label, color) for a person."""
    if climbing:
        return "CLIMBING", COLOR_CLIMB
    if crossed:
        return "CROSSED", COLOR_CROSS
    return "OK", COLOR_OK


# ----------------------------------------------------------------------------
# MODEL LOADING
# ----------------------------------------------------------------------------
def load_model(size):
    """Load YOLOv8-pose, keeping weights inside the models/ folder."""
    size = size.lower().strip()
    if size not in ("n", "s"):
        print(f"Unknown model size '{size}', falling back to 'n'.")
        size = "n"
    model_name = f"yolov8{size}-pose.pt"
    weights = os.path.join(MODELS_DIR, model_name)

    if not os.path.exists(weights):
        print(f"'{model_name}' not found in models/. Downloading official weights...")
        _ = YOLO(model_name)
        if os.path.exists(model_name) and os.path.abspath(model_name) != os.path.abspath(weights):
            shutil.move(model_name, weights)

    print(f"Loading pose model: {weights}")
    model = YOLO(weights)
    print("Model loaded.")
    return model


# ----------------------------------------------------------------------------
# EXTRACT PER-PERSON DATA FROM A YOLO RESULT
# ----------------------------------------------------------------------------
def extract_people(result):
    """Yield (track_id, kpts(17,2), confs(17,), bbox(4,)) for each person."""
    people = []
    if result.keypoints is None or result.boxes is None:
        return people

    kpts_xy = result.keypoints.xy.cpu().numpy()            # (N, 17, 2)
    kpts_cf = (result.keypoints.conf.cpu().numpy()
               if result.keypoints.conf is not None
               else np.ones(kpts_xy.shape[:2]))            # (N, 17)
    boxes = result.boxes.xyxy.cpu().numpy()                # (N, 4)
    box_cf = result.boxes.conf.cpu().numpy()               # (N,)
    ids = (result.boxes.id.cpu().numpy().astype(int)
           if result.boxes.id is not None else [None] * len(boxes))

    for i in range(len(boxes)):
        if box_cf[i] < PERSON_CONF:
            continue
        people.append((ids[i], kpts_xy[i], kpts_cf[i], boxes[i]))
    return people


# ----------------------------------------------------------------------------
# BUILD THE BOUNDARY FOR A GIVEN FRAME
# ----------------------------------------------------------------------------
def build_boundary(first_frame, show):
    """Return a Boundary either drawn interactively or from the config."""
    if INTERACTIVE_BOUNDARY and show and first_frame is not None:
        b = draw_boundary_interactive(first_frame)
        if b is not None:
            print(f"Boundary set: A={tuple(b.a.astype(int))} "
                  f"B={tuple(b.b.astype(int))} "
                  f"restricted~{tuple(b.restricted_point.astype(int))}")
            return b
        print("Boundary drawing cancelled/skipped; using config boundary.")
    return Boundary(BOUNDARY_A, BOUNDARY_B, RESTRICTED_POINT)


# ----------------------------------------------------------------------------
# RUN ON A STREAM (video file or webcam)
# ----------------------------------------------------------------------------
def run_stream(model, source, show=True, out_name="fence_detection_output.mp4"):
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print(f"Error: could not open source '{source}'.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = int(fps) if fps and fps > 0 else 30
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Grab the first frame so the boundary can be drawn on the real scene.
    ret, first_frame = cap.read()
    if not ret:
        print("Error: could not read the first frame.")
        cap.release()
        return
    boundary = build_boundary(first_frame, show)

    out_path = os.path.join(OUTPUT_DIR, out_name)
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))

    # Per-track rolling history of the raw single-frame flags.
    cross_hist = defaultdict(lambda: deque(maxlen=HISTORY_LEN))
    climb_hist = defaultdict(lambda: deque(maxlen=HISTORY_LEN))

    print("Processing... press 'q' in the window to stop.")
    # Rewind so the first frame is also processed.
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        results = model.track(
            frame, persist=True, conf=PERSON_CONF, imgsz=IMG_SIZE,
            verbose=False, classes=[0]
        )
        result = results[0]

        boundary.draw(frame)
        any_cross = False
        any_climb = False

        for track_id, kpts, confs, bbox in extract_people(result):
            fp = feet_point(kpts, confs, bbox)
            raw_cross = boundary.is_restricted(fp, margin_px=CROSS_MARGIN_PX)
            climb_info = analyze_climb(kpts, confs, bbox, boundary)
            raw_climb = climb_info["is_climb"]

            key = track_id if track_id is not None else -1
            cross_hist[key].append(raw_cross)
            climb_hist[key].append(raw_climb)

            recent_cross = list(cross_hist[key])[-CROSS_CONFIRM_FRAMES:]
            recent_climb = list(climb_hist[key])[-CLIMB_CONFIRM_FRAMES:]
            crossed = (len(recent_cross) >= CROSS_CONFIRM_FRAMES and all(recent_cross))
            climbing = (len(recent_climb) >= CLIMB_CONFIRM_FRAMES and all(recent_climb))

            any_cross = any_cross or crossed
            any_climb = any_climb or climbing

            label, color = decide_display(crossed, climbing)
            draw_person(frame, kpts, confs, bbox, label, color, track_id)

        if any_climb:
            draw_alert_banner(frame, "CLIMB DETECTED", COLOR_CLIMB)
        elif any_cross:
            draw_alert_banner(frame, "BOUNDARY CROSSED", COLOR_CROSS)

        writer.write(frame)
        if show:
            disp = cv2.resize(frame, (960, 540)) if w > 960 else frame
            cv2.imshow("Fence / Wall Climb Detection", disp)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    cap.release()
    writer.release()
    if show:
        cv2.destroyAllWindows()
    print(f"Done. Saved: {out_path}")


# ----------------------------------------------------------------------------
# RUN ON A SINGLE IMAGE
# ----------------------------------------------------------------------------
def run_image(model, img_path, show=True, boundary=None):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"Error: could not read image '{img_path}'.")
        return

    if boundary is None:
        boundary = build_boundary(frame, show)

    result = model(frame, conf=PERSON_CONF, imgsz=IMG_SIZE, verbose=False, classes=[0])[0]

    boundary.draw(frame)
    any_cross = False
    any_climb = False

    for _tid, kpts, confs, bbox in extract_people(result):
        fp = feet_point(kpts, confs, bbox)
        crossed = boundary.is_restricted(fp, margin_px=CROSS_MARGIN_PX)
        climb_info = analyze_climb(kpts, confs, bbox, boundary)
        climbing = climb_info["is_climb"]

        any_cross = any_cross or crossed
        any_climb = any_climb or climbing

        m = climb_info["metrics"]
        ta = f"{m['torso_angle']:.0f}" if m["torso_angle"] is not None else "-"
        print(f"  person: crossed={crossed}  climbing={climbing}  "
              f"near_fence={climb_info['near']}  hands_up={climb_info['hands_up']}  "
              f"feet_lifted={climb_info['feet_lifted']}  torso_angle={ta}  "
              f"fence_dist={m['fence_dist']:.0f}")

        label, color = decide_display(crossed, climbing)
        draw_person(frame, kpts, confs, bbox, label, color)

    if any_climb:
        draw_alert_banner(frame, "CLIMB DETECTED", COLOR_CLIMB)
    elif any_cross:
        draw_alert_banner(frame, "BOUNDARY CROSSED", COLOR_CROSS)

    out_path = os.path.join(OUTPUT_DIR, "output_" + os.path.basename(img_path))
    cv2.imwrite(out_path, frame)
    print(f"Saved: {out_path}")

    if show:
        cv2.imshow("Fence / Wall Climb Detection", frame)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run_folder(model, folder, show=False):
    exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    imgs = sorted(f for f in os.listdir(folder) if f.lower().endswith(exts))
    if not imgs:
        print(f"No images found in '{folder}'.")
        return

    # Draw one boundary on the first image and reuse it for the whole folder
    # (assumes the folder is one camera / scene).
    first = cv2.imread(os.path.join(folder, imgs[0]))
    boundary = build_boundary(first, show) if first is not None else None

    print(f"Running on {len(imgs)} image(s) in {folder} ...")
    for name in imgs:
        print(f"\n{name}")
        run_image(model, os.path.join(folder, name), show=show, boundary=boundary)


# ----------------------------------------------------------------------------
# CLI / INTERACTIVE ENTRY POINT
# ----------------------------------------------------------------------------
def interactive(model):
    print("\nSelect input source:")
    print("  1) Live webcam")
    print("  2) Video file")
    print("  3) Single image")
    print("  4) Folder of images")
    choice = input("Enter choice (1-4): ").strip()

    if choice == "1":
        run_stream(model, 0, show=True)
    elif choice == "2":
        path = input(f"Enter video path [{VIDEO_PATH}]: ").strip() or VIDEO_PATH
        if not os.path.exists(path):
            print(f"Video not found: {path}")
            return
        run_stream(model, path, show=True,
                   out_name=os.path.splitext(os.path.basename(path))[0] + "_fence.mp4")
    elif choice == "3":
        path = input(f"Enter image path [{IMAGE_PATH}]: ").strip() or IMAGE_PATH
        if not os.path.exists(path):
            print(f"Image not found: {path}")
            return
        run_image(model, path, show=True)
    elif choice == "4":
        path = input(f"Enter folder path [{DATA_DIR}]: ").strip() or DATA_DIR
        if not os.path.isdir(path):
            print(f"Folder not found: {path}")
            return
        run_folder(model, path, show=False)
    else:
        print("Invalid choice.")


def main():
    parser = argparse.ArgumentParser(
        description="Pose-based fence/wall climb detection (YOLOv8-Pose)")
    parser.add_argument("--source", help="0 for webcam, or path to video/image/folder")
    parser.add_argument("--model", default=DEFAULT_MODEL_SIZE, choices=["n", "s"],
                        help="YOLOv8-pose size: n (fast) or s (accurate)")
    parser.add_argument("--no-show", action="store_true", help="run headless (no window)")
    args = parser.parse_args()

    model = load_model(args.model)
    show = not args.no_show

    if args.source is None:
        interactive(model)
        return

    src = args.source
    if src == "0" or src == "1":
        run_stream(model, int(src), show=show)
    elif os.path.isdir(src):
        run_folder(model, src, show=show)
    elif src.lower().endswith((".jpg", ".jpeg", ".png", ".bmp", ".webp")):
        run_image(model, src, show=show)
    else:
        run_stream(model, src, show=show,
                   out_name=os.path.splitext(os.path.basename(src))[0] + "_fence.mp4")


if __name__ == "__main__":
    main()
