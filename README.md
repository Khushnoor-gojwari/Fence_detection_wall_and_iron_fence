# Pose-Based Fence / Wall Climb Detection (YOLOv8-Pose)

Watch a wall or fence in a camera feed and raise two events:

- **BOUNDARY CROSSED** — a person's feet move onto the restricted side of the
  fence/wall line you defined.
- **CLIMB DETECTED** — a person is up against the fence/wall and their body
  posture matches climbing (reaching up, feet lifted off the ground, body
  vertical, feet rising above the fence line).

## The problem

Watching a perimeter is a security task, but the naive approach — training a
plain object detector on "person climbing" boxes — is fragile. A person simply
standing in front of a fence produces almost the same bounding box as a person
climbing it, so an appearance-only detector fires constant false alarms.

This project instead reasons about **body geometry** using pose estimation, so
it can tell a real climb (hands gripping up high, feet lifted, body stacked
vertically against the surface) apart from someone just standing nearby.

## Why pose-based

We use **YOLOv8-Pose** (`yolov8n-pose.pt` or `yolov8s-pose.pt`), which returns
the 17 COCO body key-points per person (shoulders, hips, knees, ankles, wrists,
etc.). From those key-points we compute features a bounding box alone cannot
capture:

| Feature | Meaning |
|---|---|
| `near_fence` | Person's box is against the fence/wall line. |
| `hands_up` | A wrist is raised above the shoulder (reaching to grip the top). |
| `feet_lifted` | Ankles pulled up toward the hips, or a strongly bent knee (leg hooked on). |
| `ankle_above_line` | An ankle has risen above the fence line itself (feet over the top). |
| `body_vertical` | Torso is upright — climbing is a vertical effort. |

## How events are decided

**Boundary crossing.** The fence/wall is a line with two endpoints plus a marker
for the restricted side. Each person's feet point (mean of the confident ankles,
or the bottom-centre of the box as a fallback) is classified by the sign of the
2D cross-product against the line. If it sits on the restricted side (by a small
margin), that's a crossing. This works for horizontal, vertical, or diagonal
fence lines.

**Climb.** A **CLIMB** is flagged when:

```
person is against the fence  AND  body is vertical  AND  (hands raised OR feet lifted)
```

or immediately when an ankle rises above the fence line (feet climbing over the
top).

### Temporal confirmation

For video and webcam, each person is tracked with a stable ID and an event is
only declared after the condition holds for several consecutive frames
(`CROSS_CONFIRM_FRAMES` / `CLIMB_CONFIRM_FRAMES`). This removes single-frame
noise and ignores momentary poses (e.g. briefly reaching up near the fence).

## Project structure

```
fence_detection/
  fence_detection.py   # main script (boundary + crossing + climb logic)
  requirements.txt
  README.md
  models/              # YOLOv8-pose weights (auto-downloaded here on first run)
  data/                # put your test images / videos here
  output/              # annotated results are written here
```

## Setup

Uses the shared D-Fire virtual environment (Python 3.12). From the `D-Fire`
folder:

```bash
source venv/bin/activate
pip install -r fence_detection/requirements.txt
```

`ultralytics`, `torch`, `opencv-python` and `numpy` are already present in the
project venv, so this is usually a no-op. On first run the chosen pose model
(`yolov8n-pose.pt` by default) is downloaded automatically into `models/`.

## Defining the boundary

By default (`INTERACTIVE_BOUNDARY = True`) you draw the fence/wall line on the
first frame with the mouse:

1. Click point **A**, then point **B** — the fence/wall line.
2. Click a third point inside the **restricted** area — fixes which side is
   off-limits.
3. Press **ENTER** to confirm  (**R** to redo, **ESC** to cancel).

Prefer to skip drawing? Hard-code the line near the top of `fence_detection.py`
(pixel coordinates in the original frame resolution) and set
`INTERACTIVE_BOUNDARY = False`:

```python
BOUNDARY_A = (100, 300)
BOUNDARY_B = (1180, 300)
RESTRICTED_POINT = (640, 60)     # any point in the off-limits area
```

## Usage

Interactive menu (webcam / video / image / folder):

```bash
python fence_detection.py
```

Command line:

```bash
python fence_detection.py --source 0                    # live webcam
python fence_detection.py --source data/clip.mp4         # video file
python fence_detection.py --source data/photo.jpg        # single image
python fence_detection.py --source data/                 # folder of images
python fence_detection.py --model s                      # use yolov8s-pose (more accurate)
python fence_detection.py --source data/clip.mp4 --no-show   # headless
```

Annotated output (skeleton + per-person label, plus a banner when an event is
confirmed) is saved to `output/`.

### Preset input paths

If you would rather not pass `--source` each time, set the two constants near the
top of `fence_detection.py` and just run `python fence_detection.py`:

```python
VIDEO_PATH = os.path.join(DATA_DIR, "fence.mp4")        # e.g. data/test_video.mp4
IMAGE_PATH = os.path.join(DATA_DIR, "test_image.jpg")   # e.g. data/test_image.jpg
```

In the interactive menu, options 2 (video) and 3 (image) show these as the
default in brackets — press Enter to use them or type another path to override.

## Labels shown on screen

Per person:

| Label | Meaning | Color |
|---|---|---|
| `OK` | Person present, no event | green |
| `CROSSED` | Feet on the restricted side of the line | orange |
| `CLIMBING` | Climbing posture confirmed against the fence | red |

Plus a top banner: `BOUNDARY CROSSED` (orange) or `CLIMB DETECTED` (red), with
climb taking priority. The fence/wall line is drawn in magenta with a small
arrow pointing into the restricted side.

## Tuning

All thresholds live in the `CONFIG` block at the top of `fence_detection.py`.
Camera angle matters, so adjust these for your scene:

- `NEAR_FENCE_PX` — how close the person's box must come to the line to count as
  "against the fence".
- `HANDS_UP_FRAC` — how far above the shoulder a wrist must be to count as
  reaching up (fraction of box height).
- `FEET_LIFTED_FRAC` — how much the ankles must rise from their standing spot to
  count as lifted.
- `KNEE_BENT_DEG` — how bent a knee must be to count as a leg hooked onto the
  fence.
- `TORSO_VERTICAL_DEG` — how upright the torso must stay to count as climbing.
- `CROSS_MARGIN_PX` — pixels past the line before a crossing counts (anti-flicker).
- `CROSS_CONFIRM_FRAMES` / `CLIMB_CONFIRM_FRAMES` — higher = fewer false alarms
  but slower to react.

## Camera placement matters

The climb geometry (torso angle, hands-vs-shoulders, ankle-vs-hip) is derived
from **2D image geometry**, which best matches reality from a roughly
**side-on / eye-level** view of the fence. From a steep overhead or heavily
angled camera the geometry is foreshortened, so you may need to loosen or tighten
the thresholds above. The boundary-crossing test is robust to angle as long as
the fence line is drawn correctly on the frame.

## Notes and limitations

- Accuracy depends on camera height/angle and on key-point confidence; heavy
  occlusion or very small/distant people reduce reliability.
- The rules are geometric and interpretable by design. To tune for a specific
  deployment, the per-image run prints the computed metrics
  (`near_fence`, `hands_up`, `feet_lifted`, `torso_angle`, `fence_dist`) so you
  can fit thresholds to your own footage.
- The boundary is a straight line. For a curved or multi-segment perimeter you
  would extend `Boundary` to hold multiple segments or a polygon.
