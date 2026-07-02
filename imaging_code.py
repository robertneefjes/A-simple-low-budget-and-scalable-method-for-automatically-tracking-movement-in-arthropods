#!/usr/bin/env python3

import time
import csv
import subprocess
from pathlib import Path
from datetime import datetime

import cv2
import numpy as np
from picamera2 import Picamera2
from libcamera import controls

# =====================
# USER SETTINGS
# =====================

CAPTURE_INTERVAL_SECONDS = 10 * 60      # 10 minutes between frames
IMAGE_SAVE_INTERVAL_SECONDS = 30 * 60   # 30 minutes between saved JPEGs
PREVIEW_SCALE = 0.25                    # scale factor for preview windows

ROWS = 16
COLS = 6

BLUR_KERNEL_SIZE = (3, 3)      # blur to reduce noise

# Different diff thresholds for visible vs IR
PIX_DIFF_THRESHOLD_VISIBLE = 10
PIX_DIFF_THRESHOLD_IR = 10

# Manual focus settings
LENS_POSITION = 0.600          # fixed manual focus for all cameras
MIN_LENS = 0.0
MAX_LENS = 20.0                # safety clamp; real max may be lower

# Camera indices on the Pi (adjust if your order is different)
CAMERA_CONFIGS = [
    {"index": 0, "label": "cam0"},
    {"index": 1, "label": "cam1"},
]

# =====================
# BACKUP SETTINGS
# =====================

BACKUP_ENABLED = True
RCLONE_REMOTE = "antbackup"           # name of your rclone remote
REMOTE_BASE_DIR = "ant_experiment"    # base folder on Google Drive

# =====================
# WIFI CHECK SETTINGS
# =====================

WIFI_CHECK_ENABLED = True
WIFI_CHECK_INTERVAL_SECONDS = 5 * 60    # every 5 minutes
WIFI_CHECK_HOST = "8.8.8.8"             # or your router's IP, e.g. "192.168.1.1"
# NOTE: this uses sudo and assumes no password prompt; adjust if needed
WIFI_RECONNECT_SHELL = "sudo ifconfig wlan0 down && sleep 3 && sudo ifconfig wlan0 up"

# =====================
# EXPERIMENT METADATA (filled at startup)
# =====================

DEVICE_ID = 1                 # which Raspberry Pi: 1 or 2
DAY_START_HOUR = 8            # default "day" start hour
NIGHT_START_HOUR = 20         # default "day" end hour (= night start)
CAMERA_TREATMENTS = {}        # filled interactively, e.g. {"cam0": "control", "cam1": "treatment"}

# =====================
# GLOBAL OUTPUT ROOT
# =====================

RUN_ID = datetime.now().strftime("%Y%m%d_%H%M%S")
ROOT_BASE_DIR = Path.home() / "ant_experiment" / RUN_ID
ROOT_BASE_DIR.mkdir(parents=True, exist_ok=True)


# =====================
# CAMERA MONITOR CLASS
# =====================

class CameraMonitor:
    """
    Holds all state for one camera:
      - Picamera2 instance
      - ROI grid parameters (crop rectangle) and ROIs
      - previous frame (blurred grayscale)
      - CSV writer and image directory
      - last image save time
      - experiment metadata (device, treatment, ROI labels/codes)
    """

    def __init__(self, cam_index: int, label: str):
        self.index = cam_index
        self.label = label

        # Per-camera directories and CSV
        self.base_dir = ROOT_BASE_DIR / label
        self.image_dir = self.base_dir / "images"
        self.image_dir.mkdir(parents=True, exist_ok=True)

        self.csv_path = self.base_dir / "movement_log.csv"
        self.csv_file = open(self.csv_path, "w", newline="")
        self.csv_writer = csv.writer(self.csv_file)
        # CSV header will be written after metadata (set_metadata) is called
        # because ROI codes depend on treatment, device, date.

        # Camera setup
        self.picam = Picamera2(cam_index)
        config = self.picam.create_still_configuration(main={"format": "RGB888"})
        self.picam.configure(config)
        self.picam.start()

        print(f"[{self.label}] Warming up camera...")
        time.sleep(2.0)

        # Set manual focus and enable AE/AWB initially
        lp = float(max(MIN_LENS, min(MAX_LENS, LENS_POSITION)))
        self.picam.set_controls({
            "AfMode": controls.AfModeEnum.Manual,
            "AfTrigger": controls.AfTriggerEnum.Cancel,
            "LensPosition": lp,
            "AeEnable": True,
            "AwbEnable": True,
        })
        print(f"[{self.label}] Manual focus set to LensPosition={lp:.3f}, AE/AWB enabled for initial lock.")
        time.sleep(2.0)

        # First frame and metadata (for initial exposure/white balance lock)
        first_rgb = self.picam.capture_array("main")
        meta = self.picam.capture_metadata()

        self._lock_exposure_from_metadata(meta)
        print(f"[{self.label}] Initial exposure/white balance locked from metadata.")

        # Frame size and default crop
        self.frame_height, self.frame_width = first_rgb.shape[:2]

        # Crop rectangle parameters (for ROI tool)
        self.crop_x = 0
        self.crop_y = 0
        self.crop_w = self.frame_width
        self.crop_h = self.frame_height

        self.set_default_crop()   # sets crop_x/y/w/h and ROIS

        # Row layout factors: 0..1 positions within the crop (top to bottom)
        self.row_factors = [r / ROWS for r in range(ROWS + 1)]
        self.update_rois()

        # Previous frame (blurred grayscale) for motion diff
        prev_gray = cv2.cvtColor(first_rgb, cv2.COLOR_RGB2GRAY)
        self.prev_gray = cv2.GaussianBlur(prev_gray, BLUR_KERNEL_SIZE, 0)

        # For image-saving interval
        self.last_image_save = time.time()

        # Metadata placeholders
        self.device_id = None
        self.treatment = None
        self.run_date = None
        self.roi_labels = [str(i) for i in range(ROWS * COLS)]
        self.roi_codes = []

    def _lock_exposure_from_metadata(self, meta):
        """
        Lock exposure/gain/AWB based on metadata.
        Does NOT touch focus (lens position).
        """
        controls_dict = {
            "AeEnable": False,
            "AwbEnable": False,
        }

        if meta.get("ExposureTime") is not None:
            controls_dict["ExposureTime"] = int(meta["ExposureTime"])
        controls_dict["AnalogueGain"] = 1.0
        if meta.get("ColourGains") is not None:
            controls_dict["ColourGains"] = meta["ColourGains"]

        self.picam.set_controls(controls_dict)

    def set_default_crop(self):
        """
        Set crop rectangle to a centered 3:2 region and update ROIs.
        """
        target_aspect = 3 / 2
        fw, fh = self.frame_width, self.frame_height

        if fw / fh > target_aspect:
            new_width = int(fh * target_aspect)
            new_height = fh
        else:
            new_width = fw
            new_height = int(fw / target_aspect)

        self.crop_w = new_width
        self.crop_h = new_height
        self.crop_x = (fw - new_width) // 2
        self.crop_y = (fh - new_height) // 2

        # Reset row factors to equal rows
        self.row_factors = [r / ROWS for r in range(ROWS + 1)]
        self.update_rois()

    def update_rois(self):
        """
        Recompute ROIS list from current crop rectangle and row_factors.
        row_factors define relative positions (0..1) of horizontal boundaries.
        """
        x0 = int(self.crop_x)
        y0 = int(self.crop_y)
        w = int(self.crop_w)
        h = int(self.crop_h)

        roi_w = max(1, w // COLS)

        # Ensure row_factors exist and are sorted
        if not hasattr(self, "row_factors") or len(self.row_factors) != ROWS + 1:
            self.row_factors = [r / ROWS for r in range(ROWS + 1)]
        self.row_factors = sorted(self.row_factors)
        # Clamp to [0,1]
        self.row_factors[0] = 0.0
        self.row_factors[-1] = 1.0

        # Compute absolute y positions
        row_positions = [y0 + int(h * f) for f in self.row_factors]

        rois = []
        for r in range(ROWS):
            y1 = row_positions[r]
            y2 = row_positions[r + 1]
            # enforce at least 1 px height
            if y2 <= y1:
                y2 = y1 + 1

            for c in range(COLS):
                x1 = x0 + c * roi_w
                x2 = x1 + roi_w
                rois.append((x1, y1, x2, y2))
        self.ROIS = rois

    def shift_crop(self, dx: int, dy: int):
        """
        Move the crop rectangle by (dx, dy), clamped to frame bounds.
        """
        fw, fh = self.frame_width, self.frame_height

        self.crop_x += dx
        self.crop_y += dy

        # Clamp so crop stays inside frame
        self.crop_x = max(0, min(self.crop_x, fw - self.crop_w))
        self.crop_y = max(0, min(self.crop_y, fh - self.crop_h))

        self.update_rois()

    def scale_crop(self, dw: int, dh: int):
        """
        Change crop width/height by (dw, dh), clamped to frame bounds
        and a reasonable minimum size.
        """
        fw, fh = self.frame_width, self.frame_height
        min_w, min_h = 100, 100

        new_w = self.crop_w + dw
        new_h = self.crop_h + dh

        if new_w < min_w:
            new_w = min_w
        if new_h < min_h:
            new_h = min_h

        if new_w > fw:
            new_w = fw
        if new_h > fh:
            new_h = fh

        # Also ensure crop_x + new_w <= fw, crop_y + new_h <= fh
        if self.crop_x + new_w > fw:
            self.crop_x = fw - new_w
        if self.crop_y + new_h > fh:
            self.crop_y = fh - new_h

        self.crop_w = new_w
        self.crop_h = new_h

        self.update_rois()

    def recalibrate_for_light_change(self, mode_label: str):
        """
        Recalculate exposure/gain/AWB for new lighting (e.g. visible vs IR),
        but KEEP the current focus (lens position).
        """
        print(f"[{self.label}] Recalibrating exposure for new light mode: {mode_label}")

        # Enable auto exposure and auto white balance temporarily
        self.picam.set_controls({
            "AeEnable": True,
            "AwbEnable": True,
            # Focus remains manual; we don't touch LensPosition here.
        })

        # Let auto settle a bit
        time.sleep(2.0)

        # Capture a frame and get its metadata
        _ = self.picam.capture_array("main")
        meta = self.picam.capture_metadata()

        # Lock new exposure/gain/AWB
        self._lock_exposure_from_metadata(meta)
        print(f"[{self.label}] Exposure/white balance re-locked for {mode_label}.")

        # Refresh prev_gray to avoid huge diff after exposure jump
        frame_rgb = self.picam.capture_array("main")
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        self.prev_gray = cv2.GaussianBlur(frame_gray, BLUR_KERNEL_SIZE, 0)

    def set_metadata(self, device_id: int, treatment: str, run_date: str):
        """
        Set experiment metadata and generate ROI labels and unique codes.

        ROI labels follow per-camera mapping:

        cam0:
          index 5  -> A1
          index 95 -> A16
          index 0  -> F1
          index 90 -> F16
          => letters right-to-left (A..F) and rows 1..16 top-to-bottom.

        cam1 (mirrored):
          index 5  -> F16
          index 95 -> F1
          index 0  -> A16
          index 90 -> A1
          => letters left-to-right (A..F) and rows 16..1 top-to-bottom.

        Codes: DEV<device_id>_<treatment>_<label>_<run_date>
        """
        self.device_id = device_id
        self.treatment = treatment
        self.run_date = run_date

        roi_labels = []
        roi_codes = []

        for r in range(ROWS):
            for c in range(COLS):
                if self.label == "cam0":
                    # columns reversed, rows normal
                    col_index = COLS - 1 - c
                    row_number = r + 1
                elif self.label == "cam1":
                    # columns normal, rows reversed
                    col_index = c
                    row_number = ROWS - r
                else:
                    # default: columns normal, rows normal
                    col_index = c
                    row_number = r + 1

                col_letter = chr(ord('A') + col_index)
                label = f"{col_letter}{row_number}"
                code = f"DEV{device_id}_{treatment}_{label}_{run_date}"
                roi_labels.append(label)
                roi_codes.append(code)

        self.roi_labels = roi_labels
        self.roi_codes = roi_codes

        # If CSV header not written yet, write it now
        if self.csv_file and self.csv_file.tell() == 0:
            header_rois = [f"{code}_score" for code in self.roi_codes]
            self.csv_writer.writerow(["timestamp"] + header_rois + ["image_saved"])
            self.csv_file.flush()

    def process_frame(self, timestamp_iso: str, diff_threshold: int):
        """
        Capture one frame, compare to previous, update state, and return:
          - movement_scores (list per ROI)
          - image_saved (bool)
          - overlay_preview (RGB with ROI grid)
          - diff_preview (binary diff image)
        Uses diff_threshold passed in (depends on visible vs IR).
        """
        # Capture current frame
        frame_rgb = self.picam.capture_array("main")

        # Convert to grayscale and blur
        frame_gray = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2GRAY)
        frame_gray = cv2.GaussianBlur(frame_gray, BLUR_KERNEL_SIZE, 0)

        # Full-frame difference vs previous frame
        diff = cv2.absdiff(frame_gray, self.prev_gray)

        # Threshold to binary change image using current diff_threshold
        _, diff_binary = cv2.threshold(
            diff, diff_threshold, 255, cv2.THRESH_BINARY
        )

        # Per-ROI movement scores (count changed pixels)
        movement_scores = []
        for (x1, y1, x2, y2) in self.ROIS:
            roi_thresh = diff_binary[y1:y2, x1:x2]
            score = int(np.count_nonzero(roi_thresh))
            movement_scores.append(score)

        # Update previous frame
        self.prev_gray = frame_gray

        # Create overlay with ROI grid for preview
        overlay = frame_rgb.copy()
        for i, (x1, y1, x2, y2) in enumerate(self.ROIS):
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                8,  # thicker lines for better visibility
                lineType=cv2.LINE_AA
            )
            if hasattr(self, "roi_labels") and 0 <= i < len(self.roi_labels):
                label = self.roi_labels[i]
            else:
                label = str(i)

            cv2.putText(
                overlay,
                label,
                (x1 + 10, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 0, 0),
                4
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 10, y1 + 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.2,
                (0, 255, 0),
                2
            )

        # Resized previews
        overlay_preview = cv2.resize(
            overlay,
            (0, 0),
            fx=PREVIEW_SCALE,
            fy=PREVIEW_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
        diff_preview = cv2.resize(
            diff_binary,
            (0, 0),
            fx=PREVIEW_SCALE,
            fy=PREVIEW_SCALE,
            interpolation=cv2.INTER_NEAREST
        )

        # Save JPEG every IMAGE_SAVE_INTERVAL_SECONDS
        image_saved = False
        now = time.time()
        if now - self.last_image_save >= IMAGE_SAVE_INTERVAL_SECONDS:
            img_name = self.image_dir / f"{self.label}_frame_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
            cv2.imwrite(str(img_name), cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
            self.last_image_save = now
            image_saved = True

        # Write CSV row
        self.csv_writer.writerow([timestamp_iso] + movement_scores + [image_saved])
        self.csv_file.flush()

        # Print matrix of ROI values (still numeric, row-major)
        print(f"{timestamp_iso} | {self.label} | diff_threshold={diff_threshold}")
        for row in range(ROWS):
            row_vals = movement_scores[row * COLS:(row + 1) * COLS]
            print(" ".join(f"{v:6d}" for v in row_vals))
        print("-" * 60)

        return movement_scores, image_saved, overlay_preview, diff_preview

    def close(self):
        self.picam.stop()
        self.csv_file.close()


# =====================
# BACKUP & LIGHT MODE HELPERS
# =====================

def do_backup(tag: str):
    """Run rclone sync for the entire run directory (both cameras)."""
    remote_path = f"{RCLONE_REMOTE}:{REMOTE_BASE_DIR}/{RUN_ID}"
    try:
        subprocess.run(
            [
                "rclone",
                "sync",
                str(ROOT_BASE_DIR),
                remote_path,
                "--create-empty-src-dirs",
                "--quiet",
            ],
            check=True
        )
        print(f"{tag} | Backup completed to {remote_path}")
    except subprocess.CalledProcessError as e:
        print(f"{tag} | Backup FAILED: {e}")


def get_light_mode(now: datetime) -> str:
    """
    Return 'visible' during configured day hours, else 'ir'.

    DAY_START_HOUR and NIGHT_START_HOUR are inclusive/exclusive bounds.
    Handles both normal (e.g. 8..20) and wrap-around (e.g. 20..6).
    """
    global DAY_START_HOUR, NIGHT_START_HOUR
    hour = now.hour

    if DAY_START_HOUR == NIGHT_START_HOUR:
        # degenerate: no "day"
        return "ir"

    if DAY_START_HOUR < NIGHT_START_HOUR:
        # e.g. 8..20
        if DAY_START_HOUR <= hour < NIGHT_START_HOUR:
            return "visible"
        else:
            return "ir"
    else:
        # e.g. 20..6 (wrap around midnight)
        if hour >= DAY_START_HOUR or hour < NIGHT_START_HOUR:
            return "visible"
        else:
            return "ir"


def get_diff_threshold_for_mode(mode: str) -> int:
    if mode == "visible":
        return PIX_DIFF_THRESHOLD_VISIBLE
    else:
        return PIX_DIFF_THRESHOLD_IR


def check_wifi_and_reconnect():
    """
    Check if we can reach WIFI_CHECK_HOST; if not, attempt a reconnect.

    NOTE: WIFI_RECONNECT_SHELL uses sudo and assumes no password prompt.
    Adjust to your setup if needed.
    """
    if not WIFI_CHECK_ENABLED:
        return

    try:
        subprocess.run(
            ["ping", "-c", "1", "-W", "2", WIFI_CHECK_HOST],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        # reachable
    except subprocess.CalledProcessError:
        print("WiFi check failed, attempting reconnect...")
        try:
            subprocess.run(
                WIFI_RECONNECT_SHELL,
                shell=True,
                check=True
            )
            print("WiFi reconnect command issued.")
        except Exception as e:
            print(f"WiFi reconnect failed: {e}")


# =====================
# ROI SETUP TOOL (PER CAMERA)
# =====================

def run_roi_setup_single(mon: CameraMonitor):
    """
    Interactive ROI setup for ONE camera.

    Controls:
      w/s   : move grid up/down
      a/d   : move grid left/right
      +/=   : enlarge grid (width & height)
      -     : shrink grid (width & height)
      [ / ] : select which horizontal boundary between rows to adjust
      i/k   : move selected boundary up/down (change heights of adjacent rows)
      r     : reset to default centered crop and equal rows
      Space or Enter : accept and return True
      q     : abort and return False
    """
    move_step = 20
    size_step = 20
    row_step_pixels = 10  # how much one i/k press moves a row boundary
    selected_boundary = ROWS // 2  # index 1..ROWS-1 (0 and ROWS are fixed top/bottom)

    print()
    print(f"=== ROI SETUP: {mon.label} ===")
    print("Controls:")
    print("  w/s   : move grid up/down")
    print("  a/d   : move grid left/right")
    print("  + or =: enlarge grid (width & height)")
    print("  -     : shrink grid (width & height)")
    print("  [ / ] : select which horizontal boundary to adjust")
    print("  i/k   : move selected boundary up/down (change row heights)")
    print("  r     : reset to default centered crop and equal rows")
    print("  Space or Enter : finish ROI setup for this camera")
    print("  q     : abort setup and exit program")
    print("===============================")
    print()

    while True:
        frame = mon.picam.capture_array("main")
        mon.update_rois()

        overlay = frame.copy()
        color = (0, 255, 0)

        # Draw ROIs and labels
        for idx, (x1, y1, x2, y2) in enumerate(mon.ROIS):
            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                color,
                4,
                lineType=cv2.LINE_AA
            )
            label = mon.roi_labels[idx] if hasattr(mon, "roi_labels") else str(idx)
            cv2.putText(
                overlay,
                label,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 0),
                3
            )
            cv2.putText(
                overlay,
                label,
                (x1 + 5, y1 + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                1
            )

        # Show which boundary is selected
        if hasattr(mon, "row_factors") and 0 < selected_boundary < ROWS:
            # draw a thicker horizontal line at the selected boundary
            y0 = int(mon.crop_y)
            h = int(mon.crop_h)
            y = y0 + int(h * mon.row_factors[selected_boundary])
            cv2.line(
                overlay,
                (int(mon.crop_x), y),
                (int(mon.crop_x + mon.crop_w), y),
                (0, 0, 255),
                3,
                lineType=cv2.LINE_AA
            )
            status_text = f"Boundary between rows {selected_boundary} and {selected_boundary+1}"
        else:
            status_text = "No boundary selected"

        cv2.putText(
            overlay,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 0),
            3
        )
        cv2.putText(
            overlay,
            status_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            1
        )

        preview = cv2.resize(
            overlay,
            (0, 0),
            fx=PREVIEW_SCALE,
            fy=PREVIEW_SCALE,
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow(f"{mon.label} ROI setup", preview)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            print("ROI setup aborted by user.")
            return False

        # Space or Enter: accept
        if key == ord(' ') or key == 13:
            print(f"ROI setup complete for {mon.label}.")
            cv2.destroyWindow(f"{mon.label} ROI setup")
            return True

        # Move grid
        if key == ord('w'):
            mon.shift_crop(0, -move_step)
        elif key == ord('s'):
            mon.shift_crop(0, move_step)
        elif key == ord('a'):
            mon.shift_crop(-move_step, 0)
        elif key == ord('d'):
            mon.shift_crop(move_step, 0)

        # Resize grid
        if key in (ord('+'), ord('=')):  # enlarge
            mon.scale_crop(size_step, size_step)
        elif key == ord('-'):  # shrink
            mon.scale_crop(-size_step, -size_step)

        # Reset to default
        if key == ord('r'):
            mon.set_default_crop()
            mon.row_factors = [r / ROWS for r in range(ROWS + 1)]
            mon.update_rois()
            print(f"{mon.label}: crop and rows reset to default.")

        # Select boundary
        if key == ord('['):
            selected_boundary = max(1, selected_boundary - 1)
        elif key == ord(']'):
            selected_boundary = min(ROWS - 1, selected_boundary + 1)

        # Move selected boundary up/down (change row heights)
        if key in (ord('i'), ord('k')):
            if 0 < selected_boundary < ROWS:
                # Convert row_step_pixels to delta factor
                h = float(mon.crop_h)
                if h <= 0:
                    continue
                delta = (row_step_pixels / h) * (1 if key == ord('k') else -1)

                rf = mon.row_factors
                old_val = rf[selected_boundary]
                new_val = old_val + delta

                # Keep boundary between neighbors with a small minimum distance
                min_gap = 0.01
                lower = rf[selected_boundary - 1] + min_gap
                upper = rf[selected_boundary + 1] - min_gap
                new_val = max(lower, min(new_val, upper))

                rf[selected_boundary] = new_val
                mon.row_factors = rf
                mon.update_rois()


# =====================
# STARTUP CONFIGURATION
# =====================

def collect_experiment_metadata(monitors):
    """
    Ask the user for:
      - device ID (1 or 2),
      - day/night cycle,
      - treatment per camera,
      - optionally edit ROIs per camera.

    Fills DEVICE_ID, DAY_START_HOUR, NIGHT_START_HOUR, CAMERA_TREATMENTS
    and configures each monitor with metadata (ROI labels/codes).
    """
    global DEVICE_ID, DAY_START_HOUR, NIGHT_START_HOUR, CAMERA_TREATMENTS

    print("\n=== Ant Experiment Setup ===")

    # Device ID
    while True:
        dev = input("Device number (1 or 2, default 1): ").strip()
        if dev == "":
            DEVICE_ID = 1
            break
        if dev in ("1", "2"):
            DEVICE_ID = int(dev)
            break
        print("Please enter 1 or 2.")

    # Day/night cycle
    def ask_hour(prompt, default):
        while True:
            val = input(f"{prompt} (0–23, default {default}): ").strip()
            if val == "":
                return default
            try:
                h = int(val)
                if 0 <= h <= 23:
                    return h
            except ValueError:
                pass
            print("Please enter an integer between 0 and 23.")

    DAY_START_HOUR = ask_hour("Day (visible) starts at hour", DAY_START_HOUR)
    NIGHT_START_HOUR = ask_hour("Day (visible) ends at hour", NIGHT_START_HOUR)

    # Treatment per camera
    CAMERA_TREATMENTS = {}
    run_date = datetime.now().strftime("%Y%m%d")
    print("\nAssign treatment per camera:")
    for mon in monitors:
        while True:
            t = input(f"  Treatment name for {mon.label} (e.g. 'control', 'high_temp'): ").strip()
            if t:
                CAMERA_TREATMENTS[mon.label] = t
                break
            print("Treatment name cannot be empty.")

        mon.set_metadata(
            device_id=DEVICE_ID,
            treatment=CAMERA_TREATMENTS[mon.label],
            run_date=run_date,
        )

    print("\n=== Summary ===")
    print(f"  Run ID         : {RUN_ID}")
    print(f"  Device ID      : {DEVICE_ID}")
    print(f"  Day visible    : {DAY_START_HOUR:02d}:00 – {NIGHT_START_HOUR:02d}:00")
    for mon in monitors:
        print(f"  Camera {mon.label}: treatment='{mon.treatment}', example ROI label='{mon.roi_labels[0]}'")

    # ROI editing menu
    print("\n=== ROI Editing ===")
    print("You can edit the ROI grid per camera before starting the experiment.")
    print("Options:")
    print("  1: Edit ROIs for cam0")
    print("  2: Edit ROIs for cam1")
    print("  3: Edit ROIs for both (one after the other)")
    print("  s: Skip ROI editing and start immediately")
    print("  q: Quit without starting")

    while True:
        choice = input("Select option [1/2/3/s/q]: ").strip().lower()
        if choice == "1":
            if len(monitors) >= 1:
                if not run_roi_setup_single(monitors[0]):
                    return False
            break
        elif choice == "2":
            if len(monitors) > 1:
                if not run_roi_setup_single(monitors[1]):
                    return False
            else:
                print("No second camera configured.")
            break
        elif choice == "3":
            for mon in monitors:
                if not run_roi_setup_single(mon):
                    return False
            break
        elif choice == "s":
            # no ROI editing
            break
        elif choice == "q":
            return False
        else:
            print("Please choose 1, 2, 3, s, or q.")

    print("Startup configuration done. Starting main experiment loop.")
    return True


# =====================
# INITIALIZE CAMERAS
# =====================

monitors = []
for cfg in CAMERA_CONFIGS:
    monitors.append(CameraMonitor(cfg["index"], cfg["label"]))

print(f"Run directory: {ROOT_BASE_DIR}")
print("Dual-camera ROI monitor with manual focus and dynamic thresholds started.")

# Collect metadata & optional ROI editing
ok = collect_experiment_metadata(monitors)
cv2.destroyAllWindows()
if not ok:
    # Clean up and exit
    for mon in monitors:
        mon.close()
    print("Exiting due to setup abort.")
    raise SystemExit(0)

print("Press 'q' in a preview window or Ctrl+C in the terminal to stop.")

# Initial light mode based on current time
now = datetime.now()
current_light_mode = get_light_mode(now)
last_light_mode = current_light_mode
print(f"Initial light mode: {current_light_mode}")

last_wifi_check = 0

# =====================
# MAIN LOOP
# =====================

try:
    while True:
        loop_start = time.time()
        now = datetime.now()
        timestamp_iso = now.isoformat()

        # Current light mode and diff threshold
        current_light_mode = get_light_mode(now)
        current_diff_threshold = get_diff_threshold_for_mode(current_light_mode)

        # Check if light mode has changed (day/night crossings)
        if current_light_mode != last_light_mode:
            print(
                f"Light mode changed from {last_light_mode} to {current_light_mode} "
                f"at {timestamp_iso}. Recalibrating cameras."
            )
            for mon in monitors:
                mon.recalibrate_for_light_change(current_light_mode)
            last_light_mode = current_light_mode

        # WiFi failsafe (periodic)
        if WIFI_CHECK_ENABLED and (time.time() - last_wifi_check) >= WIFI_CHECK_INTERVAL_SECONDS:
            check_wifi_and_reconnect()
            last_wifi_check = time.time()

        # Process each camera
        for mon in monitors:
            _, _, overlay_preview, diff_preview = mon.process_frame(
                timestamp_iso,
                diff_threshold=current_diff_threshold
            )

            # Show previews in separate windows per camera
            cv2.imshow(f"{mon.label} Ant Monitor (ROIs)", overlay_preview)
            cv2.imshow(f"{mon.label} Change (binary)", diff_preview)

        # Handle key press from any OpenCV window
        if cv2.waitKey(1) & 0xFF == ord("q"):
            print("Quit requested from window.")
            break

        # Backup once per cycle (both cameras)
        if BACKUP_ENABLED:
            do_backup(timestamp_iso)

        # Maintain capture interval
        elapsed = time.time() - loop_start
        sleep_time = CAPTURE_INTERVAL_SECONDS - elapsed
        if sleep_time > 0:
            time.sleep(sleep_time)

except KeyboardInterrupt:
    print("\nStopped by user.")

finally:
    for mon in monitors:
        mon.close()

    cv2.destroyAllWindows()

    if BACKUP_ENABLED:
        do_backup("FINAL")

    print("Cleanup complete.")