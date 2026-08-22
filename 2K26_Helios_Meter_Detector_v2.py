import json
import threading
from pathlib import Path
import time

import cv2
import numpy as np

from helios import vision, meter, overlay

BASE = Path(__file__).resolve().parent
CFG = BASE / "meter_v2.json"
DEFAULT = {
    "meter_target": 100,
    "early_timing": 2,
    "late_timing": 2,
    "roi_x": 850,
    "roi_y": 400,
    "roi_w": 220,
    "roi_h": 400,
    "min_width": 8,
    "max_width": 80,
    "min_height": 100,
    "max_height": 400,
    "white_v_low": 200,
    "white_s_high": 50,
    "green_h_low": 40,
    "green_h_high": 90,
    "green_s_low": 80,
    "green_v_low": 100,
}


def load_config():
    try:
        saved = json.loads(CFG.read_text(encoding="utf-8"))
        return {**DEFAULT, **saved}
    except (OSError, ValueError, TypeError):
        return dict(DEFAULT)


def save_config(config):
    CFG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


class CVWorker:
    """Helios Vision-based meter detector for NBA 2K Arrow2 meter with RS calibration."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._lock = threading.RLock()
        self.c = load_config()
        self.last = None
        self._closed = False
        self._live = {
            "tempo": "READY",
            "meter": None,
            "confidence": 0,
            "rs_held": False,
        }
        self._release_cooldown = 0.05
        self._last_release_time = 0
        self._rs_held_start = None
        self._calibrating = False
        self._calibration_rois = []
        
        # Initialize Helios Vision components
        self._init_vision()

    def _init_vision(self):
        """Initialize Helios Vision finders for white shaft and green cap detection."""
        try:
            # White shaft detection (meter body)
            white_range = vision.BgrRange(
                low=(0, 0, self.c["white_v_low"]),
                high=(180, self.c["white_s_high"], 255),
            )
            self.white_finder = vision.ContourFinder(
                colors=[white_range],
                roi=(self.c["roi_x"], self.c["roi_y"], self.c["roi_w"], self.c["roi_h"]),
                width=(self.c["min_width"], self.c["max_width"]),
                height=(self.c["min_height"], self.c["max_height"]),
                grouping=vision.GROUP_CONNECTED,
                max_results=16,
                focus=True,
                focus_padding=12,
                adaptive_color=True,
                color_growth=12,
            )
            self.white_finder.set_visuals(
                enabled=True,
                show_roi=True,
                show_outline=True,
                roi_color=(255, 255, 255, 255),
                outline_color=(255, 255, 255, 255),
                roi_thickness=2,
            )
            
            # Green cap detection (release indicator)
            green_range = vision.BgrRange(
                low=(self.c["green_h_low"], self.c["green_s_low"], self.c["green_v_low"]),
                high=(self.c["green_h_high"], 255, 255),
            )
            self.green_finder = vision.ContourFinder(
                colors=[green_range],
                roi=(self.c["roi_x"], self.c["roi_y"], self.c["roi_w"], self.c["roi_h"]),
                width=(5, 60),
                height=(8, 80),
                grouping=vision.GROUP_CONNECTED,
                max_results=16,
                focus=False,
                adaptive_color=True,
                color_growth=8,
            )
            self.green_finder.set_visuals(
                enabled=True,
                show_roi=False,
                show_outline=True,
                outline_color=(0, 255, 0, 255),
                outline_thickness=2,
            )
            
            # Meter tracker
            self.meter_tracker = meter.Meter()
            self.meter_tracker.set_visuals(
                enabled=True,
                show_bbox=True,
                show_path=True,
                show_distance=False,
                show_speed=False,
                show_time_to_release=False,
                show_elapsed_time=False,
                bbox_color=(0, 255, 255, 255),
                path_color=(0, 255, 255, 255),
                metrics_color=(0, 255, 255, 255),
                target=overlay.BOTH,
            )
        except Exception as e:
            print(f"Vision initialization failed: {e}")

    def _release_shot(self):
        """Trigger shot release when meter is full."""
        import time
        current_time = time.time()
        if current_time - self._last_release_time < self._release_cooldown:
            return False
        self._last_release_time = current_time
        
        try:
            from helios import controls
            # Release right stick by setting to neutral (0, 0)
            controls.send_controller_input({"stick_2": (0, 0)})
            return True
        except Exception as e:
            print(f"Release failed: {e}")
            return False

    def process(self, frame):
        if frame is None or frame.size == 0:
            return frame, bytearray()

        try:
            from helios import controls
            state = controls.get_controller_state()
            rs_held = (state.get("stick_2", (0, 0))[0] != 0 or 
                      state.get("stick_2", (0, 0))[1] != 0)
        except Exception:
            rs_held = False

        config = self._config()
        
        # Draw frame border
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), (255, 210, 0), 2)
        
        tempo = "WAITING FOR RS"
        meter_value = None
        confidence = 0.0
        released = False
        
        # Handle RS held state
        if rs_held:
            if self._rs_held_start is None:
                self._rs_held_start = time.time()
            
            # Find white shaft contours
            white_results = self.white_finder.find()
            green_results = self.green_finder.find()
            
            # Match white shaft with green cap
            if len(white_results) > 0 and len(green_results) > 0:
                best_white = white_results[0]
                best_green = None
                
                # Find green cap closest to white shaft top
                white_top = best_white.bounds[1]
                min_distance = float('inf')
                
                for green in green_results:
                    distance = abs(green.centroid[1] - white_top)
                    if distance < min_distance:
                        min_distance = distance
                        best_green = green
                
                if best_green is not None and best_white.confidence > 0.5 and best_green.confidence > 0.3:
                    confidence = (best_white.confidence + best_green.confidence) / 2
                    
                    # Calculate meter position (green cap position on white shaft)
                    shaft_top = best_white.bounds[1]
                    shaft_bottom = best_white.bounds[1] + best_white.bounds[3]
                    green_y = best_green.centroid[1]
                    
                    meter_value = max(0.0, min(100.0, 100.0 * (shaft_bottom - green_y) / max(1, best_white.bounds[3])))
                    self.last = meter_value
                    
                    # Auto-release when meter reaches full (100%)
                    if meter_value >= config["meter_target"]:
                        tempo = "RELEASING!"
                        released = self._release_shot()
                    else:
                        tempo = f"METER ACTIVE"
                    
                    # Update meter tracker for visualization
                    meter_point = (int(best_green.centroid[0]), int(green_y))
                    release_point = (int(best_white.centroid[0]), int(shaft_top))
                    bbox = best_white.bounds
                    
                    self.meter_tracker.update(
                        point=meter_point,
                        release_point=release_point,
                        bounding_box=bbox,
                        padding=8,
                    )
            else:
                tempo = "DETECTING METER..."
        else:
            # RS not held
            self._rs_held_start = None
            tempo = "WAITING FOR RS"
            
            # Draw instruction
            overlay.text(
                960, 540,
                "HOLD RS DOWN TO START METER",
                color=(255, 100, 100, 255),
                scale=3,
                target=overlay.BOTH,
                anchor=overlay.CENTER,
            )
        
        with self._lock:
            self._live = {
                "tempo": tempo,
                "meter": meter_value,
                "confidence": confidence,
                "rs_held": rs_held,
            }
        
        # Draw HUD info
        meter_text = "--" if meter_value is None else f"{meter_value:.1f}%"
        status_color = (0, 255, 0) if released else (0, 200, 255)
        
        cv2.rectangle(frame, (10, 10), (480, 140), (20, 24, 29), -1)
        cv2.putText(frame, "2K26 ARROW2 METER", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.70, (235, 240, 245), 2)
        cv2.putText(frame, f"STATUS: {tempo}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.60, status_color, 2)
        cv2.putText(frame, f"METER: {meter_text}   CONF: {confidence:.0f}%", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (225, 230, 235), 1)
        cv2.putText(frame, "Hold RS Down - Auto Release at 100%", (20, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (145, 155, 165), 1)
        
        return frame, bytearray()

    def _config(self):
        with self._lock:
            return dict(self.c)

    def close(self):
        self._closed = True

    def __del__(self):
        self.close()
