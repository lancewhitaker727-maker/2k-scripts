import json
import threading
from pathlib import Path

import cv2
import numpy as np

from helios import vision, meter, overlay

BASE = Path(__file__).resolve().parent
CFG = BASE / "meter_v2.json"
DEFAULT = {
    "meter_target": 99,
    "early_timing": 2,
    "late_timing": 2,
    "roi_x": 700,
    "roi_y": 180,
    "roi_w": 520,
    "roi_h": 760,
    "min_width": 8,
    "max_width": 240,
    "min_height": 20,
    "max_height": 700,
    "white_v_low": 205,
    "white_s_high": 70,
    "green_h_low": 35,
    "green_h_high": 95,
    "green_s_low": 130,
    "green_v_low": 120,
    "focus": True,
    "adaptive_color": True,
    "color_growth": 0,
}
SLIDERS = (
    ("Meter Target", "meter_target"),
    ("Early Timing", "early_timing"),
    ("Late Timing", "late_timing"),
    ("ROI X", "roi_x"),
    ("ROI Y", "roi_y"),
    ("ROI Width", "roi_w"),
    ("ROI Height", "roi_h"),
    ("Min Width", "min_width"),
    ("Max Width", "max_width"),
    ("Min Height", "min_height"),
    ("Max Height", "max_height"),
    ("White V Low", "white_v_low"),
    ("White S High", "white_s_high"),
    ("Green H Low", "green_h_low"),
    ("Green H High", "green_h_high"),
    ("Green S Low", "green_s_low"),
    ("Green V Low", "green_v_low"),
)


def load_config():
    try:
        saved = json.loads(CFG.read_text(encoding="utf-8"))
        return {**DEFAULT, **saved}
    except (OSError, ValueError, TypeError):
        return dict(DEFAULT)


def save_config(config):
    CFG.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")


class CVWorker:
    """Helios Vision-based meter detector for NBA 2K Arrow2 meter."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._lock = threading.RLock()
        self.c = load_config()
        self.last = None
        self._closed = False
        self._live = {
            "tempo": "INITIALIZING",
            "meter": None,
            "confidence": 0,
        }
        self._settings_started = False
        self._release_cooldown = 0.1
        self._last_release_time = 0
        
        # Initialize Helios Vision components
        self._init_vision()
        self._start_settings_window()

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
                focus=self.c["focus"],
                focus_padding=16,
                adaptive_color=self.c["adaptive_color"],
                color_growth=self.c["color_growth"],
            )
            self.white_finder.set_visuals(
                enabled=True,
                show_roi=True,
                show_outline=True,
                roi_color=(255, 255, 255, 255),
                outline_color=(255, 255, 255, 255),
            )
            
            # Green cap detection (release indicator)
            green_range = vision.BgrRange(
                low=(self.c["green_h_low"], self.c["green_s_low"], self.c["green_v_low"]),
                high=(self.c["green_h_high"], 255, 255),
            )
            self.green_finder = vision.ContourFinder(
                colors=[green_range],
                roi=(self.c["roi_x"], self.c["roi_y"], self.c["roi_w"], self.c["roi_h"]),
                width=(5, 42),
                height=(4, 50),
                grouping=vision.GROUP_CONNECTED,
                max_results=16,
                focus=False,
                adaptive_color=self.c["adaptive_color"],
                color_growth=self.c["color_growth"],
            )
            self.green_finder.set_visuals(
                enabled=True,
                show_roi=False,
                show_outline=True,
                outline_color=(0, 255, 0, 255),
            )
            
            # Meter tracker
            self.meter_tracker = meter.Meter()
            self.meter_tracker.set_visuals(
                enabled=True,
                show_bbox=True,
                show_path=True,
                show_distance=False,
                show_speed=True,
                show_time_to_release=True,
                show_elapsed_time=False,
                bbox_color=(104, 244, 255, 255),
                path_color=(104, 244, 255, 255),
                metrics_color=(104, 244, 255, 255),
                target=overlay.BOTH,
            )
        except Exception as e:
            print(f"Vision initialization failed: {e}")
            raise

    def _start_settings_window(self):
        if self._settings_started:
            return
        self._settings_started = True
        threading.Thread(target=self._settings_main, name="Helios Meter Settings", daemon=True).start()

    def _settings_main(self):
        try:
            import tkinter as tk
        except ImportError:
            return

        try:
            root = tk.Tk()
            root.title("2K26 Helios Vision Meter")
            root.resizable(False, False)
            root.configure(padx=14, pady=12)

            title = tk.Label(root, text="Arrow2 Meter Tracker", font=("Segoe UI", 13, "bold"))
            title.pack(anchor="w")
            note = tk.Label(root, text="Hold RS down; releases when meter turns green.")
            note.pack(anchor="w", pady=(0, 8))

            value_labels = {}
            for label, key in SLIDERS:
                group = tk.Frame(root)
                group.pack(fill="x", pady=4)
                tk.Label(group, text=label, anchor="w").pack(fill="x")
                value_label = tk.Label(group, font=("Segoe UI", 11, "bold"))
                value_label.pack(anchor="center")
                value_labels[key] = value_label
                with self._lock:
                    initial = int(self.c[key])
                slider = tk.Scale(
                    group,
                    from_=0,
                    to=255 if key.startswith(("white", "green")) else 999,
                    orient="horizontal",
                    showvalue=False,
                    resolution=1,
                    command=lambda value, setting=key: self._set_setting(setting, value),
                )
                slider.set(initial)
                slider.pack(fill="x")

            live_label = tk.Label(root, justify="left", font=("Consolas", 11, "bold"))
            live_label.pack(anchor="w", pady=(10, 0))

            def refresh():
                if self._closed:
                    root.destroy()
                    return
                with self._lock:
                    config = dict(self.c)
                    live = dict(self._live)
                for key, widget in value_labels.items():
                    widget.configure(text=str(int(config[key])))
                meter = "--" if live["meter"] is None else f"{live['meter']:.0f}%"
                live_label.configure(
                    text=f"STATUS: {live['tempo']}\nMETER: {meter}\nCONFIDENCE: {live['confidence']:.0f}%"
                )
                root.after(100, refresh)

            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.after(0, refresh)
            root.mainloop()
        except Exception:
            return

    def _set_setting(self, key, value):
        with self._lock:
            self.c[key] = max(0, min(255, int(float(value))))
            save_config(self.c)
            
            # Update vision settings if changed
            if key.startswith("roi") or key.startswith(("min_", "max_")):
                try:
                    self.white_finder.set_roi((self.c["roi_x"], self.c["roi_y"], self.c["roi_w"], self.c["roi_h"]))
                    self.white_finder.set_size_range(
                        width=(self.c["min_width"], self.c["max_width"]),
                        height=(self.c["min_height"], self.c["max_height"]),
                    )
                    self.green_finder.set_roi((self.c["roi_x"], self.c["roi_y"], self.c["roi_w"], self.c["roi_h"]))
                except Exception:
                    pass
            elif key.startswith(("white_", "green_")):
                try:
                    white_range = vision.BgrRange(
                        low=(0, 0, self.c["white_v_low"]),
                        high=(180, self.c["white_s_high"], 255),
                    )
                    self.white_finder.set_colors([white_range])
                    
                    green_range = vision.BgrRange(
                        low=(self.c["green_h_low"], self.c["green_s_low"], self.c["green_v_low"]),
                        high=(self.c["green_h_high"], 255, 255),
                    )
                    self.green_finder.set_colors([green_range])
                except Exception:
                    pass

    def _config(self):
        with self._lock:
            return dict(self.c)

    @staticmethod
    def _tempo(meter, config):
        target = config["meter_target"]
        if meter < target - config["early_timing"]:
            return "EARLY"
        if meter > target + config["late_timing"]:
            return "LATE"
        return "PERFECT"

    def _release_shot(self):
        """Trigger shot release when in perfect range."""
        import time
        current_time = time.time()
        if current_time - self._last_release_time < self._release_cooldown:
            return
        self._last_release_time = current_time
        
        try:
            from helios import controls
            controls.send_controller_input({"rs": 0})  # Release right stick
        except Exception:
            pass

    def process(self, frame):
        if frame is None or frame.size == 0:
            return frame, bytearray()

        config = self._config()
        
        # Find white shaft contours
        white_results = self.white_finder.find()
        green_results = self.green_finder.find()
        
        tempo = "SEARCHING FOR ARROW2"
        meter_value = None
        confidence = 0.0
        release_shot = False
        
        # Draw frame border
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), (255, 210, 0), 2)
        
        # Match white shaft with green cap
        if len(white_results) > 0 and len(green_results) > 0:
            best_white = white_results[0]
            best_green = None
            
            # Find green cap closest to white shaft top
            white_top = best_white.centroid[1] - best_white.bounds[3] / 2
            min_distance = float('inf')
            
            for green in green_results:
                distance = abs(green.centroid[1] - white_top)
                if distance < min_distance:
                    min_distance = distance
                    best_green = green
            
            if best_green is not None:
                confidence = (best_white.confidence + best_green.confidence) / 2
                
                # Calculate meter position (green cap position on white shaft)
                shaft_top = best_white.bounds[1]
                shaft_bottom = best_white.bounds[1] + best_white.bounds[3]
                green_y = best_green.centroid[1]
                
                meter_value = max(0.0, min(100.0, 100.0 * (shaft_bottom - green_y) / best_white.bounds[3]))
                self.last = meter_value
                
                # Check if meter is in perfect range
                target = config["meter_target"]
                if target - config["early_timing"] <= meter_value <= target + config["late_timing"]:
                    tempo = "PERFECT"
                    release_shot = True
                    self._release_shot()
                else:
                    tempo = self._tempo(meter_value, config)
                
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
        
        with self._lock:
            self._live = {
                "tempo": tempo,
                "meter": meter_value,
                "confidence": confidence,
            }
        
        # Draw HUD info
        meter_text = "--" if meter_value is None else f"{meter_value:.0f}%"
        color = (0, 255, 0) if release_shot else (0, 200, 255)
        cv2.rectangle(frame, (10, 10), (440, 126), (20, 24, 29), -1)
        cv2.putText(frame, "2K26 ARROW2 METER", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 240, 245), 2)
        cv2.putText(frame, f"STATUS: {tempo}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"METER: {meter_text}   CONF: {confidence:.0f}%", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 230, 235), 1)
        cv2.putText(frame, "HOLD RS DOWN - AUTO RELEASE", (20, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (145, 155, 165), 1)
        
        return frame, bytearray()

    def close(self):
        self._closed = True

    def __del__(self):
        self.close()
