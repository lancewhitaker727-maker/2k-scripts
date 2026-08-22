import json
import threading
from pathlib import Path

import cv2
import numpy as np

try:
    from helios import controls as _helios_controls
except Exception:
    _helios_controls = None

BASE = Path(__file__).resolve().parent
CFG = BASE / "meter_v2.json"
DEFAULT = {
    "meter_target": 99,
    "early_timing": 2,
    "late_timing": 2,
    "detection_tolerance": 35,
    "smoothing": 0,
    "minimum_detection_confidence": 20,
    "min_height": 25,
    "max_height": 220,
    "min_width": 5,
    "max_width": 70,
    "green_h_low": 35,
    "green_h_high": 95,
    "green_s_low": 130,
    "green_v_low": 120,
    "white_v_low": 205,
    "white_s_high": 70,
}
SLIDERS = (
    ("Meter Target", "meter_target"),
    ("Early Timing", "early_timing"),
    ("Late Timing", "late_timing"),
    ("Detection Tolerance", "detection_tolerance"),
    ("Smoothing", "smoothing"),
    ("Minimum Detection Confidence", "minimum_detection_confidence"),
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
    """Visual-only Helios CV worker. Detects meter and sends controller input to release shot."""

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self._lock = threading.RLock()
        self._packet_sequence = 0
        self._smoothed_box = None
        self._green_frames = 0
        self.c = load_config()
        self.last = None
        self._closed = False
        self._live = {
            "tempo": "WAITING FOR METER",
            "meter": None,
            "confidence": 0,
        }
        self._settings_started = False
        self._last_release_time = 0
        self._release_cooldown = 0.1  # Prevent rapid repeated releases
        self._start_settings_window()

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
            root.title("2K26 Helios Meter Settings")
            root.resizable(False, False)
            root.configure(padx=14, pady=12)

            title = tk.Label(root, text="Arrow2 Auto Release", font=("Segoe UI", 13, "bold"))
            title.pack(anchor="w")
            note = tk.Label(root, text="Hold RS down; Helios releases it when the Arrow2 meter turns green.")
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
                    to=99,
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
                    text=f"TEMPO: {live['tempo']}\nMETER: {meter}\nCONFIDENCE: {live['confidence']:.0f}%"
                )
                root.after(100, refresh)

            root.protocol("WM_DELETE_WINDOW", root.destroy)
            root.after(0, refresh)
            root.mainloop()
        except Exception:
            # Helios can run without a desktop session; CV detection remains available.
            return

    def _set_setting(self, key, value):
        with self._lock:
            self.c[key] = max(0, min(99, int(float(value))))
            save_config(self.c)

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

    def _release_right_stick(self):
        """Send a Helios override that returns Right Stick Y to neutral (releases the shot)."""
        packet = bytearray(48)
        self._packet_sequence = (self._packet_sequence % 254) + 1
        packet[0] = self._packet_sequence  # Helios CV frame marker
        packet[29] = 101  # legacy stick-Y: valid zero / released
        packet[36:40] = (0).to_bytes(4, byteorder="big", signed=True)
        
        if _helios_controls is not None:
            try:
                _helios_controls.send_cvdata(packet)
            except Exception:
                pass
        return packet

    def _send_controller_release(self):
        """Send controller input to release the shot when meter is in optimal range."""
        import time
        current_time = time.time()
        
        # Cooldown to prevent spamming releases
        if current_time - self._last_release_time < self._release_cooldown:
            return
        
        self._last_release_time = current_time
        
        try:
            self._release_right_stick()
        except Exception:
            pass

    def process(self, frame):
        if frame is None or frame.size == 0:
            return frame, bytearray()

        config = self._config()
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        white_mask = cv2.inRange(
            hsv,
            np.array([0, 0, config["white_v_low"]], np.uint8),
            np.array([180, config["white_s_high"], 255], np.uint8),
        )
        green_mask = cv2.inRange(
            hsv,
            np.array([config["green_h_low"], config["green_s_low"], config["green_v_low"]], np.uint8),
            np.array([config["green_h_high"], 255, 255], np.uint8),
        )
        kernel = np.ones((3, 3), np.uint8)
        white_mask = cv2.morphologyEx(white_mask, cv2.MORPH_CLOSE, kernel)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)

        # NBA 2K Arrow2 Large is a narrow white shaft with a small pointed cap.
        # We locate it from that local two-part geometry, not from screen position.
        white_contours, _ = cv2.findContours(white_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        green_contours, _ = cv2.findContours(green_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bodies = []
        caps = []
        for contour in white_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 60 and 40 <= h <= 220 and h / max(1, w) >= 2.0:
                width_fit = max(0.0, 1.0 - abs(w - 20.0) / 18.0)
                height_fit = max(0.0, 1.0 - abs(h - 105.0) / 90.0)
                bodies.append((100.0 * (0.55 * width_fit + 0.45 * height_fit), (x, y, w, h)))
        for contour in green_contours:
            x, y, w, h = cv2.boundingRect(contour)
            if 5 <= w <= 42 and 4 <= h <= 50 and 0.20 <= h / max(1, w) <= 3.0:
                caps.append((x, y, w, h))

        tolerance = config["detection_tolerance"]
        best = None
        for body_score, (bx, by, bw, bh) in bodies:
            body_center = bx + bw / 2.0
            for cx, cy, cw, ch in caps:
                cap_center = cx + cw / 2.0
                horizontal_error = abs(cap_center - body_center)
                vertical_offset = cy + ch / 2.0 - by
                max_horizontal = bw * 0.75 + tolerance * 0.25
                # The pointed cap sits at the body top; allow a small overlap.
                if horizontal_error > max_horizontal or not (-35 - tolerance * 0.2 <= vertical_offset <= bh * 0.30):
                    continue
                align = max(0.0, 1.0 - horizontal_error / max_horizontal)
                top_fit = max(0.0, 1.0 - abs(vertical_offset) / max(35.0, bh * 0.35))
                confidence = 0.55 * body_score + 45.0 * (0.60 * align + 0.40 * top_fit)
                if best is None or confidence > best[0]:
                    best = (confidence, (bx, by, bw, bh), (cx, cy, cw, ch))

        tempo = "SEARCHING FOR ARROW2"
        meter = None
        confidence = 0.0
        release_shot = False
        cv2.rectangle(frame, (2, 2), (frame.shape[1] - 3, frame.shape[0] - 3), (255, 210, 0), 2)

        if best is not None:
            confidence, body, cap = best
            bx, by, bw, bh = body
            cx, cy, cw, ch = cap
            raw_box = np.array([bx, by, bw, bh], dtype=np.float32)
            if self._smoothed_box is None:
                self._smoothed_box = raw_box
            else:
                alpha = max(0.05, (100 - config["smoothing"]) / 100.0)
                self._smoothed_box = alpha * raw_box + (1.0 - alpha) * self._smoothed_box
            sx, sy, sw, sh = self._smoothed_box

            # Cap at the shaft's top is 100%; cap moving down the shaft reduces fill.
            cap_y = cy + ch / 2.0
            meter = max(0.0, min(100.0, 100.0 * ((sy + sh) - cap_y) / max(1.0, sh)))
            self.last = meter
            self._last_arrow = (bx, by, bw, bh)

            if confidence >= config["minimum_detection_confidence"]:
                self._green_frames += 1
                
                # Check if meter is in the perfect range
                target = config["meter_target"]
                if target - config["early_timing"] <= meter <= target + config["late_timing"]:
                    tempo = "PERFECT"
                    release_shot = True
                    # Send controller release input
                    self._send_controller_release()
                else:
                    tempo = self._tempo(meter, config)
                
                cv2.rectangle(frame, (int(sx), int(sy)), (int(sx + sw), int(sy + sh)), (255, 255, 255), 2)
                cv2.rectangle(frame, (cx, cy), (cx + cw, cy + ch), (0, 255, 0), 2)
            else:
                self._green_frames = 0
                tempo = "LOW CONFIDENCE"
        else:
            self._green_frames = 0
            self._smoothed_box = None
            self._last_arrow = None
            self.last = None

        with self._lock:
            self._live = {"tempo": tempo, "meter": meter, "confidence": confidence}

        meter_text = "--" if meter is None else f"{meter:.0f}%"
        color = (0, 255, 0) if release_shot else (0, 200, 255)
        cv2.rectangle(frame, (10, 10), (440, 126), (20, 24, 29), -1)
        cv2.putText(frame, "2K26 ARROW2 METER", (20, 34), cv2.FONT_HERSHEY_SIMPLEX, 0.60, (235, 240, 245), 2)
        cv2.putText(frame, f"TEMPO: {tempo}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(frame, f"METER: {meter_text}   CONF: {confidence:.0f}%", (20, 88), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (225, 230, 235), 1)
        cv2.putText(frame, "HOLD RS DOWN - AUTO RELEASE", (20, 111), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (145, 155, 165), 1)

        return frame, bytearray()

    def close(self):
        self._closed = True
