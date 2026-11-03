import cv2
import mediapipe as mp
import numpy as np
import threading
import time

from pathlib import Path
from collections import deque

from backend.alert_service import alert_service
from backend.temporal_features import (
    TemporalFeatureExtractor,
    FEATURES,
    YawnDetector,
)
from backend.model_service import predict_from_features


# --------------------------------------------------
# CONFIGURATION
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

MODEL_PATH = PROJECT_ROOT / "models" / "face_landmarker.task"

CAMERA_INDEX = 0

SAMPLE_INTERVAL_SECONDS = 1.0

MAR_THRESHOLD = 0.45
YAWN_DURATION_SECONDS = 1.0

DROWSY_DURATION_SECONDS = 6.0
YAWN_COUNT_THRESHOLD = 3
YAWN_WINDOW_SECONDS = 60.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14


class CameraService:

    def __init__(self):

        self.camera = None
        self.thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        self.running = False
        self.error = None
        self.latest_frame = None

        # Shared Arduino controller assigned in main.py
        self.arduino_controller = None

        self.status = {
            "camera": "Not started",
            "face_detected": False,
            "model_status": "Warming up",
            "ear": None,
            "mar": None,
            "drowsy_probability": None,
            "recent_yawns": 0,
            "warning": False,
            "message": "Camera is not started",
        }

    # --------------------------------------------------
    # CAMERA STATUS
    # --------------------------------------------------

    def get_status(self):

        with self.lock:
            return self.status.copy()

    # --------------------------------------------------
    # START CAMERA
    # --------------------------------------------------

    def start(self):

        if self.running:
            return {"message": "Camera already running"}

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Face Landmarker model not found: {MODEL_PATH}"
            )

        self.stop_event.clear()
        self.error = None

        print(
            f"[CAMERA START] Arduino controller attached: "
            f"{self.arduino_controller is not None}",
            flush=True,
        )

        # Ensure Arduino is connected and outputs are OFF.
        if self.arduino_controller is not None:

            try:

                if not self.arduino_controller.get_status()["connected"]:
                    self.arduino_controller.connect()

                sent = self.arduino_controller.send("CLEAR")

                print(
                    f"[CAMERA START] Arduino CLEAR command: {sent}",
                    flush=True,
                )

            except Exception as exc:

                print(
                    f"[ARDUINO ERROR] Could not initialize Arduino: {exc}",
                    flush=True,
                )

        self.thread = threading.Thread(
            target=self._run,
            daemon=True
        )

        self.running = True
        self.thread.start()

        return {"message": "Camera starting"}

    # --------------------------------------------------
    # STOP CAMERA
    # --------------------------------------------------

    def stop(self):

        self.stop_event.set()

        alert_service.stop()

        # Turn OFF LEDs and buzzer.
        if self.arduino_controller is not None:

            try:

                sent = self.arduino_controller.send("CLEAR")

                print(
                    f"[CAMERA STOP] Arduino CLEAR command: {sent}",
                    flush=True,
                )

            except Exception as exc:

                print(f"[ARDUINO ERROR] {exc}", flush=True)

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5)

        self.running = False

        with self.lock:

            self.latest_frame = None

            self.status = {
                "camera": "Stopped",
                "face_detected": False,
                "model_status": "Stopped",
                "ear": None,
                "mar": None,
                "drowsy_probability": None,
                "recent_yawns": 0,
                "warning": False,
                "message": "Camera stopped",
            }

        return {"message": "Camera stopped"}

    # --------------------------------------------------
    # GET CAMERA FRAME
    # --------------------------------------------------

    def get_frame(self):

        with self.lock:

            if self.latest_frame is None:
                return None

            return self.latest_frame

    # --------------------------------------------------
    # MAIN CAMERA PROCESSING
    # --------------------------------------------------

    def _run(self):

        extractor = TemporalFeatureExtractor()

        yawn_detector = YawnDetector(
            mar_threshold=MAR_THRESHOLD,
            duration_seconds=YAWN_DURATION_SECONDS,
            sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
        )

        yawn_times = deque()

        last_sample_time = 0
        drowsy_start_time = None
        previous_warning = False
        last_arduino_command = None

        timestamp_ms = 0

        try:

            # ------------------------------------------
            # MEDIAPIPE INITIALIZATION
            # ------------------------------------------

            BaseOptions = mp.tasks.BaseOptions
            FaceLandmarker = mp.tasks.vision.FaceLandmarker

            FaceLandmarkerOptions = (
                mp.tasks.vision.FaceLandmarkerOptions
            )

            RunningMode = mp.tasks.vision.RunningMode

            options = FaceLandmarkerOptions(
                base_options=BaseOptions(
                    model_asset_path=str(MODEL_PATH)
                ),
                running_mode=RunningMode.VIDEO,
                num_faces=1,
            )

            # ------------------------------------------
            # OPEN WEBCAM
            # ------------------------------------------

            self.camera = cv2.VideoCapture(CAMERA_INDEX)

            if not self.camera.isOpened():
                raise RuntimeError("Could not open webcam")

            print("[CAMERA] Webcam started successfully.", flush=True)

            with FaceLandmarker.create_from_options(options) as landmarker:

                while not self.stop_event.is_set():

                    success, frame = self.camera.read()

                    if not success:
                        raise RuntimeError(
                            "Unable to read webcam frame"
                        )

                    frame = cv2.flip(frame, 1)

                    height, width = frame.shape[:2]

                    rgb = cv2.cvtColor(
                        frame,
                        cv2.COLOR_BGR2RGB
                    )

                    image = mp.Image(
                        image_format=mp.ImageFormat.SRGB,
                        data=rgb
                    )

                    result = landmarker.detect_for_video(
                        image,
                        timestamp_ms
                    )

                    timestamp_ms += 33

                    now = time.monotonic()

                    previous_status = self.get_status()

                    model_status = previous_status["model_status"]

                    drowsy_probability = previous_status[
                        "drowsy_probability"
                    ]

                    face_detected = bool(result.face_landmarks)

                    ear = None
                    mar = None

                    # ----------------------------------
                    # FACE DETECTION
                    # ----------------------------------

                    if face_detected:

                        landmarks = result.face_landmarks[0]

                        def points(indices):

                            return np.array(
                                [
                                    [
                                        landmarks[i].x * width,
                                        landmarks[i].y * height,
                                    ]
                                    for i in indices
                                ],
                                dtype=np.float32
                            )

                        # ----------------------------------
                        # EAR / MAR CALCULATION
                        # ----------------------------------

                        def ratio(p):

                            horizontal = np.linalg.norm(
                                p[0] - p[3]
                            )

                            if horizontal < 1e-6:
                                return None

                            v1 = np.linalg.norm(
                                p[1] - p[5]
                            )

                            v2 = np.linalg.norm(
                                p[2] - p[4]
                            )

                            return float(
                                (v1 + v2) / (2 * horizontal)
                            )

                        left_ear = ratio(points(LEFT_EYE))
                        right_ear = ratio(points(RIGHT_EYE))

                        valid_ears = [
                            x for x in (left_ear, right_ear)
                            if x is not None
                        ]

                        if valid_ears:
                            ear = float(np.mean(valid_ears))

                        mouth = points([
                            MOUTH_LEFT,
                            MOUTH_RIGHT,
                            MOUTH_TOP,
                            MOUTH_BOTTOM,
                        ])

                        mouth_width = np.linalg.norm(
                            mouth[0] - mouth[1]
                        )

                        if mouth_width > 1e-6:

                            mar = float(
                                np.linalg.norm(
                                    mouth[2] - mouth[3]
                                ) / mouth_width
                            )

                        # ----------------------------------
                        # FEATURE EXTRACTION
                        # ----------------------------------

                        if ear is not None and mar is not None:

                            if (
                                now - last_sample_time
                                >= SAMPLE_INTERVAL_SECONDS
                            ):

                                last_sample_time = now

                                yawn_event = yawn_detector.update(mar)

                                extractor.add_sample(ear, mar)

                                if yawn_event:
                                    yawn_times.append(now)

                                while (
                                    yawn_times
                                    and now - yawn_times[0]
                                    > YAWN_WINDOW_SECONDS
                                ):
                                    yawn_times.popleft()

                                # ----------------------------------
                                # MODEL PREDICTION
                                # ----------------------------------

                                if extractor.ready():

                                    features = extractor.get_features()

                                    values = [
                                        features[name]
                                        for name in FEATURES
                                    ]

                                    prediction = predict_from_features(
                                        values
                                    )

                                    model_status = prediction["status"]

                                    probabilities = prediction[
                                        "probabilities"
                                    ]

                                    drowsy_probability = (
                                        probabilities.get("1", 0.0)
                                    )

                                    if model_status == "Drowsy":

                                        if drowsy_start_time is None:
                                            drowsy_start_time = now

                                    else:

                                        drowsy_start_time = None

                                else:

                                    model_status = "Warming up"
                                    drowsy_probability = None

                        else:

                            model_status = "Warming up"
                            drowsy_probability = None

                        # ----------------------------------
                        # DRAW FACIAL LANDMARKS
                        # ----------------------------------

                        for point in landmarks:

                            x = int(point.x * width)
                            y = int(point.y * height)

                            if 0 <= x < width and 0 <= y < height:

                                cv2.circle(
                                    frame,
                                    (x, y),
                                    1,
                                    (0, 255, 0),
                                    -1
                                )

                    else:

                        yawn_detector.reset()
                        extractor.reset()

                        drowsy_start_time = None

                        model_status = "No face detected"
                        drowsy_probability = None

                    # ----------------------------------
                    # WARNING CALCULATION
                    # ----------------------------------

                    recent_yawns = len(yawn_times)

                    model_warning = (
                        model_status == "Drowsy"
                        and drowsy_start_time is not None
                        and now - drowsy_start_time
                        >= DROWSY_DURATION_SECONDS
                    )

                    yawn_warning = (
                        recent_yawns >= YAWN_COUNT_THRESHOLD
                    )

                    warning = model_warning or yawn_warning

                    # ----------------------------------
                    # VOICE ALERT
                    # ----------------------------------

                    if warning and not previous_warning:

                        print(
                            "[CAMERA] Drowsiness warning detected.",
                            flush=True
                        )

                        alert_service.trigger_alert()

                    previous_warning = warning

                    # ----------------------------------
                    # ARDUINO CONTROL
                    # ----------------------------------

                    # Drowsy  -> ALERT
                    # Alert   -> CLEAR
                    # Warming up -> CLEAR
                    # No face -> CLEAR

                    if model_status == "Drowsy":
                        desired_arduino_command = "ALERT"

                    elif model_status == "Alert":
                        desired_arduino_command = "NORMAL"

                    else:
                        desired_arduino_command = "CLEAR"
    

                    # Diagnostic log on every camera loop.
                    print(
                        f"[DEBUG] Model={model_status}, "
                        f"Arduino controller attached="
                        f"{self.arduino_controller is not None}",
                        flush=True,
                    )

                    if desired_arduino_command != last_arduino_command:

                        if self.arduino_controller is not None:

                            try:

                                sent = self.arduino_controller.send(
                                    desired_arduino_command
                                )

                                print(
                                    f"[CAMERA] Model={model_status}; "
                                    f"Arduino {desired_arduino_command}: {sent}",
                                    flush=True,
                                )

                            except Exception as exc:

                                print(
                                    f"[ARDUINO ERROR] {exc}",
                                    flush=True
                                )

                        else:

                            print(
                                "[ARDUINO ERROR] Controller is not attached!",
                                flush=True
                            )

                        last_arduino_command = desired_arduino_command

                    # ----------------------------------
                    # DISPLAY STATUS
                    # ----------------------------------

                    if warning:

                        display_text = (
                            "DROWSINESS ALERT - PULL OVER SAFELY"
                        )

                        color = (0, 0, 255)

                    else:

                        display_text = f"Status: {model_status}"

                        color = (0, 200, 0)

                    cv2.putText(
                        frame,
                        display_text,
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        color,
                        2
                    )

                    if ear is not None and mar is not None:

                        cv2.putText(
                            frame,
                            f"EAR: {ear:.3f}  MAR: {mar:.3f}",
                            (20, 70),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 0),
                            2
                        )

                    cv2.putText(
                        frame,
                        f"Recent yawns: {recent_yawns}/3",
                        (20, 105),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 255),
                        2
                    )

                    # ----------------------------------
                    # ENCODE VIDEO FRAME
                    # ----------------------------------

                    encoded, buffer = cv2.imencode(
                        ".jpg",
                        frame,
                        [cv2.IMWRITE_JPEG_QUALITY, 80]
                    )

                    if encoded:

                        with self.lock:

                            self.latest_frame = buffer.tobytes()

                            self.status = {
                                "camera": "Running",
                                "face_detected": face_detected,
                                "model_status": model_status,
                                "ear": ear,
                                "mar": mar,
                                "drowsy_probability": (
                                    drowsy_probability
                                ),
                                "recent_yawns": recent_yawns,
                                "warning": warning,
                                "message": display_text,
                            }

        # ----------------------------------------------
        # ERROR HANDLING
        # ----------------------------------------------

        except Exception as exc:

            self.error = str(exc)

            print(
                f"[CAMERA ERROR] {exc}",
                flush=True
            )

            with self.lock:

                self.status["camera"] = "Error"
                self.status["message"] = str(exc)

        # ----------------------------------------------
        # CLEANUP
        # ----------------------------------------------

        finally:

            # Always switch OFF LEDs and buzzer.
            if self.arduino_controller is not None:

                try:

                    sent = self.arduino_controller.send("CLEAR")

                    print(
                        f"[CAMERA EXIT] Arduino CLEAR command: {sent}",
                        flush=True,
                    )

                except Exception as exc:

                    print(
                        f"[ARDUINO ERROR] {exc}",
                        flush=True
                    )

            if self.camera:

                self.camera.release()
                self.camera = None

            self.running = False

            print("[CAMERA] Camera processing stopped.", flush=True)


# --------------------------------------------------
# SINGLE CAMERA SERVICE INSTANCE
# --------------------------------------------------

camera_service = CameraService()