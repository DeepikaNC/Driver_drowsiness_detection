import threading
import time
from collections import deque
from pathlib import Path
from backend.arduino_controller import ArduinoController
import cv2
import mediapipe as mp
import numpy as np

from backend.voice_service import run_conversation
from backend.temporal_features import (
    TemporalFeatureExtractor,
    FEATURES,
    YawnDetector,
)
from backend.model_service import predict_from_features


# ======================== CONFIGURATION ========================

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_landmarker.task"
)

VOICE_ALERT_COOLDOWN_SECONDS = 30.0
_last_voice_alert_time = 0.0

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH_LEFT = 61
MOUTH_RIGHT = 291
MOUTH_TOP = 13
MOUTH_BOTTOM = 14

SAMPLE_INTERVAL_MS = 1000
SAMPLE_INTERVAL_SECONDS = 1.0

# Initial webcam-testing settings; calibrate with recorded examples.
MAR_THRESHOLD = 0.45
YAWN_DURATION_SECONDS = 1.0

# Alert conditions
DROWSY_DURATION_SECONDS = 6.0
YAWN_COUNT_THRESHOLD = 3
YAWN_WINDOW_SECONDS = 60.0


# ======================== FEATURE CALCULATIONS ========================

def get_points(landmarks, indices, width, height):
    return np.array(
        [[landmarks[i].x * width, landmarks[i].y * height]
         for i in indices],
        dtype=np.float32,
    )


def aspect_ratio(points):
    horizontal = np.linalg.norm(points[0] - points[3])
    if horizontal < 1e-6:
        return None

    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    return float((vertical_1 + vertical_2) / (2.0 * horizontal))


def get_ear(landmarks, indices, width, height):
    return aspect_ratio(get_points(landmarks, indices, width, height))


def get_mar(landmarks, width, height):
    points = get_points(
        landmarks,
        [MOUTH_LEFT, MOUTH_RIGHT, MOUTH_TOP, MOUTH_BOTTOM],
        width,
        height,
    )
    horizontal = np.linalg.norm(points[0] - points[1])
    if horizontal < 1e-6:
        return None

    vertical = np.linalg.norm(points[2] - points[3])
    return float(vertical / horizontal)


# ======================== VOICE ASSISTANT ========================

def start_voice_assistant(alert_event, stop_event):
    try:
        run_conversation(alert_event, stop_event)
    except Exception as exc:
        print(f"[VOICE ERROR] {type(exc).__name__}: {exc}", flush=True)


def signal_voice_alert(reason, alert_event, stop_event, voice_thread):
    """Start or signal the voice assistant, with a cooldown."""
    global _last_voice_alert_time

    now = time.monotonic()

    if now - _last_voice_alert_time < VOICE_ALERT_COOLDOWN_SECONDS:
        print("[VOICE] Alert suppressed by cooldown.", flush=True)
        return voice_thread

    if stop_event.is_set():
        return voice_thread

    _last_voice_alert_time = now
    print(f"[ALERT] {reason}", flush=True)

    if voice_thread is None or not voice_thread.is_alive():
        alert_event.clear()
        voice_thread = threading.Thread(
            target=start_voice_assistant,
            args=(alert_event, stop_event),
            daemon=True,
        )
        voice_thread.start()
        print("[VOICE] Assistant started.", flush=True)
    else:
        alert_event.set()
        print("[VOICE] Alert sent to active assistant.", flush=True)

    return voice_thread



# ======================== LIVE DETECTION ========================

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Face Landmarker model not found: {MODEL_PATH}")

    BaseOptions = mp.tasks.BaseOptions
    FaceLandmarker = mp.tasks.vision.FaceLandmarker
    FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
    RunningMode = mp.tasks.vision.RunningMode

    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=str(MODEL_PATH)),
        running_mode=RunningMode.VIDEO,
        num_faces=1,
    )

    extractor = TemporalFeatureExtractor()
    yawn_detector = YawnDetector(
        mar_threshold=MAR_THRESHOLD,
        duration_seconds=YAWN_DURATION_SECONDS,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    )

    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError("Could not open webcam.")

    last_sample_ms = -SAMPLE_INTERVAL_MS
    timestamp_ms = 0
    last_prediction = None

    # Yawns are counted as distinct events in a rolling time window.
    yawn_times = deque()
    yawn_event_count = 0
    yawn_alert_active = False

    alert_event = threading.Event()
    stop_event = threading.Event()
    voice_thread = None

    face_missing_start_time = None
    FACE_MISSING_GRACE_SECONDS = 1.5
    drowsy_start_time = None
    continuous_drowsy_seconds = 0.0
    drowsiness_alert_active = False
    drowsiness_alert_sent = False

    # Only these event messages are printed; EAR/MAR debug logs are omitted.
    arduino = ArduinoController(port="COM7")
    last_arduino_command = None
    print("Live drowsiness detection started.")
    print("Press Q in the webcam window to quit.")
    print(
        f"Triggers: model Drowsy for {DROWSY_DURATION_SECONDS:.0f}s OR "
        f"{YAWN_COUNT_THRESHOLD}+ yawns in {YAWN_WINDOW_SECONDS:.0f}s."
    )

    try:
        with FaceLandmarker.create_from_options(options) as landmarker:
            while True:
                success, frame = camera.read()
                if not success:
                    print("[CAMERA ERROR] Could not read webcam frame.")
                    break

                frame = cv2.flip(frame, 1)
                height, width = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                result = landmarker.detect_for_video(image, timestamp_ms)
                timestamp_ms += 33

                model_status = (
                    last_prediction["status"] if last_prediction else "Warming up"
                )
                alert_reasons = []

                if result.face_landmarks:
                    face_missing_start_time = None
                    landmarks = result.face_landmarks[0]
                    left = get_ear(landmarks, LEFT_EYE, width, height)
                    right = get_ear(landmarks, RIGHT_EYE, width, height)
                    mar = get_mar(landmarks, width, height)

                    valid_ears = [
                        value for value in (left, right)
                        if value is not None
                    ]
                    ear = float(np.mean(valid_ears)) if valid_ears else None

                    if ear is not None and mar is not None:
                        now_ms = timestamp_ms

                        if now_ms - last_sample_ms >= SAMPLE_INTERVAL_MS:
                            # Update yawn detector silently; no per-sample EAR/MAR logs.
                            yawn_event = yawn_detector.update(mar)
                            last_sample_ms = now_ms
                            extractor.add_sample(ear, mar)

                            if yawn_event:
                                yawn_event_count += 1
                                yawn_times.append(time.monotonic())
                                print(
                                    f"[YAWN] Distinct yawn detected "
                                    f"({yawn_event_count} total).",
                                    flush=True,
                                )

                            current_time = time.monotonic()
                            while (
                                yawn_times
                                and current_time - yawn_times[0] > YAWN_WINDOW_SECONDS
                            ):
                                yawn_times.popleft()

                            recent_yawns = len(yawn_times)

                            # Independent yawn trigger: does not depend on model result.
                            if (
                                recent_yawns >= YAWN_COUNT_THRESHOLD
                                and not yawn_alert_active
                            ):
                                yawn_alert_active = True
                                alert_reasons.append(
                                    f"Repeated yawns ({recent_yawns} within "
                                    f"{YAWN_WINDOW_SECONDS:.0f}s)"
                                )
                                print(
                                    "[DROWSY ALERT] Repeated-yawn rule triggered.",
                                    flush=True,
                                )
                                voice_thread = signal_voice_alert(
                                    "Repeated yawns detected. This is a possible "
                                    "drowsiness warning, even if the model predicts Alert.",
                                    alert_event,
                                    stop_event,
                                    voice_thread,
                                )

                            # Allow another yawn alert only after count drops below threshold.
                            if recent_yawns < YAWN_COUNT_THRESHOLD:
                                yawn_alert_active = False

                            # Keep the model prediction independent and unchanged.
                            if extractor.ready():
                                features = extractor.get_features()
                                values = [features[name] for name in FEATURES]
                                last_prediction = predict_from_features(values)
                                model_status = last_prediction["status"]
                                probabilities = last_prediction["probabilities"]

                                if model_status == "Drowsy":
                                    if drowsy_start_time is None:
                                        drowsy_start_time = time.monotonic()
                                    continuous_drowsy_seconds = (
                                        time.monotonic() - drowsy_start_time
                                    )
                                else:
                                    drowsy_start_time = None
                                    continuous_drowsy_seconds = 0.0
                                    drowsiness_alert_active = False
                                    drowsiness_alert_sent = False

                                if (
                                    model_status == "Drowsy"
                                    and continuous_drowsy_seconds >= DROWSY_DURATION_SECONDS
                                    and not drowsiness_alert_sent
                                ):
                                    drowsiness_alert_active = True
                                    drowsiness_alert_sent = True
                                    alert_reasons.append("Model sustained Drowsy prediction")
                                    voice_thread = signal_voice_alert(
                                        f"The model has predicted Drowsy continuously for "
                                        f"{continuous_drowsy_seconds:.1f} seconds.",
                                        alert_event,
                                        stop_event,
                                        voice_thread,
                                    )

                    # Draw facial landmarks without printing numerical EAR/MAR.
                    for point in landmarks:
                        x = int(point.x * width)
                        y = int(point.y * height)
                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                else:
                    yawn_detector.reset()

                    if face_missing_start_time is None:
                        face_missing_start_time = time.monotonic()

                    if (
                        time.monotonic() - face_missing_start_time
                        >= FACE_MISSING_GRACE_SECONDS
                    ):
                        drowsy_start_time = None
                        continuous_drowsy_seconds = 0.0
                        drowsiness_alert_active = False
                        drowsiness_alert_sent = False

                # Webcam status: the red warning is separate from the model output.
                current_time = time.monotonic()
                while (
                    yawn_times
                    and current_time - yawn_times[0] > YAWN_WINDOW_SECONDS
                ):
                    yawn_times.popleft()
                recent_yawns = len(yawn_times)

                yawn_triggered = recent_yawns >= YAWN_COUNT_THRESHOLD
                model_triggered = (
                    drowsiness_alert_active
                    and model_status == "Drowsy"
                    and continuous_drowsy_seconds >= DROWSY_DURATION_SECONDS
                )
                drowsy_warning = yawn_triggered or model_triggered
                if drowsy_warning:
                    new_command = "WARNING"
                elif model_status == "Drowsy":
                    new_command = "DROWSY"
                elif model_status == "Alert":
                    new_command = "ALERT"
                else:
                    new_command = "RESET"

                if new_command != last_arduino_command:
                    arduino.send(new_command)
                    last_arduino_command = new_command
                if drowsy_warning:
                    warning_text = "DROWSINESS ALERT - PULL OVER SAFELY"
                    warning_color = (0, 0, 255)  # Red in OpenCV BGR
                else:
                    warning_text = "Status: Monitoring"
                    warning_color = (0, 200, 0)

                cv2.putText(
                    frame, warning_text, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, warning_color, 2,
                )
                cv2.putText(
                    frame,
                    f"Recent yawns: {recent_yawns}/{YAWN_COUNT_THRESHOLD} "
                    f"in {YAWN_WINDOW_SECONDS:.0f}s",
                    (20, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (255, 255, 0), 2,
                )
                cv2.putText(
                    frame, f"Model: {model_status}", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    (0, 0, 255) if model_status == "Drowsy" else (255, 255, 255),
                    2,
                )

                cv2.imshow("Live Drowsiness Detection", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        if arduino:
            arduino.close()
        stop_event.set()
        alert_event.set()
        camera.release()
        cv2.destroyAllWindows()

        if voice_thread is not None and voice_thread.is_alive():
            voice_thread.join(timeout=2)

        print("Live detection stopped.")


if __name__ == "__main__":
    main()
