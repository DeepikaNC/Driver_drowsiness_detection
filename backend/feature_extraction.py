
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np


# --------------------------------------------------
# Configuration
# --------------------------------------------------

MODEL_PATH = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_landmarker.task"
)

CAMERA_INDEX = 0

# MediaPipe Face Landmarker indices.
# Six points ordered around each eye.
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

MOUTH = [78, 81, 13, 308, 14, 178]


# --------------------------------------------------
# Landmark utilities
# --------------------------------------------------

def landmark_xy(landmarks, indices, width, height):
    """Convert normalized landmark coordinates to pixel coordinates."""
    return np.array(
        [
            [landmarks[i].x * width, landmarks[i].y * height]
            for i in indices
        ],
        dtype=np.float32,
    )


def calculate_ratio(points):
    """Calculate a six-point vertical-to-horizontal ratio."""
    vertical_1 = np.linalg.norm(points[1] - points[5])
    vertical_2 = np.linalg.norm(points[2] - points[4])
    horizontal = np.linalg.norm(points[0] - points[3])

    if horizontal < 1e-6:
        return None

    return float(
        (vertical_1 + vertical_2) / (2.0 * horizontal)
    )


# --------------------------------------------------
# EAR and MAR calculations
# --------------------------------------------------

def calculate_ear(landmarks, eye_indices, width, height):
    """Calculate EAR for one eye."""
    points = landmark_xy(
        landmarks, eye_indices, width, height
    )
    return calculate_ratio(points)




def calculate_mar(landmarks, width, height):
    """Calculate six-point mouth aspect ratio."""
    points = landmark_xy(
        landmarks, MOUTH, width, height
    )
    return calculate_ratio(points)




# --------------------------------------------------
# Webcam and face-landmark detection
# --------------------------------------------------

def main():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Face Landmarker model not found: {MODEL_PATH}"
        )

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

    camera = cv2.VideoCapture(CAMERA_INDEX)

    if not camera.isOpened():
        print(
            "Could not open webcam. "
            "Check camera permissions."
        )
        return

    print("Webcam started. Press Q to quit.")

    try:
        with FaceLandmarker.create_from_options(
            options
        ) as landmarker:

            frame_timestamp_ms = 0

            while True:
                success, frame = camera.read()

                if not success:
                    print("Could not read a webcam frame.")
                    break

                # Mirror the webcam image.
                frame = cv2.flip(frame, 1)

                height, width = frame.shape[:2]

                rgb_frame = cv2.cvtColor(
                    frame, cv2.COLOR_BGR2RGB
                )

                mp_image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                result = landmarker.detect_for_video(
                    mp_image,
                    frame_timestamp_ms,
                )

                # VIDEO mode requires increasing timestamps.
                frame_timestamp_ms += 33

                if result.face_landmarks:
                    landmarks = result.face_landmarks[0]

                    # Calculate EAR for both eyes.
                    left_ear = calculate_ear(
                        landmarks,
                        LEFT_EYE,
                        width,
                        height,
                    )

                    right_ear = calculate_ear(
                        landmarks,
                        RIGHT_EYE,
                        width,
                        height,
                    )

                    valid_ears = [
                        value
                        for value in (left_ear, right_ear)
                        if value is not None
                    ]

                    ear = (
                        float(np.mean(valid_ears))
                        if valid_ears
                        else None
                    )

                    # Calculate MAR.
                    mar = calculate_mar(
                        landmarks, width, height
                    )

                    # Draw facial landmarks.
                    for point in landmarks:
                        x = int(point.x * width)
                        y = int(point.y * height)

                        if 0 <= x < width and 0 <= y < height:
                            cv2.circle(
                                frame,
                                (x, y),
                                1,
                                (0, 255, 0),
                                -1,
                            )

                    # Display live measurements.
                    if ear is not None and mar is not None:
                        display_text = (
                            f"EAR: {ear:.3f}  "
                            f"MAR: {mar:.3f}"
                        )

                        cv2.putText(
                            frame,
                            display_text,
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.8,
                            (0, 255, 0),
                            2,
                        )

                        cv2.putText(
                            frame,
                            "Face detected",
                            (20, 75),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 255, 0),
                            2,
                        )

                    else:
                        cv2.putText(
                            frame,
                            "Could not calculate EAR/MAR",
                            (20, 40),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (0, 0, 255),
                            2,
                        )

                else:
                    cv2.putText(
                        frame,
                        "No face detected",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2,
                    )

                cv2.imshow(
                    "Driver Drowsiness - EAR and MAR",
                    frame,
                )

                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

    finally:
        camera.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
