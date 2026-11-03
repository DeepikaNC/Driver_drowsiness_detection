
from collections import deque

import numpy as np

SAMPLE_INTERVAL_SECONDS = 1.0
MAX_SAMPLES = 15
WINDOW_SECONDS = MAX_SAMPLES * SAMPLE_INTERVAL_SECONDS


FEATURES = [
    "EAR_mean",
    "EAR_std",
    "EAR_low_percent",
    "MAR_mean",
    "MAR_max",
    "MAR_q90",
    "dEAR_std",
    "dMAR_std",
]


class YawnDetector:
    """
    Detect a possible yawn when MAR remains above a threshold
    for a configured duration.
    """

    def __init__(
        self,
        mar_threshold=0.60,
        duration_seconds=1.5,
        sample_interval_seconds=SAMPLE_INTERVAL_SECONDS,
    ):
        self.mar_threshold = mar_threshold
        self.duration_seconds = duration_seconds
        self.sample_interval_seconds = sample_interval_seconds

        self.open_samples = 0
        self.yawn_detected = False

    def update(self, mar):
        """
        Call once for every new MAR measurement.
        Returns True once when a possible yawn is detected.
        """
        if mar is None or not np.isfinite(mar):
            self.reset()
            return False

        if mar > self.mar_threshold:
            self.open_samples += 1

            elapsed_seconds = (
                self.open_samples * self.sample_interval_seconds
            )

            if (
                elapsed_seconds >= self.duration_seconds
                and not self.yawn_detected
            ):
                self.yawn_detected = True
                return True

        else:
            self.reset()

        return False

    def reset(self):
        self.open_samples = 0
        self.yawn_detected = False


class TemporalFeatureExtractor:
    def __init__(self):
        self.samples = deque(maxlen=MAX_SAMPLES)

    def add_sample(self, ear, mar):
        """Add one valid EAR/MAR measurement."""
        if ear is None or mar is None:
            return

        if not np.isfinite(ear) or not np.isfinite(mar):
            return

        self.samples.append({
            "EAR": float(ear),
            "MAR": float(mar),
        })

    def ready(self):
        """Return True after the window is filled."""
        return len(self.samples) == MAX_SAMPLES

    def reset(self):
        self.samples.clear()

    def get_features(self):
        """Calculate the eight features in model input order."""
        if not self.ready():
            raise ValueError(
                f"Need {MAX_SAMPLES} samples; "
                f"currently have {len(self.samples)}."
            )

        ear = np.array(
            [sample["EAR"] for sample in self.samples],
            dtype=float,
        )

        mar = np.array(
            [sample["MAR"] for sample in self.samples],
            dtype=float,
        )

        d_ear = np.diff(ear)
        d_mar = np.diff(mar)

        values = [
            float(np.mean(ear)),
            float(np.std(ear)),
            float(np.mean(ear < 0.20) * 100.0),
            float(np.mean(mar)),
            float(np.max(mar)),
            float(np.percentile(mar, 90)),
            float(np.std(d_ear)),
            float(np.std(d_mar)),
        ]

        return dict(zip(FEATURES, values))
