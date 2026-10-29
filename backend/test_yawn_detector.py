
from temporal_features import YawnDetector

detector = YawnDetector(
    mar_threshold=0.60,
    duration_seconds=1.5,
    sample_interval_seconds=0.5,
)

test_values = [
    0.30, 0.35,  # Mouth closed
    0.70, 0.72, 0.75,  # Sustained mouth opening
    0.30,  # Mouth closes
]

for i, mar in enumerate(test_values, start=1):
    detected = detector.update(mar)
    print(
        f"Sample {i}: MAR={mar:.2f}, "
        f"Possible yawn={detected}"
    )
