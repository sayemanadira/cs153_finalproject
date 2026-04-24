import cv2
import joblib
import mediapipe as mp
import math
from pathlib import Path
from typing import Any


def _get_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "gesture_landmark_classifier.joblib"


def _get_expected_feature_count(classifier: Any) -> int | None:
    if hasattr(classifier, "n_features_in_"):
        return int(classifier.n_features_in_)
    return None


def _dist3(a: tuple[float, float, float], b: tuple[float, float, float]) -> float:
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    dz = a[2] - b[2]
    return math.sqrt(dx * dx + dy * dy + dz * dz)


def _extract_features(hand_landmarks: Any, expected_features: int | None) -> list[float]:
    points = [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]

    xyz_features: list[float] = []
    for x, y, z in points:
        xyz_features.extend([x, y, z])

    if expected_features == 42:
        xy_features: list[float] = []
        for x, y, _ in points:
            xy_features.extend([x, y])
        return xy_features

    if expected_features == 72:
        wrist = points[0]
        thumb_tip = points[4]
        index_tip = points[8]
        middle_tip = points[12]
        ring_tip = points[16]
        pinky_tip = points[20]

        geo_features = [
            _dist3(wrist, thumb_tip),
            _dist3(wrist, index_tip),
            _dist3(wrist, middle_tip),
            _dist3(wrist, ring_tip),
            _dist3(wrist, pinky_tip),
            _dist3(thumb_tip, index_tip),
            _dist3(index_tip, middle_tip),
            _dist3(middle_tip, ring_tip),
            _dist3(ring_tip, pinky_tip),
        ]
        return xyz_features + geo_features

    if expected_features is None or expected_features == 63:
        return xyz_features

    if len(xyz_features) >= expected_features:
        return xyz_features[:expected_features]

    # Fallback when the model expects a non-standard feature count.
    return xyz_features + [0.0] * (expected_features - len(xyz_features))


def _predict_label(classifier, features: list[float]) -> tuple[str | None, str | None]:
    try:
        pred = classifier.predict([features])
        if pred is None or len(pred) == 0:
            return None, None
        return str(pred[0]), None
    except Exception as exc:
        return None, str(exc)


def main() -> None:
    """Run webcam preview with MediaPipe landmarks and gesture classification."""
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    classifier = None
    expected_features = None
    model_path = _get_model_path()
    if model_path.exists():
        try:
            classifier = joblib.load(model_path)
            expected_features = _get_expected_feature_count(classifier)
            print(f"Loaded classifier: {model_path}")
            if expected_features is not None:
                print(f"Classifier expects {expected_features} features")
        except Exception as exc:
            print(f"Could not load classifier at {model_path}: {exc}")
    else:
        print(f"Classifier file not found at {model_path}. Running landmarks only.")

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )
    prediction_warning_printed = False

    cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

    if not cap.isOpened():
        cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError(
            "Could not open webcam. Check macOS Camera permission for Terminal/VS Code and ensure no other app is using the camera."
        )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                )

                label = None
                if classifier is not None:
                    features = _extract_features(hand_landmarks, expected_features)
                    label, prediction_error = _predict_label(classifier, features)
                    if prediction_error is not None and not prediction_warning_printed:
                        print(
                            "Classifier prediction failed. This usually means your training feature format does not match the runtime feature extraction. "
                            f"Details: {prediction_error}"
                        )
                        prediction_warning_printed = True

                if label is None:
                    label = "Unknown"

                wrist = hand_landmarks.landmark[0]
                h, w, _ = frame.shape
                x = max(10, min(int(wrist.x * w), w - 180))
                y = max(30, min(int(wrist.y * h) - 10, h - 10))

                cv2.putText(
                    frame,
                    f"Hand {idx + 1}: {label}",
                    (x, y),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

        cv2.imshow("MediaPipe Hands", frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
