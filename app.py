"""A local webcam meme-mask demo powered by MediaPipe face landmarks."""

import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp


APP_FOLDER = Path(__file__).resolve().parent
MEME_FOLDER = APP_FOLDER / "memes"
MEME_PHOTO_FOLDER = MEME_FOLDER / "photos"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
FEATURE_SCALES = (0.12, 0.6, 0.12, 0.5, 0.5, 0.15, 0.15, 0.1, 0.4)
MEME_CONTEXT_SCALE = 2.6
MATCH_STABILITY_SECONDS = 0.9
MATCH_SWITCH_MARGIN = 0.2
SIGNATURE_SMOOTHING = 0.28
WHITE = (250, 250, 250)
DARK = (30, 36, 46)
MUTED = (174, 184, 194)


@dataclass(frozen=True)
class MemeEntry:
    """One independent photo and the face measurements found inside it."""

    path: Path
    signature: tuple[float, ...] | None
    face_box: tuple[float, float, float, float] | None


FACE_CONNECTIONS = (
    mp.solutions.face_mesh.FACEMESH_TESSELATION
    | mp.solutions.face_mesh.FACEMESH_CONTOURS
    | mp.solutions.face_mesh.FACEMESH_IRISES
)
MESH_STYLE = mp.solutions.drawing_utils.DrawingSpec(color=(90, 245, 220), thickness=1, circle_radius=1)


def safe_ratio(numerator: float, denominator: float) -> float:
    """Divide measurements without crashing on a nearly zero-width feature."""
    return numerator / max(abs(denominator), 0.000001)


def extract_face_signature(landmarks) -> tuple[float, ...]:
    """Describe one face with normalized eye, brow, lip, and mouth measurements."""
    face_height = abs(landmarks[10].y - landmarks[152].y)
    face_width = abs(landmarks[234].x - landmarks[454].x)
    if face_height == 0 or face_width == 0:
        return (0.0,) * len(FEATURE_SCALES)

    mouth_open = safe_ratio(landmarks[13].y - landmarks[14].y, face_height)
    mouth_width = safe_ratio(abs(landmarks[61].x - landmarks[291].x), face_width)
    left_eye_open = safe_ratio(
        abs(landmarks[159].y - landmarks[145].y),
        abs(landmarks[33].x - landmarks[133].x),
    )
    right_eye_open = safe_ratio(
        abs(landmarks[386].y - landmarks[374].y),
        abs(landmarks[362].x - landmarks[263].x),
    )
    lip_center_y = (landmarks[13].y + landmarks[14].y) / 2
    mouth_corner_y = (landmarks[61].y + landmarks[291].y) / 2
    smile_lift = (lip_center_y - mouth_corner_y) / face_height
    left_brow_raise = (landmarks[159].y - landmarks[105].y) / face_height
    right_brow_raise = (landmarks[386].y - landmarks[334].y) / face_height
    mouth_asymmetry = (landmarks[61].y - landmarks[291].y) / face_height
    eye_asymmetry = left_eye_open - right_eye_open
    return (
        mouth_open,
        mouth_width,
        smile_lift,
        left_eye_open,
        right_eye_open,
        left_brow_raise,
        right_brow_raise,
        mouth_asymmetry,
        eye_asymmetry,
    )


def find_face_box(landmarks, frame_width: int, frame_height: int) -> tuple[int, int, int, int] | None:
    """Get a slightly padded pixel rectangle around all face landmarks."""
    points = [
        (int(point.x * frame_width), int(point.y * frame_height))
        for point in landmarks
    ]
    left = min(point[0] for point in points)
    top = min(point[1] for point in points)
    right = max(point[0] for point in points)
    bottom = max(point[1] for point in points)
    padding_x = int((right - left) * 0.12)
    padding_y = int((bottom - top) * 0.12)

    left = max(0, left - padding_x)
    top = max(0, top - padding_y)
    right = min(frame_width, right + padding_x)
    bottom = min(frame_height, bottom + padding_y)
    if right <= left or bottom <= top:
        return None
    return left, top, right, bottom


def find_meme_files(folder: Path = MEME_PHOTO_FOLDER) -> list[Path]:
    """List every photo as its own gallery item; do not group by expression."""
    if not folder.is_dir():
        return []
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def build_meme_index(paths: list[Path]) -> list[MemeEntry]:
    """Measure the first detectable face in each local photo once at startup."""
    if not paths:
        return []

    entries = []
    face_mesh = mp.solutions.face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
    )
    try:
        for path in paths:
            image = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
            if image is None:
                entries.append(MemeEntry(path, None, None))
                continue
            if image.ndim == 2:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGB)
            else:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            image_height, image_width = rgb_image.shape[:2]
            scale = min(1.0, 640 / max(image_width, image_height))
            if scale < 1:
                rgb_image = cv2.resize(rgb_image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            analysis_height, analysis_width = rgb_image.shape[:2]
            results = face_mesh.process(rgb_image)
            if not results.multi_face_landmarks:
                entries.append(MemeEntry(path, None, None))
                continue

            face = results.multi_face_landmarks[0]
            box = find_face_box(face.landmark, analysis_width, analysis_height)
            normalized_box = None
            if box is not None:
                left, top, right, bottom = box
                normalized_box = (
                    left / analysis_width,
                    top / analysis_height,
                    right / analysis_width,
                    bottom / analysis_height,
                )
            entries.append(MemeEntry(path, extract_face_signature(face.landmark), normalized_box))
    finally:
        face_mesh.close()
    return entries


def meme_feature_distance(
    live_signature: tuple[float, ...], meme_signature: tuple[float, ...]
) -> float:
    """Return weighted squared distance; lower means geometrically more similar."""
    return sum(
        ((live_value - meme_value) / feature_scale) ** 2
        for live_value, meme_value, feature_scale in zip(
            live_signature, meme_signature, FEATURE_SCALES, strict=True
        )
    )


def smooth_signature(
    previous: tuple[float, ...] | None,
    current: tuple[float, ...],
    weight: float = SIGNATURE_SMOOTHING,
) -> tuple[float, ...]:
    """Blend frame-to-frame measurements to reduce tiny tracking fluctuations."""
    if previous is None:
        return current
    return tuple(
        old_value * (1 - weight) + new_value * weight
        for old_value, new_value in zip(previous, current, strict=True)
    )


def is_clear_match_improvement(candidate_distance: float, active_distance: float | None) -> bool:
    """Require a meaningful score improvement before replacing the current photo."""
    return active_distance is None or candidate_distance + MATCH_SWITCH_MARGIN < active_distance


def find_best_meme(signature: tuple[float, ...], entries: list[MemeEntry]) -> MemeEntry | None:
    """Find the individual meme photo with the closest face-feature measurements."""
    candidates = [entry for entry in entries if entry.signature is not None]
    if not candidates:
        return None
    return min(candidates, key=lambda entry: meme_feature_distance(signature, entry.signature))


def load_meme_image(entry: MemeEntry):
    """Load an image with extra scene context around its detected face."""
    image = cv2.imread(str(entry.path), cv2.IMREAD_UNCHANGED)
    if image is None or entry.face_box is None:
        return image

    height, width = image.shape[:2]
    left, top, right, bottom = entry.face_box
    center_x = (left + right) * width / 2
    center_y = (top + bottom) * height / 2
    crop_width = (right - left) * width * MEME_CONTEXT_SCALE
    crop_height = (bottom - top) * height * MEME_CONTEXT_SCALE
    left_pixel = max(0, round(center_x - crop_width / 2))
    top_pixel = max(0, round(center_y - crop_height / 2))
    right_pixel = min(width, round(center_x + crop_width / 2))
    bottom_pixel = min(height, round(center_y + crop_height / 2))
    return image[top_pixel:bottom_pixel, left_pixel:right_pixel]


def resize_to_cover(image, target_width: int, target_height: int):
    """Resize and center-crop an image so it fills the face rectangle."""
    image_height, image_width = image.shape[:2]
    scale = max(target_width / image_width, target_height / image_height)
    resized_width = max(target_width, round(image_width * scale))
    resized_height = max(target_height, round(image_height * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (resized_width, resized_height), interpolation=interpolation)
    crop_x = (resized_width - target_width) // 2
    crop_y = (resized_height - target_height) // 2
    return resized[crop_y:crop_y + target_height, crop_x:crop_x + target_width]


def overlay_image(frame, image, face_box: tuple[int, int, int, int]) -> None:
    """Place a local image over the face, respecting transparent PNG pixels."""
    left, top, right, bottom = face_box
    target_width = right - left
    target_height = bottom - top
    sticker = resize_to_cover(image, target_width, target_height)
    face_region = frame[top:bottom, left:right]

    if sticker.ndim == 2:
        sticker = cv2.cvtColor(sticker, cv2.COLOR_GRAY2BGR)
    if sticker.shape[2] == 4:
        alpha = sticker[:, :, 3:4].astype("float32") / 255
        foreground = sticker[:, :, :3].astype("float32")
        background = face_region.astype("float32")
        frame[top:bottom, left:right] = (alpha * foreground + (1 - alpha) * background).astype("uint8")
    else:
        frame[top:bottom, left:right] = cv2.addWeighted(sticker, 0.93, face_region, 0.07, 0)


def draw_face_mesh(frame, face_landmarks) -> None:
    """Draw the tracked face triangles, feature outlines, and refined irises."""
    mp.solutions.drawing_utils.draw_landmarks(
        image=frame,
        landmark_list=face_landmarks,
        connections=FACE_CONNECTIONS,
        landmark_drawing_spec=MESH_STYLE,
        connection_drawing_spec=MESH_STYLE,
    )


def draw_status(
    frame,
    meme_name: str,
    match_distance: float | None,
    overlay_on: bool,
    mesh_on: bool,
    image_count: int,
    indexed_count: int,
) -> None:
    """Draw app status and shortcuts without covering the central camera view."""
    width = frame.shape[1]
    cv2.rectangle(frame, (0, 0), (width, 100), DARK, -1)
    cv2.putText(frame, "FACE MEME STUDIO", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.72, WHITE, 2)
    mask_mode = "MASK ON" if overlay_on else "MASK OFF"
    distance_text = f"distance {match_distance:.2f}" if match_distance is not None else "manual / no face match"
    cv2.putText(frame, f"{mask_mode}  |  {meme_name[:25]}  |  {distance_text}", (21, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.46, MUTED, 1)
    mesh_mode = "MESH ON" if mesh_on else "MESH OFF"
    cv2.putText(frame, f"{mesh_mode}  |  M: mask  D: mesh  S: next photo  Q: quit  |  {indexed_count}/{image_count} faces", (21, 84), cv2.FONT_HERSHEY_SIMPLEX, 0.4, WHITE, 1)


def main() -> None:
    camera = cv2.VideoCapture(0)
    if not camera.isOpened():
        raise RuntimeError(
            "Could not open the webcam. On macOS, allow Camera access for VS Code or Terminal in "
            "System Settings > Privacy & Security > Camera, then try again."
        )

    face_mesh = mp.solutions.face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )
    meme_paths = find_meme_files()
    library = build_meme_index(meme_paths)
    indexed_count = sum(entry.signature is not None for entry in library)
    print(f"Indexed {indexed_count} of {len(library)} photos with detectable faces.")
    candidate_entry = None
    candidate_started = 0.0
    smoothed_signature = None
    active_entry = None
    active_image = None
    active_distance = None
    current_index = -1
    manual_override_until = 0.0
    overlay_on = True
    mesh_on = False
    running = True

    try:
        while running:
            success, frame = camera.read()
            if not success:
                break

            frame = cv2.flip(frame, 1)
            height, width = frame.shape[:2]
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            face = results.multi_face_landmarks[0] if results.multi_face_landmarks else None
            face_box = None

            if face is not None:
                landmarks = face.landmark
                face_box = find_face_box(landmarks, width, height)

                if time.monotonic() >= manual_override_until:
                    raw_signature = extract_face_signature(landmarks)
                    smoothed_signature = smooth_signature(smoothed_signature, raw_signature)
                    live_signature = smoothed_signature
                    best_entry = find_best_meme(live_signature, library)
                    if best_entry is not None:
                        best_distance = meme_feature_distance(live_signature, best_entry.signature)
                        active_signature = active_entry.signature if active_entry is not None else None
                        live_active_distance = (
                            meme_feature_distance(live_signature, active_signature)
                            if active_signature is not None
                            else None
                        )

                        if active_entry is not None and best_entry.path == active_entry.path:
                            active_distance = best_distance
                            candidate_entry = None
                        elif is_clear_match_improvement(best_distance, live_active_distance):
                            if candidate_entry is None or best_entry.path != candidate_entry.path:
                                candidate_entry = best_entry
                                candidate_started = time.monotonic()
                            elif time.monotonic() - candidate_started >= MATCH_STABILITY_SECONDS:
                                active_entry = candidate_entry
                                active_image = load_meme_image(active_entry)
                                active_distance = best_distance
                                print(
                                    f"Matched meme: {active_entry.path.name} "
                                    f"(feature distance {active_distance:.2f}; lower is closer)"
                                )
                                current_index = next(
                                    index for index, entry in enumerate(library)
                                    if entry.path == active_entry.path
                                )
                                candidate_entry = None
                        else:
                            candidate_entry = None

                if mesh_on:
                    draw_face_mesh(frame, face)

                if overlay_on and face_box is not None and active_image is not None:
                    overlay_image(frame, active_image, face_box)

            meme_name = active_entry.path.name if active_entry is not None else "ADD PHOTOS TO memes/photos"
            draw_status(
                frame,
                meme_name,
                active_distance,
                overlay_on,
                mesh_on,
                len(library),
                indexed_count,
            )
            cv2.imshow("Face Meme Studio", frame)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                running = False
            elif key == ord("m"):
                overlay_on = not overlay_on
            elif key == ord("d"):
                mesh_on = not mesh_on
            elif key == ord("s") and library:
                current_index = (current_index + 1) % len(library)
                active_entry = library[current_index]
                active_image = load_meme_image(active_entry)
                active_distance = None
                candidate_entry = active_entry
                manual_override_until = time.monotonic() + 2.0
                smoothed_signature = None

    except KeyboardInterrupt:
        pass
    finally:
        camera.release()
        face_mesh.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()