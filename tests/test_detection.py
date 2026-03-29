from drive_text.detection import TrackState


def build_track(points, boxes):
    track = TrackState(track_id=1, class_name="car")
    for idx, point in enumerate(points):
        frame_index, x, y = point
        track.points.append((frame_index, x, y))
        track.bboxes.append(boxes[idx])
        track.confidences.append(0.9)
    return track


def test_speed_bucket_stopped():
    track = build_track(
        [(0, 100.0, 100.0), (3, 101.0, 100.5), (6, 101.2, 100.8)],
        [(90.0, 90.0, 130.0, 120.0), (90.5, 90.2, 130.5, 120.2), (91.0, 90.4, 131.0, 120.4)],
    )
    assert track.speed_bucket(1280, 720) == "stopped"


def test_speed_bucket_fast():
    track = build_track(
        [(0, 100.0, 100.0), (3, 180.0, 130.0), (6, 270.0, 170.0)],
        [(90.0, 90.0, 130.0, 120.0), (170.0, 120.0, 210.0, 150.0), (260.0, 160.0, 300.0, 190.0)],
    )
    assert track.speed_bucket(1280, 720) in {"moderate", "fast"}
