import numpy as np
import pytest

from skellytracker.trackers.apriltag_tracker.apriltag_model_info import (
    AprilTagModelInfo,
    april_tag_landmark_names,
    april_tag_model_info_for,
)
from skellytracker.trackers.apriltag_tracker.apriltag_recorder import AprilTagRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


def test_april_tag_landmark_names_corners_order():
    names = april_tag_landmark_names((2, 0), "corners")
    assert names == [
        "tag2_corner0",
        "tag2_corner1",
        "tag2_corner2",
        "tag2_corner3",
        "tag0_corner0",
        "tag0_corner1",
        "tag0_corner2",
        "tag0_corner3",
    ]


def test_april_tag_landmark_names_center_mode():
    names = april_tag_landmark_names((10, 11), "center")
    assert names == ["tag10_center", "tag11_center"]


def test_april_tag_model_info_defaults():
    assert AprilTagModelInfo.name == "apriltag"
    assert AprilTagModelInfo.num_tracked_points == 16
    assert len(AprilTagModelInfo.landmark_names) == 16


def test_april_tag_model_info_for_matches_params_length():
    tag_ids = (0, 1)
    info = april_tag_model_info_for(tag_ids, "corners")
    assert info.name == "apriltag"
    assert info.tracker_name == "AprilTagTracker"
    assert info.num_tracked_points == 8
    assert len(info.landmark_names) == 8


def test_april_tag_recorder_nan_when_missing():
    names = april_tag_landmark_names((0,), "corners")
    recorder = AprilTagRecorder(num_tracked_points=len(names))
    frame_objects = [TrackedObject(object_id=n) for n in names]
    recorder.record({n: o for n, o in zip(names, frame_objects)})
    out = recorder.process_tracked_objects()
    assert out.shape == (1, 4, 3)
    assert np.isnan(out).all()


def test_april_tag_recorder_fills_xyz():
    names = april_tag_landmark_names((0,), "center")
    recorder = AprilTagRecorder(num_tracked_points=len(names))
    obj = TrackedObject(object_id=names[0], pixel_x=12.5, pixel_y=34.0)
    recorder.record({names[0]: obj})
    out = recorder.process_tracked_objects()
    assert out.shape == (1, 1, 3)
    assert out[0, 0, 0] == 12.5
    assert out[0, 0, 1] == 34.0
    assert out[0, 0, 2] == 0.0


def test_apriltag_tracker_blank_frame_no_detection():
    pytest.importorskip("pupil_apriltags")
    from skellytracker.trackers.apriltag_tracker.apriltag_tracker import AprilTagTracker

    tracker = AprilTagTracker(tag_ids=(0,), point_mode="center")
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    tracked = tracker.process_image(frame)
    assert tracked["tag0_center"].pixel_x is None
    assert tracked["tag0_center"].pixel_y is None


def test_get_tracker_apriltag_factory():
    pytest.importorskip("pupil_apriltags")
    from skellytracker.process_folder_of_videos import get_tracker, get_tracker_params
    from skellytracker.trackers.apriltag_tracker.apriltag_tracking_params import (
        AprilTagTrackingParams,
    )

    params = get_tracker_params("AprilTagTracker")
    assert isinstance(params, AprilTagTrackingParams)
    tracker = get_tracker("AprilTagTracker", params)
    assert tracker.__class__.__name__ == "AprilTagTracker"


def test_apriltag_tracker_process_video_tmp_path(tmp_path):
    pytest.importorskip("pupil_apriltags")
    import cv2

    from skellytracker.trackers.apriltag_tracker.apriltag_tracker import AprilTagTracker

    video_path = tmp_path / "cam0.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(video_path), fourcc, 5.0, (64, 48))
    if not writer.isOpened():
        pytest.skip("OpenCV VideoWriter not available for mp4 on this system")
    for _ in range(3):
        writer.write(np.zeros((48, 64, 3), dtype=np.uint8))
    writer.release()

    tracker = AprilTagTracker(tag_ids=(0,), point_mode="center")
    out = tracker.process_video(
        input_video_filepath=video_path,
        output_video_filepath=None,
        save_data_bool=False,
        use_tqdm=False,
    )
    assert out is not None
    assert out.shape[1:] == (1, 3)
    assert out.shape[0] >= 1
    assert np.isnan(out).all()
