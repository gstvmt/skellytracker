from types import SimpleNamespace
from typing import Literal, Tuple, Union

from skellytracker.trackers.base_tracker.model_info import ModelInfo


def april_tag_landmark_names(
    tag_ids: Tuple[int, ...],
    point_mode: Literal["corners", "center", "both"] = "corners",
) -> list[str]:
    """Stable ordering: iterate ``tag_ids`` in order; corners 0..3 per tag when ``point_mode=='corners'``."""
    names: list[str] = []
    for tag_id in tag_ids:
        if point_mode in ["center", "both"]:
            names.append(f"tag{tag_id}_center")
        if point_mode in ["corners", "both"]:
            for c in range(4):
                names.append(f"tag{tag_id}_corner{c}")
    return names


def april_tag_segment_connections(
    tag_ids: Tuple[int, ...],
    point_mode: Literal["corners", "center", "both"] = "corners",
) -> dict:
    """Generate connections for drawing squares around tags."""
    connections = {}
    if point_mode in ["corners", "both"]:
        for tag_id in tag_ids:
            for i in range(4):
                segment_name = f"tag{tag_id}_edge{i}"
                connections[segment_name] = {
                    "proximal": f"tag{tag_id}_corner{i}",
                    "distal": f"tag{tag_id}_corner{(i+1)%4}"
                }
    return connections


def april_tag_model_info_for(
    tag_ids: Tuple[int, ...],
    point_mode: Literal["corners", "center", "both"] = "corners",
) -> ModelInfo:
    """
    Build a model-info object compatible with ``process_folder_of_videos`` / FreeMoCap
    for a given tag layout (use the same ``tag_ids`` and ``point_mode`` in :class:`AprilTagTrackingParams`).
    """
    landmark_names = april_tag_landmark_names(tag_ids, point_mode)
    model_info = ModelInfo()
    model_info.name = "apriltag"
    model_info.tracker_name = "AprilTagTracker"
    model_info.landmark_names = landmark_names
    model_info.num_tracked_points = len(landmark_names)
    model_info.segment_connections = april_tag_segment_connections(tag_ids, point_mode)
    
    # Extra fields for custom reference
    model_info["tag_ids"] = tag_ids
    model_info["point_mode"] = point_mode
    return model_info


class AprilTagModelInfo(ModelInfo):
    """Default AprilTag layout: tags ``0..3``, four corners + center each."""

    name = "apriltag"
    tracker_name = "AprilTagTracker"
    tag_ids: Tuple[int, ...] = (0, 1, 2, 3)
    point_mode: Literal["corners", "center", "both"] = "both"
    landmark_names = april_tag_landmark_names(tag_ids, point_mode)
    num_tracked_points = len(landmark_names)
    segment_connections = april_tag_segment_connections(tag_ids, point_mode)
