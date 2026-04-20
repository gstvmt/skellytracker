from copy import deepcopy
from typing import Dict

import numpy as np

from skellytracker.trackers.base_tracker.base_recorder import BaseRecorder
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class AprilTagRecorder(BaseRecorder):
    """Stacks AprilTag points as ``(num_frames, num_tracked_points, 3)`` with ``z=0`` when visible."""

    def __init__(self, num_tracked_points: int):
        super().__init__()
        self._num_tracked_points = num_tracked_points

    def record(self, tracked_objects: Dict[str, TrackedObject]) -> None:
        self.recorded_objects.append(
            [deepcopy(tracked_object) for tracked_object in tracked_objects.values()]
        )

    def process_tracked_objects(self, **kwargs) -> np.ndarray:
        out = np.full(
            (len(self.recorded_objects), self._num_tracked_points, 3),
            np.nan,
            dtype=float,
        )
        for frame_index, tracked_object_list in enumerate(self.recorded_objects):
            for point_index, tracked_object in enumerate(tracked_object_list):
                if (
                    tracked_object.pixel_x is not None
                    and tracked_object.pixel_y is not None
                ):
                    out[frame_index, point_index, 0] = tracked_object.pixel_x
                    out[frame_index, point_index, 1] = tracked_object.pixel_y
                    out[frame_index, point_index, 2] = 0.0
        self.recorded_objects_array = out
        return self.recorded_objects_array
