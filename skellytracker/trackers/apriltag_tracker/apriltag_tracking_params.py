from typing import Literal, Tuple

from pydantic import Field

from skellytracker.trackers.base_tracker.base_tracking_params import BaseTrackingParams

DEFAULT_TAG_IDS: Tuple[int, ...] = (0, 1, 2, 3)


class AprilTagTrackingParams(BaseTrackingParams):
    """Parameters for :class:`AprilTagTracker` (must stay aligned with the paired model info)."""

    tag_ids: Tuple[int, ...] = Field(default=DEFAULT_TAG_IDS)
    point_mode: Literal["corners", "center", "both"] = "corners"
    families: str = "tag36h11"
    nthreads: int = 1
    quad_decimate: float = 1.0
