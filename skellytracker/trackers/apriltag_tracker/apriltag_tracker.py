from __future__ import annotations

from typing import Dict, Literal, Tuple

import cv2
import numpy as np

from skellytracker.trackers.apriltag_tracker.apriltag_recorder import AprilTagRecorder
from skellytracker.trackers.apriltag_tracker.apriltag_model_info import april_tag_landmark_names
from skellytracker.trackers.base_tracker.base_tracker import BaseTracker
from skellytracker.trackers.base_tracker.tracked_object import TrackedObject


class AprilTagTracker(BaseTracker):
    """
    AprilTag detection (``pupil-apriltags``) with a fixed set of ``tag_ids`` and stable point ordering.

    Install optional dependency: ``pip install skellytracker[apriltag]``.
    """

    def __init__(
        self,
        tag_ids: Tuple[int, ...],
        point_mode: Literal["corners", "center", "both"] = "corners",
        families: str = "tag36h11",
        nthreads: int = 4,
        quad_decimate: float = 1.0,
    ):
        try:
            from pupil_apriltags import Detector
        except ImportError as exc:  # pragma: no cover - exercised when extra not installed
            raise ImportError(
                "AprilTagTracker requires `pupil-apriltags`. "
                "Install with: pip install 'skellytracker[apriltag]'"
            ) from exc

        self._tag_ids = tuple(tag_ids)
        self._point_mode = point_mode
        tracked_object_names = list(april_tag_landmark_names(self._tag_ids, point_mode))
        self.tracked_object_names = tracked_object_names
        super().__init__(
            recorder=AprilTagRecorder(num_tracked_points=len(tracked_object_names)),
            tracked_object_names=tracked_object_names,
        )
        self._detector = Detector(
            families=families,
            nthreads=nthreads,
            quad_decimate=quad_decimate,
        )
        self._last_tag_detections: list = []

    def cleanup(self) -> None:
        # pupil-apriltags uses ctypes; letting Detector.__del__ run during interpreter
        # shutdown can segfault (unload order vs OpenCV). Drop the native detector while
        # the runtime is still fully up — BaseTracker.process_video calls this before return.
        from logging import getLogger
        _log = getLogger(__name__)
        _log.debug("Cleaning up AprilTagTracker...")
        self._last_tag_detections = []
        if hasattr(self, "_detector") and self._detector is not None:
            _log.debug("Dropping pupil_apriltags Detector...")
            self._detector = None
            _log.debug("Detector dropped.")
        super().cleanup()

    def reinitialize_tracked_objects(self) -> None:
        for name in self.tracked_object_names:
            self.tracked_objects[name] = TrackedObject(object_id=name)

    def process_image(self, image: np.ndarray, **kwargs) -> Dict[str, TrackedObject]:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        tags = self._detector.detect(gray)
        self._last_tag_detections = list(tags)

        self.reinitialize_tracked_objects()
        by_id = {int(t.tag_id): t for t in tags}

        for tag_id in self._tag_ids:
            det = by_id.get(tag_id)
            if self._point_mode in ["center", "both"]:
                name = f"tag{tag_id}_center"
                if det is not None:
                    self.tracked_objects[name].pixel_x = float(det.center[0])
                    self.tracked_objects[name].pixel_y = float(det.center[1])
            if self._point_mode in ["corners", "both"]:
                for corner_index in range(4):
                    name = f"tag{tag_id}_corner{corner_index}"
                    if det is not None:
                        self.tracked_objects[name].pixel_x = float(
                            det.corners[corner_index, 0]
                        )
                        self.tracked_objects[name].pixel_y = float(
                            det.corners[corner_index, 1]
                        )

        self.annotated_image = self.annotate_image(
            image=image, tracked_objects=self.tracked_objects
        )
        return self.tracked_objects

    def annotate_image(
        self, image: np.ndarray, tracked_objects: Dict[str, TrackedObject], **kwargs
    ) -> np.ndarray:
        annotated = image.copy()
        for tag in self._last_tag_detections:
            for i in range(4):
                pt1 = (int(tag.corners[i][0]), int(tag.corners[i][1]))
                pt2 = (
                    int(tag.corners[(i + 1) % 4][0]),
                    int(tag.corners[(i + 1) % 4][1]),
                )
                cv2.line(annotated, pt1, pt2, (0, 255, 0), 2)
            center = (int(tag.center[0]), int(tag.center[1]))
            cv2.circle(annotated, center, 5, (0, 0, 255), -1)
            cv2.putText(
                annotated,
                str(tag.tag_id),
                (center[0] - 10, center[1] - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                2,
            )

        for obj in tracked_objects.values():
            if obj.pixel_x is not None and obj.pixel_y is not None:
                cv2.drawMarker(
                    annotated,
                    (int(obj.pixel_x), int(obj.pixel_y)),
                    (255, 255, 0),
                    markerType=cv2.MARKER_CROSS,
                    markerSize=12,
                    thickness=2,
                )
        return annotated
