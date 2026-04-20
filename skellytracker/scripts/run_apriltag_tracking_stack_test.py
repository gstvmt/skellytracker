"""
Smoke test for AprilTag tracking through ``process_folder_of_videos`` (same stack shape as FreeMoCap).

Install the detector extra, then point at a folder of synchronized videos (one video per camera)::

    pip install -e ".[apriltag]"
    python -m skellytracker.scripts.run_apriltag_tracking_stack_test /path/to/synchronized_videos

Optional: ``--tag-ids 0 1 2 3`` and ``--point-mode corners|center`` must match between
``AprilTagTrackingParams`` and ``april_tag_model_info_for`` (this script keeps them aligned).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "synchronized_videos",
        type=Path,
        help="Folder of synchronized camera videos (e.g. FreeMoCap ``synchronized_videos``).",
    )
    parser.add_argument(
        "--tag-ids",
        type=int,
        nargs="+",
        default=[0, 1, 2, 3],
        help="AprilTag ids to reserve slots for (stable order in output tensor).",
    )
    parser.add_argument(
        "--point-mode",
        choices=("corners", "center", "both"),
        default="corners",
        help="Track four corners per tag or a single center per tag.",
    )
    parser.add_argument(
        "--num-processes",
        type=int,
        default=1,
        help="Parallel processes (1 = sequential).",
    )
    parser.add_argument(
        "--nthreads",
        type=int,
        default=1,
        help="Number of threads for the AprilTag detector (reducing to 1 often prevents segfaults).",
    )
    args = parser.parse_args(argv)

    sync_path = args.synchronized_videos.expanduser().resolve()
    if not sync_path.is_dir():
        print(f"Not a directory: {sync_path}", file=sys.stderr)
        return 1

    tag_ids = tuple(args.tag_ids)
    try:
        from skellytracker.process_folder_of_videos import process_folder_of_videos
        from skellytracker.trackers.apriltag_tracker.apriltag_model_info import (
            april_tag_model_info_for,
        )
        from skellytracker.trackers.apriltag_tracker.apriltag_tracking_params import (
            AprilTagTrackingParams,
        )
    except ImportError as exc:
        print(exc, file=sys.stderr)
        return 1

    tracking_params = AprilTagTrackingParams(
        tag_ids=tag_ids,
        point_mode=args.point_mode,
        num_processes=args.num_processes,
        nthreads=args.nthreads,
    )
    model_info = april_tag_model_info_for(
        tag_ids=tracking_params.tag_ids,
        point_mode=tracking_params.point_mode,
    )

    print(f"Model: {model_info.name}, points={model_info.num_tracked_points}")
    print(f"Videos folder: {sync_path}")

    array_result = process_folder_of_videos(
        model_info=model_info,
        tracking_params=tracking_params,
        synchronized_video_path=sync_path,
        output_folder_path=sync_path.parent / "output_data" / "raw_data",
        num_processes=tracking_params.num_processes,
    )

    print(f"Output array shape (numCams, numFrames, numPoints, xyz): {array_result.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
