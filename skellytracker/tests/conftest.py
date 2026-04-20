from typing import Optional

import numpy as np
import pytest

from skellytracker.system.default_paths import FIGSHARE_CHARUCO_TEST_IMAGE_URL
from skellytracker.utilities.download_test_image import download_test_image

# Lazy caches so running a single file (e.g. test_apriltag_tracker.py) does not
# hit the network unless a test actually requests these fixtures.
_test_image_cache: Optional[np.ndarray] = None
_charuco_test_image_cache: Optional[np.ndarray] = None


@pytest.fixture()
def test_image():
    global _test_image_cache
    if _test_image_cache is None:
        _test_image_cache = download_test_image()
    return _test_image_cache


@pytest.fixture
def charuco_test_image():
    global _charuco_test_image_cache
    if _charuco_test_image_cache is None:
        _charuco_test_image_cache = download_test_image(
            test_image_url=FIGSHARE_CHARUCO_TEST_IMAGE_URL
        )
    return _charuco_test_image_cache
