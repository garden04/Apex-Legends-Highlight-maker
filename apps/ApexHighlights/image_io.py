"""Bounded retries for transient Windows image-file sharing violations."""
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np


def retry_access(operation, check_cancel=lambda: None, timeout=8.0):
    deadline = time.monotonic() + timeout
    while True:
        check_cancel()
        try:
            return operation()
        except PermissionError:
            if time.monotonic() >= deadline:
                raise
            time.sleep(0.05)


def read_completed_image(path, check_cancel=lambda: None):
    # FFmpeg publishes completed files by atomic rename, but a Windows handle
    # can still temporarily deny opening that published name. Never skip it.
    data = retry_access(lambda: Path(path).read_bytes(), check_cancel)
    frame = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None:
        raise ValueError(f'추출된 이미지를 읽지 못했습니다: {Path(path).name}')
    return frame


class ImageTemporaryDirectory(tempfile.TemporaryDirectory):
    def cleanup(self):
        # The producer is stopped and its handles closed before cleanup.
        retry_access(super().cleanup)
