"""Video decoding + the training-time crop for one camera, run in a subprocess per view."""
import cv2
import numpy as np
import torch

from multismpl.inference.network import image_preprocessor


def video_worker(path, cam, n, queue):
    """Decode + crop one camera's video in a subprocess; returns the crop intrinsics with frame 0."""
    try:
        torch.set_num_threads(1)
        cv2.setNumThreads(1)
        pre = image_preprocessor()
        cap = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG, [cv2.CAP_PROP_N_THREADS, 2])
        for f in range(n):
            ok, im = cap.read()
            if not ok:
                raise RuntimeError(f'{path}: decode failed at frame {f}')
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
            out = pre.process_one_image(im, None, cam['cam_ext'][:3], cam['cam_int'], np.array(im.shape[:2]), np.array([518, 518]))
            K = np.asarray(out[3], np.float64) if f == 0 else None
            queue.put((f, np.ascontiguousarray(out[0]), K))
        cap.release()
    except Exception as exc:  # noqa: BLE001
        queue.put((-1, str(exc), None))
