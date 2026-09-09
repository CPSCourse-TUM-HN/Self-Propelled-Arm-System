import os
import sys
import time

import cv2
import numpy as np

from .camera_geometry import RAW_FRAME_SPACE


JETSON_INFERENCE_ROOT = "/workspace/jetson-inference"


def setup_jetson_inference_paths(root=JETSON_INFERENCE_ROOT):
    paths = [
        os.path.join(root, "build/aarch64/lib/python/3.6"),
        os.path.join(root, "python/examples"),
    ]
    for path in paths:
        if path not in sys.path:
            sys.path.insert(0, path)

    lib_path = os.path.join(root, "build/aarch64/lib")
    os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")


def summarize_region(depth_array, x1_ratio=0.4, y1_ratio=0.4, x2_ratio=0.6, y2_ratio=0.6):
    height, width = depth_array.shape[:2]
    x1 = max(0, min(width - 1, int(width * x1_ratio)))
    x2 = max(x1 + 1, min(width, int(width * x2_ratio)))
    y1 = max(0, min(height - 1, int(height * y1_ratio)))
    y2 = max(y1 + 1, min(height, int(height * y2_ratio)))

    region = depth_array[y1:y2, x1:x2]
    finite = region[np.isfinite(region)]
    if finite.size == 0:
        return {
            "mean": 0.0,
            "min": 0.0,
            "max": 0.0,
            "count": 0,
            "region": (x1, y1, x2, y2),
        }

    return {
        "mean": float(np.mean(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "count": int(finite.size),
        "region": (x1, y1, x2, y2),
    }


class DepthCamera(object):
    depth_enabled = True

    def __init__(self, width=320, height=240, network="fcn-mobilenet", rectifier=None):
        setup_jetson_inference_paths()

        from jetbot import Camera
        from jetson_inference import depthNet
        from jetson_utils import cudaDeviceSynchronize, cudaFromNumpy, cudaToNumpy

        self.width = width
        self.height = height
        self.network = network
        self.rectifier = rectifier
        self._camera_cls = Camera
        self._cuda_from_numpy = cudaFromNumpy
        self._cuda_sync = cudaDeviceSynchronize

        print("[depth] loading network {}".format(network))
        self.net = depthNet(network)
        self.depth_field = self.net.GetDepthField()
        self.depth_array = cudaToNumpy(self.depth_field)
        self.camera = None

    def start(self, warmup_frames=2, max_warmup_attempts=20):
        print("[camera] starting JetBot Camera {}x{}".format(self.width, self.height))
        try:
            self.camera = self._camera_cls.instance(width=self.width, height=self.height)
        except RuntimeError:
            self.camera = None
            print("[camera] failed to initialize JetBot Camera")
            print("[camera] common fixes: stop other notebook kernels, restart this kernel, or reboot the board")
            raise
        time.sleep(1.0)

        done = 0
        attempts = 0
        try:
            while done < warmup_frames and attempts < max_warmup_attempts:
                attempts += 1
                frame = self.read_frame()
                if frame is None:
                    print("[warmup] no camera frame ({}/{})".format(attempts, max_warmup_attempts))
                    time.sleep(0.2)
                    continue
                self.process_frame(frame)
                done += 1
                print("[warmup] frame {}/{} processed".format(done, warmup_frames))
                time.sleep(0.1)
        except Exception:
            self.stop()
            raise

        if done < warmup_frames:
            self.stop()
            raise RuntimeError("camera warmup failed: no usable frame after {} attempts".format(attempts))

    def stop(self):
        if self.camera is None:
            return
        try:
            self.camera.stop()
            time.sleep(0.5)
            print("[camera] stopped")
        finally:
            self.camera = None

    def read_frame(self):
        if self.camera is None:
            return None
        return self.camera.value

    def prepare_frame(self, frame, frame_space=RAW_FRAME_SPACE):
        if self.rectifier is not None and frame_space == RAW_FRAME_SPACE:
            return self.rectifier.rectify(frame)
        return frame

    def process_frame(self, frame, frame_space=RAW_FRAME_SPACE):
        prepared = self.prepare_frame(frame, frame_space)
        rgb = cv2.cvtColor(prepared, cv2.COLOR_BGR2RGB)
        cuda_img = self._cuda_from_numpy(rgb)
        self.net.Process(cuda_img)
        self._cuda_sync()
        return self.depth_array

    def observe(self, region=(0.4, 0.4, 0.6, 0.6), frame=None, frame_space=RAW_FRAME_SPACE):
        if frame is None:
            frame = self.read_frame()
        if frame is None:
            return None
        depth = self.process_frame(frame, frame_space=frame_space)
        stats = summarize_region(depth, region[0], region[1], region[2], region[3])
        stats["timestamp"] = time.time()
        return stats


class CameraOnly(object):
    depth_enabled = False

    def __init__(self, width=320, height=240):
        from jetbot import Camera

        self.width = int(width)
        self.height = int(height)
        self._camera_cls = Camera
        self.camera = None

    def start(self, warmup_frames=2, max_warmup_attempts=20):
        print("[camera] starting camera-only mode {}x{} (DepthNet disabled)".format(self.width, self.height))
        self.camera = self._camera_cls.instance(width=self.width, height=self.height)
        time.sleep(1.0)
        attempts = 0
        frames = 0
        while frames < warmup_frames and attempts < max_warmup_attempts:
            attempts += 1
            if self.read_frame() is None:
                time.sleep(0.2)
                continue
            frames += 1
            print("[warmup] camera-only frame {}/{} ready".format(frames, warmup_frames))
            time.sleep(0.1)
        if frames < warmup_frames:
            self.stop()
            raise RuntimeError("camera-only warmup failed after {} attempts".format(attempts))

    def read_frame(self):
        if self.camera is None:
            return None
        return self.camera.value

    def stop(self):
        if self.camera is None:
            return
        try:
            self.camera.stop()
            time.sleep(0.5)
            print("[camera] camera-only stopped")
        finally:
            self.camera = None
