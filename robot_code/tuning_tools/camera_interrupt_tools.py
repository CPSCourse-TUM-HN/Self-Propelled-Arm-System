"""Interactive camera and DepthNet lifecycle diagnostics for the tuning notebook."""

from __future__ import print_function

import gc
import threading
import time
import traceback

import cv2
import ipywidgets as widgets
from IPython.display import display

from demo_core import load_config
from demo_core.depth_vision import CameraOnly, setup_jetson_inference_paths, summarize_region
from demo_core.camera_geometry import CameraRectifier


class CameraInterruptUI(object):
    """Independent CameraOnly and DepthNet lifecycle interruption test."""

    def __init__(self):
        config = load_config()
        camera = config.section("camera")
        self.depth_rectification_enabled = bool(camera.get("depth_rectification_enabled", False))
        self.rectifier = CameraRectifier(config) if self.depth_rectification_enabled else None
        self.width = widgets.IntText(value=int(camera["width"]), description="width")
        self.height = widgets.IntText(value=int(camera["height"]), description="height")
        self.network = widgets.Text(value=str(camera.get("depth_network", "fcn-mobilenet")), description="depth_network")
        self.output = widgets.Output(layout={"border": "1px solid #ccc", "height": "430px", "overflow_y": "auto"})
        self.image = widgets.Image(format="jpeg", width=int(camera["width"]), height=int(camera["height"]))
        self.camera = None
        self.depth_net = None
        self.depth_array = None
        self.cuda_from_numpy = None
        self.cuda_sync = None
        self.model_loading = False
        self.unload_after_load = False
        self.worker = None
        self.lock = threading.RLock()

    def _run_async(self, name, operation):
        if self.worker is not None and self.worker.is_alive():
            with self.output:
                print("[interrupt-test] worker busy; stop/release remains available")
            return

        def target():
            with self.output:
                print("\n>>> {}".format(name))
                try:
                    result = operation()
                    print("[result] {}".format(result))
                except Exception as exc:
                    print("[error] {}".format(exc))
                    traceback.print_exc()

        self.worker = threading.Thread(target=target)
        self.worker.daemon = True
        self.worker.start()

    def start_camera(self):
        def operation():
            with self.lock:
                if self.camera is not None and self.camera.camera is not None:
                    return "camera already started"
                self.camera = CameraOnly(width=int(self.width.value), height=int(self.height.value))
                camera = self.camera
            camera.start(warmup_frames=2)
            frame = camera.read_frame()
            return {"camera": "started", "shape": None if frame is None else frame.shape}

        self._run_async("start_camera", operation)

    def capture(self):
        def operation():
            with self.lock:
                camera = self.camera
            if camera is None:
                raise RuntimeError("camera is not started")
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("camera returned no frame")
            ok, encoded = cv2.imencode(".jpg", frame)
            if ok:
                self.image.value = encoded.tobytes()
            return {"shape": frame.shape, "dtype": str(frame.dtype)}

        self._run_async("capture", operation)

    def load_model(self):
        def operation():
            with self.lock:
                if self.depth_net is not None:
                    return "DepthNet already loaded"
                self.model_loading = True
                self.unload_after_load = False
            setup_jetson_inference_paths()
            from jetson_inference import depthNet
            from jetson_utils import cudaDeviceSynchronize, cudaFromNumpy, cudaToNumpy

            try:
                net = depthNet(str(self.network.value))
                field = net.GetDepthField()
                array = cudaToNumpy(field)
                with self.lock:
                    if self.unload_after_load:
                        print("[depth] unload was requested during load; discarding model")
                    else:
                        self.depth_net = net
                        self.depth_array = array
                        self.cuda_from_numpy = cudaFromNumpy
                        self.cuda_sync = cudaDeviceSynchronize
                        return "DepthNet loaded"
                del array
                del field
                del net
                gc.collect()
                return "DepthNet loaded then unloaded"
            finally:
                with self.lock:
                    self.model_loading = False

        self._run_async("load_depthnet", operation)

    def unload_model(self):
        with self.lock:
            self.unload_after_load = True
            net = self.depth_net
            self.depth_net = None
            self.depth_array = None
            self.cuda_from_numpy = None
            self.cuda_sync = None
            loading = self.model_loading
        if net is not None:
            del net
        gc.collect()
        with self.output:
            print("[depth] unload requested loading={} model_present={}".format(loading, net is not None))

    def test_depth(self):
        def operation():
            with self.lock:
                camera = self.camera
                net = self.depth_net
                depth_array = self.depth_array
                cuda_from_numpy = self.cuda_from_numpy
                cuda_sync = self.cuda_sync
            if camera is None or camera.camera is None:
                raise RuntimeError("camera is not started")
            if net is None:
                raise RuntimeError("DepthNet is not loaded")
            frame = camera.read_frame()
            if frame is None:
                raise RuntimeError("camera returned no frame")
            if self.rectifier is not None:
                frame = self.rectifier.rectify(frame)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            net.Process(cuda_from_numpy(rgb))
            cuda_sync()
            stats = summarize_region(depth_array, 0.4, 0.4, 0.6, 0.6)
            stats["frame_space"] = "rectified" if self.rectifier is not None else "raw"
            return stats

        self._run_async("test_depth_single_frame", operation)

    def stop_release_camera(self):
        with self.lock:
            camera = self.camera
            self.camera = None
        if camera is not None:
            try:
                camera.stop()
            finally:
                del camera
                gc.collect()
        with self.output:
            print("[camera] stop -> release complete")

    def stop_release_all(self):
        self.stop_release_camera()
        self.unload_model()

    def clear_log(self):
        self.output.clear_output()

    def _button(self, label, callback, style=""):
        button = widgets.Button(description=label, button_style=style, layout=widgets.Layout(width="160px"))
        button.on_click(lambda _: callback())
        return button

    def display(self):
        start = self._button("Start Camera", self.start_camera, "success")
        capture = self._button("Capture", self.capture)
        load = self._button("Load DepthNet", self.load_model, "success")
        unload = self._button("Unload DepthNet", self.unload_model, "warning")
        test = self._button("Test Depth Frame", self.test_depth)
        stop_camera = self._button("STOP + RELEASE CAMERA", self.stop_release_camera, "danger")
        stop_all = self._button("STOP + RELEASE ALL", self.stop_release_all, "danger")
        clear = self._button("Clear Log", self.clear_log)
        display(widgets.VBox([
            widgets.HBox([self.width, self.height, self.network]),
            widgets.HBox([start, capture, stop_camera]),
            widgets.HBox([load, unload, test]),
            widgets.HBox([stop_all, clear]),
            self.image,
            self.output,
        ]))
        print("[interrupt-test] ready; CameraOnly and DepthNet have separate lifecycles")


def build_ui():
    panel = CameraInterruptUI()
    panel.display()
    return panel
