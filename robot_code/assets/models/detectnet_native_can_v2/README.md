# Native DetectNet can detector: SSD MobileNet V2 SSD-Lite

Deployment candidate; TensorRT 7.1 build and camera detection are not yet
verified on the Jetson Nano. The working V1 files remain in
`../detectnet_native_can/` for rollback.

- Model: `can_ssd_mobilenet_v2.onnx`, ONNX opset 11.
- Input: `input_0`, float32 `[1, 3, 300, 300]`.
- Outputs: `scores` `[1, 3000, 2]`, decoded `boxes` `[1, 3000, 4]`.
- Labels: `BACKGROUND`, `can` (class ID 1).
- Existing confidence 0.20 and clustering 0.30 are unchanged.
- Standard DetectNet handles postprocessing; no custom backend is required.
- The graph includes 53 Clip nodes; ONNX checker passes, but this does not
  establish compatibility with the board's TensorRT parser.

## Notebook smoke test

Restart the kernel and run `tuning_tools/dual_can_vague_map_recording_test.ipynb`.
Enable `camera_real` and `live_stream`, leave `record_camera` unchecked, and
click `Start Monitor`. Base and arm are dry-run in this notebook. Confirm the
model-load log names V2 and visible cans have plausible boxes/confidence.
First startup may take time to build the TensorRT engine. Finish with
`STOP + Release`. A successful smoke test establishes basic integration,
not accuracy or performance superiority over V1.

To roll back, restore `detectors.can.model_path` and `labels_path` in
`empirical_parameters.json` to the V1 directory and restart the session.

## Integrity

SHA256 (`can_ssd_mobilenet_v2.onnx`):
`64bdb2a0c79ea057a2a1697d0dd082c1ebc55a4d5ca9aabc173e55b2ce35549a`

Only runtime files are included; the training checkpoint and shell test
scripts are intentionally not copied from the delivery bundle.
