# Real-Time Object Detection for Security Monitoring

YOLOv8 inference on live video with OpenCV, plus train and export (ONNX / INT8 TFLite) entry points in `detect.py`.

```bash
pip install ultralytics opencv-python
python detect.py --source 0
```

**Status:** untested here. It needs a camera/video and a model download that this repo's CI does not have. No mAP or speed-up numbers are claimed; train on your own labelled dataset (`data.yaml`) and report your own measured results.
