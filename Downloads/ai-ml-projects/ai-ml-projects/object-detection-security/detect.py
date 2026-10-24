"""YOLOv8 detection on a video file or webcam with OpenCV overlay.

NOT executed in this repo's tests (needs `pip install ultralytics opencv-python`, a model
download and a video source).

    python detect.py --source 0                       # webcam
    python detect.py --source clip.mp4 --weights best.pt --conf 0.4
    python detect.py --train data.yaml                # train on your labelled dataset
    python detect.py --export best.pt                 # export ONNX (add --int8 for quantised TFLite)
"""
import argparse

import cv2
from ultralytics import YOLO


def run(source, weights, conf):
    model = YOLO(weights)
    cap = cv2.VideoCapture(int(source) if str(source).isdigit() else source)
    while cap.isOpened():
        ok, frame = cap.read()
        if not ok:
            break
        res = model(frame, conf=conf, verbose=False)[0]
        cv2.imshow("detections (q to quit)", res.plot())
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0")
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--train", metavar="DATA_YAML")
    ap.add_argument("--export", metavar="WEIGHTS")
    ap.add_argument("--int8", action="store_true")
    a = ap.parse_args()
    if a.train:
        YOLO(a.weights).train(data=a.train, epochs=50, imgsz=640)
    elif a.export:
        YOLO(a.export).export(format="tflite" if a.int8 else "onnx", int8=a.int8)
    else:
        run(a.source, a.weights, a.conf)
