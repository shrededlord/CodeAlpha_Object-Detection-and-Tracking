import cv2
from ultralytics import YOLO
from enum import Enum
import numpy as np
from ultralytics.utils.plotting import Annotator, colors
from collections import defaultdict

class ModelVariant(Enum):
    YOLOv8n = "yolov8n.pt"
    YOLOv8s = "yolov8s.pt"
    YOLOv8m = "yolov8m.pt"
    YOLOv8l = "yolov8l.pt"
    YOLOv8x = "yolov8x.pt"

def process_video(input_path: str):
    model = YOLO(ModelVariant.YOLOv8n.value)
    label_names = model.model.names

    tracking_history = defaultdict(lambda: [])

    video_capture = cv2.VideoCapture(input_path)
    assert video_capture.isOpened(), "Error opening video file"

    frame_width, frame_height, frame_rate = (
        int(video_capture.get(prop)) 
        for prop in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT, cv2.CAP_PROP_FPS)
    )

    output_writer = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc(*"mp4v"), frame_rate, (frame_width, frame_height))

    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if ret:
            detection_results = model.track(frame, persist=True, verbose=False)
            bounding_boxes = detection_results[0].boxes.xyxy.cpu()
            if detection_results[0].boxes.id is not None:
                classes = detection_results[0].boxes.cls.cpu().tolist()
                ids = detection_results[0].boxes.id.int().cpu().tolist()
                confidences = detection_results[0].boxes.conf.float().cpu().tolist()
                annotator = Annotator(frame, line_width=2)
                for box, cls, obj_id in zip(bounding_boxes, classes, ids):
                    annotator.box_label(box, color=colors(int(cls), True), label=label_names[int(cls)])
                    track = tracking_history[obj_id]
                    track.append((int((box[0] + box[2]) / 2), int((box[1] + box[3]) / 2)))
                    if len(track) > 30:
                        track.pop(0)
                    points = np.array(track, dtype=np.int32).reshape((-1, 1, 2))
                    cv2.circle(frame, track[-1], 7, colors(int(cls), True), -1)
                    cv2.polylines(frame, [points], isClosed=False, color=colors(int(cls), True), thickness=2)
            output_writer.write(frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            break

    output_writer.release()
    video_capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    process_video("your_video_file.mp4")
