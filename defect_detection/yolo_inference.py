import torch
import numpy as np
import cv2
from time import time


class YOLODetector:
    """
    YOLOv5 Object Detector for video input using OpenCV.
    """

    def __init__(self, capture_source=0, model_path='best.pt', yolov5_repo='./yolov5'):
        self.capture_source = capture_source
        self.model_path = model_path
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = self.load_model(yolov5_repo)
        self.classes = self.model.names
        print(f"[INFO] Model loaded on {self.device}")

    def load_model(self, yolov5_repo):
        """
        Loads YOLOv5 custom model from a local path.
        """
        model = torch.hub.load(yolov5_repo, 'custom', path=self.model_path, source='local', trust_repo=True)
        model.to(self.device)
        model.eval()
        return model

    def get_video_capture(self):
        """
        Initializes OpenCV video stream.
        """
        return cv2.VideoCapture(self.capture_source)

    def score_frame(self, frame):
        """
        Predict objects in a given frame.
        """
        results = self.model([frame])
        labels = results.xyxyn[0][:, -1]
        cords = results.xyxyn[0][:, :-1]
        return labels, cords

    def class_to_label(self, label_id):
        """
        Convert numeric label to string label.
        """
        return self.classes[int(label_id)]

    def plot_boxes(self, frame, labels, cords, conf_thres=0.3):
        """
        Draw bounding boxes and class labels on the frame.
        """
        x_shape, y_shape = frame.shape[1], frame.shape[0]
        for i, row in enumerate(cords):
            if row[4] >= conf_thres:
                x1, y1 = int(row[0] * x_shape), int(row[1] * y_shape)
                x2, y2 = int(row[2] * x_shape), int(row[3] * y_shape)
                label = self.class_to_label(labels[i])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f'{label}', (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return frame

    def __call__(self):
        """
        Run the object detection on video stream.
        """
        cap = self.get_video_capture()
        assert cap.isOpened(), "[ERROR] Video source not accessible"

        while True:
            ret, frame = cap.read()
            if not ret:
                print("[INFO] End of video stream or error.")
                break

            frame = cv2.resize(frame, (640, 640))
            start_time = time()
            labels, cords = self.score_frame(frame)
            frame = self.plot_boxes(frame, labels, cords)
            fps = 1 / (time() - start_time)

            cv2.putText(frame, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.imshow("YOLOv5 Detection", frame)

            if cv2.waitKey(1) & 0xFF == 27:  # ESC key to exit
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    # Example usage:
    detector = YOLODetector(
        capture_source='/Test data/close.mp4',                        # or ='rtsp://192.168.1.101:8554/test'incase of Blueye ROV camera.
        model_path='detection model/yolov5/best.pt',        # your custom .pt file
        yolov5_repo='yolov5'             # path to local yolov5 repo
    )
    detector()

