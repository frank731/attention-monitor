import cv2
import dotenv
from typing import Dict, List
import numpy as np
import threading
from detection.face_detector import FaceDetector
from analysis.attention_classifier import AttentionClassifier, AttentionStatus
from utils.visualization import draw_detections
from time import time

dotenv.load_dotenv()

class AsyncAttentionMonitor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.names_to_filepath = self.face_detector.load_reference_images("class_images")
        self.attention_classifier = AttentionClassifier()
        self.last_time = time()

        # store frames of distraction/focus by name
        self.stats: Dict[str, List[AttentionStatus]] = {}
    
    def classify_person(self, face: np.ndarray, name: str):
        cur_time = time()
        
        result = self.attention_classifier.classify_attention(face)
        if name not in self.stats:
            self.stats[name] = []
        
        self.stats[name].append(result)
    
    def scale_bbox(self, bbox, screen_height, scale_x=3):
        x, y, w, h = bbox
        new_w = int(w * scale_x)
        new_x = int(x - (new_w - w)/2)
        return np.array([new_x, 0, new_w, screen_height])

    def run(self):
        video_capture = cv2.VideoCapture(1)
        frame_count = 0
        
        while True:
            success, frame = video_capture.read()
            if not success:
                break

            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            screen_height = frame.shape[0]
            
            faces_bounding, names = self.face_detector.detect_faces(gray_frame, rgb_frame)
            classification_bboxes = []
            for bounding in faces_bounding:
                classification_bboxes.append(self.scale_bbox(bounding, screen_height))

            extracted_faces = self.face_detector.extract_face_images(frame, classification_bboxes)

            if frame_count % 10 == 0 and extracted_faces:
                for extracted_face, name in zip(extracted_faces, names):
                    thread = threading.Thread(target=self.classify_person,
                        args=(extracted_face, name)
                    )
                    thread.start()

            draw_detections(frame, faces_bounding, names)
            cv2.imshow('Attention Monitor', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

            frame_count += 1

        video_capture.release()
        cv2.destroyAllWindows()

def main():
    monitor = AsyncAttentionMonitor()
    monitor.run()

if __name__ == "__main__":
    main()
