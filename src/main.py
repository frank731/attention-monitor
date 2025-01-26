import cv2
import math
import dotenv
from typing import Dict, List, Tuple
from dataclasses import dataclass
import numpy as np
import threading
from detection.face_detector import FaceDetector
from analysis.attention_classifier import AttentionClassifier, AttentionStatus
from utils.visualization import draw_detections
from time import time

dotenv.load_dotenv()

@dataclass
class AttentionTimes:
    focused: float
    distracted: float
    on_phone: float

@dataclass
class AttentionStrings:
    focused: str
    distracted: str
    on_phone: str

def format_seconds(seconds):
   """Convert seconds into human readable string, showing only needed units."""
   if seconds < 60:
       return f"{math.floor(seconds)}s"
   
   minutes, seconds = divmod(seconds, 60)
   if minutes < 60:
       return f"{math.floor(minutes)}m"
       
   hours, minutes = divmod(minutes, 60)
   if hours < 24:
       return f"{math.floor(hours)}h {math.floor(minutes)}m"
       
   days, hours = divmod(hours, 24)
   return f"{math.floor(days)}d {math.floor(hours)}h {math.floor(minutes)}m"

class AsyncAttentionMonitor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.names_to_filepath = self.face_detector.load_reference_images("class_images")
        self.attention_classifier = AttentionClassifier()

        # store frames of distraction/focus by name
        self.stats: Dict[str, List[Tuple[AttentionStatus, float]]] = {}
    
    def classify_person(self, face: np.ndarray, name: str):
        cur_time = time()
        result = self.attention_classifier.classify_attention(face)
        if name not in self.stats:
            self.stats[name] = []
        print(result)
        self.stats[name].append((result, cur_time))

    def count_attention(self, name: str) -> AttentionTimes:
        focused = 0
        distracted = 0
        on_phone = 0
        last_status = None
        last_time = None
        for status, time in self.stats[name]:
            if last_status is None:
                last_status = status
                last_time = time
                continue
            if last_status == AttentionStatus.FOCUSED:
                focused += time - last_time
            elif last_status == AttentionStatus.DISTRACTED:
                distracted += time - last_time
            elif last_status == AttentionStatus.ON_PHONE:
                on_phone += time - last_time
            last_status = status
            last_time = time

        return AttentionTimes(focused, distracted, on_phone)
    
    def format_attention_as_str(self, times: AttentionTimes) -> AttentionStrings:
        return AttentionStrings(
            focused=format_seconds(times.focused),
            distracted=format_seconds(times.distracted),
            on_phone=format_seconds(times.on_phone)
        )
    
    def scale_bbox(self, bbox, screen_height, screen_width, scale_x=3):
        x, y, w, h = bbox
        
        new_w = int(w * scale_x)
        new_x = int(x - (new_w - w)/2)
        
        # Handle left edge
        if new_x < 0:
            new_w += new_x  # Reduce width by overflow amount
            new_x = 0
            
        # Handle right edge
        if new_x + new_w > screen_width:
            new_w = screen_width - new_x

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
            screen_width = frame.shape[1]
            
            faces_bounding, names = self.face_detector.detect_faces(gray_frame, rgb_frame)
            classification_bboxes = []
            for bounding in faces_bounding:
                classification_bboxes.append(self.scale_bbox(bounding, screen_height, screen_width))

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
