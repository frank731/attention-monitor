import cv2
import math
import dotenv
from typing import Dict, List, Tuple
from PIL import Image, ImageTk
from dataclasses import dataclass
import numpy as np
import threading
from detection.face_detector import FaceDetector
from analysis.attention_classifier import AttentionClassifier, AttentionStatus, STATUS_TO_TEXT
from gui.gui import create_ui
from utils.visualization import draw_detections
from time import time
from tkinter import Tk, Canvas, Button, PhotoImage

dotenv.load_dotenv()


@dataclass
class AttentionTimes:
    focused: float
    distracted: float
    on_phone: float


@dataclass
class AttentionStrings:
    focused_str: str
    distracted_str: str
    on_phone_str: str


def format_seconds(seconds):
    """Convert seconds into human readable string, showing only needed units."""
    if seconds < 60:
        return f"{math.floor(seconds)}s"

    minutes, seconds = divmod(seconds, 60)
    if minutes < 60:
        return f"{math.floor(minutes)}m {math.floor(seconds):02}s"

    hours, minutes = divmod(minutes, 60)
    if hours < 24:
        return f"{math.floor(hours)}h {math.floor(minutes):02}m"

    days, hours = divmod(hours, 24)
    return f"{math.floor(days)}d {math.floor(hours)}h {math.floor(minutes)}m"


class AsyncAttentionMonitor:
    def __init__(self):
        self.face_detector = FaceDetector()
        self.names_to_filepath = self.face_detector.load_reference_images(
            "class_images"
        )
        self.attention_classifier = AttentionClassifier()

        # store frames of distraction/focus by name
        self.stats: Dict[str, List[Tuple[AttentionStatus, float]]] = {}
        for name in self.names_to_filepath.keys():
            self.stats[name] = [(AttentionStatus.Focused, time())]

        self.image_tk = None
        self.window_tk = Tk()
        self.window_tk.geometry("1440x960")
        self.window_tk.configure(bg="#FFFFFF")

        self.canvas_tk = Canvas(
            self.window_tk,
            bg="#FFFFFF",
            height=960,
            width=1440,
            bd=0,
            highlightthickness=0,
            relief="ridge",
        )

        self.video_capture = cv2.VideoCapture(1)

        self.frame_count = 0
        self._images = []

    def create_tkinter(self):
        (
            self.image_tk,
            self.cumulative,
            self.david_time,
            self.jonathan_time,
            self.frank_time,
            self.djf_status,
            self._images,
        ) = create_ui(self.canvas_tk)

        self.name_mapping = {
            "Jonathan Li": (self.jonathan_time, self.djf_status[0]),
            "David Li": (self.david_time, self.djf_status[1]),
            "Frank Li": (self.frank_time, self.djf_status[2]),
        }

    def classify_person(self, face: np.ndarray, name: str):
        cur_time = time()
        result = self.attention_classifier.classify_attention(face)
        if name not in self.stats:
            self.stats[name] = []
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
            if last_status == AttentionStatus.Focused:
                focused += time - last_time
            elif last_status == AttentionStatus.Distracted:
                distracted += time - last_time
            elif last_status == AttentionStatus.OnPhone:
                on_phone += time - last_time
            last_status = status
            last_time = time

        return AttentionTimes(focused, distracted, on_phone)

    def format_attention_as_str(self, times: AttentionTimes, append_names: bool = False) -> AttentionStrings:
        return AttentionStrings(
            focused_str=format_seconds(times.focused) + (" attentive" if append_names else ""),
            distracted_str=format_seconds(times.distracted) + (" distracted" if append_names else ""),
            on_phone_str=format_seconds(times.on_phone) + (" on phone" if append_names else ""),
        )

    def get_cumulative_attention(self) -> AttentionTimes:
        names = list(self.stats.keys())
        total_attention = AttentionTimes(0, 0, 0)

        for name in names:
            times = self.count_attention(name)
            total_attention.focused += times.focused
            total_attention.distracted += times.distracted
            total_attention.on_phone += times.on_phone

        return total_attention

    def scale_bbox(self, bbox, screen_height, screen_width, scale_x=3):
        x, y, w, h = bbox

        new_w = int(w * scale_x)
        new_x = int(x - (new_w - w) / 2)

        # Handle left edge
        if new_x < 0:
            new_w += new_x  # Reduce width by overflow amount
            new_x = 0

        # Handle right edge
        if new_x + new_w > screen_width:
            new_w = screen_width - new_x

        return np.array([new_x, 0, new_w, screen_height])

    def run(self):
        success, frame = self.video_capture.read()
        if not success:
            return

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        screen_height = frame.shape[0]
        screen_width = frame.shape[1]

        faces_bounding, names = self.face_detector.detect_faces(gray_frame, rgb_frame)
        classification_bboxes = []
        for bounding in faces_bounding:
            classification_bboxes.append(
                self.scale_bbox(bounding, screen_height, screen_width)
            )

        extracted_faces = self.face_detector.extract_face_images(
            frame, classification_bboxes
        )

        if self.frame_count % 10 == 0 and extracted_faces:
            for extracted_face, name in zip(extracted_faces, names):
                thread = threading.Thread(
                    target=self.classify_person, args=(extracted_face, name)
                )
                thread.start()

        statuses = [
            (
                self.stats[name][-1][0]
                if name in self.stats and len(self.stats[name]) > 0
                else AttentionStatus.Focused
            )
            for name in names
        ]
        old_frame = frame
        frame = cv2.resize(frame, None, fx=0.53, fy=0.53, interpolation=cv2.INTER_CUBIC)

        resize_factor_x = frame.shape[1] / old_frame.shape[1]
        resize_factor_y = frame.shape[0] / old_frame.shape[0]

        adjusted_faces_bounding = [
            (
                int(bbox[0] * resize_factor_x),
                int(bbox[1] * resize_factor_y),
                int(bbox[2] * resize_factor_x),
                int(bbox[3] * resize_factor_y),
            )
            for bbox in faces_bounding
        ]


        draw_detections(frame, adjusted_faces_bounding, names, statuses)
        # cv2.imshow("Attention Monitor", frame)

        self.frame_count += 1

        for name in self.stats:
            attn = self.format_attention_as_str(self.count_attention(name))
            # print(
            #     f"{name}: {attn.focused_str} focused, {attn.distracted_str} distracted, {attn.on_phone_str} on phone"
            # )

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.img_update = ImageTk.PhotoImage(Image.fromarray(frame))
        self.canvas_tk.itemconfig(self.image_tk, image=self.img_update)
        self.canvas_tk.coords(self.image_tk, 908.0, 453.0)

        cumulative_attention = self.format_attention_as_str(self.get_cumulative_attention())
        self.update_ui(cumulative_attention, self.cumulative)

        for person_name in self.names_to_filepath.keys():
            times = self.count_attention(person_name)
            attn = self.format_attention_as_str(times, append_names=True)
            self.update_ui(attn, self.name_mapping[person_name][0])
            last_status = self.stats[person_name][-1][0]
            self.canvas_tk.itemconfig(
                self.name_mapping[person_name][1],
                text=STATUS_TO_TEXT[last_status],
            )

        self.canvas_tk.after(1, self.run)
    
    def update_ui(self, attn: AttentionStrings, ids: List[int]):
        attentive, distracted, on_phone = ids
        self.canvas_tk.itemconfig(
            attentive,
            text=attn.focused_str,
        )
        self.canvas_tk.itemconfig(
            distracted,
            text=attn.distracted_str,
        )
        self.canvas_tk.itemconfig(
            on_phone,
            text=attn.on_phone_str,
        )


def main():
    monitor = AsyncAttentionMonitor()
    monitor.create_tkinter()
    monitor.run()

    monitor.window_tk.mainloop()
    monitor.video_capture.release()


if __name__ == "__main__":
    main()
